import csv,json
from collections import Counter,defaultdict
import numpy as np
from .scenario import OUT,maps
from patrol_aimpoints.experiment import read


def write(name,rows):
    with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def report():
    paths=sorted((OUT/'cases').glob('*.gz'));assert len(paths)==200
    runs=[];persons=[];alarms=[];branches=[];diff=[];changes=[];examples={};values=defaultdict(list)
    for path in paths:
        d=read(path);meta=d['meta'];p,n=meta['parent'],meta['layout'];tags=d['tags']
        alarms.append(dict(**meta,hash=d['hash']))
        records={(r['row']['arm'],r['row']['seed']):r for r in d['records']}
        for (arm,s),r in records.items():
            row=r['row'];assert row['scenario_hash']==d['hash']
            runs.append(row);values[p,arm].append(row['saved'])
            for v,tag in zip(r['persons'],tags):
                persons.append(dict(parent=p,layout=n,arm=arm,seed=s,**v,**tag,
                                    remaining=v['birth']+240-v['found'] if v['found']>=0 else ''))
            if arm=='G':
                for v in r['div']:diff.append(dict(parent=p,layout=n,seed=s,**v))
                other=records['T',s]
                for i,(a,b) in enumerate(zip(r['persons'],other['persons'])):
                    changes.append(dict(parent=p,layout=n,seed=s,id=i,**tags[i],
                          found=int(a['found']>=0)-int(b['found']>=0),saved=int(a['saved']>=0)-int(b['saved']>=0)))
        for br in d['branches']:
            s,t=br['seed'],br['t'];g=records['G',s];b=br['alternative']
            def ids(r,key,lo,hi):return {i for i,v in enumerate(r['persons']) if lo<v[key]<=hi}
            gf=ids(g,'found',t,t+60);bf=ids(b,'found',t,t+60);gs=ids(g,'saved',-1,300);bs=ids(b,'saved',-1,300)
            gain=sorted(gs-bs);loss=sorted(bs-gs)
            gz=g['trace'][t//5]['z'];tz=b['trace'][t//5]['z']
            downstream=any(a in meta['corridor'] and a!=c for a,c in zip(gz,tz))
            qualifying=[i for i in gain if i<16 and i in gf-bf and tags[i]['crossed']]
            item=dict(parent=p,layout=n,seed=s,t=t,g_found=sorted(gf-bf),t_found=sorted(bf-gf),g_saved=gain,t_saved=loss,
                      downstream=downstream,downstream_gained=qualifying if downstream else [])
            branches.append(item)
            for label,ok in [('downstream',downstream and qualifying),('loss',loss)]:
                if ok and label not in examples:
                    examples[label]=item
                    maps.save_json(OUT/(label+'-replay.json'),dict(case=str(path.name),**item,alarms=meta,
                              baseline=g,alternative=b,person_tags=tags))
    assert len(runs)==1400 and len(persons)==56000
    for name,rows in [('runs',runs),('people',persons),('alarms',alarms),('first-branches',branches),('disagreements',diff),('person-pairs',changes)]:write(name,rows)
    summary=[];strata=[];paired=[]
    for arm in 'HTG':
        rr=[r for r in runs if r['arm']==arm]
        summary.append(dict(arm=arm,n=len(rr),found=np.mean([r['found'] for r in rr]),saved=np.mean([r['saved'] for r in rr]),
                        near=sum(r['near'] for r in rr),all40=sum(r['all40'] for r in rr)))
        pp=[v for v in persons if v['arm']==arm]
        for label,sub in [(k,[v for v in pp if v['kind']==k]) for k in ['落水','浅滩','村落一','村落二']]+[
                (f'落水_跨区{x}_挂靠{y}',[v for v in pp if v['kind']=='落水' and v['crossed']==x and v['snag']==y]) for x in (False,True) for y in (False,True)]:
            found=[v for v in sub if v['found']>=0]
            strata.append(dict(arm=arm,label=label,n=len(sub),found=len(found),saved=sum(v['saved']>=0 for v in sub),
                          mean_found_s=np.mean([v['found'] for v in found]) if found else '',remaining=np.mean([v['remaining'] for v in found]) if found else '',
                          **{r:sum(v['reason']==r for v in sub) for r in ('unseen_timeout','seen_timeout','left')}))
    for a,b in [('G','T'),('G','H'),('T','H')]:
        x=np.array([np.mean(values[p,a])-np.mean(values[p,b]) for p in range(184001,184101)])
        lo,hi=np.quantile(np.random.default_rng(20261104).choice(x,(10000,100)).mean(1),[.025,.975])
        paired.append(dict(a=a,b=b,delta=x.mean(),lo=lo,hi=hi,positive=int((x>0).sum())))
    for name,rows in [('summary',summary),('strata',strata),('paired',paired)]:write(name,rows)
    lines=['# 贵州三类险情的四机配对实验','', '100 张合成河段、200 个布局，H 共 200 场，T、G 各 600 场。真实案例与模拟设置见入口 README。','',
           '|策略|每场发现|每场按期投放|完成率|','|---|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['arm']}|{r['found']:.2f}|{r['saved']:.2f}|{r['saved']/40:.2%}|")
    for r in paired:lines.append(f"\n{r['a']}−{r['b']} 为 {r['delta']:+.3f} 人/场，父图配对 95% 区间 [{r['lo']:+.3f}, {r['hi']:+.3f}]。")
    lines+=['','GMRF 提高了综合场景投放人数。' if paired[0]['lo']>0 else '综合场景未达到 GMRF 增益标准。','',
            '|G 相对 T|新增投放人次|失去投放人次|净变化|','|---|---:|---:|---:|']
    for k in ['落水','浅滩','村落一','村落二']:
        vals=[e['saved'] for e in changes if e['kind']==k]
        lines.append(f'|{k}|{vals.count(1)}|{vals.count(-1)}|{sum(vals):+d}|')
    lines+=['',f"首次端点分歧 {len(branches)} 场。G 单步改派与下游新发现、最终新增跨区落水者投放同时出现 {sum(bool(b['downstream_gained']) for b in branches)} 场；至少失去一名投放 {sum(bool(b['t_saved']) for b in branches)} 场。两支随后都使用 G。"]
    (OUT/'实验报告.md').write_text('\n'.join(lines),encoding='utf8')

