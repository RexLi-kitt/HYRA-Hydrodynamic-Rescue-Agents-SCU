import csv,json
from collections import defaultdict
import math
import numpy as np
from .study import OUT,SEEDS,read,maps


def export(name,rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)


def report():
    runs=[];people=[];scenes=[]
    for f in sorted((OUT/'test').glob('*.gz')):
        d=read(f);scenes.append(d['meta'])
        for r in d['records']:
            row=r['row'];runs.append(row)
            people.extend(dict(**v)|{k:row[k] for k in ['parent','layout','orientation','arm','seed']} for v in r['persons'])
    assert len(runs)==480 and len(people)==19200
    export('runs.csv',runs);export('people.csv',people);maps.save_json(OUT/'scenarios.json',scenes)
    summary=[];results=[];goals=[]
    lines=['# 逐秒三波洪水投放验证','',
           '我方独立实现，外部报告的 91.20% 与 +12.66 个百分点仅作背景。主测试按 24 张父图、48 个共同布局评估。GRU 的人数与场次按三个训练种子平均。']
    for ori in ('main','rot','mirror'):
        rr=[r for r in runs if r['orientation']==ori]
        ps=sorted({r['parent'] for r in rr});n=len(ps)*(2 if ori=='main' else 1)
        assert len(rr)==8*n
        scores={}
        for a in ('P0','P1','M0','M1'):
            subset=[r for r in rr if r['arm']==a]
            for s in [None]+(list(SEEDS) if a.startswith('M') else []):
                sub=[r for r in subset if s is None or r['seed']==s]
                assert len(sub)==n*(3 if a.startswith('M') and s is None else 1)
                summary.append(dict(orientation=ori,arm=a,seed='mean' if s is None else s,unique_layouts=n,
                    found=np.mean([r['found'] for r in sub]),saved=np.mean([r['saved'] for r in sub]),pct=2.5*np.mean([r['saved'] for r in sub]),
                    all40=n*np.mean([r['all40'] for r in sub]),near=n*np.mean([r['near'] for r in sub])))
            scores[a]=np.array([np.mean([r['saved'] for r in subset if r['parent']==p])*2.5 for p in ps])
        rng=np.random.default_rng(20261103);ix=rng.integers(0,len(ps),(10000,len(ps)))
        bs={a:v[ix].mean(1) for a,v in scores.items()};m={a:v.mean() for a,v in scores.items()}
        for a,b in [('M1','best'),('M1','M0'),('M0','best'),('M1','P0'),('M1','P1')]:
            base=max(m['P0'],m['P1']) if b=='best' else m[b]
            db=bs[a]-(np.maximum(bs['P0'],bs['P1']) if b=='best' else bs[b])
            lo,hi=np.quantile(db,[.025,.975])
            results.append(dict(orientation=ori,a=a,b=b,delta=m[a]-base,lo=lo,hi=hi))
        best=max(m['P0'],m['P1']); gap=m['M1']-best
        stronger='P0' if m['P0']>=m['P1'] else 'P1'
        goals.append(dict(orientation=ori,total=n*40,comp=stronger,baseline_people=round(best*n*.4),M1_people=m['M1']*n*.4,
                          target_people=math.ceil((best+15)*n*.4-1e-8),required_more=max(0.,(best+15-m['M1'])*n*.4),
                          margin_15pp=bool(gap>=15-1e-9),in_85_95=bool(85<=m['M1']<=95)))
        lines+=['',f"## {ori}",'','|方法|平均发现/场|按期完成率|全员场数|≥38人场数|','|---|---:|---:|---:|---:|']
        for s in summary:
            if s['orientation']==ori and s['seed']=='mean':
                lines.append(f"|{s['arm']}|{s['found']:.2f}|{s['pct']:.2f}%|{s['all40']:.2f}/{n}|{s['near']:.2f}/{n}|")
        for x in results:
            if x['orientation']==ori and x['b'] in ('best','M0'):
                lines.append(f"\n{x['a']}−{x['b']} {x['delta']:+.2f} 个百分点，配对 95% 区间 [{x['lo']:+.2f}, {x['hi']:+.2f}]。")
        lines.append(f"\n15 个百分点目标{'达到' if goals[-1]['margin_15pp'] else '未达到'}，按单种子等效分母还差 {goals[-1]['required_more']:.2f} 人次。")
    for name,rows in [('summary.csv',summary),('paired.csv',results),('target-15pp.csv',goals)]:export(name,rows)
    (OUT/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

