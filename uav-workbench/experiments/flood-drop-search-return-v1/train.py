"""Full-return supervised action ranking. Isolated pilot, no TD or policy-gradient RL."""
import sys, copy, json, time, contextlib, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from probe import (ROOT,REPO,HERE,BASE,PREVIOUS,SegmentEnv,FlowFogEnv,proposals,context,
                   public_features,clone,fingerprint,result,read,write,sha)
from long_environment import Config,make_scenario
from long_water import generate
from memory import MemoryGRU,FineEnv,predictor
from run import transform

OUT=ROOT/'uav-search-return-train-20261104/run01'
PROBE=ROOT/'uav-search-return-20261104/run01'
TIMES=[240,480,720,960,1200,1440,1560,1680]
SEEDS=[202611151,202611152]
OLD_MODEL=PREVIOUS/'fine-s202611142-e16.pt'

def action_features(e,ps):
    """Same current-public-only inputs for linear and deep; shape [actions,225]."""
    cc=context(e,ps);f=public_features(e)
    _,aa,oo,_=e.choices()
    i=ps[1]['uav'] if len(ps)>1 else 0
    xy=e.pos[i];c0,r0=np.clip((xy/100).astype(int),0,9)
    rows=[]
    for k,p in enumerate(ps):
        goal=np.array(oo[i][aa[i]]['xy'] if p is None else p['xy'])
        c,r=np.clip((goal/100).astype(int),0,9)
        delta=goal-xy;endpoint=xy+delta*min(1.,30*e.cfg.speed/max(np.linalg.norm(delta),1e-9))
        path=xy[None]+np.linspace(0,1,17)[:,None]*(endpoint-xy)[None]
        cr=np.clip((path/100).astype(int),0,9)
        rows.append(np.r_[cc[k],f[:,r,c],f[:,r0,c0],f[:,cr[:,1],cr[:,0]].mean(1)])
    x=np.array(rows,np.float32)
    assert x.shape==(len(ps),225) and np.isfinite(x).all()
    return x

class Ranker(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.net=nn.Linear(225,1) if kind=='linear' else nn.Sequential(nn.Linear(225,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        head=self.net if kind=='linear' else self.net[-1]
        nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def forward(self,x):
        v=self.net(x).squeeze(-1)
        return v-v[...,0:1]  # Default is exactly zero, also in training.

def predict(m,x,mu,sd):
    with torch.inference_mode():return m(torch.from_numpy((x-mu)/sd)).numpy()

def choose(v,margin):
    if margin is None:return 0
    a=int(np.argmax(v));return a if v[a]>margin else 0

def aggregate(rows):
    return dict(scenes=len(rows),people=sum(r['people'] for r in rows),supplied=sum(r['supplied'] for r in rows),
                all_scenes=sum(r['all_scene'] for r in rows),p95_scenes=sum(r['p95_scene'] for r in rows),
                overrides=sum(r.get('overrides',0) for r in rows),refloats=sum(r['refloats'] for r in rows))

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic()
    counts=dict(label_states=0,new_branches=0,reused_branches=0,epochs=0,evaluations=0)
    cfg=Config(**read(PREVIOUS/'protocol.json')['config'])
    fit=read(PREVIOUS/'scenarios.json')['fit'][12:20]
    source={str(p):sha(p) for d in (HERE,BASE,HERE.parent/'flood-drop-fine-memory-v1',HERE.parent/'flood-drop-search-redesign-v1') for p in d.glob('*.py')}
    source.update({str(p):sha(p) for p in (REPO/'terrain-workbench/terrain-lab').glob('flood_*.py')})
    inputs={str(p):sha(p) for p in (PREVIOUS/'scenarios.json',PREVIOUS/'input-lock.json',OLD_MODEL,PROBE/'protocol.json',PROBE/'summary.json',PROBE/'labels.jsonl')}
    # Prior source lock guarantees reused branch results have the same executor.
    for p,h in read(PROBE/'protocol.json')['source_sha256'].items():assert sha(p)==h,p
    oldlock=read(PREVIOUS/'input-lock.json')
    assert sha(PREVIOUS/'scenarios.json')==oldlock['scenarios_sha256']
    for c in fit:assert sha(c['field']['path'])==oldlock['fields_sha256'][c['field']['path']]
    protocol=dict(date='2026-11-04',version='search-return-train-v1',config=cfg.__dict__,source_sha256=source,input_sha256=inputs,
        fit_parents=[c['field']['parent'] for c in fit],cal_parents=list(range(1104701,1104704)),test_parents=list(range(1104801,1104807)),times=TIMES,
        splits='8existing FIT parents1104501..08,8fixed states each.3new CAL parents1104701..03 for offline checkpoint/margin selection.6new TEST parents1104801..06,one layout each; no selection or tuning on these results. Rotated first3 TEST after ordinary panel.',
        data='Roll-in and continuation use full coverage100. At each state default +3other-region search segments,<=30s, one idle UAV; discovery/rescue interrupts. Final supplied gain labels; same hash-counter random events. Reuse exactly12previous FIT states/48branches after exact proposals and default result checks.',
        features='225identical public features:33current context,64channel current public memory at goal and current tile,64channel mean along17samples of next<=300m straight segment. No future/truth/runtime labels. Default features describe its current goal but default keeps replanning.',
        training='Linear225->1 and MLP225->128->64->1;2seeds202611151/152;32supervised epochs. Adam1e-3,weight_decay1e-4,Huber on default-centered final-people gains,8states/batch,gradient clip5. FIT-only normalization. No reward tweaking, TD or forbidden algorithms. No claim of RL episodes.',
        selection='Evaluate epoch0/8/16/32 offline CAL gains with margins0/.5/1/2/never. Select per family/seed by total chosen gain,then fewer changes,larger margin,earlier epoch. Correlated teacher-state values not additive achieved outcome. Never opt-out ensures selection can retain full planner.',
        deployment='One decision at the same8fixed times; repeated macro choices may shift state distribution, so full-episode CAL and TEST mandatory. No hidden future used at deployment.',
        controls='Complete coverage100,flow_coverage,prior fineGRU seed142 epoch16 frozen before new test. Four independently CAL-selected trained policies. All same cases.7methods*3CAL +7*6TEST +7*3rotatedTEST=84logicalrollouts. Never-margin policies reuse exact coverage outcome and disclose alias.',
        gates='FIT signs>=5positive/5negative in>=2parents per sign; otherwise stop. Successful model must improve people on new TEST, retain full/95%scene metrics,check seed/rotation consistency.15pp scene target not promised. No automatic64/128 on one noisy result.',
        cost='One CPU4thread workload,read-only shared water,<=3600s activeguard,STOP supported. Worst352branchlabels minus48reused=304new full continuations;84logicalfull evaluations. Environment dominates; no GPU benefit assumed.',
        limitations='Only8FIT parents; hidden targets may make return unpredictable. No data aggregation iteration yet.6TEST parents small; people not independent samples. Synthetic water uncalibrated. All failure/opt-out/checkpoints retained.',
        unchanged='No changes to difficulty,original planner,environment/production defaults or old weights; no push.')
    write(OUT/'protocol.json',protocol)
    def status(state,**kw):write(OUT/'status.json',dict(state=state,elapsed_s=time.monotonic()-start,**counts,**kw))
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('STOP requested')
        if time.monotonic()-start>3600:raise TimeoutError('3600s cap')
    def finish(e):
        while e.t<e.cfg.duration and not (e.status>=2).all():
            if e.t%120==0:guard()
            if e.step(e.choices()[1])[2]:break
        return result(e)
    scenarios=dict(fit=fit,calibration=[],test=[]);audits=[];records=[];testrows=[]
    try:
        for split,seeds in (('calibration',range(1104701,1104704)),('test',range(1104801,1104807))):
            for si,seed in enumerate(seeds):
                guard();status('generating',split=split,index=si+1)
                with (OUT/'generation.log').open('a',encoding='utf-8') as fp,contextlib.redirect_stdout(fp):
                    field=generate(OUT/'fields',REPO/'terrain-workbench/terrain-lab',seed,split)
                scenario=make_scenario(np.load(field['path'],mmap_mode='r'),2026111500+(0 if split=='calibration' else 100)+si,cfg,fall_end=1680,pool_size=120)
                scenarios[split].append(dict(id=f'{split}-{si}',field=field,scenario=scenario))
                write(OUT/'scenarios.json',scenarios)
        groups=[{c['field']['parent'] for c in v} for v in scenarios.values()]
        assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        waterhash={c['field']['path']:sha(c['field']['path']) for group in scenarios.values() for c in group}
        write(OUT/'input-lock.json',dict(scenarios_sha256=sha(OUT/'scenarios.json'),water_sha256=waterhash))
        oldrows=[json.loads(line) for line in (PROBE/'labels.jsonl').read_text().splitlines()]
        reuse={(r['case'],r['time']):r for r in oldrows}
        for split in ('fit','calibration'):
            for ci,c in enumerate(scenarios[split]):
                status('roll_in',split=split,case=c['id']);water=np.load(c['field']['path'],mmap_mode='r')
                e=SegmentEnv(water,c['scenario']);snapshots=[]
                while True:
                    if e.t%120==0:guard()
                    if e.t in TIMES:snapshots.append(clone(e))
                    if e.step(e.choices()[1])[2]:break
                baseline=result(e);write(OUT/f'baseline-{split}-{ci}.json',baseline)
                assert len(snapshots)==len(TIMES),'Missing fixed state; retain failure rather than replace time'
                for st in snapshots:
                    ps=proposals(st);x=action_features(st,ps);stamp=fingerprint(st)
                    other=clone(st);other.person[:]=999;other.status[:]=3;other.fall[:]+=1;other.deadline[:]+=1;other.water=None
                    assert proposals(other)==ps;np.testing.assert_array_equal(x,action_features(other,ps))
                    rr=reuse.get((c['id'],st.t)) if split=='fit' else None
                    vals=[];sources=[]
                    for ai,p in enumerate(ps):
                        if rr is not None:
                            assert rr['proposals']==ps
                            oldpath=PROBE/f'branch-{ci}-{st.t}-{ai}.json'
                            assert sha(oldpath)==read(PROBE/'manifest.json')[oldpath.name]
                            row=read(oldpath);counts['reused_branches']+=1
                            sources.append(dict(path=str(oldpath),sha256=sha(oldpath)))
                        else:
                            status('branching',split=split,case=c['id'],time=st.t,branch=ai)
                            b=clone(st);b.arm(p);row=finish(b);counts['new_branches']+=1
                            path=OUT/f'branch-{split}-{ci}-{st.t}-{ai}.json';write(path,dict(proposal=p,**row))
                            sources.append(dict(path=str(path),sha256=sha(path)))
                        assert fingerprint(st)==stamp
                        if ai==0:assert row['persons']==baseline['persons'] and row['final_position']==baseline['final_position']
                        assert row['segment_steps']<=30
                        vals.append(row['supplied'])
                    key=f'{split}-{ci}-{st.t}';y=np.array(vals,np.float32)-vals[0]
                    np.savez_compressed(OUT/f'state-{key}.npz',x=x,y=y)
                    r=dict(key=key,split=split,case=c['id'],parent=c['field']['parent'],time=st.t,
                        proposals=ps,gain=y.tolist(),supplied=vals,sources=sources)
                    records.append(r);counts['label_states']+=1
                    with (OUT/'labels.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(r)+'\n')
                    audits.append(dict(state=key,public_input_isolation=True,origin_unchanged=True,default_exact=True))
                    write(OUT/'audits.json',audits)
                print(json.dumps(dict(stage='labels',split=split,case=c['id'],**counts)),flush=True)
        fitr=[r for r in records if r['split']=='fit'];calr=[r for r in records if r['split']=='calibration']
        gains=[g for r in fitr for g in r['gain'][1:]]
        pg={r['parent'] for r in fitr if any(g>0 for g in r['gain'][1:])};ng={r['parent'] for r in fitr if any(g<0 for g in r['gain'][1:])}
        gate=dict(positive=sum(g>0 for g in gains),negative=sum(g<0 for g in gains),ties=sum(g==0 for g in gains),positive_parents=sorted(pg),negative_parents=sorted(ng))
        write(OUT/'label-gate.json',gate)
        if gate['positive']<5 or gate['negative']<5 or len(pg)<2 or len(ng)<2:
            status('insufficient_label_signal');write(OUT/'summary.json',dict(status='insufficient_label_signal',gate=gate));return
        def data(rr):
            rows=[]
            for r in rr:
                with np.load(OUT/f'state-{r["key"]}.npz') as z:rows.append((z['x'],z['y']))
            return rows
        fd=data(fitr);cd=data(calr)
        allx=np.concatenate([x for x,y in fd]);mu=allx.mean(0);sd=allx.std(0).clip(.01)
        write(OUT/'normalization.json',dict(mean=mu.tolist(),std=sd.tolist(),fit_states=len(fd),fit_actions=len(allx)))
        tx=[torch.from_numpy((x-mu)/sd) for x,y in fd];ty=[torch.from_numpy(y) for x,y in fd]
        selected={};models={};cal_tables=[]
        for kind in ('linear','deep'):
            for seed in SEEDS:
                torch.manual_seed(seed);model=Ranker(kind);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
                name=f'{kind}-s{seed}';candidates=[]
                for epoch in range(33):
                    guard();status('training',model=name,epoch=epoch)
                    if epoch:
                        model.train();losses=[]
                        for ids in torch.randperm(len(tx)).split(8):
                            loss=torch.stack([nn.functional.smooth_l1_loss(model(tx[int(i)]),ty[int(i)]) for i in ids]).mean()
                            assert torch.isfinite(loss)
                            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
                        counts['epochs']+=1
                        with (OUT/'training.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(model=name,epoch=epoch,loss=float(np.mean(losses))))+'\n')
                    if epoch in (0,8,16,32):
                        model.eval();preds=[predict(model,x,mu,sd) for x,y in cd]
                        path=OUT/f'{name}-e{epoch}.pt'
                        torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),mean=mu.tolist(),std=sd.tolist(),kind=kind,seed=seed,epoch=epoch,parameters=sum(p.numel() for p in model.parameters()),protocol_sha256=sha(OUT/'protocol.json')),path)
                        for margin in (0.,.5,1.,2.,None):
                            actions=[choose(v,margin) for v in preds];gain=sum(float(cd[i][1][a]) for i,a in enumerate(actions))
                            row=dict(model=name,epoch=epoch,margin=margin,total_offline_gain=gain,changes=sum(a!=0 for a in actions),actions=actions,checkpoint=str(path))
                            candidates.append(row);cal_tables.append(row)
                        write(OUT/f'predictions-{name}-e{epoch}.json',[dict(state=r['key'],prediction=v.tolist(),actual_gain=y.tolist()) for r,v,(x,y) in zip(calr,preds,cd)])
                best=max(candidates,key=lambda r:(r['total_offline_gain'],-r['changes'],1e9 if r['margin'] is None else r['margin'],-r['epoch']))
                selected[name]=best;ck=torch.load(best['checkpoint'],map_location='cpu',weights_only=True)
                m=Ranker(kind);m.load_state_dict(ck['state']);m.eval();models[name]=m
                print(json.dumps(dict(stage='selected',**best)),flush=True)
        write(OUT/'calibration-selection-all.json',cal_tables);write(OUT/'selected.json',selected)
        oldck=torch.load(OLD_MODEL,map_location='cpu',weights_only=True)
        oldm=MemoryGRU(oldck['mean'],oldck['coarse']);oldm.load_state_dict(oldck['state']);oldm.eval()
        oldpred=predictor(oldm,np.array(oldck['mu'],np.float32),np.array(oldck['sd'],np.float32))
        methods=['coverage100','flow_coverage','prior-fine142-e16',*models]
        for phase,cases,rotation in (('calibration_full',scenarios['calibration'],False),('test',scenarios['test'],False),('rotation',scenarios['test'][:3],True)):
            for c in cases:
                w=np.load(c['field']['path'],mmap_mode='r');sc=c['scenario']
                if rotation:w,sc=transform(w,sc)
                baseline=None
                for name in methods:
                    guard();status('evaluating',phase=phase,case=c['id'],model=name)
                    if name in selected and selected[name]['margin'] is None:
                        row=copy.deepcopy(baseline);row.update(method=name,alias='coverage100: CAL selected never override',overrides=0,decisions=[])
                    else:
                        if name=='coverage100':e=SegmentEnv(w,sc)
                        elif name=='flow_coverage':e=FlowFogEnv(w,sc,'flow_coverage')
                        elif name=='prior-fine142-e16':e=FineEnv(w,sc,oldpred,.25)
                        else:e=SegmentEnv(w,sc)
                        decisions=[];overrides=0
                        while True:
                            if e.t%120==0:guard()
                            if name in models and e.t in TIMES:
                                ps=proposals(e);x=action_features(e,ps);v=predict(models[name],x,mu,sd)
                                if e.t==480:
                                    other=clone(e);other.person[:]=999;other.status[:]=3;other.fall[:]+=1;other.deadline[:]+=1;other.water=None
                                    np.testing.assert_array_equal(x,action_features(other,ps))
                                a=choose(v,selected[name]['margin']);e.arm(ps[a]);overrides+=int(a!=0)
                                decisions.append(dict(time=e.t,choice=a,proposals=ps,predicted_gains=v.tolist()))
                            if e.step(e.choices()[1])[2]:break
                        if not hasattr(e,'segment_steps'):e.segment_steps=0;e.segment_end=None
                        row=dict(phase=phase,case=c['id'],parent=c['field']['parent'],method=name,overrides=overrides,decisions=decisions,**result(e))
                        if name=='coverage100':baseline=copy.deepcopy(row)
                    testrows.append(row);counts['evaluations']+=1
                    with (OUT/'results.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(row)+'\n')
                print(json.dumps(dict(stage='evaluated',phase=phase,case=c['id'],evaluations=counts['evaluations'])),flush=True)
        for p,h in {**source,**inputs,**waterhash}.items():assert sha(p)==h,p
        summary={phase:{name:aggregate([r for r in testrows if r['phase']==phase and r['method']==name]) for name in methods} for phase in ('calibration_full','test','rotation')}
        write(OUT/'summary.json',dict(status='completed',results=summary,selected=selected,counts=counts,elapsed_s=time.monotonic()-start,source_water_unchanged=True,new_test_not_used_for_selection=True))
        status('completed');print(json.dumps(summary),flush=True)
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':main()
