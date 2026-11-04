"""Bounded 20m ConvLSTM pilot: new parents, common physics, calibration-only selection."""
from pathlib import Path
import argparse,copy,contextlib,hashlib,json,sys,time,traceback,platform
import numpy as np
import torch
from torch import nn
from long_environment import Config,make_scenario
from long_water import generate,sha
from long_flow import FlowFogEnv
from history_model import History,features,Model
from models import DemandModel,maps,augment
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];ROOT=REPO.parent

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf8-sig'))
def aggregate(rr):
    return dict(cases=len(rr),supplied=sum(r['supplied'] for r in rr),people=sum(r['people'] for r in rr),people_rate=sum(r['supplied'] for r in rr)/sum(r['people'] for r in rr),all_rate=np.mean([r['supplied']==r['people'] for r in rr]).item(),p95_rate=np.mean([r['supplied']>=np.ceil(.95*r['people']) for r in rr]).item(),refloats=sum(r['refloats'] for r in rr))
def transform(w,s):
    w=np.rot90(w,1,(-2,-1)).copy();u=w[:,1].copy();w[:,1]=w[:,2];w[:,2]=-u;s=copy.deepcopy(s)
    def xy(p):return [p[1],1000-p[0]]
    s['traps']=[xy(p) for p in s['traps']]
    for e in s['events']:
        e['xy']=xy(e['xy']);e['field'][1:]=[e['field'][2],-e['field'][1]]
    return w,s

def audit(e):
    a=copy.copy(e);a.history=copy.deepcopy(e.history);a.prediction=None;a.predicted_at=None
    b=copy.copy(a);b.history=copy.deepcopy(a.history);b.person=np.full_like(e.person,888);b.status=np.full_like(e.status,3);b.deadline=e.deadline+13;b.fall=e.fall+17
    b.water=np.full_like(e.water,12.);b.scenario=copy.deepcopy(e.scenario);b.scenario['events']=[]
    np.testing.assert_array_equal(features(a),features(b))
    aa=a.choices();bb=b.choices()
    for j in (0,1,3):np.testing.assert_array_equal(aa[j],bb[j])
    np.testing.assert_array_equal(a.prediction,b.prediction)
    return dict(time=e.t,hidden_state_future_water_invariant=True)

def run(out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[];completed=0;torch.set_num_threads(4)
    cfg=Config(duration=2100,window=420,people=40,uavs=4,step=1,sensor=20.,trigger_depth=2.,trigger_speed=1.,fall_pattern='three_bursts')
    seeds={'fit':list(range(1104001,1104013)),'calibration':list(range(1104101,1104105)),'development':list(range(1104201,1104207))}
    old=ROOT/'uav-regional-refloat-20261103/run01/gru-32.pt'
    sources={str(p):sha(p) for p in HERE.glob('*.py')}
    sources.update({str(p):sha(p) for p in (REPO/'terrain-workbench/terrain-lab').glob('flood_*.py')})
    protocol=dict(version='convlstm20-long-v1',created='2026-11-04',config=cfg.__dict__,parent_seeds=seeds,
        sampling='12fit parents x2layouts,4calibration x1layout,6development x2layouts. Event seed2026110400+100*split_index+2*parent_index+layout. fall_end1680=T-window; pool120; no roster/map exclusions.',
        hydraulics='Actual SWE0..2100 every5s. Same hydrograph to1500 then zero source to2100, closed domain; water continues evolving. Synthetic generator, not real flood. Separate module; old inputs/defaults untouched.',
        training='Both new models32 supervised epochs,Adam1e-3,Poisson count loss,batch32maps,gradclip5,CPU4threads,seed20261104. Label current pending undiscovered counts only during FIT collection, every30s. 4causal frames. coverage100 rollout data. Not RL episodes.',
        models='ConvLSTM hidden16 3x3 gates +1x1 count head vs shared14->32 per-tile GRU; same14features except coordinates8/9 zeroed in BOTH. Same rotation/reflection augmentation and normalization. OldGRU32 unchanged additional linkage control.',
        normalization='Fit only; u/v common zero-centered scale; zero position channels; other channels fitmean/std. No calibration/development optimization labels.',
        selection='All0/16/32 checkpoints evaluated on4calibration scenes. Select separately by supplied people,then >=95% scenes,then earlier checkpoint. Save selection before any development result. Never select or change config using development.',
        methods=['coverage100','flow_coverage','old_gru32','new_gru','convlstm'],
        planning='Same complete common planner,1second sensing/execution,10second continuous service,50:50 coverage/learned demand and now/60s observed-flow forecast for all learned methods. No privileged truth/future water/deadline input.',
        evaluation='12development scenes x5methods=60; first3development parents layout0 rotated90 x5methods=15 separate stress. Calibration4 x(2planning+oldGRU+6newcheckpoints)=36. Parent-disjoint. Paired parent bootstrap,small synthetic pilot,not final independent validation.',
        target='People first,report all/95%-people scene rates. Desired people85..95%; no people regression vs both strong planners; >=15percentage points95%-scene gap. Higher than95% is not performance failure; indicates easier regime. No guaranteed target.',
        stop='STOPfile,hard7200sec budget; failure preserves partial files. One serial workload. No production change or remote push.',
        source_hashes=sources,old_model_sha256=sha(old),runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda_available=torch.cuda.is_available()))
    write(out/'protocol.json',protocol)
    def status(state,**kw):write(out/'status.json',dict(state=state,elapsed_s=time.monotonic()-start,evaluation_rollouts=completed,**kw))
    def guard():
        if (out/'STOP').exists():raise InterruptedError('User STOP')
        if time.monotonic()-start>7200:raise TimeoutError('Two-hour pilot cap')
    def evaluate(c,method,predict=None,phase='development',collect=False,rotation=False):
        nonlocal completed
        guard();status('collecting' if collect else 'evaluating',phase=phase,case=c['id'],method=method)
        w=np.load(c['field']['path'],mmap_mode='r');s=c['scenario']
        if rotation:w,s=transform(w,s)
        envmethod=method if method in ('coverage100','flow_coverage') else method+'_flowblend'
        e=FlowFogEnv(w,s,envmethod,predict);history=History();xx=[];yy=[];times=[];audits=[]
        while True:
            if e.t%30==0 and collect:
                xx.append(maps(history.get(e)));yy.append(e.sector_counts(100).astype(np.float32));times.append(e.t)
            if phase=='development' and method=='convlstm' and c['id']=='development-0-0' and e.t in (0,120,480):audits.append(audit(e))
            _,_,done,_=e.step(e.choices()[1])
            if done:break
        persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(cfg.people)]
        for p in persons:
            if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
        assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
        r=dict(case=c['id'],parent=c['field']['parent'],phase=phase,method=method,people=cfg.people,supplied=int((e.status==2).sum()),expired=int((e.status==3).sum()),outside=int((e.status==4).sum()),refloats=int(e.refloat_counts.sum()),unseen_expired=sum(p['status']==3 and p['first_seen']<0 for p in persons),persons=persons)
        if collect:return np.stack(xx),np.stack(yy),times,r
        rows.append(r);completed+=1
        with (out/'results.jsonl').open('a',encoding='utf8') as fp:fp.write(json.dumps(r,allow_nan=False)+'\n')
        if audits:write(out/'leakage-audit.json',audits)
        return r
    try:
        cases={};fields=[]
        for si,(split,ss) in enumerate(seeds.items()):
            cases[split]=[]
            for i,seed in enumerate(ss):
                guard();status('generating_water',split=split,parent_index=i+1,parent_total=len(ss))
                with (out/'generation.log').open('a',encoding='utf8') as log,contextlib.redirect_stdout(log):field=generate(out/'fields',REPO/'terrain-workbench/terrain-lab',seed,split)
                assert field['finite_nonnegative'];fields.append(field)
                for j in range(1 if split=='calibration' else 2):
                    guard();status('generating_roster',split=split,parent_index=i+1,layout=j)
                    scenario=make_scenario(np.load(field['path'],mmap_mode='r'),2026110400+100*si+2*i+j,cfg,fall_end=1680,pool_size=120)
                    cases[split].append(dict(id=f'{split}-{i}-{j}',field=field,scenario=scenario))
                write(out/'scenarios.json',cases);write(out/'fields.json',fields)
        write(out/'scenario-lock.json',dict(sha256=sha(out/'scenarios.json'),before_training=True))
        xs=[];ys=[];index=[];fitrows=[]
        for c in cases['fit']:
            x,y,t,r=evaluate(c,'coverage100',collect=True,phase='fit_collection');xs.append(x);ys.append(y);fitrows.append(r);index.append(dict(case=c['id'],times=t,frames=len(t)))
            write(out/'collection-progress.json',dict(completed=len(xs),planned=len(cases['fit'])))
        x=np.concatenate(xs);y=np.concatenate(ys);np.savez_compressed(out/'fit-labels.npz',x=x,y=y);write(out/'fit-index.json',index);write(out/'fit-rollouts.json',fitrows)
        mu=x.mean((0,1,3,4));sd=x.std((0,1,3,4)).clip(.01)
        mu[3:5]=0;sd[3:5]=max(float(np.sqrt(np.mean(x[:,:,3:5]**2))),.01);mu[8:10]=0;sd[8:10]=1
        a=torch.from_numpy(x);b=torch.from_numpy(y);mt=torch.tensor(mu)[None,None,:,None,None];st=torch.tensor(sd)[None,None,:,None,None];mean=float(y.mean())
        loss=nn.PoissonNLLLoss(log_input=False);records=[];predictors={};checkpoints={}
        def predictor(m,muv,sdv):
            def predict(z):
                with torch.inference_mode():return m(torch.from_numpy((maps(z)-muv[None,:,None,None])/sdv[None,:,None,None])[None])[0].numpy().reshape(-1)
            return predict
        for kind in ('gru','convlstm'):
            torch.manual_seed(20261104);m=DemandModel(kind,mean);opt=torch.optim.Adam(m.parameters(),lr=.001);rng=np.random.default_rng(20261104)
            for epoch in range(33):
                guard();status('training',model=kind,epoch=epoch,total_epochs=32)
                if epoch:
                    m.train();weighted=0.;n=0
                    for ids in torch.randperm(len(a)).split(32):
                        z,q=augment(a[ids],b[ids],int(rng.integers(4)),bool(rng.integers(2)))
                        opt.zero_grad();l=loss(m((z-mt)/st),q);assert torch.isfinite(l);l.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();weighted+=float(l.detach())*len(ids);n+=len(ids)
                    records.append(dict(model=kind,epoch=epoch,fit_augmented_poisson=weighted/n))
                    write(out/'training-history.json',records)
                if epoch in (0,16,32):
                    m.eval();name=f'{kind}-{epoch}';path=out/(name+'.pt')
                    torch.save(dict(state=m.state_dict(),kind=kind,epoch=epoch,mu=mu.tolist(),sd=sd.tolist(),mean=mean),path)
                    checkpoints[name]=dict(sha256=sha(path),parameters=sum(p.numel() for p in m.parameters()),epoch=epoch)
                    predictors[name]=predictor(copy.deepcopy(m),mu,sd)
        ck=torch.load(old,map_location='cpu',weights_only=True);om=Model(ck['kind'],ck['mean']);om.load_state_dict(ck['state']);om.eval()
        def old_predict(z):
            with torch.inference_mode():return om(torch.tensor((z-np.asarray(ck['mu'],np.float32))/np.asarray(ck['sd'],np.float32))).numpy()
        controls={'coverage100':None,'flow_coverage':None,'old_gru32':old_predict}
        for method,p in {**controls,**predictors}.items():
            for c in cases['calibration']:evaluate(c,method,p,'calibration')
        cal={method:aggregate([r for r in rows if r['phase']=='calibration' and r['method']==method]) for method in {**controls,**predictors}}
        selected={kind:max([kind+'-'+str(ep) for ep in (0,16,32)],key=lambda name:(cal[name]['supplied'],cal[name]['p95_rate'],-checkpoints[name]['epoch'])) for kind in ('gru','convlstm')}
        write(out/'selection.json',dict(selected=selected,calibration=cal,checkpoint_hashes=checkpoints,before_development=True))
        methods={**controls,'new_gru':predictors[selected['gru']],'convlstm':predictors[selected['convlstm']]}
        for method,p in methods.items():
            for c in cases['development']:evaluate(c,method,p)
            for c in cases['development'][::2][:3]:evaluate(c,method,p,'rotate90',rotation=True)
        stats={phase:{m:aggregate([r for r in rows if r['phase']==phase and r['method']==m]) for m in methods} for phase in ('development','rotate90')}
        # Parent paired intervals; all-zero scene outcomes do not establish certainty.
        rr=[r for r in rows if r['phase']=='development'];parents=sorted({r['parent'] for r in rr});keys=['people_rate','p95_rate','all_rate'];names=list(methods)
        values=np.array([[[aggregate([r for r in rr if r['parent']==p and r['method']==m])[k] for k in keys] for m in names] for p in parents])
        ids=np.random.default_rng(20261104).integers(len(parents),size=(5000,len(parents)));boot=values[ids].mean(1)
        comparisons={}
        for method in ('new_gru','convlstm'):
            j=names.index(method);delta=boot[:,j]-np.maximum(boot[:,0],boot[:,1]);point=values.mean(0)[j]-np.maximum(values.mean(0)[0],values.mean(0)[1]);comparisons[method]={k:dict(gap=float(point[i]),ci95=np.quantile(delta[:,i],[.025,.975]).tolist()) for i,k in enumerate(keys)}
        for p,h in sources.items():assert sha(p)==h,p
        for f in fields:assert sha(f['path'])==f['sha256']
        assert sha(old)==protocol['old_model_sha256'];assert sha(out/'scenarios.json')==read(out/'scenario-lock.json')['sha256']
        write(out/'summary.json',dict(state='completed',training_epochs_per_model=32,total_supervised_epochs=64,evaluation_rollouts=completed,fit_collection_rollouts=len(xs),fit_maps=len(x),selected=selected,statistics=stats,paired_parent_intervals=comparisons,source_integrity=True,limitations='6development parents same synthetic family; single training seed; 3rotation parents. No independent final test; no claim field efficacy. Actual refloats reported, not implied.'))
        status('completed',selected=selected)
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output)
