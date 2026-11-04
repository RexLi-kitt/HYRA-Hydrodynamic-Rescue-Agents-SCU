"""Phase1: matched-capacity GRU footprint/data-aggregation pilot."""
from pathlib import Path
import sys,copy,json,time,contextlib,traceback,hashlib
import numpy as np
import torch
from torch import nn
from memory import MemoryGRU,FineEnv,FineHistory,augment,normalizer,predictor,public_features
from adapter import FlowFogEnv
from long_environment import Config,make_scenario
from long_water import generate,sha
from models import DemandModel,maps
from run import transform
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];ROOT=REPO.parent
OLD=ROOT/'uav-convlstm20-20261104/run01'
OUT=ROOT/'uav-fine-memory-20261104/run01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def restore_old():
    ck=torch.load(OLD/'gru-32.pt',map_location='cpu',weights_only=True)
    m=DemandModel('gru',ck['mean']);m.load_state_dict(ck['state']);m.eval()
    mu=np.array(ck['mu'],np.float32);sd=np.array(ck['sd'],np.float32)
    def f(x):
        with torch.inference_mode():return m(torch.from_numpy((maps(x)-mu[None,:,None,None])/sd[None,:,None,None])[None])[0].numpy().ravel()
    return f
def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    m=MemoryGRU(ck['mean'],ck['coarse']);m.load_state_dict(ck['state']);m.eval()
    return m,predictor(m,np.array(ck['mu'],np.float32),np.array(ck['sd'],np.float32)),ck
def balanced(a,n):
    assert len(a)>0
    return a[np.linspace(0,len(a)-1,n).round().astype(int)]
def aggregate(rr):
    return dict(cases=len(rr),people=sum(r['people'] for r in rr),supplied=sum(r['supplied'] for r in rr),
        all_scenes=sum(r['supplied']==r['people'] for r in rr),p95_scenes=sum(r['supplied']>=.95*r['people'] for r in rr),
        failed_not_seen_before_deadline=sum(r['failed_not_seen_before_deadline'] for r in rr),
        refloats=sum(r['refloats'] for r in rr))

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic()
    oldcases=read(OLD/'scenarios.json');oldp=read(OLD/'protocol.json')
    cfg=Config(**oldp['config']);seeds=[202611141,202611142]
    sources={str(p):sha(p) for folder in (HERE,HERE.parent/'flood-drop-search-redesign-v1',HERE.parent/'flood-drop-convlstm20-v1')
        for p in folder.glob('*.py')}
    sources.update({str(p):sha(p) for p in (REPO/'terrain-workbench/terrain-lab').glob('flood_*.py')})
    inputs={str(p):sha(p) for p in (OLD/'scenarios.json',OLD/'protocol.json',OLD/'gru-32.pt')}
    protocol=dict(date='2026-11-04',version='fine-memory-v1',config=cfg.__dict__,source_sha256=sources,input_sha256=inputs,
        fit='24parents:12oldFIT layout0 +12new1104501..1104512 one layout each. No oldCAL/DEV/test moved into FIT.',
        calibration='8entirely new parents1104601..1104608,1layout each. Fixed before any fitting. This is model-selection CAL, not final test.',
        model='All arms64input->32GRU+counthead,matched nominal parameters; coarse arm masks50finechannels. Fine adds5x5subtile20m knownfraction andobservationage per100mtile. No hidden location/deadline/future input.',
        training='Two seeds202611141/142. Acoarse andBfine32supervisedepochs; Cfine_mixed warm-start exactB16+optimizer/RNG then16more. Adam.0005,batch32,clip5,Poisson countloss. D4 rotates fine grid and u/v. Not reinforcement-learning episodes.',
        datasets='Eachparent48uniform-time samples. A/B base flowcoverage trajectory; C24base+24B16ownrollout on sameFITscenario. Same1152samples; normalizationFITbaseonly andshared. Selfrollout labels are currenthiddenpendingcounts TRAINonly. Aggregation-inspired,not fullDAgger.',
        intervention='Same residual adapteralpha.25 chosen prior pilot,all arms. Coarsebaseline is matched capacity,oldGRU legacy additionalcontrol. No new difficulty tuning.',
        evaluation='CAL A16/B16/A32/B32/C32 eachseed=80rollouts; C16 aliases B16. flowcoverage,coverage100,oldGRU32=24controls. Then CAL-selected model eachseed and2planners on first3CALrotated=12stress. Total116.',
        selection='Within eachseed CAL totalpeople first,95%scenes second,earlier epoch thencoarse/fine/mixed. Allresults kept. No selection on rotation and no automatic64/128.',
        stop='One serialCPU4threads workload,STOP file,7200s activecap. Save atscene/epoch; never silently restart. No remote push orproductiondefaultchange.')
    write(OUT/'protocol.json',protocol);evaluated=[];collections=[];trainlog=[];audits=[];counts=dict(collection=0,evaluation=0,epoch_updates=0)
    def status(state,**kw):write(OUT/'status.json',dict(state=state,elapsed_s=time.monotonic()-start,**counts,**kw))
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('User STOP')
        if time.monotonic()-start>7200:raise TimeoutError('7200s cap')
    def evaluate(c,method,pred=None,collect=False,phase='calibration',rotation=False):
        guard();status('collecting' if collect else 'evaluating',case=c['id'],method=method,phase=phase)
        w=np.load(c['field']['path'],mmap_mode='r');s=c['scenario']
        if rotation:w,s=transform(w,s)
        if method in ('flow_coverage','coverage100'):e=FlowFogEnv(w,s,method)
        elif method=='old_gru32':e=FlowFogEnv(w,s,'gru_flowblend',pred)
        else:e=FineEnv(w,s,pred,.25)
        history=FineHistory();xs=[];ys=[];tt=[]
        while True:
            if e.t%120==0:guard()
            if collect and e.t%30==0:
                xs.append(history.get(e));ys.append(e.sector_counts(100).astype(np.float32));tt.append(e.t)
            if isinstance(e,FineEnv) and phase=='calibration' and c['id']=='cal-0' and e.t in (0,480):
                a=copy.copy(e);a.history=copy.deepcopy(e.history);a.prediction=None;a.predicted_at=None
                b=copy.copy(a);b.history=copy.deepcopy(a.history)
                b.person=np.full_like(e.person,888);b.status=np.full_like(e.status,3);b.water=None;b.fall=e.fall+10;b.deadline=e.deadline+10
                np.testing.assert_array_equal(public_features(a),public_features(b))
                aa=a.choices();bb=b.choices()
                for j in (0,1,3):np.testing.assert_array_equal(aa[j],bb[j])
                audits.append(dict(method=method,time=e.t,hidden_truth_invariant=True))
            if e.step(e.choices()[1])[2]:break
        persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),
            supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(cfg.people)]
        assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
        for p in persons:
            if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
        row=dict(case=c['id'],parent=c['field']['parent'],phase=phase,method=method,people=cfg.people,supplied=int((e.status==2).sum()),
             refloats=int(e.refloat_counts.sum()),persons=persons,
             failed_not_seen_before_deadline=sum(p['status']!=2 and (p['first_seen']<0 or p['first_seen']>=p['deadline']) for p in persons))
        if collect:
            counts['collection']+=1;collections.append(row)
            write(OUT/'collection-results.json',collections)
            return np.stack(xs),np.stack(ys),tt
        counts['evaluation']+=1;evaluated.append(row)
        with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        return row
    try:
        fit=copy.deepcopy(oldcases['fit'][::2]);cal=[];oldparents={c['field']['parent'] for g in oldcases.values() for c in g}
        for split,newseeds in (('fit',range(1104501,1104513)),('cal',range(1104601,1104609))):
            for i,seed in enumerate(newseeds):
                guard();status('generating_water',split=split,index=i+1)
                with (OUT/'generation.log').open('a',encoding='utf-8') as f,contextlib.redirect_stdout(f):
                    field=generate(OUT/'fields',REPO/'terrain-workbench/terrain-lab',seed,split)
                assert field['parent'] not in oldparents
                scenario=make_scenario(np.load(field['path'],mmap_mode='r'),2026111400+(0 if split=='fit' else 100)+i,cfg,fall_end=1680,pool_size=120)
                c=dict(id=f'{split}-new-{i}' if split=='fit' else f'cal-{i}',field=field,scenario=scenario)
                (fit if split=='fit' else cal).append(c)
                write(OUT/'scenarios.json',dict(fit=fit,calibration=cal))
        assert len(fit)==24 and len(cal)==8
        assert not ({c['field']['parent'] for c in fit}&{c['field']['parent'] for c in cal})
        fieldhash={c['field']['path']:sha(c['field']['path']) for c in fit+cal}
        write(OUT/'input-lock.json',dict(scenarios_sha256=sha(OUT/'scenarios.json'),fields_sha256=fieldhash,before_training=True))
        base=[]
        for i,c in enumerate(fit):
            x,y,t=evaluate(c,'flow_coverage',collect=True,phase='fit_base')
            np.savez_compressed(OUT/f'fit-base-{i:02d}.npz',x=x,y=y,times=np.array(t))
            base.append((balanced(x,48),balanced(y,48)))
        x=np.concatenate([a for a,b in base]);y=np.concatenate([b for a,b in base])
        mu,sd=normalizer(x);mean=float(y.mean())
        write(OUT/'normalization.json',dict(mu=mu.tolist(),sd=sd.tolist(),fitparents=24,samples=len(x)))
        xt=torch.from_numpy(x);yt=torch.from_numpy(y)
        mt=torch.from_numpy(mu)[None,None,:,None,None];st=torch.from_numpy(sd)[None,None,:,None,None]
        loss=nn.PoissonNLLLoss(log_input=False);paths={}
        def train(seed,arm,a,b,start_epoch=1,ck=None):
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            m=MemoryGRU(mean,arm=='coarse');opt=torch.optim.Adam(m.parameters(),lr=.0005)
            if ck is not None:
                m.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer'])
                torch.set_rng_state(ck['torch_rng']);rng.bit_generator.state=ck['aug_rng']
            for ep in range(start_epoch,33):
                guard();status('training',seed=seed,arm=arm,epoch=ep,total_epochs=32);m.train();total=0.
                for ids in torch.randperm(len(a)).split(32):
                    z,q=augment(a[ids],b[ids],int(rng.integers(4)),bool(rng.integers(2)))
                    opt.zero_grad();l=loss(m((z-mt)/st),q);assert torch.isfinite(l)
                    l.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();total+=float(l.detach())*len(ids)
                counts['epoch_updates']+=1;trainlog.append(dict(seed=seed,arm=arm,epoch=ep,poisson=total/len(a)))
                write(OUT/'training-history.json',trainlog)
                if ep in (16,32):
                    m.eval();name=f'{arm}-s{seed}-e{ep}';path=OUT/(name+'.pt')
                    torch.save(dict(state=m.state_dict(),coarse=arm=='coarse',mean=mean,mu=mu.tolist(),sd=sd.tolist(),epoch=ep,seed=seed,
                        optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),aug_rng=rng.bit_generator.state),path)
                    paths[name]=path
        for seed in seeds:
            train(seed,'coarse',xt,yt);train(seed,'fine',xt,yt)
            m,p,ck=restore(paths[f'fine-s{seed}-e16']);mixedx=[];mixedy=[]
            for i,c in enumerate(fit):
                sx,sy,t=evaluate(c,f'fine-s{seed}-e16',p,collect=True,phase='fit_self')
                np.savez_compressed(OUT/f'fit-self-s{seed}-{i:02d}.npz',x=sx,y=sy,times=np.array(t))
                bx,by=base[i];mixedx.extend([balanced(bx,24),balanced(sx,24)]);mixedy.extend([balanced(by,24),balanced(sy,24)])
            mx=np.concatenate(mixedx);my=np.concatenate(mixedy);assert mx.shape==x.shape and my.shape==y.shape
            train(seed,'mixed',torch.from_numpy(mx),torch.from_numpy(my),17,ck)
            del mx,my,mixedx,mixedy,m,p,ck
        write(OUT/'checkpoint-lock.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
        oldpredict=restore_old()
        for method,p in (('coverage100',None),('flow_coverage',None),('old_gru32',oldpredict)):
            for c in cal:evaluate(c,method,p)
        stats={}
        for name,path in paths.items():
            m,p,ck=restore(path)
            for c in cal:evaluate(c,name,p)
            stats[name]=aggregate([r for r in evaluated if r['phase']=='calibration' and r['method']==name])
            write(OUT/'calibration-statistics.json',stats)
        nominated={}
        for seed in seeds:
            names=[n for n in paths if f's{seed}-' in n]
            def key(n):
                arm=n.split('-')[0];ep=int(n.rsplit('e',1)[1]);r=stats[n]
                return (r['supplied'],r['p95_scenes'],-ep,-['coarse','fine','mixed'].index(arm))
            nominated[str(seed)]=max(names,key=key)
        write(OUT/'selection.json',dict(selected=nominated,rule=protocol['selection'],before_rotation=True))
        for method,p in (('coverage100',None),('flow_coverage',None)):
            for c in cal[:3]:evaluate(c,method,p,phase='rotation',rotation=True)
        for seed,name in nominated.items():
            m,p,ck=restore(paths[name])
            for c in cal[:3]:evaluate(c,name,p,phase='rotation',rotation=True)
        write(OUT/'leakage-audit.json',audits)
        for p,h in {**sources,**inputs,**fieldhash}.items():assert sha(p)==h,p
        assert counts['evaluation']==116 and counts['collection']==72 and counts['epoch_updates']==160
        summaries={phase:{method:aggregate([r for r in evaluated if r['phase']==phase and r['method']==method])
            for method in sorted({r['method'] for r in evaluated if r['phase']==phase})} for phase in ('calibration','rotation')}
        write(OUT/'summary.json',dict(state='completed',selected=nominated,statistics=summaries,source_inputs_unchanged=True,
            counts=counts,limits='8newCALparents,selectiondata not finaltest;2trainingseeds;same synthetic generator;no guaranteed improvement;no production change.'))
        status('completed',selected=nominated);print(json.dumps(summaries))
    except BaseException:
        write(OUT/'failure.json',dict(traceback=traceback.format_exc()))
        status('failed');raise
if __name__=='__main__':main()

