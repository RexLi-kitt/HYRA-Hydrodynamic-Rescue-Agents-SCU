"""Observed-history regional demand inference; truth is TRAIN label only."""
from pathlib import Path
import sys,json,time,copy,argparse,traceback
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-search-probe-v1'))
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-coordination-v1'))
from search_env import SearchEnv
from train_coordination import read,write,sha,summarize

def pool(a):return np.asarray(a).reshape(10,10,10,10).mean((1,3))

def features(e):
    """Only public observations/configuration; one row per 100m tile."""
    known=e.known
    age=np.where(known,np.clip((e.t-e.last)/240,0,2),2.)
    seen=np.zeros((10,10));pending=seen.copy()
    for j in np.flatnonzero(e.discovered):
        c,r=np.clip((e.seen_xy[j]/100).astype(int),0,9)
        seen[r,c]+=1
        if e.belief[j]==1:pending[r,c]+=1
    rr,cc=np.mgrid[:10,:10]
    values=[pool(known),pool(age),pool(np.where(known,e.measured[0]/2,0)),
            pool(np.where(known,e.measured[1],0)),pool(np.where(known,e.measured[2],0)),
            pool(known&(e.measured[0]>=e.cfg.wet_depth)),seen/5,pending/5,
            (cc+.5)/10,(rr+.5)/10]
    values += [np.full((10,10),v) for v in (e.t/e.cfg.duration,e.cfg.sensor/100,e.cfg.people/80,e.cfg.window/300)]
    x=np.stack(values,-1).reshape(100,14).astype(np.float32)
    assert np.isfinite(x).all()
    return x

class History:
    def __init__(self):self.frames=[];self.time=None
    def get(self,e):
        if self.time!=e.t and (not self.frames or e.t%30==0):
            self.frames.append(features(e));self.frames=self.frames[-4:];self.time=e.t
        # Repeat the first actual observation, never future padding.
        return np.stack([self.frames[0]]*(4-len(self.frames))+self.frames,axis=1)

def statistic(x):
    a=x[:,-1];return (.1+a[:,1])*(.25+2*a[:,5]+1-a[:,0])

class Model(nn.Module):
    def __init__(self,kind,mean):
        super().__init__();self.kind=kind
        if kind=='gru':self.recurrent=nn.GRU(14,32,batch_first=True);self.head=nn.Linear(32,1)
        else:self.head=nn.Linear(56,1)
        nn.init.zeros_(self.head.weight);nn.init.constant_(self.head.bias,float(np.log(np.expm1(max(mean,1e-5)))))
    def forward(self,x):
        h=self.recurrent(x)[0][:,-1] if self.kind=='gru' else x.flatten(1)
        return torch.nn.functional.softplus(self.head(h).squeeze(-1))+1e-6

class LearnedEnv(SearchEnv):
    def __init__(self,water,scenario,predict):
        self.history=History();self.predict=predict;self.prediction=None;self.predicted_at=None
        super().__init__(water,scenario,'coverage100')
    def search_score(self,stale,risk):
        x=self.history.get(self)
        if self.prediction is None or self.predicted_at!=self.history.time:
            self.prediction=np.maximum(0,np.asarray(self.predict(x))).reshape(10,10)
            assert np.isfinite(self.prediction).all()
            self.predicted_at=self.history.time
        return stale*(1+risk)*(1+8*np.repeat(np.repeat(self.prediction,10,0),10,1))

def run(out,previous,probe):
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();counts=dict(training_layouts=0,supervised_epochs=0,evaluation_rollouts=0)
    def status(state,**kw):write(out/'status.json',dict(status=state,elapsed_s=time.monotonic()-started,**counts,**kw))
    def guard():
        if (out/'STOP').exists():raise InterruptedError('User STOP')
        if time.monotonic()-started>3600:raise TimeoutError('One hour budget')
    old=read(previous/'protocol.json');cases=read(previous/'scenarios.json');train=cases['train'];dev=cases['development']
    parents=sorted({c['field']['parent'] for c in train});assert len(parents)==16
    fit=set(parents[:12]);cal=set(parents[12:]);dp={c['field']['parent'] for group in dev.values() for c in group};assert not set(parents)&dp
    sources={**old['source_hashes'],str(HERE.parent/'flood-drop-search-probe-v1/search_env.py'):sha(HERE.parent/'flood-drop-search-probe-v1/search_env.py'),str(Path(__file__).resolve()):sha(__file__)}
    protocol=dict(version='regional-demand-v1',date='2026-11-03',scenario_sha256=sha(previous/'scenarios.json'),source_hashes=sources,water_hashes=old['water_hashes'],fit_parents=sorted(fit),calibration_parents=sorted(cal),dev_parents=sorted(dp),
        features='14 public per100m tile: known fraction, observed age, observed h/2,u,v,wet fraction, discovered and believed pending counts/5,xy,t/T,sensor/100,N/80,window/300. Four causal observations30s apart; initial repeat padding.',
        labels='Current hidden pending undiscovered count per100m tile; TRAIN simulation truth only. Collect every30s following coverage100 on32 archivedTRAIN layouts. DEV never labels/fits/selects.',
        models='Statistical age*wet/unknown prior calibrated by fit-total count; softplus linear56 and shared GRU14->32->1, same inputs. Adam1e-3 Poisson loss,32 supervised epochs each,seed20261103,CPU4threads,batch2048. Save0/16/32. These are not RL episodes.',
        selection='Use minimum calibration-parent Poisson loss among0/16/32 within each model, tie earlier. Check calibration improvement vs constant and statistical predictors. All three checkpoints evaluated on all DEV regardless of gate; no DEV-selected version.',
        scoring='Same original patrol score multiplied1+8*predicted count, matching oracle functional form; known-person dispatch and physical executor unchanged. Prediction every30s. No hidden input/future water/true fall/deadline.',
        baseline='Keep coverage100 plus four historical full planners; no weakening. All24 reusedDEV:60m12main,40m6,100m6pressure. Primary supplied people; all and95% scenes also. No final test.',
        success='SelectedGRU must exceed coverage100 main people and not regress either pressure view; scene15pp objective reported separately. Oracle not attainable promise. Keep all failures; no default upgrade.',
        limitations='Synthetic fixed acceptedN, all accepted falls random in these archives;2m/1mps thresholds unexercised. Random individual events need not be inferable. Off-policy training search history may shift at deployment. Coordinates may memorizeTRAIN, so hydraulic-parent holdout required.')
    write(out/'protocol.json',protocol);write(out/'scenarios.json',cases)
    def verify():
        assert sha(previous/'scenarios.json')==protocol['scenario_sha256']
        for p,h in {**sources,**old['water_hashes']}.items():assert sha(p)==h,p
    try:
        torch.set_num_threads(4);torch.manual_seed(20261103);np.random.seed(20261103);verify()
        xs=[];ys=[];splits=[];index=[]
        for c in train:
            guard();status('collecting',case=c['id']);e=SearchEnv(np.load(c['field']['path'],mmap_mode='r'),c['scenario'],'coverage100');history=History()
            while True:
                if e.t%30==0:
                    x=history.get(e);y=e.sector_counts(100).ravel().astype(np.float32)
                    # Non-initial public-history check: hidden truth mutation must not alter features.
                    if e.t==120 and counts['training_layouts']==0:
                        q=copy.copy(e);q.person=np.full_like(e.person,999);q.status=np.ones_like(e.status);q.fall=e.fall+100;q.deadline=e.deadline+100
                        assert np.array_equal(features(e),features(q))
                        write(out/'readiness.json',dict(public_features_hidden_truth_invariant=True,parent_split_disjoint=True,causal_padding=True,status='passed'))
                    xs.append(x.copy());ys.append(y);splits.extend([c['field']['parent'] in fit]*100);index.append(dict(case=c['id'],parent=c['field']['parent'],t=e.t,rows=[(len(xs)-1)*100,len(xs)*100]))
                _,_,done,_=e.step(e.choices()[1])
                if done:break
            counts['training_layouts']+=1
        x=np.concatenate(xs);y=np.concatenate(ys);mask=np.array(splits);assert mask.any() and (~mask).any() and y[mask].sum()>0 and y[~mask].sum()>0
        np.savez_compressed(out/'training-labels.npz',x=x,y=y,fit=mask);write(out/'training-index.json',index)
        mu=x[mask].mean((0,1));sd=x[mask].std((0,1)).clip(.01);np.savez(out/'normalization.npz',mean=mu,std=sd)
        a=torch.tensor((x[mask]-mu)/sd);b=torch.tensor(y[mask]);v=torch.tensor((x[~mask]-mu)/sd);w=torch.tensor(y[~mask]);mean=float(b.mean())
        scale=float(y[mask].sum()/statistic(x[mask]).sum())
        loss=nn.PoissonNLLLoss(log_input=False,full=False)
        with torch.no_grad():constant_loss=float(loss(torch.full_like(w,mean),w));stat_loss=float(loss(torch.tensor(statistic(x[~mask])*scale),w))
        records=[];models={};chosen={}
        for kind in ('linear','gru'):
            torch.manual_seed(20261103);m=Model(kind,mean);optimizer=torch.optim.Adam(m.parameters(),lr=.001);models[kind]={}
            for epoch in range(33):
                guard();status('training',model=kind,epoch=epoch)
                if epoch:
                    m.train();order=torch.randperm(len(a))
                    for ids in order.split(2048):
                        optimizer.zero_grad();l=loss(m(a[ids]),b[ids]);assert torch.isfinite(l);l.backward();nn.utils.clip_grad_norm_(m.parameters(),5);optimizer.step()
                    counts['supervised_epochs']+=1
                m.eval()
                with torch.no_grad():vl=float(loss(m(v),w))
                records.append(dict(model=kind,epoch=epoch,calibration_poisson=vl))
                if epoch in (0,16,32):
                    torch.save(dict(state=m.state_dict(),kind=kind,epoch=epoch,mu=mu.tolist(),sd=sd.tolist(),mean=mean),out/f'{kind}-{epoch}.pt');models[kind][epoch]=copy.deepcopy(m)
                write(out/'training-history.json',records)
            selected=min([r for r in records if r['model']==kind and r['epoch'] in (0,16,32)],key=lambda r:(r['calibration_poisson'],r['epoch']))
            chosen[kind]={**selected,'beats_constant_and_statistic':selected['calibration_poisson']<min(constant_loss,stat_loss)}
        write(out/'selection.json',dict(selected=chosen,constant_poisson=constant_loss,statistic_poisson=stat_loss,statistic_scale=scale,selection_before_dev=True))
        methods={'statistic':lambda z:statistic(z)*scale}
        def predictor(m):
            def predict(z):
                with torch.no_grad():return m(torch.tensor((z-mu)/sd)).numpy()
            return predict
        for kind,checkpoints in models.items():
            for epoch,m in checkpoints.items():methods[f'{kind}-{epoch}']=predictor(m)
        results={name:{} for name in methods}
        for name,predict in methods.items():
            for panel,group in dev.items():
                rows=[]
                for c in group:
                    guard();status('evaluating',model=name,panel=panel,case=c['id']);e=LearnedEnv(np.load(c['field']['path'],mmap_mode='r'),c['scenario'],predict)
                    while True:
                        _,_,done,_=e.step(e.choices()[1])
                        if done:break
                    row=dict(**e.metrics(),case=c['id']);row['persons']=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(e.cfg.people)]
                    assert row['supplied']+row['expired']+row['outside']==e.cfg.people
                    for p in row['persons']:
                        if p['status']==2:assert p['fall']+e.cfg.drop_seconds<=p['supplied_at']<=p['deadline']
                    rows.append(row);counts['evaluation_rollouts']+=1
                    with (out/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(model=name,panel=panel,**row),allow_nan=False)+'\n')
                results[name][panel]=summarize(rows)
        baseline=read(probe/'summary.json');coverage=baseline['results']['coverage100'];selected=results[f"gru-{chosen['gru']['epoch']}"]
        gate=selected['view60']['people_supplied']>coverage['view60']['people_supplied'] and all(selected[p]['people_supplied']>=coverage[p]['people_supplied'] for p in ('view40','view100'))
        verify();write(out/'summary.json',dict(status='completed',counts=counts,results=results,selection=chosen,coverage100=coverage,old_full_baselines=baseline['full_baselines'],selected_gru_people_gate=gate,source_water_unchanged=True));status('completed')
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);a=p.parse_args();run(a.output,a.previous,a.probe)
