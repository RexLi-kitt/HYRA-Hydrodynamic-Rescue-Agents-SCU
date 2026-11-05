"""Instrument existing rotated DEV rollouts; unchanged actions and weights."""
import time,json,copy
import numpy as np
import torch
from train import ROOT,HERE,read,write,sha,MemoryGRU,predictor,FlowFogEnv,FineEnv,transform
PRIOR=ROOT/'uav-retrain660-20261104/run01';OUT=ROOT/'uav-route-audit-20261104/run01'

def main():
 OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic()
 name='statistical-s202611171-e8';path=PRIOR/f'{name}.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
 m=MemoryGRU(ck['mean'],False);m.load_state_dict(ck['state']);m.eval();pred=predictor(m,np.array(ck['mu'],np.float32),np.array(ck['sd'],np.float32))
 sc=read(PRIOR/'scenarios.json')['test'][:3];old=[json.loads(l) for l in (PRIOR/'results.jsonl').read_text(encoding='utf-8').splitlines()]
 hashes={str(p):sha(p) for p in (path,PRIOR/'scenarios.json',PRIOR/'results.jsonl',HERE/'route_audit.py')}
 hashes.update(read(PRIOR/'protocol.json')['source_sha256'])
 write(OUT/'protocol.json',dict(scope='3existing rotated DEV cases,GRUseed171andfullflowplanner.6instrumented reruns,not new accuracy evidence. No policychange,trainingorfuture access. Offlinehiddenpositions separate NPZ.',source_sha256=hashes,
  metrics='Patrol target jumps>100m,adjacent patrol movement reversals cos<-.5,actual displacement,stoppedpatrolsteps,conflicts,10s coverage andfailedpersonclosestdistance duringactive life. Stored traces verify summary. Hidden quantities log only,never used bycontroller.',success='Each replay must reproduce prior supplied and per-personstatus exactly before interpretation. No stage acceptance orsceneexclusion.'))
 results=[]
 for c in sc:
  w,s=transform(np.load(c['field']['path'],mmap_mode='r'),c['scenario'])
  for method in ('flow_coverage',name):
   e=FlowFogEnv(w,s,method) if method=='flow_coverage' else FineEnv(w,s,pred,.25)
   traces=[];truth=[];closest=np.full(40,np.inf);prevgoal=None;prevpatrol=None;prevmove=None;jumps=reverse=patroln=stopped=0;distance=0.
   while True:
    if time.monotonic()-start>600 or (OUT/'STOP').exists():raise InterruptedError('budgetorSTOP')
    mask,chosen,options,_=e.choices();orders=[options[i][chosen[i]] for i in range(4)]
    goals=np.stack([o['xy'] for o in orders]);patrol=np.array([o['person']<0 and o['region']>=0 for o in orders]);pos=e.pos.copy();oldconf=e.conflicts
    active=e.status==1;d=np.linalg.norm(e.person[:,None]-pos[None],axis=-1).min(1);closest[active]=np.minimum(closest[active],d[active])
    if prevgoal is not None:jumps+=int(((np.linalg.norm(goals-prevgoal,axis=1)>100)&patrol&prevpatrol).sum())
    if e.t%10==0:truth.append((e.t,e.person.copy(),e.status.copy()))
    done=e.step(chosen)[2];move=e.pos-pos;length=np.linalg.norm(move,axis=1);distance+=float(length.sum());patroln+=int(patrol.sum());stopped+=int(((length<1e-6)&patrol).sum())
    if prevmove is not None:
     denom=np.linalg.norm(prevmove,axis=1)*length;cos=np.sum(prevmove*move,axis=1)/np.maximum(denom,1e-12)
     reverse+=int(((cos<-.5)&(denom>1e-6)&patrol&prevpatrol).sum())
    traces.append(dict(time=int(e.t),position=e.pos.tolist(),goals=goals.tolist(),patrol=patrol.tolist(),conflicts=int(e.conflicts-oldconf),new_total_seen=int(e.discovered.sum()),covered_cells=int(e.known.sum())))
    prevgoal=goals;prevpatrol=patrol;prevmove=move
    if done:break
   original=next(r for r in old if r['phase']=='rotation' and r['case']==c['id'] and r['method']==method)
   assert int((e.status==2).sum())==original['supplied']
   assert all(int(e.status[p['id']])==p['status'] and float(e.delivered_at[p['id']])==p['supplied_at'] and float(e.first_seen[p['id']])==p['first_seen'] for p in original['persons'])
   stem=c['id']+'-'+method
   write(OUT/(stem+'-trace.json'),traces)
   np.savez_compressed(OUT/(stem+'-diagnostic-truth.npz'),time=np.array([t[0] for t in truth]),position=np.stack([t[1] for t in truth]),status=np.stack([t[2] for t in truth]))
   result=dict(case=c['id'],method=method,supplied=original['supplied'],duration=e.t,patrol_steps=patroln,patrol_target_jumps=jumps,patrol_reversals=reverse,patrol_stopped_steps=stopped,conflicts=int(e.conflicts),distance=distance,covered_cells=int(e.known.sum()),failed=[dict(id=int(i),closest_while_active=float(closest[i]),seen=float(e.first_seen[i]),fall=int(e.fall[i]),deadline=int(e.deadline[i])) for i in np.flatnonzero(e.status!=2)],replay_exact=True)
   results.append(result);write(OUT/'status.json',dict(state='running',completed=len(results),elapsed_s=time.monotonic()-start));print(json.dumps(result),flush=True)
 for p,h in hashes.items():assert sha(p)==h,p
 write(OUT/'summary.json',dict(results=results,elapsed_s=time.monotonic()-start,source_weights_unchanged=True));write(OUT/'status.json',dict(state='completed',completed=len(results)))

if __name__=='__main__':main()
