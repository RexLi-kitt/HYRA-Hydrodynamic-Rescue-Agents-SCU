"""Privileged diagnostic only: separate count prediction and search-actuation limits."""
import copy,json,time,traceback
import numpy as np
from train import ROOT,HERE,read,write,sha,transform
from joint_avoidance import SafeFineEnv,SafeFlowEnv,swept_distances
from joint_trial import aggregate
OUT=ROOT/'uav-information-probe-20261104/run01'
PRIOR=ROOT/'uav-joint-avoidance-20261104/run01'

class OracleResidual(SafeFineEnv):
    """Current hidden counts, same 30s cache/flow blend/25% residual as GRU."""
    def __init__(self,w,s):
        super().__init__(w,s,lambda _:self.sector_counts(100).reshape(-1),.25)

class OracleDirect(SafeFlowEnv):
    def __init__(self,w,s,size):
        self.oracle_size=size
        super().__init__(w,s,'flow_coverage')
    def search_score(self,stale,risk):
        base=super().search_score(stale,risk)
        counts=self.sector_counts(self.oracle_size);cells=self.oracle_size//10
        return base*(1+8*np.repeat(np.repeat(counts,cells,0),cells,1))

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[]
    cases=read(PRIOR/'confirmation-scenarios.json')[:6]
    sources={**read(PRIOR/'protocol.json')['source_sha256'],str(HERE/'information_probe.py'):sha(HERE/'information_probe.py')}
    inputs={str(PRIOR/n):sha(PRIOR/n) for n in ('confirmation-scenarios.json','results.jsonl','summary.json')}
    water={c['field']['path']:sha(c['field']['path']) for c in cases}
    write(OUT/'protocol.json',dict(source_sha256=sources,input_sha256=inputs,water_sha256=water,
        purpose='Diagnostic only, NOT deployable performance or theoretical upper bound. Old 20-parent confirmation FAILED and is now development. Use first6 by index, not outcome.',
        config=cases[0]['scenario']['config'],cases=[c['id'] for c in cases],
        arms=['oracle-residual25: true current100m counts, identical30s cache/flow blend/alpha.25',
              'oracle-direct100: fullflow score times(1+8*true current100m counts)',
              'oracle-direct20: same direct score with20m counts'],
        common='Shared joint avoidance, speed10, sensor20, service10s, no exact hidden coordinates given to known-person assignment; no future births or deadlines. Hidden counts confined to explicitly privileged classes.',
        decision='6original+first3rotations x3diagnostic arms=27new. Reuse27baseline/GRU rows. If exact counts with unchanged residual beat both planners by>=15pp p95scene and >=85%people on original and no people deficit on rotation, prediction is a viable bottleneck. If only strong100m works, investigate score coupling. If only20m works, output resolution needs reconsideration. These finite heuristics are not optimal bounds; no oracle result is claimed as neural gain.',
        cost='OneCPU workload,1200s guard,STOP, no training/physics change/production/push. Source and water locks rechecked.'))
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('STOP')
        if time.monotonic()-start>1200:raise TimeoutError('1200s')
    def record(r):
        rows.append(r)
        with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
    try:
        for r in [json.loads(s) for s in (PRIOR/'results.jsonl').read_text(encoding='utf-8').splitlines()]:
            if r['case'] in [c['id'] for c in cases] and (r['phase']=='confirmation' or (r['phase']=='confirmation_rotation' and r['case'] in [c['id'] for c in cases[:3]])):
                q=copy.deepcopy(r);q['reused']=True;record(q)
        for phase,cc,rot in [('confirmation',cases,False),('confirmation_rotation',cases[:3],True)]:
            for c in cc:
                for method in ['oracle-residual25','oracle-direct100','oracle-direct20']:
                    guard();write(OUT/'status.json',dict(state='running',case=c['id'],phase=phase,method=method,completed_new=sum(not r['reused'] for r in rows),elapsed_s=time.monotonic()-start))
                    w=np.load(c['field']['path'],mmap_mode='r');s=c['scenario']
                    if rot:w,s=transform(w,s)
                    e=OracleResidual(w,s) if method=='oracle-residual25' else OracleDirect(w,s,int(method.removeprefix('oracle-direct')))
                    minimum=float('inf')
                    while True:
                        if e.t%120==0:guard()
                        p=e.pos.copy();done=e.step(e.choices()[1])[2]
                        minimum=min(minimum,float(swept_distances(p,e.pos).min()));assert minimum>=5-1e-8
                        if done:break
                    pp=[dict(id=i,fall=int(e.fall[i]),deadline=int(e.deadline[i]),first_seen=float(e.first_seen[i]),supplied_at=float(e.delivered_at[i]),status=int(e.status[i])) for i in range(40)]
                    for p in pp:
                        if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
                    assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
                    r=dict(phase=phase,case=c['id'],parent=c['field']['parent'],method=method,supplied=int((e.status==2).sum()),people=40,all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=38),persons=pp,reused=False,privileged=True,min_swept_separation=minimum,refloats=int(e.refloat_counts.sum()),conflicts=int(e.conflicts))
                    record(r);print(json.dumps({k:v for k,v in r.items() if k!='persons'}),flush=True)
        results={p:{m:aggregate([r for r in rows if r['phase']==p and r['method']==m]) for m in sorted({r['method'] for r in rows})} for p in ['confirmation','confirmation_rotation']}
        for p,h in {**sources,**inputs,**water}.items():assert sha(p)==h,p
        write(OUT/'summary.json',dict(results=results,elapsed_s=time.monotonic()-start,new_rollouts=27,reused_rollouts=27,source_inputs_unchanged=True,privileged_diagnostic_only=True))
        write(OUT/'status.json',dict(state='completed',elapsed_s=time.monotonic()-start))
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
