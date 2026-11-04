"""Pre-registered paired old-interface ablation; does not select a new model."""
import json,time,traceback
import numpy as np
import torch
from train import ROOT,HERE,read,write,sha,transform
from window_adapter import load
from joint_avoidance import SafeFineEnv,swept_distances
from joint_trial import aggregate
OUT=ROOT/'uav-count-interface-ablation-20261104/run01'
CONFIRM=ROOT/'uav-count-confirmation-20261104/run01'

def register():
    OUT.mkdir(parents=True,exist_ok=False)
    p=read(CONFIRM/'protocol.json');assert p['chosen']=='old142-adapter'
    write(OUT/'protocol.json',dict(source_sha256={**p['source_sha256'],str(HERE/'coupling_ablation.py'):sha(HERE/'coupling_ablation.py')},
        confirmation_protocol_sha256=sha(CONFIRM/'protocol.json'),registration_time=time.time(),
        current_confirmation_status=read(CONFIRM/'status.json'),
        purpose='Before reading final fresh outcomes, preregister same selectedold142 weights andnormalization, same20freshscenes+5rotation, reverting ONLY last score mapping tooriginalresidual25. Sharedavoidance/cache/flowblend/physicalscenario unchanged. Directly attribute paired differences to interface ratherthan selectingdifferentcheckpoint.',
        report='25additionalold-interface runs, compare everyperson andsupplied/p95/full with main CountCoupled results. No parameterselection/gatechange/freshTESTpromotion; original75run objective stands.',
        budget='OneCPU4aftermaincompletion,STOP/1200s guard,no newfields/training/production/push.'))

def main():
    assert read(CONFIRM/'status.json')['state']=='completed';protocol=read(OUT/'protocol.json');assert sha(CONFIRM/'protocol.json')==protocol['confirmation_protocol_sha256']
    start=time.monotonic();torch.set_num_threads(4);pred=load(202611142,True)[0];cases=read(CONFIRM/'scenarios.json');rows=[];inputs=read(CONFIRM/'input-lock.json')
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('STOP')
        if time.monotonic()-start>1200:raise TimeoutError('1200s')
    try:
        for phase,cc,rot in [('confirmation',cases,False),('rotation',cases[:5],True)]:
            for c in cc:
                guard();write(OUT/'status.json',dict(state='running',case=c['id'],phase=phase,completed=len(rows),elapsed_s=time.monotonic()-start))
                w=np.load(c['field']['path'],mmap_mode='r');s=c['scenario']
                if rot:w,s=transform(w,s)
                e=SafeFineEnv(w,s,pred,.25);minimum=float('inf')
                while True:
                    if e.t%120==0:guard()
                    p=e.pos.copy();done=e.step(e.choices()[1])[2];minimum=min(minimum,float(swept_distances(p,e.pos).min()));assert minimum>=5-1e-8
                    if done:break
                pp=[dict(id=i,fall=int(e.fall[i]),deadline=int(e.deadline[i]),first_seen=float(e.first_seen[i]),supplied_at=float(e.delivered_at[i]),status=int(e.status[i])) for i in range(40)]
                for p in pp:
                    if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
                assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
                r=dict(phase=phase,case=c['id'],parent=c['field']['parent'],method='old142-original-residual25',supplied=int((e.status==2).sum()),people=40,all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=38),persons=pp,min_swept_separation=minimum,refloats=int(e.refloat_counts.sum()))
                rows.append(r)
                with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
                print(json.dumps({k:v for k,v in r.items() if k!='persons'}),flush=True)
        new=[json.loads(v) for v in (CONFIRM/'results.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(v)['method']=='old142-adapter']
        results={p:{'old_interface':aggregate([r for r in rows if r['phase']==p]),'new_interface':aggregate([r for r in new if r['phase']==p])} for p in ['confirmation','rotation']}
        for p,h in {**protocol['source_sha256'],**inputs['water_sha256']}.items():assert sha(p)==h,p
        write(OUT/'summary.json',dict(results=results,elapsed_s=time.monotonic()-start,new_rollouts=25,source_inputs_unchanged=True))
        write(OUT/'status.json',dict(state='completed',elapsed_s=time.monotonic()-start))
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    import sys
    register() if '--register' in sys.argv else main()
