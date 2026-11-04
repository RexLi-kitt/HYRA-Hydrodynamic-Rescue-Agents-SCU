"""Replay archived policies unchanged; diagnose task duplication and blocking."""
from pathlib import Path
import sys,json,argparse,collections
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-joint-v1'))
from suite import ResidualEnv,model_for,read,write,sha

def replay(model,case):
    e=ResidualEnv(np.load(case['field']['path'],mmap_mode='r'),case['scenario']);trace=[];streak=np.zeros(4,int);longest=0
    counts=collections.Counter()
    while True:
        obs=e.observe();mask,_,options,_=e.choices();a=model.act(obs,mask,True)[0]
        orders=[options[i][a[i]] for i in range(4)]
        targets=[o['person'] for o in orders if o['person']>=0]
        counts['duplicate_person_requests']+=len(targets)-len(set(targets))
        known=int(((e.belief==1)&e.discovered&(e.t-e.seen_at<=120)).sum())
        counts['patrol_while_known']+=sum(o['person']<0 for o in orders) if known else 0
        before=e.pos.copy();c=e.conflicts;t=e.t
        _,_,done,_=e.step(a)
        intended=np.array([np.linalg.norm(o['xy']-before[i])>1e-6 for i,o in enumerate(orders)])
        blocked=intended & (np.linalg.norm(e.pos-before,axis=1)<1e-9)
        conflict=e.conflicts-c
        streak=np.where(blocked,streak+e.cfg.step,0);longest=max(longest,int(streak.max()))
        counts['blocked_uav_decisions']+=int(blocked.sum());counts['conflict_steps']+=int(conflict>0);counts['steps']+=1
        if conflict or len(targets)>len(set(targets)) or blocked.any():
            trace.append(dict(t=t,actions=a.tolist(),known=known,targets=[o['person'] for o in orders],
                goals=[o['xy'].tolist() for o in orders],positions=before.tolist(),next_positions=e.pos.tolist(),
                conflicts=conflict,blocked=blocked.tolist()))
        if done:break
    return dict(case=case['id'],metrics=e.metrics(),diagnostics=dict(counts),max_stationary_with_goal_s=longest,trace=trace)

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);old=read(a.previous/'scenarios.json');results={}
    inputs=[a.previous/'protocol.json',a.previous/'scenarios.json',HERE/'diagnose.py']
    specs=[('balanced_initial','steady-happo',0),('happo_regressed','steady-happo',32),('mat_regressed','explore-mat',32)]
    write(out/'protocol.json',dict(scope='Fixed archived main12 cases; no retraining or policy/environment modifications; diagnostics not final test',
        hypotheses=['duplicate person task requests','stationary repeated conflicts despite nonzero goal','patrol selected while observed people pending'],
        specs=specs,source_hashes={str(p):sha(p) for p in inputs}))
    torch.set_num_threads(2)
    for label,folder,ep in specs:
        checkpoint=a.previous/folder/f'checkpoint-{ep:04}.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=model_for(state['learner']['algorithm'],state['recipe']);model.restore(state['learner']);model.eval()
        expected=read(a.previous/folder/f'development-{ep:04}.json')['main40']['rows'];rows=[]
        for case,ref in zip(old['development']['main40'],expected):
            r=replay(model,case)
            for k in ('supplied','conflicts','expired_unseen'):assert r['metrics'][k]==ref[k],(label,k)
            rows.append(r);write(out/'status.json',dict(status='diagnosing',variant=label,completed=len(rows),total=12))
        write(out/f'{label}.json',dict(checkpoint_sha256=sha(checkpoint),rows=rows))
        totals=collections.Counter()
        for r in rows:totals.update(r['diagnostics'])
        results[label]=dict(people=sum(r['metrics']['supplied'] for r in rows),conflicts=sum(r['metrics']['conflicts'] for r in rows),
            totals=dict(totals),max_stationary_with_goal_s=max(r['max_stationary_with_goal_s'] for r in rows))
        print(json.dumps({label:results[label]}),flush=True)
    write(out/'summary.json',results);write(out/'status.json',dict(status='completed',replays=36,archived_metrics_reproduced=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
