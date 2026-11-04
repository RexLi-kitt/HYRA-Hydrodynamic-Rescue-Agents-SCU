from pathlib import Path
import argparse,copy,json,sys,os,time,traceback
import numpy as np
import torch
from reservation_env import ResidualEnv,ReservationEnv,NearestReservationEnv
from suite import read,write,sha,model_for
import rollout
rollout.ResidualEnv=ResidualEnv

HERE=Path(__file__).resolve().parent

def test(case):
    water=np.load(case['field']['path'],mmap_mode='r');env=ReservationEnv(water,case['scenario'])
    rows=[]
    for i in range(4):
        row=[None]*9
        for r in range(4):row[4+r]=dict(xy=np.array([250+500*(r%2),250+500*(r//2)],float),person=-1,region=r,utility=1.)
        row[8]=dict(xy=env.pos[i].copy(),person=-1,region=-1,utility=0.)
        rows.append(row)
    orders=[rows[i][4] for i in range(4)]
    fixed,changes=env.reserve(orders,rows);assert len({o['region'] for o in fixed})==4 and len(changes)==3
    unique=[rows[i][4+i] for i in range(4)];same,changes=env.reserve(unique,rows)
    assert not changes and all(np.array_equal(o['xy'],r['xy']) for o,r in zip(unique,same))
    env.visible[0]=True;env.seen_xy[0]=env.pos[0]+[1,0]
    target=dict(xy=env.seen_xy[0].copy(),person=0,region=-1,utility=1.)
    fixed,changes=env.reserve([target,target,unique[2],unique[3]],rows)
    assert fixed[0]['person']==0 and fixed[1]['person']!=0
    before=env.reserve(orders,rows)
    env.fall+=5;env.deadline+=5;env.person[:]=800
    after=env.reserve(orders,rows)
    assert before[1]==after[1] and all(np.array_equal(x['xy'],y['xy']) for x,y in zip(before[0],after[0]))
    # No-reservation path must reproduce the parent's first8 physical steps.
    from candidate_env import DispatchEnv
    parent=DispatchEnv(water,case['scenario']);guard=ReservationEnv(water,case['scenario'])
    for _ in range(8):
        a=parent.choices()[1];b=guard.choices()[1];assert np.array_equal(a,b)
        parent.step(a);guard.step(b);assert np.array_equal(parent.pos,guard.pos)
    return dict(status='passed',duplicate_patrol_unique=True,duplicate_person_service_preserved=True,unique_orders_unchanged=True,hidden_truth_ignored=True,initial_physical_trace_unchanged=True)

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);cases=read(a.previous/'scenarios.json')['development']['main40']
    specs=[('balanced_initial','steady-happo',0),('happo_regressed','steady-happo',32),('mat_regressed','explore-mat',32)]
    p=dict(scope='Single shared reservation repair; frozen weights/12 main cases; not retraining or algorithm gain',
        source_hashes={str(f):sha(f) for f in HERE.glob('*.py')},previous_protocol_sha256=sha(a.previous/'protocol.json'),
        rules='one observed person perUAV assignment; one patrol region perUAV; service/nearer request precedence; unresolved loser holds; unchanged collision geometry.',
        decision='Proceed to32 training only if at least one regressed policy gains people and initial balanced planning does not lose people; report guarded nearest4 baseline as well. Otherwise no automatic training.',specs=specs)
    write(out/'protocol.json',p);write(out/'readiness.json',test(cases[0]));torch.set_num_threads(2);summaries={};start=time.monotonic()
    try:
        for label,folder,ep in specs:
            path=a.previous/folder/f'checkpoint-{ep:04}.pt';state=torch.load(path,map_location='cpu',weights_only=False)
            model=model_for(state['learner']['algorithm'],state['recipe']);model.restore(state['learner']);model.eval()
            result=rollout.evaluate(model,cases);result['checkpoint_sha256']=sha(path);write(out/f'{label}.json',result);summaries[label]=result['summary']
            write(out/'status.json',dict(status='probing',pid=os.getpid(),completed=len(summaries),elapsed_s=time.monotonic()-start))
            print(json.dumps({label:result['summary']}),flush=True)
        rows=[]
        for case in cases:
            e=NearestReservationEnv(np.load(case['field']['path'],mmap_mode='r'),case['scenario'])
            while True:
                _,_,done,_=e.step(e.choices()[1])
                if done:break
            rows.append(dict(**e.metrics(),case=case['id']))
        result=dict(rows=rows,summary=rollout.summarize(rows));write(out/'nearest_reserved.json',result);summaries['nearest_reserved']=result['summary']
        improved=summaries['happo_regressed']['people_supplied']>53 or summaries['mat_regressed']['people_supplied']>375
        approved=improved and summaries['balanced_initial']['people_supplied']>=457
        for f,h in p['source_hashes'].items():assert sha(f)==h
        write(out/'summary.json',dict(results=summaries,training_gate_passed=approved,original_people=dict(balanced_initial=457,happo_regressed=53,mat_regressed=375,nearest=462),
            interpretation='Frozen-policy repair benefit, not learned improvement. Matched guarded planners required for retraining.'))
        write(out/'status.json',dict(status='completed',replays=48,training_gate_passed=approved,elapsed_s=time.monotonic()-start))
    except BaseException as e:write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
