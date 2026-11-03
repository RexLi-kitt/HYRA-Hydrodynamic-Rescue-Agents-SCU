"""Registered repair for learned unforced waiting. Retain planning exactly at0.
No score threshold tuning; all five levels and all three methods remain in scope.
"""
from pathlib import Path
import argparse,copy,os,random,time,traceback,json
import numpy as np
import torch
from residual_env import ResidualEnv,initialize_planning_default
from train import read,write,sha,seed,Learner,summary,quality,METHODS,paired_summary
from metrics import summary,quality

HERE=Path(__file__).resolve().parent

def episode(model,case,collect=False):
    env=ResidualEnv(np.load(case['field']['path'],mmap_mode='r'),case['scenario'])
    obs=env.observe();mask,_,options,_=env.choices();done=False;rows=[];ret=0.;free=used=0
    while not done:
        action,logs,value=model.act(obs,mask,not collect)
        choices=mask.sum(1)>1;free+=int(choices.sum())
        used+=sum(bool(choices[i] and (options[i][action[i]]['person']!=options[i][8]['person'] or
            not np.allclose(options[i][action[i]]['xy'],options[i][8]['xy'],rtol=0,atol=1e-9))) for i in range(4))
        nxt,reward,done,_=env.step(action);nm,_,noptions,_=env.choices();ret+=reward
        if collect:rows.append(dict(obs=obs,mask=mask,action=action,reward=reward,done=float(done),next_obs=nxt,next_mask=nm,logp=logs,value=value))
        obs=nxt;mask=nm;options=noptions
    for i,r in enumerate(rows):r['next_action']=rows[i+1]['action'] if i+1<len(rows) else np.full(4,8,dtype=np.int64)
    result=dict(**env.metrics(),case=case['id'],parent=case['field']['parent'],level=case['level'],
        return_sum=ret,alternative_task_choice_fraction=used/max(1,free),
        expired_seen_at_deadline=int((env.status==3).sum())-env.expired_unseen)
    assert result['supplied']+result['expired']+result['outside']==20
    if not collect:result['persons']=[dict(id=j,fall=float(env.fall[j]),deadline=float(env.deadline[j]),first_seen=float(env.first_seen[j]),
        supplied_at=float(env.delivered_at[j]),status=int(env.status[j])) for j in range(20)]
    return result,rows

def evaluate(model,cases):
    saved=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    try:
        rows=[episode(model,c)[0] for c in cases];levels=list(dict.fromkeys(c['level'] for c in cases))
        return dict(rows=rows,summary=summary(rows),by_level={lv:summary([r for r in rows if r['level']==lv]) for lv in levels})
    finally:random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2])

def run(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def status(state,**kw):write(out/'status.json',dict(status=state,pid=os.getpid(),elapsed_s=time.monotonic()-start,**kw))
    def guard():
        if (out/'STOP_AFTER_EPISODE').exists():raise InterruptedError('Stop requested')
        if time.monotonic()-start>3600:raise TimeoutError('Pilot cap')
    try:
        assert read(a.previous/'status.json')['status']=='completed'
        p=read(a.previous/'protocol.json');cases=read(a.previous/'scenarios.json')
        hashes=dict(p['source_hashes']);hashes.update({str(f):sha(f) for f in HERE.glob('*.py')})
        for path,h in hashes.items():assert sha(path)==h
        for path,h in p['water_hashes'].items():assert sha(path)==h
        plans={k:read(a.previous/f'planner-{k}.json') for k in ('original','dispatch')}
        for result in plans.values():
            result['summary']=summary(result['rows'])
            result['by_level']={lv['name']:summary([r for r in result['rows'] if r['level']==lv['name']]) for lv in p['levels']}
        protocol=dict(version='planning-default-residual-64-01',date='2026-11-02',methods=list(METHODS),episodes=64,seed=8301,
            diagnosis='Prior MAAC episode32 selected free hold for all4 UAVs, zero movement for40s despite valid patrol goals; failure retained.',
            change='Choice8 executes the intact coordinated planner suggestion instead of unforced hold. Other observed task candidates remain. Initialize heads to exact default, bias5.',
            interpretation='Hybrid planning+learned task correction. Episode0 is exactly planning, NOT a trained neural gain. Only paired excess over the stronger planner counts.',
            reward_unchanged=True,physical_conditions_unchanged=True,roster_unchanged=True,baseline_unmodified=True,
            primary_metric='Maximize total people supplied before deadlines; ties use all20 scenes then at-least19 scenes; exact ties keep earlier',
            secondary_metrics=['20/20 scene completion','at least19/20 scene completion'],
            metric_amendment='Explicit user request2026-11-02 during preceding dispatch pilot; change of measurement is not algorithm improvement. Original all-person records remain.',
            prior_scene_gap_targets=p['targets_percentage_points'],
            gap_interpretation='User confirmed people first and15-point target for scene success. Here the15/25/50-point aspirations apply to ALL20 scene completion; at-least19 remains reported. They cannot override a lower total people count.',
            no_bc_updates=True,no_prior_neural_weights=True,source_hashes=hashes,water_hashes=p['water_hashes'],
            parent_protocol_sha256=sha(a.previous/'protocol.json'),scenarios_sha256=sha(a.previous/'scenarios.json'),
            levels=p['levels'],scope='Development repair experiment,1 training seed,6 hydraulic parents,12 layouts across5 levels; not a formal final test',
            preregistered_cost='3x64 training; one60-case planning identity check;360 learned-checkpoint evaluations. Episode0 results reused across3 methods only after exact-default proof.',
            stop='No automatic256/1024 or further parameter changes. If no gain, retain planning and report failure.')
        write(out/'protocol.json',protocol);write(out/'scenarios.json',cases)
        for k,v in plans.items():write(out/f'planner-{k}.json',v)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed(8301)
        leaders={};identity=None
        for m in METHODS:
            guard();seed(8301);folder=out/m;folder.mkdir();model=Learner(m,760);initialize_planning_default(model,5.);replay=[];best=None
            for ep in range(65):
                guard()
                if ep:
                    model.train();metric,rows=episode(model,cases['train'][ep-1],True)
                    if m=='maac':
                        replay.extend(rows);del replay[:-2048]
                        for _ in range(8):loss=model.update_off_policy(random.sample(replay,min(64,len(replay))))
                    else:loss=model.update_on_policy(rows)
                    metric.update(episode=ep,updates=model.updates,diagnostics=loss)
                    with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
                    status('training',method=m,episode=ep,completed_methods=list(leaders))
                if ep in (0,32,64):
                    status('evaluation',method=m,episode=ep,completed_methods=list(leaders));model.eval()
                    if ep==0 and identity is not None:
                        result=copy.deepcopy(identity);result['initial_result_reused_after_exact_default_proof']=True
                    else:
                        result=evaluate(model,cases['development'])
                        if ep==0:
                            refs={(r['level'],r['case']):r for r in plans['dispatch']['rows']}
                            for r in result['rows']:
                                ref=refs[(r['level'],r['case'])]
                                for key in ['supplied','expired','outside','expired_unseen','conflicts','max_drift_m','coverage','return_sum']:
                                    assert abs(r[key]-ref[key])<1e-9,(key,r[key],ref[key])
                                assert r['persons']==ref['persons']
                                assert r['alternative_task_choice_fraction']==0
                            identity=copy.deepcopy(result);write(out/'planning-identity.json',dict(status='passed',cases=60,full_person_outcomes_match=True))
                    write(folder/f'development-{ep:04}.json',result)
                    ck=folder/f'checkpoint-{ep:04}.pt'
                    torch.save(dict(learner=model.checkpoint(),episode=ep,protocol_sha256=sha(out/'protocol.json'),
                        exact_planning_initialization=True,bc_updates=0,replay=replay,
                        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())),ck)
                    if best is None or quality(result)>tuple(best['quality']):
                        best=dict(episode=ep,quality=quality(result),summary=result['summary'],path=str(ck),sha256=sha(ck));write(folder/'best.json',best)
                    print(json.dumps(dict(method=m,episode=ep,summary=result['summary'])),flush=True)
            leaders[m]=best;write(out/'leaders.json',leaders)
        for path,h in hashes.items():assert sha(path)==h
        for path,h in p['water_hashes'].items():assert sha(path)==h
        ref=max(plans,key=lambda k:quality(plans[k]))
        records=[dict(r,method='planner') for r in plans[ref]['rows']]
        for m,b in leaders.items():records.extend(dict(r,method=m) for r in read(out/m/f'development-{b["episode"]:04}.json')['rows'])
        write(out/'summary.json',dict(status='completed',training_episodes=192,development_rollouts_executed=420,
            initial_results_reused=120,stronger_planning_baseline=ref,baselines={k:v['summary'] for k,v in plans.items()},
            learners=leaders,paired_differences={m:paired_summary(records,m) for m in METHODS},
            paired_people_differences={m:paired_summary([dict(r,scene_success=r['supply_fraction']) for r in records],m) for m in METHODS},
            paired_at_least_19_differences={m:paired_summary([dict(r,scene_success=r['supplied']>=19) for r in records],m) for m in METHODS},
            planning_identity_verified=True,source_hashes_unchanged=True,formal_claim=False))
        status('completed',training_episodes=192,development_rollouts_executed=420,initial_results_reused=120)
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required')
    run(a)
