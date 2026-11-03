"""Paired count/relative-count reward with shared100/60/40m visibility."""
from pathlib import Path
import sys,json,random,time,traceback,copy,argparse,shutil
from dataclasses import asdict,replace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-coordination-v1'))
from train_coordination import model_for,baseline_candidate,read,write,sha,seed,episode,evaluate,quality,Config,run_case,summarize
import numpy as np
import torch
from fog_rollout import episode,evaluate,ResidualEnv
import fog_rollout
from variable_dispatch import DispatchEnv as ReferenceEnv
from variable_environment import make_scenario

PROFILES={
 'main40':Config(people=40,duration=1320,window=240,fall_pattern='three_bursts',trigger_depth=2.,trigger_speed=1.),
 'people60':Config(people=60,duration=1320,window=240,fall_pattern='three_bursts',trigger_depth=2.,trigger_speed=1.),
 'people80':Config(people=80,duration=1320,window=240,fall_pattern='three_bursts',trigger_depth=2.,trigger_speed=1.),
 'window180':Config(people=40,duration=1320,window=180,fall_pattern='three_bursts',trigger_depth=2.,trigger_speed=1.),
 'window300':Config(people=40,duration=1320,window=300,fall_pattern='three_bursts',trigger_depth=2.,trigger_speed=1.)}
RECIPE=dict(id='steady',bias=1.,lr=1e-4,gamma=.999)

def prepare(out,previous):
 out.mkdir(parents=True,exist_ok=False)
 old=read(previous/'protocol.json');assert read(previous/'status.json')['status']=='completed'
 assert sha(previous/'scenarios.json')==old['scenario_sha256']
 source=read(previous/'scenarios.json')
 def fog(c,radius):
  c=copy.deepcopy(c);c['id']+=f'-view{radius}';c['scenario']['config']['sensor']=float(radius)
  return c
 cases=dict(train=[fog(c,(40,60,100)[i%3]) for i,c in enumerate(source['train'])],development={})
 for name,radius,count in [('view60',60,12),('view40',40,6),('view100',100,6)]:
  cases['development'][name]=[fog(c,radius) for c in source['development']['main40'][:count]]
 for i in range(6):
  assert cases['development']['view60'][i]['scenario']['events']==cases['development']['view40'][i]['scenario']['events']==cases['development']['view100'][i]['scenario']['events']
 write(out/'scenarios.json',cases);shutil.copy2(previous/'trigger-audit.json',out/'source-trigger-audit.json')
 p=copy.deepcopy(old)
 p.update(version='fog-count-vs-relative-count-v1',date='2026-11-03',seed=9601,
  runs=[dict(id=tag+'-'+m,method=m,recipe=dict(id=tag,bias=1.,lr=1e-4,gamma=1.)) for tag in ('absolute_people','relative_people') for m in ('maac','mat','happo')],
  profiles={name:dict(people=40,sensor_m=r,window=240,duration=1320,cases=n) for name,r,n in [('view60',60,12),('view40',40,6),('view100',100,6)]},
  primary='view60 people supplied; all andceil95% retained. Other views separate. All four full planners retained.',
  previous=str(previous),previous_protocol_sha256=sha(previous/'protocol.json'),scenario_sha256=sha(out/'scenarios.json'),
  changes='Training reward absolute=10*new supplied; relative=10*(new supplied-reference new supplied). Discount1 for exact final count difference. Same map/roster and visibility for all methods. No coverage/conflict shaping in either reward, physical safety unchanged.',
  reward_caveat='Subtracting action-independent same-case planning return does not inherently change expected optimal policy. It may change finite-sample credit/variance; not a guaranteed advantage. Controls differ ONLY by reference subtraction.',
  reference='Full nearest4 dispatcher independent trajectory on identical exogenous scenario and sensor radius. Reference totals only TRAIN reward, never learner observation. On policy early termination, subtract all remaining reference rescues to preserve exact final difference.',
  initialization='Bias1/zerohead defaults to exact nearest4 planning,0 is not learning.',
  budget='Six configurations32 RL episodes each=192;32 extra TRAIN reference runs;24DEV*3checkpoints*6=432 neural evaluations plus96 four-planner evaluations=528.',
  fairness='Same visibility cap, targets, water andphysics. Separate planner paths naturally produce different observations; neither gets hidden truth. No weakened planner.',
  limitations='Fixed radius visibility restriction only, not complete fog optics/weather/flight safety.2m/1mps rule unchanged; existing accepted events allrandom. Known DEV reuse/one model seed. No final test.',
  next_rule='No automatic extension. Need32>16 and>=0 in primary people with scene/other-view nonregression; include failures. No target15pp guarantee.')
 p['source_hashes'].update({str(f):sha(f) for f in HERE.glob('*.py')})
 p['source_hashes'].update({str(HERE.parent/'flood-drop-strong-default-v1/strong_env.py'):sha(HERE.parent/'flood-drop-strong-default-v1/strong_env.py')})
 write(out/'protocol.json',p);write(out/'status.json',dict(status='prepared',training_episodes=0))

def references(out,cases,status,guard):
 result={}
 for index,c in enumerate(cases):
  guard();e=ReferenceEnv(np.load(c['field']['path'],mmap_mode='r',allow_pickle=False),c['scenario']);curve=[0]
  while True:
   _,_,done,_=e.step(e.choices()[1]);curve.append(int((e.status==2).sum()))
   if done:break
  result[c['id']]=dict(cumulative=curve,final_supplied=curve[-1],metrics=e.metrics())
  status('training_reference',completed=index+1,total=len(cases))
 write(out/'training-reference.json',result);return result


def execute(out):
 p=read(out/'protocol.json');cases=read(out/'scenarios.json');start=time.monotonic();count=evals=0;plans={};leaders={}
 assert read(out/'readiness.json')['status']=='passed'
 def verify():
  assert sha(out/'scenarios.json')==p['scenario_sha256']
  for path,h in {**p['source_hashes'],**p['water_hashes']}.items():assert sha(path)==h,path
 def status(s,**kw):write(out/'status.json',dict(status=s,training_episodes=count,evaluation_rollouts=evals,elapsed_s=time.monotonic()-start,**kw))
 def guard():
  if (out/'STOP_AFTER_EPISODE').exists():raise InterruptedError('Stop requested')
  if time.monotonic()-start>7200:raise TimeoutError('Runtime budget exceeded')
 try:
  verify();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
  fog_rollout.REFERENCES=references(out,cases['train'],status,guard)
  for method in ('original','dispatch','balanced','nearest_reserved'):
   plans[method]={}
   for name,group in cases['development'].items():
    rows=[]
    for c in group:
     guard();status('planning',method=method,profile=name)
     rows.append(baseline_candidate(c,method=='nearest_reserved') if method in ('balanced','nearest_reserved') else run_case(c,method));evals+=1
    plans[method][name]=dict(rows=rows,summary=summarize(rows))
   write(out/f'planner-{method}.json',plans[method])
  for run in p['runs']:
   method=run['method'];key=run['id'];recipe=run['recipe'];fog_rollout.REWARD_MODE=recipe['id']
   folder=out/key;folder.mkdir();seed(p['seed']);model=model_for(method,recipe);replay=[];results={};best=None
   for ep in range(33):
    guard();status('training' if ep else 'initialization',method=key,episode=ep)
    if ep:
     model.train();metric,rows=episode(model,cases['train'][ep-1],True)
     if method=='maac':
      replay.extend(rows);del replay[:-2048]
      for _ in range(8):loss=model.update_off_policy(random.sample(replay,min(64,len(replay))))
     else:loss=model.update_on_policy(rows)
     model.check_finite();count+=1;metric.update(episode=ep,updates=model.updates,diagnostics=loss)
     with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
    if ep in p['checkpoints']:
     model.eval();status('evaluation',method=key,episode=ep)
     result={name:evaluate(model,g) for name,g in cases['development'].items()};evals+=24;results[ep]=result
     if ep==0:
      for name in result:
       for r,b in zip(result[name]['rows'],plans['dispatch'][name]['rows']):assert r['persons']==b['persons']
     write(folder/f'development-{ep:04}.json',result)
     ck=folder/f'checkpoint-{ep:04}.pt';torch.save(dict(learner=model.checkpoint(),episode=ep,recipe=recipe,replay=replay,protocol_sha256=sha(out/'protocol.json'),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())),ck)
     q=quality(result['view60'])
     if best is None or q>tuple(best['quality']):
      best=dict(episode=ep,quality=q,summaries={k:v['summary'] for k,v in result.items()},path=str(ck),sha256=sha(ck));write(folder/'best.json',best)
     print(json.dumps(dict(method=key,episode=ep,main=result['view60']['summary'])),flush=True)
   trend=dict(checkpoints={str(e):{k:v['summary'] for k,v in r.items()} for e,r in results.items()},people_up_16_to32=quality(results[32]['view60'])[0]>quality(results[16]['view60'])[0])
   write(folder/'trend.json',trend);leaders[key]=dict(best=best,trend=trend);write(out/'leaders.json',leaders)
  verify();write(out/'summary.json',dict(status='completed',training_episodes=count,evaluation_rollouts=evals,baselines={m:{n:v['summary'] for n,v in b.items()} for m,b in plans.items()},learners=leaders,source_and_water_hashes_unchanged=True,formal_test=False));status('completed')
 except BaseException as e:
  status('failed',error=repr(e));write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--previous',type=Path);a.add_argument('--prepare',action='store_true');args=a.parse_args()
 prepare(args.output,args.previous) if args.prepare else execute(args.output)
