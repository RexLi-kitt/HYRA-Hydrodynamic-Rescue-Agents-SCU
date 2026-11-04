"""Strong exact planning default and paired bias1/.1 pilots on frozen threshold2 cases."""
from pathlib import Path
import sys,json,random,time,traceback,copy,argparse,shutil
from dataclasses import asdict,replace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-coordination-v1'))
from train_coordination import model_for,baseline_candidate,read,write,sha,seed,episode,evaluate,quality,Config,run_case,summarize
import numpy as np
import torch
from strong_rollout import episode,evaluate,ResidualEnv
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
 shutil.copy2(previous/'scenarios.json',out/'scenarios.json')
 shutil.copy2(previous/'trigger-audit.json',out/'trigger-audit.json')
 p=copy.deepcopy(old)
 p.update(version='strong-default-bias-paired-v1',seed=9401,
  runs=[dict(id=f'{tag}-{m}',method=m,recipe=dict(id=tag,bias=b,lr=1e-4,gamma=.999)) for tag,b in [('control',1.),('lowbias',.1)] for m in ('maac','mat','happo')],
  previous=str(previous),previous_protocol_sha256=sha(previous/'protocol.json'),
  changes='Use exact nearest4 dispatcher as action8 default and identical physics, without extra reservation layer. Bias1 vs.1 paired from same model seed; all else fixed. Neither recipe uses new reward or hydraulic fields.',
  fairness='Keep all four full archived planners, rerun on unchanged cases. Episode0 must equal nearest4 dispatch person by person on all36 cases. Stronger planner choice based on prior DEV data, no independent-test claim.',
  budget='Six configurations32 episodes each=192;0/16/32 all36 cases,648 neural+144 planning=792 evaluation rollouts; no automatic extension.',
  limitations='Fixed previously used layouts for paired diagnosis, one training seed. Default-action bias lowering is a hypothesis, not a confirmed fix. All accepted falls in these fields are random;2m/1mps rule preserved but unexercised.',
  next_rule='Report every checkpoint and variant. Extend only after people32>16 and>=0 plus nonregression in scene and other panels. No weak planner substitute.')
 p['source_hashes'].update({str(f):sha(f) for f in HERE.glob('*.py')})
 write(out/'protocol.json',p);write(out/'status.json',dict(status='prepared',training_episodes=0))


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
   method=run['method'];key=run['id'];recipe=run['recipe']
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
     result={name:evaluate(model,g) for name,g in cases['development'].items()};evals+=36;results[ep]=result
     if ep==0:
      for name in result:
       for r,b in zip(result[name]['rows'],plans['dispatch'][name]['rows']):assert r['persons']==b['persons']
     write(folder/f'development-{ep:04}.json',result)
     ck=folder/f'checkpoint-{ep:04}.pt';torch.save(dict(learner=model.checkpoint(),episode=ep,recipe=recipe,replay=replay,protocol_sha256=sha(out/'protocol.json'),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())),ck)
     q=quality(result['main40'])
     if best is None or q>tuple(best['quality']):
      best=dict(episode=ep,quality=q,summaries={k:v['summary'] for k,v in result.items()},path=str(ck),sha256=sha(ck));write(folder/'best.json',best)
     print(json.dumps(dict(method=key,episode=ep,main=result['main40']['summary'])),flush=True)
   trend=dict(checkpoints={str(e):{k:v['summary'] for k,v in r.items()} for e,r in results.items()},people_up_16_to32=quality(results[32]['main40'])[0]>quality(results[16]['main40'])[0])
   write(folder/'trend.json',trend);leaders[key]=dict(best=best,trend=trend);write(out/'leaders.json',leaders)
  verify();write(out/'summary.json',dict(status='completed',training_episodes=count,evaluation_rollouts=evals,baselines={m:{n:v['summary'] for n,v in b.items()} for m,b in plans.items()},learners=leaders,source_and_water_hashes_unchanged=True,formal_test=False));status('completed')
 except BaseException as e:
  status('failed',error=repr(e));write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--previous',type=Path);a.add_argument('--prepare',action='store_true');args=a.parse_args()
 prepare(args.output,args.previous) if args.prepare else execute(args.output)
