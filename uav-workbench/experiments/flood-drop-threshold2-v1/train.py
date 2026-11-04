"""User-specified depth2m OR speed1m/s threshold,32-episode pilots."""
from pathlib import Path
import sys,json,random,time,traceback,copy,argparse
from dataclasses import asdict,replace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-coordination-v1'))
from train_coordination import model_for,baseline_candidate,read,write,sha,seed,episode,evaluate,quality,Config,run_case,summarize
import numpy as np
import torch
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
 old=read(previous/'scenarios.json')
 trainfields=list({c['field']['parent']:c['field'] for c in old['train']}.values())
 devfields=list({c['field']['parent']:c['field'] for g in old['development'].values() for c in g}.values())
 assert len(trainfields)==16 and len(devfields)==6
 assert not {f['parent'] for f in trainfields}&{f['parent'] for f in devfields}
 fields={f['path']:f for f in trainfields+devfields}
 for path,f in fields.items():
  assert sha(path)==f['sha256'] and f['finite_nonnegative']
  assert abs(f['mass_residual_m3'])<=max(1e-6,abs(f['source_volume_m3'])*1e-9)
 def case(f,cfg,s,name):
  water=np.load(f['path'],mmap_mode='r',allow_pickle=False)
  scene=make_scenario(water,s,cfg,fall_end=960,pool_size=120)
  assert all(e['deadline']<=1320 for e in scene['events'])
  for e in scene['events']:
   forced=e['field'][0]>=2. or np.linalg.norm(e['field'][1:])>=1.
   assert (e['trigger']=='threshold')==forced
  return dict(id=f'{name}-{s}',field=f,scenario=scene)
 # All nine workload combinations enter training, in a fixed order, before outcomes.
 mix=[replace(PROFILES['main40'],people=n,window=w) for n in (40,60,80) for w in (180,240,300)]
 groups=dict(train=[case(trainfields[i%16],mix[i%9],2601000+i,'train') for i in range(32)],development={})
 for name,cfg in PROFILES.items():
  groups['development'][name]=[case(devfields[i%6],cfg,2711000+i,name) for i in range(12 if name=='main40' else 6)]
 # Exact fixed-event checks across population and deadline panels.
 for i in range(6):
  common={e['exposure_id']:e for e in groups['development']['people80'][i]['scenario']['events']}
  for g in groups['development'].values():
   for e in g[i]['scenario']['events']:
    assert all(e[k]==common[e['exposure_id']][k] for k in ('fall','xy','field','trigger'))
 c=groups['train'][0];water=np.load(c['field']['path'],mmap_mode='r')
 cfg=Config(**c['scenario']['config'])
 # Count the actual accepted roster, without pretending this is population risk.
 audit={}
 for name,g in [('train',groups['train']),*groups['development'].items()]:
  events=[e for c in g for e in c['scenario']['events']]
  audit[name]=dict(total=len(events),threshold=sum(e['trigger']=='threshold' for e in events),random=sum(e['trigger']=='random' for e in events),attempted_exposures=sum(c['scenario']['attempted_exposures'] for c in g))
 write(out/'trigger-audit.json',audit)
 write(out/'scenarios.json',groups)
 sources=dict(read(previous/'protocol.json')['source_hashes'])
 sources.update({str(p):sha(p) for p in HERE.glob('*.py')})
 p=dict(version='user-threshold2m-1mps-v1',date='2026-11-02',methods=['maac','mat','happo'],recipe=RECIPE,
  profiles={k:asdict(v) for k,v in PROFILES.items()},training_configs=[asdict(c) for c in mix],
  training_episodes_per_model=32,checkpoints=[0,16,32],seed=9401,
  primary='Number supplied on main40, tie all then ceil95%; pressure panels all retained.',
  scenario_meaning='User assumption: h>=2m OR hypot(u,v)>=1m/s forces fall for a sampled wet exposure; otherwise existing random_per_minute=.05 mechanism. Not an empirical human stability model.',
  legacy_config='Original conditional fixed-N generator restored. These fields also normalize existing planning water scores: all planners and learners share the change. Not an isolated event-only intervention; no raw water scaling.',
  fairness='Four complete planners, unchanged information/candidates/reservation. No deliberately weakened baseline. No scenario selected using learner advantage.',
  limitations='Fixed N accepted falls, not all residents being continuously monitored; threshold/random change conditions roster. Map unchanged for this threshold batch; 6 reused DEV hydraulic parents with new layouts, not final generalization. One model seed. Short pilot cannot establish15pp.',
  budget='Three models32 episodes each, no automatic extension;0/16/32 evaluation. One heavy process2threads,7200s cap.',
  success_rule='People first. Report both scene metrics against strongest people-first planner AND maxima of all planners.15pp is exploratory target only; no post-hoc switching metric.',
  next_rule='If32 people>16 and>=0 with no scene or pressure regression, consider64/128 on same frozen task; otherwise diagnose. Never weaken full baseline to manufacture progress.',
  source_hashes=sources,water_hashes={k:f['sha256'] for k,f in fields.items()},scenario_sha256=sha(out/'scenarios.json'))
 write(out/'protocol.json',p)
 write(out/'event-tests.json',dict(status='passed',threshold_labels_match_2m_or_1mps=True,paired_populations_and_windows=True,full_windows=True,disjoint_hydraulic_parents=True))
 write(out/'status.json',dict(status='prepared',training_episodes=0))

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
  for method in p['methods']:
   folder=out/method;folder.mkdir();seed(p['seed']);model=model_for(method,p['recipe']);replay=[];results={};best=None
   for ep in range(33):
    guard();status('training' if ep else 'initialization',method=method,episode=ep)
    if ep:
     model.train();metric,rows=episode(model,cases['train'][ep-1],True)
     if method=='maac':
      replay.extend(rows);del replay[:-2048]
      for _ in range(8):loss=model.update_off_policy(random.sample(replay,min(64,len(replay))))
     else:loss=model.update_on_policy(rows)
     model.check_finite();count+=1;metric.update(episode=ep,updates=model.updates,diagnostics=loss)
     with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
    if ep in p['checkpoints']:
     model.eval();status('evaluation',method=method,episode=ep)
     result={name:evaluate(model,g) for name,g in cases['development'].items()};evals+=36;results[ep]=result
     if ep==0:
      for name in result:
       for r,b in zip(result[name]['rows'],plans['balanced'][name]['rows']):assert r['persons']==b['persons']
     write(folder/f'development-{ep:04}.json',result)
     ck=folder/f'checkpoint-{ep:04}.pt';torch.save(dict(learner=model.checkpoint(),episode=ep,recipe=p['recipe'],replay=replay,protocol_sha256=sha(out/'protocol.json'),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())),ck)
     q=quality(result['main40'])
     if best is None or q>tuple(best['quality']):
      best=dict(episode=ep,quality=q,summaries={k:v['summary'] for k,v in result.items()},path=str(ck),sha256=sha(ck));write(folder/'best.json',best)
     print(json.dumps(dict(method=method,episode=ep,main=result['main40']['summary'])),flush=True)
   trend=dict(checkpoints={str(e):{k:v['summary'] for k,v in r.items()} for e,r in results.items()},people_up_16_to32=quality(results[32]['main40'])[0]>quality(results[16]['main40'])[0])
   write(folder/'trend.json',trend);leaders[method]=dict(best=best,trend=trend);write(out/'leaders.json',leaders)
  verify();write(out/'summary.json',dict(status='completed',training_episodes=count,evaluation_rollouts=evals,baselines={m:{n:v['summary'] for n,v in b.items()} for m,b in plans.items()},learners=leaders,source_and_water_hashes_unchanged=True,formal_test=False));status('completed')
 except BaseException as e:
  status('failed',error=repr(e));write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--previous',type=Path);a.add_argument('--prepare',action='store_true');args=a.parse_args()
 prepare(args.output,args.previous) if args.prepare else execute(args.output)
