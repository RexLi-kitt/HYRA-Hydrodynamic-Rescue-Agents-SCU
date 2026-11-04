from pathlib import Path
import sys,json,time,copy,traceback,argparse
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-coordination-v1'))
from train_coordination import read,write,sha,summarize
from search_env import SearchEnv
MODES=['original','wet_age','coverage100','oracle200','oracle100']

def run(out,previous):
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();p0=read(previous/'protocol.json');cases=read(previous/'scenarios.json')['development'];results={m:{} for m in MODES};count=0
 def status(state,**kw):write(out/'status.json',dict(status=state,evaluation_rollouts=count,elapsed_s=time.monotonic()-start,**kw))
 def guard():
  if (out/'STOP').exists():raise InterruptedError('Stop requested')
  if time.monotonic()-start>3600:raise TimeoutError('Runtime cap')
 sources={**p0['source_hashes'],**{str(f):sha(f) for f in HERE.glob('*.py')}}
 protocol=dict(version='regional-search-probe-v1',date='2026-11-03',scenario_sha256=sha(previous/'scenarios.json'),source_hashes=sources,water_hashes=p0['water_hashes'],modes=MODES,
  scope='24 unchanged DEV cases;60m12,40m6,100m6.120 rollouts,0 training. Only patrol score changes; known-person nearest4 priority and execution unchanged.',
  score='Original stale*(1+risk).wet_age prior knownwet2/knowndry.5/unknown1. coverage100 average capped search-age*prior within100m tiles, then multiply original by.25+coarse. Oracle multiplies original by1+8*current undiscovered pending count per200/100m tile.',
  oracle='NONDEPLOYABLE diagnostic; current hidden pending counts aggregated only, no exact coordinates/future events/deadlines offered. Same persons/service/sensing/physics. Not an optimal upper bound or learned result.',
  gate='To justify learned regional inference: oracle main supplied>best of four old full planners AND new deployable heuristics, main expired_unseen less than its best deployable comparator, no people regression vs strongest deployable in either pressure visibility. If not, inspect search/action execution; do not force training.',
  future='Only if this gate passes, separately test whether observation history predicts sector demand better than simple wet-area/age prior on held-out TRAIN parents before considering a temporal network.',
  limitations='Reused DEV, no independent test, oracle artificially privileged; score constants preregistered, not tuned for observed performance. Negative result rejects this scorer/candidate setup only. Fixed threshold2m/1mps remains unexercised in original accepted events.')
 write(out/'protocol.json',protocol);write(out/'cases.json',cases)
 def verify():
  assert sha(previous/'scenarios.json')==protocol['scenario_sha256']
  for path,h in {**sources,**p0['water_hashes']}.items():assert sha(path)==h,path
 try:
  verify();first=cases['view60'][0];water=np.load(first['field']['path'],mmap_mode='r',allow_pickle=False)
  for mode in MODES[:3]:
   e=SearchEnv(water,first['scenario'],mode);a=e.choices();e.person[:]=999;e.fall[:]+=100;e.deadline[:]+=100;e.status[:]=1;b=e.choices()
   assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1]) and np.array_equal(a[3],b[3]),'Hidden truth leak'
  e=SearchEnv(water,first['scenario'],'oracle100');e.status[:]=0;e.status[0]=1;e.person[0]=[110,110];a=e.sector_counts(100);e.person[0]=[190,190];b=e.sector_counts(100);assert np.array_equal(a,b)
  e.person[0]=[210,110];assert not np.array_equal(a,e.sector_counts(100))
  e.status[0]=0;assert e.sector_counts(100).sum()==0
  write(out/'readiness.json',dict(status='passed',deployable_hidden_truth_isolated=True,oracle_within_tile_invariant=True,oracle_ignores_future_people=True))
  full={m:read(previous/f'planner-{m}.json') for m in ('original','dispatch','balanced','nearest_reserved')}
  for panel,group in cases.items():
   for m in MODES:results[m][panel]=[]
   for ci,c in enumerate(group):
    water=np.load(c['field']['path'],mmap_mode='r',allow_pickle=False)
    for mode in MODES:
     guard();status('evaluating',mode=mode,panel=panel,case=c['id']);env=SearchEnv(water,c['scenario'],mode)
     while True:
      _,_,done,_=env.step(env.choices()[1])
      if done:break
     row=dict(**env.metrics(),case=c['id'],privileged=mode.startswith('oracle'))
     row['persons']=[dict(id=j,fall=int(env.fall[j]),deadline=int(env.deadline[j]),first_seen=float(env.first_seen[j]),supplied_at=float(env.delivered_at[j]),status=int(env.status[j])) for j in range(env.cfg.people)]
     assert row['supplied']+row['expired']+row['outside']==env.cfg.people
     for p in row['persons']:
      if p['status']==2:assert p['fall']+env.cfg.drop_seconds<=p['supplied_at']<=p['deadline']
     if mode=='original':
      old=full['dispatch'][panel]['rows'][ci]
      for k in ('supplied','expired','outside','conflicts','expired_unseen'):assert row[k]==old[k],k
      for p,q in zip(row['persons'],old['persons']):
       for k in p:assert p[k]==q[k],k
     results[mode][panel].append(row);count+=1
     with (out/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(panel=panel,mode=mode,**row),allow_nan=False)+'\n')
  summary={m:{n:summarize(rows) for n,rows in panels.items()} for m,panels in results.items()}
  baselines={m:{n:v['summary'] for n,v in panels.items()} for m,panels in full.items()}
  deploy={**baselines,**{m:summary[m] for m in ('wet_age','coverage100')}}
  strongest={n:max(deploy,key=lambda m:(deploy[m][n]['people_supplied'],-deploy[m][n]['expired_unseen'])) for n in cases}
  gates={}
  for m in ('oracle200','oracle100'):
   o=summary[m];main=deploy[strongest['view60']]['view60']
   gates[m]=dict(justifies_regional_learning=o['view60']['people_supplied']>main['people_supplied'] and o['view60']['expired_unseen']<main['expired_unseen'] and all(o[n]['people_supplied']>=deploy[strongest[n]][n]['people_supplied'] for n in ('view40','view100')),
    main_people_gain=o['view60']['people_supplied']-main['people_supplied'],main_unseen_expiry_change=o['view60']['expired_unseen']-main['expired_unseen'])
  verify();write(out/'summary.json',dict(status='completed',training_episodes=0,evaluation_rollouts=count,results=summary,full_baselines=baselines,strongest_deployable=strongest,gates=gates,source_water_unchanged=True));status('completed')
 except BaseException as ex:
  write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run(a.output,a.previous)
