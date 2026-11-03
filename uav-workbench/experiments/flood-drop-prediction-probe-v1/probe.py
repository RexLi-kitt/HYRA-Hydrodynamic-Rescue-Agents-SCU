from pathlib import Path
import sys,json,time,copy,traceback,argparse
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-coordination-v1'))
from train_coordination import read,write,sha,summarize
from variable_environment import sample_water,first_hit
from prediction_env import PredictionEnv

MODES=['original','hold','linear_30','linear_60','oracle_now','oracle_30','oracle_60']

def drift(water,scene):
 cfg=scene['config'];pos=np.array([e['xy'] for e in scene['events']],float);fall=np.array([e['fall'] for e in scene['events']])
 stuck=np.zeros(len(pos),bool);outside=np.zeros(len(pos),bool);traps=np.array(scene['traps'],float).reshape(-1,2)
 trajectory=np.empty((cfg['duration']+1,len(pos),2),float);trajectory[0]=pos
 for now in range(cfg['duration']):
  ids=np.flatnonzero((fall<=now)&~stuck&~outside)
  if len(ids):
   values=sample_water(water,now,pos[ids]);stuck[ids[values[:,0]<cfg['wet_depth']]]=True
   for k,j in enumerate(ids):
    if stuck[j]:continue
    nxt=pos[j]+values[k,1:];hit=first_hit(pos[j],nxt,traps,cfg['trap_radius'])
    if hit is not None:nxt=pos[j]+(nxt-pos[j])*hit;stuck[j]=True
    outside[j]=((nxt<0)|(nxt>=1000)).any();pos[j]=nxt
  trajectory[now+1]=pos
 assert np.isfinite(trajectory).all();return trajectory

def run(out,previous):
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();p0=read(previous/'protocol.json');cases=read(previous/'scenarios.json')['development'];results={m:{} for m in MODES};count=0
 def status(s,**kw):write(out/'status.json',dict(status=s,elapsed_s=time.monotonic()-start,evaluation_rollouts=count,**kw))
 def guard():
  if (out/'STOP').exists():raise InterruptedError('Stop requested')
  if time.monotonic()-start>3600:raise TimeoutError('One-hour diagnostic cap')
 sources={**p0['source_hashes'],**{str(f):sha(f) for f in HERE.glob('*.py')}}
 protocol=dict(version='prediction-value-probe-v1',date='2026-11-03',previous=str(previous),source_hashes=sources,water_hashes=p0['water_hashes'],scenario_sha256=sha(previous/'scenarios.json'),modes=MODES,
  scope='24 existing DEV scenarios: view60 main12;view40/100 six each;168 rollouts,0 training. Same known-person nearest4 candidates and global assignment, only predicted target location differs.',
  oracle='Only already discovered, believed pending and recent<=120s person candidates receive true positions. No unseen-person reveal, no future fall/deadline/status input. Physical sensing/service/collision unchanged. NONDEPLOYABLE diagnostic, not theoretical optimum or model performance.',
  horizons='linear/oracle use same estimated travel time from last-seen position divided by UAV speed, capped30/60s. Oracle trajectory rounded to nearest second; current-position oracle separately diagnoses stale tracking.',
  gate='For a horizon to justify trajectory learning here, oracle main supplied must exceed strongest full planner AND corresponding linear baseline; no people regression on either other visibility panel. Otherwise do not automatically train GRU; inspect or move to search inference. Passing only warrants next experiment, not superiority claim.',
  limitations='A negative result only rejects benefit under this candidate/execution rule, not all prediction architectures. Known reused DEV,one set of layouts; privileged oracle not deployable. Passive drift ignores later rescue/expiry deliberately, so no future rescue/deadline status is supplied.',
  references=[dict(title='Trajectron++',url='https://arxiv.org/abs/2001.03093',use='Research direction for recurrent trajectory prediction with dynamics/maps; not evidence of flood rescue performance.'),dict(title='Deep Sets',url='https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html',use='Permutation-invariant variable-size person representations; not implemented in this diagnostic.')])
 write(out/'protocol.json',protocol);write(out/'cases.json',cases)
 def verify():
  assert sha(previous/'scenarios.json')==protocol['scenario_sha256']
  for path,h in {**sources,**p0['water_hashes']}.items():assert sha(path)==h,path
 try:
  verify();baseline=read(previous/'planner-dispatch.json');full={m:read(previous/f'planner-{m}.json') for m in ('original','dispatch','balanced','nearest_reserved')}
  for panel,group in cases.items():
   for m in MODES:results[m][panel]=[]
   for ci,c in enumerate(group):
    guard();status('building_passive_trajectory',panel=panel,case=c['id']);water=np.load(c['field']['path'],mmap_mode='r',allow_pickle=False);trajectory=drift(water,c['scenario'])
    # Save privileged trajectories as clearly labelled diagnostic-only artifacts.
    target=out/'oracle-trajectories'/f'{panel}-{ci:02}.npz';target.parent.mkdir(exist_ok=True);np.savez_compressed(target,positions=trajectory)
    for mode in MODES:
     guard();status('evaluating',panel=panel,case=c['id'],mode=mode)
     env=PredictionEnv(water,c['scenario'],mode,trajectory if mode.startswith('oracle') else None)
     # Hidden oracle target array cannot alter choices when nobody is known.
     if mode.startswith('oracle'):
      before=env.choices();saved=env.trajectory;env.trajectory=np.full_like(trajectory,999.)
      after=env.choices();env.trajectory=saved
      if not env.discovered.any():assert np.array_equal(before[1],after[1]) and np.array_equal(before[3],after[3])
     while True:
      _,_,done,_=env.step(env.choices()[1])
      active=env.status==1
      assert np.allclose(env.person[active],trajectory[env.t,active],rtol=0,atol=1e-8),'passive trajectory mismatch'
      if done:break
     row=dict(**env.metrics(),case=c['id'],privileged=mode.startswith('oracle'))
     row['persons']=[dict(id=j,fall=int(env.fall[j]),deadline=int(env.deadline[j]),supplied_at=float(env.delivered_at[j]),status=int(env.status[j])) for j in range(env.cfg.people)]
     assert row['supplied']+row['expired']+row['outside']==env.cfg.people
     if mode=='original':
      old=baseline[panel]['rows'][ci]
      for key in ('supplied','expired','outside','conflicts','expired_unseen'):assert row[key]==old[key],(key,row[key],old[key])
      assert row['persons']==[{k:v for k,v in person.items() if k in ('id','fall','deadline','supplied_at','status')} for person in old['persons']]
     results[mode][panel].append(row);count+=1
     with (out/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(panel=panel,mode=mode,**row),allow_nan=False)+'\n')
  summaries={m:{n:summarize(rows) for n,rows in panels.items()} for m,panels in results.items()}
  strongest={n:max(v[n]['summary']['people_supplied'] for v in full.values()) for n in cases};gates={}
  for h in (30,60):
   o=summaries[f'oracle_{h}'];linear=summaries[f'linear_{h}']
   gates[str(h)]=dict(justifies_next_trajectory_trial=o['view60']['people_supplied']>max(strongest['view60'],linear['view60']['people_supplied']) and all(o[n]['people_supplied']>=max(strongest[n],linear[n]['people_supplied']) for n in ('view40','view100')),
    main_gain_vs_best_full=o['view60']['people_supplied']-strongest['view60'],main_gain_vs_linear=o['view60']['people_supplied']-linear['view60']['people_supplied'])
  verify();write(out/'summary.json',dict(status='completed',training_episodes=0,evaluation_rollouts=count,results=summaries,full_baselines={m:{n:v['summary'] for n,v in panels.items()} for m,panels in full.items()},gates=gates,source_water_unchanged=True,deployable_oracle=False));status('completed')
 except BaseException as ex:
  write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run(a.output,a.previous)
