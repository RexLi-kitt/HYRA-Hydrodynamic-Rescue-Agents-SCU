from pathlib import Path
import json,sys,time,argparse,traceback
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-coordination-v1'))
from train_coordination import read,write,sha,summarize
from refloat_search import RefloatSearch
from search_env import SearchEnv

def run(out,previous):
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();data=read(previous/'scenarios.json');cases=data['train'];old=read(previous/'protocol.json');count=0
 sources={**old['source_hashes'],**{str(p):sha(p) for p in HERE.glob('*.py')}}
 sources[str(HERE.parent/'flood-drop-search-probe-v1/search_env.py')]=sha(HERE.parent/'flood-drop-search-probe-v1/search_env.py')
 protocol=dict(date='2026-11-03',version='refloat-screen-v1',source_hashes=sources,water_hashes=old['water_hashes'],scenario_sha256=sha(previous/'scenarios.json'),scope='32 archived TRAIN cases only, no DEV selection. Legacy coverage100 vs refloat coverage100 vs refloat oracle100;96 rollouts. Two legacy-off trajectory regressions additional. Reuse all archived N/window/view settings rather than regenerate18 configurations: cheaper and exact accepted-roster pairing.',
  rules='Stranded current h<.01m; rewet h>=.01 and (h>=2 or |v|>=1) forced; else1-(1-.05)^(1/60) per second. Obstacle capture remains. Counter randomness seed/exposure_id/second; original fall/deadline immutable. All supplied/expired/outside stop.',
  gate='Oracle total supplied exceeds refloat coverage and no aggregate people regression in any40/60/100 visibility; this justifies limited prediction fitting, not deployability or learning success. Keep every paired gain/loss.',
  limits='Synthetic field family; force thresholds may remain unexercised. Both ground and obstacle event counts now separate; water not recomputed. No model training in this executable.')
 write(out/'protocol.json',protocol);write(out/'scenarios.json',data)
 def status(state,**kw):write(out/'status.json',dict(status=state,evaluation_rollouts=count,elapsed_s=time.monotonic()-start,**kw))
 def verify():
  assert sha(previous/'scenarios.json')==protocol['scenario_sha256']
  for p,h in {**sources,**old['water_hashes']}.items():assert sha(p)==h,p
 try:
  verify();status('regression')
  for c in cases[:2]:
   water=np.load(c['field']['path'],mmap_mode='r');a=SearchEnv(water,c['scenario'],'coverage100');b=RefloatSearch(water,c['scenario']);b.refloat_enabled=False
   while True:
    aa=a.choices()[1];bb=b.choices()[1];assert np.array_equal(aa,bb)
    _,_,done,_=a.step(aa);b.step(bb)
    for k in ('person','pos','status','stuck','progress','deadline','delivered_at'):assert np.array_equal(getattr(a,k),getattr(b,k)),k
    if done:break
  write(out/'regression.json',dict(disabled_refloat_full_trajectories=2,status='passed'))
  results={k:[] for k in ('legacy','refloat','oracle')}
  for c in cases:
   for mode in results:
    if (out/'STOP').exists():raise InterruptedError('User STOP')
    if time.monotonic()-start>1800:raise TimeoutError('30minute cap')
    status('evaluating',case=c['id'],mode=mode)
    water=np.load(c['field']['path'],mmap_mode='r');e=SearchEnv(water,c['scenario'],'coverage100') if mode=='legacy' else RefloatSearch(water,c['scenario'],'oracle100' if mode=='oracle' else 'coverage100')
    deadlines=e.deadline.copy();falls=e.fall.copy()
    while True:
     _,_,done,_=e.step(e.choices()[1])
     assert np.array_equal(deadlines,e.deadline) and np.array_equal(falls,e.fall)
     if done:break
    r=dict(**e.metrics(),case=c['id'],parent=c['field']['parent'],sensor=e.cfg.sensor,mode=mode,people=e.cfg.people,window=e.cfg.window,refloat_events=sum(x['type']=='refloat' for x in e.events),ground_events=sum(x['type']=='stranded' for x in e.events),obstacle_events=sum(x['type']=='obstacle_capture' for x in e.events))
    r['persons']=[dict(id=j,exposure_id=c['scenario']['events'][j].get('exposure_id',j),fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j]),refloat_count=int(e.refloat_counts[j]) if mode!='legacy' else 0) for j in range(e.cfg.people)]
    r['events']=e.events
    assert r['supplied']+r['expired']+r['outside']==e.cfg.people
    for p in r['persons']:
     if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
    results[mode].append(r);count+=1
    with (out/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
  summary={k:summarize(v) for k,v in results.items()};views={k:{str(s):summarize([r for r in v if r['sensor']==s]) for s in (40.,60.,100.)} for k,v in results.items()}
  gate=summary['oracle']['people_supplied']>summary['refloat']['people_supplied'] and all(views['oracle'][s]['people_supplied']>=views['refloat'][s]['people_supplied'] for s in views['oracle'])
  verify();write(out/'summary.json',dict(status='completed',totals=summary,by_sensor=views,prediction_pilot_gate=gate,refloat_counts={k:sum(r['refloat_events'] for r in v) for k,v in results.items()},new_training=0));status('completed')
 except BaseException as ex:
  write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run(a.output,a.previous)
