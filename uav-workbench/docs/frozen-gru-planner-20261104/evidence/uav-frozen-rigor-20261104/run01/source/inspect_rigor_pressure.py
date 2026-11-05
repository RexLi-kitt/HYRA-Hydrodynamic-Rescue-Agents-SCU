"""Exact frozen trajectory replay, no changed model or environment parameters."""
from pathlib import Path
import os,sys,json,time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
HERE=ROOT/'hyra-uav-pr-20261101/uav-workbench/experiments/flood-drop-search-return-v1'
sys.path.insert(0,str(HERE))
from rigor_support import *
torch.set_num_threads(1)
start=time.monotonic();dest=OUT/'pressure-replay';dest.mkdir(exist_ok=False)
case=next(c for c in read(OUT/'scenarios.json') if c['id']=='manufactured-0')
write(dest/'protocol.json',dict(scope='Post-test exploratory diagnostic, not new independent confirmation',case=case['id'],methods=['coverage100','statistic','gru-frozen'],change='None. Exact replay and trajectory logging only.',question='Does restricted exploration appear before the first dry/refloat episode? Separate lack of exploration from an unsupported claim that refloating alone caused failure.',first_dry_start=480,snapshots=[0,30,120,240,479,600,900,1200,1500,1800,2100,2399],source_sha256=sha(__file__),source_inputs_locked_by=str(OUT/'protocol.json'),no_training=True))
initialize_worker();original=[json.loads(x) for x in (OUT/'results.jsonl').read_text(encoding='utf-8').splitlines()]
results=[]
for method in ['coverage100','statistic','gru-frozen']:
 w=np.load(case['field']['path'],mmap_mode='r')
 e=SafeFlowEnv(w,case['scenario'],method) if method=='coverage100' else CountCoupledEnv(w,case['scenario'],MODELS[method],.25)
 def forbid(*a,**k):raise AssertionError('Hidden counts requested')
 e.sector_counts=forbid
 snapshots=[];distance=np.zeros(4);minimum=float('inf')
 while True:
  choice=e.choices()
  if e.t in read(dest/'protocol.json')['snapshots']:
   # Hidden counts below are audit output only, never an input to the policy.
   row=dict(t=e.t,positions=e.pos.tolist(),ever_observed_cells=int(e.known.sum()),discovered_by_now=int(e.discovered.sum()),supplied=int((e.status==2).sum()),refloats=int(e.refloat_counts.sum()),selected_targets=[choice[2][i][a]['xy'].tolist() for i,a in enumerate(choice[1])],selected_utilities=[float(choice[2][i][a]['utility']) for i,a in enumerate(choice[1])],audit_only_pending_unseen=int(((e.status==1)&~e.discovered).sum()))
   if method!='coverage100':
    p=e.prediction;row['prediction']=dict(sum=float(p.sum()),min=float(p.min()),max=float(p.max()),grid=p.tolist())
   snapshots.append(row)
  old=e.pos.copy();done=e.step(choice[1])[2];distance+=np.linalg.norm(e.pos-old,axis=1);minimum=min(minimum,float(swept_distances(old,e.pos).min()))
  if done:break
 baseline=next(r for r in original if r['case']==case['id'] and r['method']==method)
 pp=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j]),refloats=int(e.refloat_counts[j])) for j in range(e.cfg.people)]
 assert pp==baseline['persons'] and int(e.refloat_counts.sum())==baseline['refloats']
 results.append(dict(method=method,exact_person_records=True,supplied=baseline['supplied'],refloats=baseline['refloats'],distance_m=distance.tolist(),last_t=e.t,final_observed_cells=int(e.known.sum()),min_separation=minimum,snapshots=snapshots))
 print(json.dumps(dict(method=method,supplied=baseline['supplied'],observed_cells=int(e.known.sum()),before_first_dry=next(s for s in snapshots if s['t']==479)),ensure_ascii=False),flush=True)
write(dest/'result.json',dict(elapsed_s=time.monotonic()-start,results=results,scope='Three exact replays of ONE already viewed manufactured layout, diagnostic only. No extra independent sample. Refloating, depth, spatial distribution and global wetness are confounded across this stress family vs main SWE family; do not infer a unique cause.'))
