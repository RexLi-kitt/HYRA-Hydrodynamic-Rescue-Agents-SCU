"""Portable inference/replay entry for the explicitly frozen research snapshot."""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'uav-workbench/experiments/flood-drop-search-return-v1'))
from count_coupling import CountCoupledEnv
from joint_avoidance import SafeFlowEnv,swept_distances
from memory import MemoryGRU
from window_adapter import normalized

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--water',type=Path);p.add_argument('--case',default='original-1105501');p.add_argument('--method',choices=['gru-frozen','coverage100','flow_coverage'],default='gru-frozen');p.add_argument('--output',type=Path);a=p.parse_args()
 torch.set_num_threads(1)
 modelpath=HERE/'model/fine-s202611142-e16.pt'
 assert sha(modelpath)=='3bed74682c5d7b163814b1585d162b9b5485e040c8756c3bcfed93756ff6284a'
 ck=torch.load(modelpath,map_location='cpu',weights_only=True);m=MemoryGRU(ck['mean'],ck['coarse']);m.load_state_dict(ck['state']);m.eval()
 mu=np.array(ck['mu'],np.float32);sd=np.array(ck['sd'],np.float32)
 def pred(x):
  with torch.inference_mode():return m(torch.from_numpy(normalized(x,mu,sd,True))[None])[0].numpy().ravel()
 if a.check:
  v=pred(np.zeros((4,64,10,10),np.float32));assert v.shape==(100,) and np.isfinite(v).all() and (v>=0).all()
  print(json.dumps(dict(weights_sha256=sha(modelpath),parameters=sum(t.numel() for t in m.parameters()),finite_nonnegative_output=True,training=False)));return
 if a.water is None or a.output is None:p.error('--water and --output are required for a replay')
 evidence=HERE/'evidence/uav-frozen-rigor-20261104/run01'
 case=next(c for c in read(evidence/'scenarios.json') if c['id']==a.case)
 expected_hash=read(evidence/'input-lock.json')['water_sha256'][case['field']['path']]
 assert sha(a.water)==expected_hash,'Water file does not match the frozen case'
 w=np.load(a.water,mmap_mode='r');e=CountCoupledEnv(w,case['scenario'],pred,.25) if a.method=='gru-frozen' else SafeFlowEnv(w,case['scenario'],a.method)
 def forbid(*x,**kw):raise AssertionError('Hidden target count requested')
 e.sector_counts=forbid;minimum=float('inf')
 while True:
  old=e.pos.copy();done=e.step(e.choices()[1])[2];minimum=min(minimum,float(swept_distances(old,e.pos).min()))
  assert minimum>=5-1e-8
  if done:break
 persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j]),refloats=int(e.refloat_counts[j])) for j in range(e.cfg.people)]
 original=next(json.loads(l) for l in (evidence/'results.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(l)['case']==a.case and json.loads(l)['method']==a.method)
 result=dict(case=a.case,method=a.method,supplied=int((e.status==2).sum()),people=e.cfg.people,min_separation=minimum,persons=persons,exact_archived_person_records=persons==original['persons'])
 with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
 assert result['exact_archived_person_records'],'Numerical/runtime replay differs from archived record; inspect output'
 print(json.dumps({k:v for k,v in result.items() if k!='persons'}))

if __name__=='__main__':main()
