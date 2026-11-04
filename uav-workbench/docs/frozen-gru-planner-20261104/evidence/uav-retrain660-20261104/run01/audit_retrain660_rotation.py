"""Read-only 90-degree prediction consistency audit on FIT public histories."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;REPO=ROOT/'hyra-uav-pr-20261101';RUN=ROOT/'uav-retrain660-20261104/run01'
HERE=REPO/'uav-workbench/experiments/flood-drop-search-return-v1'
sys.path.insert(0,str(HERE))
from retrain660 import PREVIOUS,MemoryGRU,augment
torch.set_num_threads(4)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(RUN/'status.json')['state']=='completed'
sources={str(p):sha(p) for p in (RUN/'selection.json',ROOT/'audit_retrain660_rotation.py')}
xs=[];keys=[]
for i in range(12):
 p=RUN/f'data-fit-{i}.npz';sources[str(p)]=sha(p)
 with np.load(p) as d:
  ids=np.linspace(0,len(d['x'])-1,16).round().astype(int);xs.append(d['x'][ids]);keys.extend([(i,int(d['times'][j])) for j in ids])
x=torch.from_numpy(np.concatenate(xs));dummy=torch.zeros(len(x),10,10)
xr,_=augment(x,dummy,1,False)
back,_=augment(xr,dummy,3,False)
assert torch.equal(x,back)
selected=read(RUN/'selection.json')['selected']
models={n:(RUN/f'{n}.pt',False) for n in selected.values()}
models.update({'old142-raw':(PREVIOUS/'fine-s202611142-e16.pt',False),'old142-adapter':(PREVIOUS/'fine-s202611142-e16.pt',True)})
rows=[];stats={}
for name,(path,adapted) in models.items():
 sources[str(path)]=sha(path);ck=torch.load(path,map_location='cpu',weights_only=True)
 m=MemoryGRU(ck['mean'],False);m.load_state_dict(ck['state']);m.eval()
 mu=torch.tensor(ck['mu'])[None,None,:,None,None];sd=torch.tensor(ck['sd'])[None,None,:,None,None]
 def predict(z):
  out=[]
  for chunk in z.split(32):
   q=(chunk-mu)/sd
   if adapted:
    anchor=float(np.float32(420/300));q[:,:,13]=(anchor-mu[:,:,13])/sd[:,:,13]+(chunk[:,:,13]-anchor)
   with torch.inference_mode():out.append(m(q))
  return torch.cat(out)
 a=predict(x);b=torch.rot90(predict(xr),3,(-2,-1));assert torch.isfinite(a).all() and torch.isfinite(b).all()
 # Compare normalized spatial allocation; constant scale changes alone are irrelevant to residual adapter.
 an=a/(a.mean((-2,-1),keepdim=True)+1e-6);bn=b/(b.mean((-2,-1),keepdim=True)+1e-6)
 delta=(an-bn).abs().mean((-2,-1));v0,j0=a.flatten(1).topk(2);v1,j1=b.flatten(1).topk(2)
 valid=(v0[:,0]-v0[:,1]>1e-5)&(v1[:,0]-v1[:,1]>1e-5)
 for k,(parent,t) in enumerate(keys):rows.append(dict(model=name,fit_index=parent,time=t,normalized_spatial_l1=float(delta[k]),top_unambiguous=bool(valid[k]),top_same=bool(j0[k,0]==j1[k,0])))
 stats[name]=dict(histories=len(x),normalized_spatial_l1_mean=float(delta.mean()),normalized_spatial_l1_max=float(delta.max()),unambiguous=int(valid.sum()),top_same=int(((j0[:,0]==j1[:,0])&valid).sum()))
for p,h in sources.items():assert sha(Path(p))==h
result=dict(protocol='Supplementary post-result diagnosis, not preregistered acceptance.12FITparents x16uniform archived public histories. Samephysical660/2400 state, rotate90 using existing feature augmentation and inverse-transform predictedmap. No simulation/training/labels or checkpoint selection. Top ties within1e-5excluded from top comparison. Sensitivity is not proof of error or unique rescue-failure cause.',
 source_sha256=sources,rotation_inverse_exact=True,stats=stats,rows=rows)
(RUN/'rotation-prediction-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(stats))
