"""Read-only mechanism audit of frozen 20m search; no training or event replay."""
from pathlib import Path
from types import SimpleNamespace
import sys, json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'uav-workbench/experiments/flood-drop-convlstm20-v1'
OUT=ROOT/'terrain-workbench/docs/uav-literature-redesign-20261104'
sys.path.insert(0,str(SRC))
from long_flow import FlowFogEnv, transport
from history_model import features

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def bare(mask):
    e=object.__new__(FlowFogEnv)
    e.t=120
    e.known=mask.copy()
    e.last=np.full((100,100),90.)
    e.measured=np.broadcast_to(np.array([.2,.3,-.1])[:,None,None],(3,100,100)).copy()
    e.cfg=SimpleNamespace(wet_depth=.01,duration=2100,sensor=20.,people=40,window=420)
    e.discovered=np.zeros(0,bool)
    e.seen_xy=np.zeros((0,2)); e.belief=np.zeros(0,int)
    e.history=SimpleNamespace(time=120,get=lambda env: np.zeros((100,4,14)))
    e.prediction=None;e.predicted_at=None;e.predict=lambda x:np.ones(100)
    return e

source_hashes={p.name:digest(p) for p in SRC.glob('*.py')}
rows=[]
rr,cc=np.mgrid[:100,:100]
positions=np.array([[20,20],[980,20],[20,980],[980,980]])
cells=np.stack([(cc+.5)*10,(rr+.5)*10],-1)
for seed in range(8):
    rng=np.random.default_rng(seed)
    e=bare(rng.random((100,100))>.4)
    e.last=rng.uniform(0,120,(100,100))
    e.measured[0]=rng.uniform(0,.5,(100,100))
    stale=np.where(e.known,(e.t-e.last)/120,2.)
    risk=e.measured[0]/2+np.linalg.norm(e.measured[1:],axis=0)
    e.method='flow_coverage'; a=e.search_score(stale,risk)
    e.method='gru_flowblend'; b=e.search_score(stale,risk)
    changes=0
    for p in positions:
        for region in range(4):
            ys=slice(region//2*50,(region//2+1)*50);xs=slice(region%2*50,(region%2+1)*50)
            denom=1+np.linalg.norm(cells[ys,xs]-p,axis=-1)/250
            changes+=int(np.argmax(a[ys,xs]/denom)!=np.argmax(b[ys,xs]/denom))
    rows.append(dict(seed=seed,normalized_score_l1=float(np.abs(a/a.sum()-b/b.sum()).sum()),
                     changed_candidate_locations=changes,total_candidate_locations=16))
assert any(r['changed_candidate_locations']>0 for r in rows)

# Same counts/mean age/water per 100m tile, distinct observed subcell positions.
mask_a=cc%10<2
mask_b=cc%10>=8
fa=features(bare(mask_a)); fb=features(bare(mask_b))
assert not np.array_equal(mask_a,mask_b)
np.testing.assert_array_equal(fa,fb)

rng=np.random.default_rng(81)
d=rng.random((10,10));u=rng.uniform(-1,1,(10,10));v=rng.uniform(-1,1,(10,10))
ref=transport(d,u,v,60)
transforms=[]
for k in (1,2,3):
    dr=np.rot90(d,k);ur=np.rot90(u,k).copy();vr=np.rot90(v,k).copy()
    for _ in range(k): ur,vr=vr,-ur
    error=float(np.max(np.abs(transport(dr,ur,vr,60)-np.rot90(ref,k))))
    assert error<1e-12
    transforms.append(dict(rotation_degrees=90*k,max_abs_error=error))
assert source_hashes=={p.name:digest(p) for p in SRC.glob('*.py')}
result=dict(
 scope='Synthetic input contracts only; no full rescue trajectories, no new performance score.',
 source_sha256=source_hashes,audit_sha256=digest(Path(__file__)),
 neutral_prediction_comparison=rows,
 footprint_alias=dict(different_fine_masks=True,identical_coarse_features=True,
                     feature_shape=list(fa.shape),observed_cells_each=int(mask_a.sum())),
 transport_rotation=transforms,
 interpretation=[
 'Constant learned branch is not baseline-equivalent and can change candidate positions; not proof it causes observed failure.',
 '100m average features lose within-tile observation footprint; planner still has fine known/last arrays, so this is predictor information loss, not absence from the whole system.',
 'Transport alone passes rotation check; model and full assignment/identity equivariance remain untested.',
 'No old source changed; no training, task parameter change, or new accuracy claim.'
 ])
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'mechanism-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('neutral_prediction_comparison','footprint_alias','transport_rotation')},ensure_ascii=False,indent=2))

