"""Public-only HYRA observation encoding, referee-clipped intent v2."""
FEATURE_VERSION="HYRA_PUBLIC_INTENT_REFEREE_CLIP_V2"
import numpy as np,torch,importlib.util
from pathlib import Path
def _local(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parent/filename);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
public_model=_local('hyra_portable_public_model','public_model.py')
boundary_geometry=_local('hyra_portable_boundary_geometry','boundary_geometry.py')
LOCAL_SIDE=15
LOCAL_DIM=2*LOCAL_SIDE**2+2
BASE=998+LOCAL_DIM
PERSON_DIM=19
ACTION_DIM=12
FEATURE_DIM=64*PERSON_DIM+64*9*ACTION_DIM
MASK_DIM=585
INPUT_DIM=BASE+FEATURE_DIM+MASK_DIM

def local_coverage(o):
    """North/east aligned 10m cells around this UAV, never current hidden water."""
    pos=np.asarray(o.pos,dtype=float);last=np.asarray(o.last)
    if pos.shape!=(4,2) or last.shape!=(100,100):raise ValueError('Public geometry shape')
    offset=np.arange(-7,8);out=[]
    for p in pos:
        col,row=np.floor(p/10).astype(int)
        rr=row+offset[:,None];cc=col+offset[None,:]
        valid=(rr>=0)&(rr<100)&(cc>=0)&(cc<100)
        known=valid&(last[np.clip(rr,0,99),np.clip(cc,0,99)]>=0)
        subcell=p/10-np.floor(p/10)-.5
        out.append(np.r_[known.ravel(),valid.ravel(),subcell])
    return np.asarray(out,np.float32)

def search_encode(o):
    x=public_model.encode(o)
    x=np.concatenate((x,local_coverage(o)),axis=1)
    mask=boundary_geometry.action_mask(o)
    return np.concatenate((x,mask.astype(np.float32)),axis=1)

def person_features(o):
    """Per-agent, per-person context built exclusively from public data.

    The 30-second, 120-meter eligibility indicator is an input feature;
    it does not gate the learned geometry score or change legal actions.
    """
    pos = np.asarray(o.pos, dtype=np.float64)
    seen = np.asarray(o.seen_at)
    active = np.asarray(o.active, dtype=bool)
    visible = np.asarray(o.visible, dtype=bool)
    owner = np.asarray(o.owner)
    xy = np.asarray(o.seen_xy, dtype=np.float64)
    v = np.asarray(o.seen_v, dtype=np.float64)
    observed = (seen >= 0) & active
    age = np.maximum(0., float(o.t) - seen)
    result = np.zeros((4, 64, PERSON_DIM), dtype=np.float32)
    for i in range(4):
        rel = np.where(observed[:, None], xy-pos[i], 0.)
        distance = np.linalg.norm(rel, axis=1)
        eligible = observed & (age <= 30.) & (distance <= 120.)
        team = np.delete(pos, i, axis=0)
        team_rel = (team-pos[i])/100.
        target_team = np.linalg.norm(xy[:, None, :]-team[None, :, :], axis=-1)
        f = np.column_stack((eligible, visible & observed, np.minimum(age, 120.)/120.,
              np.clip(rel/100., -2., 2.), np.minimum(distance, 200.)/100.,
              np.where(observed[:, None], np.clip(v/3., -2., 2.), 0.),
              owner == i, (owner >= 0) & (owner != i),
              np.broadcast_to(np.clip(team_rel.reshape(6), -10., 10.), (64, 6)),
              np.clip(target_team/100., 0., 10.)))
        assert f.shape == (64, PERSON_DIM)
        f[~observed] = 0.
        result[i] = f
    if not np.isfinite(result).all(): raise ValueError('Nonfinite public features')
    return result
def encode(o):
    old = search_encode(o)
    person = person_features(o).reshape(4, 64 * PERSON_DIM)
    speed, duration, bound, _ = boundary_geometry.intent_geometry(o)
    speed = speed.reshape(4, 9, 65, 2)[:, :, 1:, :].transpose(0, 2, 1, 3)
    duration = duration.reshape(4, 9, 65)[:, :, 1:].transpose(0, 2, 1)
    bound = bound.reshape(4, 9, 65)[:, :, 1:].transpose(0, 2, 1)
    requested_duration = duration
    actual_duration = np.maximum(0., np.minimum(requested_duration, bound))
    end = np.asarray(o.pos)[:, None, None, :] + speed * actual_duration[..., None]
    xy = np.asarray(o.seen_xy)[None, :, None, :]
    vel = np.asarray(o.seen_v)[None, :, None, :]
    observed = ((np.asarray(o.seen_at) >= 0) & np.asarray(o.active, bool))[None, :, None]
    current_gap = np.where(observed[..., None], (xy-end)/50., 0.)
    future_gap = np.where(observed[..., None], (xy+5.*vel-end)/50., 0.)
    current_distance = np.linalg.norm(current_gap, axis=-1)*50.
    future_distance = np.linalg.norm(future_gap, axis=-1)*50.
    initial_distance = np.broadcast_to(np.linalg.norm(xy - np.asarray(o.pos)[:, None, None, :], axis=-1), current_distance.shape)
    fine = np.stack((np.minimum(current_distance, 5.)/5., np.minimum(future_distance, 5.)/5.,
        current_distance <= 1., future_distance <= 1., initial_distance <= 1.), axis=-1)
    fine = np.where(observed[..., None], fine, 0.)
    action = np.concatenate((np.clip(current_gap, -20., 20.), np.clip(future_gap, -20., 20.),
        (actual_duration/5.)[..., None], (bound >= requested_duration-1e-9)[..., None],
        (actual_duration < 5.-1e-9)[..., None], fine), axis=-1).astype(np.float32)
    out = np.concatenate((old[:, :BASE], person, action.reshape(4, -1), old[:, -MASK_DIM:]), axis=1)
    assert out.shape == (4, INPUT_DIM)
    return out

def features(o):
    x=encode(o)
    return dict(obs=x[:,:1450].copy(),person=x[:,BASE:BASE+64*PERSON_DIM].reshape(4,64,19).copy(),geometry=x[:,BASE+64*PERSON_DIM:BASE+FEATURE_DIM].reshape(4,64,9,12).copy(),mask=(x[:,-585:]>.5).copy())

def tensors(d,device='cpu'):
    return {k:torch.as_tensor(v,device=device,dtype=torch.bool if k=='mask' else torch.float32) for k,v in d.items()}
