"""Public action eligibility and requested intent for HYRA 20261106.1.

This is action admissibility, not a reward, planner, sensor, or referee change.
"""
import numpy as np
from types import SimpleNamespace

VERSION = 'HYRA_PUBLIC_INTENT_REFEREE_CLIP_V2'
MOVES = np.array([[0,1],[0,-1],[-1,0],[1,0],[-1,1],[-1,-1],[1,1],[1,-1],[0,0]], float)
MOVES[:8] /= np.linalg.norm(MOVES[:8], axis=1, keepdims=True)
FIELDS = ('pos', 'seen_xy', 'seen_v', 'seen_at', 'visible', 'active')
SAFE_ACTION = 8 * 65  # move=8, target=-1: zero velocity, NOT target-following.


def public_geometry(o):
    """Exact public precision is saved; do not reconstruct geometry from float32 features."""
    return {name: np.asarray(getattr(o, name)).copy() for name in FIELDS}


def first_hit(p, d, center, radius=.5):
    # Independent copy of the declared public action geometry, not hidden truth.
    offset = p-center
    if offset@offset <= radius*radius+1e-9: return 0.
    aa = d@d
    if aa < 1e-15: return None
    bb = 2*(offset@d); cc = offset@offset-radius*radius
    disc = bb*bb-4*aa*cc
    if disc < -1e-9: return None
    x = (-bb-np.sqrt(max(0.,disc)))/(2*aa)
    return float(np.clip(x,0,1)) if -1e-9 <= x <= 1+1e-9 else None


def intent_geometry(o):
    """Return [agent, packed_action] speed, stop duration and first boundary time.

    Actual paths are straight rays followed by rest. In the convex fixed square,
    testing first boundary time covers the WHOLE intended segment, not just its
    clipped endpoint. Collision resolution only shortens/stops these paths and
    is deliberately not used to excuse an unsafe intent.
    """
    pos = np.asarray(o.pos, dtype=float)
    xy = np.asarray(o.seen_xy, dtype=float); velocity = np.asarray(o.seen_v, dtype=float)
    visible = np.asarray(o.visible, dtype=bool)
    seen = np.asarray(o.seen_at); active = np.asarray(o.active, dtype=bool)
    if pos.shape != (4,2) or xy.shape != (64,2) or velocity.shape != (64,2):
        raise ValueError('Unexpected public geometry shape')
    if any(a.shape != (64,) for a in (visible, seen, active)):
        raise ValueError('Unexpected public target shape')
    if not np.isfinite(pos).all() or np.any((pos<0)|(pos>1000)):
        raise ValueError('UAV outside the fixed referee domain')
    if not np.isfinite(xy[visible]).all() or not np.isfinite(velocity[visible]).all():
        raise ValueError('Invalid visible target geometry')
    target_allowed = np.r_[True, (seen>=0)&active]
    speed = np.broadcast_to(np.repeat(MOVES*10,65,axis=0),(4,585,2)).copy()
    duration = np.full((4,585),5.)
    # Stale/unseen positions and velocities never affect the motion mask.
    for j in np.flatnonzero(visible & (seen>=0) & active):
        current = velocity[j]
        speed[:,8*65+j+1] = current*min(1.,10/max(np.linalg.norm(current),1e-12))
        for i in range(4):
            if np.linalg.norm(pos[i]-xy[j]) <= .5+1e-7: continue
            for move in range(8):
                action=move*65+j+1
                hit=first_hit(pos[i],speed[i,action]*5,xy[j])
                if hit is not None: duration[i,action]=min(5.,hit*5)
    bound=np.full_like(speed,np.inf)
    np.divide(1000-pos[:,None,:],speed,out=bound,where=speed>0)
    np.divide(-pos[:,None,:],speed,out=bound,where=speed<0)
    boundary_time=bound.min(axis=-1)
    return speed,duration,boundary_time,np.tile(target_allowed,9)


def action_mask(o):
    """Target eligibility only: the SDK accepts and clips boundary-crossing moves."""
    _, _, _, target_allowed = intent_geometry(o)
    result=np.broadcast_to(target_allowed[None,:],(4,585)).copy()
    if not result[:,SAFE_ACTION].all(): raise AssertionError('No-target hover missing')
    return result.astype(np.uint8)


def from_geometry(data):
    if set(data)!=set(FIELDS): raise ValueError('Boundary public metadata fields changed')
    return SimpleNamespace(**data)
