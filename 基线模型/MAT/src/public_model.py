import numpy as np
DIM=998

def encode(o):
    """Encode only attributes of the referee's public observation object."""
    t=float(o.t); last=np.asarray(o.last); known=last>=0
    measured=np.asarray(o.measured)
    maps=np.stack((known.astype(float),np.where(known,np.clip(measured[0]/2.,0,1),0),
                   np.where(known,np.clip(measured[1]/3.,-1,1),0),
                   np.where(known,np.clip(measured[2]/3.,-1,1),0)))
    pooled=maps.reshape(4,10,10,10,10).mean(axis=(2,4)).reshape(-1)
    team=np.r_[np.asarray(o.pos).reshape(-1)/1000.,np.asarray(o.heading).reshape(-1),t/2400.]
    observed=np.asarray(o.seen_at)>=0;active=np.asarray(o.active,dtype=bool)
    possible=observed&active
    xy=np.where(observed[:,None],np.asarray(o.seen_xy)/1000.,0.)
    v=np.where(observed[:,None],np.clip(np.asarray(o.seen_v)/3.,-1.,1.),0.)
    age=np.where(observed,np.minimum(1.,np.maximum(0.,t-np.asarray(o.seen_at))/2400.),0.)
    rows=[]
    for i in range(4):
        person=np.column_stack((xy,v,age,np.asarray(o.visible,dtype=float),active.astype(float),
                                (np.asarray(o.owner)==i).astype(float))).reshape(-1)
        own=np.eye(4,dtype=float)[i];target_mask=np.r_[1.,possible.astype(float)]
        rows.append(np.r_[pooled,team,person,own,target_mask])
    out=np.asarray(rows,dtype=np.float32)
    if out.shape!=(4,DIM) or not np.isfinite(out).all():raise ValueError('invalid public features')
    return out
