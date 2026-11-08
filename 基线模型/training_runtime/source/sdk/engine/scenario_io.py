"""Portable data adapter for the frozen archive. No access to a user's checkout."""
import json
from pathlib import Path
import numpy as np
from motion import simulate,shape,K


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def write(p,data):
    Path(p).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8")


def load_water(folder,trial):
    # Lossless storage of the original float32 NPY values in our distribution.
    with np.load(folder/trial["water"],allow_pickle=False) as z:a=z["water"]
    if a.dtype!=np.float32 or a.shape not in ((481,3),(481,3,100,100)):raise ValueError("Invalid field")
    if not np.isfinite(a).all() or (a[:,0]<0).any():raise ValueError("Invalid water values")
    return np.broadcast_to(a[:,:,None,None],(481,3,100,100)) if a.ndim==2 else a


def load_trial(folder,trial):
    water=load_water(folder,trial)
    with np.load(folder/trial["population"],allow_pickle=False) as z:pop={k:z[k] for k in ("fall","points","available","velocity")}
    shapes=dict(fall=(64,),points=(2401,64,2),available=(2401,64),velocity=(481,64,2))
    if any(pop[k].shape!=dims for k,dims in shapes.items()):raise ValueError("Invalid population")
    return water,pop


def regenerate(folder,trial):
    field,archived=load_trial(folder,trial)
    with np.load(folder/trial["origin"],allow_pickle=False) as z:
        ids=z["eligible"];traps=z["traps"]
    seed,rep=trial["parent"],trial["roster"]
    pos,clock=map(np.random.default_rng,np.random.SeedSequence([91821,seed,rep]).spawn(2))
    # Original nested rosters: draw 48, then 32 from the complement, retain the first 64.
    first=pos.choice(ids,48,replace=False)
    selected=np.r_[first,pos.choice(np.setdiff1d(ids,first),32,replace=False)]
    clocks=clock.exponential(size=80)
    yy,xx=np.unravel_index(selected,(100,100))
    samples=field[:,:,yy,xx].astype(float);delta=samples[1:]-samples[:-1]
    hazard=np.empty((480,5,80))
    for i in range(5):
        h,u,v=np.moveaxis(samples[:-1]+((i+.5)/5)*delta,1,0)
        hazard[:,i]=K*shape(h,np.hypot(u,v))
    hit=hazard.reshape(2400,80).cumsum(0)>=clocks
    fall=np.where(hit.any(0),hit.argmax(0)+1,2401)
    got=simulate(field,selected,fall,seed,rep,traps)
    for k,expected in archived.items():
        actual=got[k][:64] if k=="fall" else got[k][:,:64]
        if not np.array_equal(actual,expected):raise AssertionError(f"{trial['id']} differs in {k}")
