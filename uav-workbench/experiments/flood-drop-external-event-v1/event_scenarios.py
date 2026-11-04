from dataclasses import asdict
import numpy as np
from variable_environment import sample_water

def make_scenario(water,seed,cfg,fall_end=None,pool_size=None):
    """Externally scheduled synthetic incidents; no depth/speed failure criterion.

    Wet support only defines where an already-in-water incident can occur.
    This is not a pedestrian stability or human fall-risk model.
    """
    end=cfg.duration-cfg.window if fall_end is None else int(fall_end)
    if not 0<=end<=cfg.duration-cfg.window:raise ValueError('Incomplete individual window')
    count=cfg.people if pool_size is None else int(pool_size)
    if count<cfg.people or count>200:raise ValueError('Invalid pool')
    streams=np.random.SeedSequence(seed).spawn(4)
    exposure=np.random.default_rng(streams[0]);trng=np.random.default_rng(streams[1])
    irng=np.random.default_rng(streams[2]);crng=np.random.default_rng(streams[3])
    mid=min(end//2,1500);wet_mid=np.argwhere(water[mid//5,0]>=cfg.wet_depth)
    if not len(wet_mid):
        wet_mid=np.argwhere(np.max(water[:end//5+1,0],axis=0)>=cfg.wet_depth)
    if not len(wet_mid):raise ValueError('No wet exposure support')
    # Prefix-stable traps; they do not change the exposure RNG or hydraulics.
    order=trng.permutation(len(wet_mid))
    centers=wet_mid[order[:min(cfg.traps,len(order))]][:,::-1]*10+5
    cluster_centers=wet_mid[crng.integers(len(wet_mid),size=max(1,cfg.spatial_clusters))][:,::-1]*10+5
    events=[];wet={};attempts=0;identities=set()
    while len(events)<count and attempts<1000000:
        attempts+=1;u=exposure.random(7)
        if cfg.fall_pattern=='uniform':t=int(u[0]*(end+1))
        else:t=int(np.clip((int(u[1]*3)+.5)*end/3+(u[0]-.5)*cfg.burst_width,0,end))
        # Exact interpolated wet support at the exposure second.
        if t not in wet:
            a=t//5;b=min(a+1,300);alpha=t%5/5
            wet[t]=np.argwhere((1-alpha)*water[a,0]+alpha*water[b,0]>=cfg.wet_depth)
        ids=wet[t]
        if not len(ids):continue
        if cfg.spatial_clusters:
            center=cluster_centers[int(u[2]*cfg.spatial_clusters)]
            radius=cfg.cluster_radius*np.sqrt(u[3]);angle=2*np.pi*u[4]
            point=center+radius*np.array([np.cos(angle),np.sin(angle)])
            xy=ids[np.argmin(np.sum((ids[:,::-1]*10+5-point)**2,axis=1))][::-1]*10+5
        else:xy=ids[int(u[2]*len(ids))][::-1]*10+5
        key=(t,*xy)
        if key in identities:continue
        field=sample_water(water,t,np.asarray([xy]))[0]
        identities.add(key)
        events.append(dict(fall=t,deadline=t+cfg.window,xy=xy.tolist(),trigger='external_scheduled_event',field=field.tolist(),exposure_id=attempts))
    if len(events)!=count:raise ValueError('Exposure generation cap; no target deletion')
    # Selection is exposure order, never fall-time order; random public IDs.
    selected=[events[i] for i in irng.permutation(count) if i<cfg.people]
    return dict(seed=seed,events=selected,traps=centers.tolist(),attempted_exposures=attempts,
                pool_size=count,fall_end=end,config=asdict(cfg),id_assignment='independent shuffle',
                conditioning='Fixed N externally supplied in-water incidents; no fall-risk estimation')
