"""Bounded public-state beam planner used only to label training samples.

Predictions assume the last observed constant velocity, stationary teammates,
and ideal intended movement. The real referee always executes transitions;
these approximate predictions are labels, not inference-time action features.
"""
import numpy as np
import core as c


def plan(o, agent, target, max_depth=6, beam=4096):
    moves=c.search.geometry.MOVES[:8]
    goal=np.asarray(o.seen_xy[target],float)
    velocity=np.asarray(o.seen_v[target],float)
    age=max(0.,float(o.t-o.seen_at[target]))
    if o.seen_at[target]<0 or not o.active[target] or age>30: return None
    goal=goal+age*velocity
    team=np.delete(np.asarray(o.pos,float),agent,axis=0)
    team_heading=np.delete(np.asarray(o.heading,float),agent,axis=0)
    origin=np.asarray(o.pos[agent],float)
    initial_heading=np.asarray(o.heading[agent],float)
    nodes=origin[None,:]; headings=initial_heading[None,:]; paths=np.empty((1,0),np.int8)
    for depth in range(max_depth):
        current=goal+depth*5.*velocity
        future=goal+(depth+1)*5.*velocity
        team_delta=current-team
        team_visible=bool(((np.linalg.norm(team_delta,axis=1)<=20.+1e-9)&
                           ((team_delta*team_heading).sum(1)>=-1e-9)).any())
        diff=current[None,:]-nodes
        visible=team_visible | ((np.linalg.norm(diff,axis=1)<=20.+1e-9)&
                                ((diff*headings).sum(1)>=-1e-9))
        # All candidate paths, with the referee's visible-target inner-circle
        # stop rule modeled from public position and velocity estimate.
        starts=np.repeat(nodes,8,axis=0)
        dirs=np.tile(moves,(len(nodes),1))
        is_visible=np.repeat(visible,8)
        d=dirs*50.; delta=starts-current
        bb=2*np.sum(delta*d,axis=1); cc=np.sum(delta*delta,axis=1)-.25
        disc=bb*bb-4*2500.*cc
        roots=(-bb-np.sqrt(np.maximum(disc,0.)))/5000.
        hit=is_visible & (cc>1e-9) & (disc>=-1e-9) & (roots>=0.) & (roots<=1.)
        ends=starts+d*np.where(hit,roots,1.)[:,None]
        valid=((ends>=0)&(ends<=1000)).all(axis=1)
        # Avoid a stationary teammate's 5 m collision envelope on the full
        # candidate segment. This is approximate teacher planning only.
        for other in team:
            offset=other[None,:]-starts; segment=ends-starts
            frac=np.clip(np.sum(offset*segment,axis=1)/np.maximum(np.sum(segment*segment,axis=1),1e-12),0.,1.)
            valid &= np.linalg.norm(starts+frac[:,None]*segment-other[None,:],axis=1)>=5.-1e-8
        gap=future[None,:]-ends
        dist=np.linalg.norm(gap,axis=1)
        next_visible=((gap*dirs).sum(1)>=-1e-9) | team_visible
        wins=valid & (dist<=.9) & next_visible
        newpaths=np.concatenate((np.repeat(paths,8,axis=0),np.tile(np.arange(8,dtype=np.int8),len(nodes))[:,None]),axis=1)
        if wins.any():
            indices=np.flatnonzero(wins); best=indices[np.argmin(dist[indices])]
            return newpaths[best].astype(int).tolist()
        indices=np.flatnonzero(valid)
        if not len(indices): return None
        # Keep the first representative of each (position, heading) in sorted
        # distance order; direction matters for next-step visibility.
        indices=indices[np.argsort(dist[indices],kind='stable')]
        key=np.column_stack((np.round(ends[indices],4),np.tile(np.arange(8),len(nodes))[indices]))
        _,first=np.unique(key,axis=0,return_index=True)
        keep=indices[np.sort(first)[:beam]]
        nodes=ends[keep];headings=dirs[keep];paths=newpaths[keep]
    return None
