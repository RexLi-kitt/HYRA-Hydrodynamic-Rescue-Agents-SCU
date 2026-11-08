"""Old rule-v2 SEARCH component adapted to the current public referee API.

Preserves quadrant routes, 26m waypoint tolerance, greedy cleanup and bounded
five-step lookahead. No old 20-person rescue logic is used. The cleanup sensor
is now the official 20m half-disc; only arrived public coverage is ingested.
"""
import math
from collections import deque
from itertools import product
import numpy as np
import core
from supply import sample_cells

old=core.module('preserved_old_carpet',core.RULE)
MAP=np.array([1,0,2,3,5,4,7,6,8])
COMBINATIONS=np.array(list(product(range(9),repeat=4)),dtype=int)


def yield_conflicts(pos,desired):
    """Minimum-change joint yielding for the new synchronous collision stop.

    Inspect only known aircraft positions and no-target action kinematics.
    Exact stop-time knots prevent a conflict being hidden between integer seconds.
    """
    from supply import raw_conflicts
    speed=core.geometry.MOVES*10
    speed=np.broadcast_to(speed,(4,9,2))
    bound=np.full_like(speed,np.inf)
    np.divide(1000-pos[:,None,:],speed,out=bound,where=speed>0)
    np.divide(-pos[:,None,:],speed,out=bound,where=speed<0)
    duration=np.maximum(0,np.minimum(5,bound.min(axis=-1)))
    knots=np.unique(np.r_[np.arange(6),duration.ravel()])
    tracks=pos[None,:,None,:]+np.minimum(knots[:,None,None],duration[None])[...,None]*speed[None]
    if not len(raw_conflicts(tracks[:,np.arange(4),desired])):return desired.copy()
    valid=np.ones(len(COMBINATIONS),bool)
    end=tracks[-1]
    allowed=(np.linalg.norm(end-pos[:,None],axis=-1)>1e-5);allowed[:,8]=True
    for i in range(4):valid &= allowed[i,COMBINATIONS[:,i]]
    for i in range(4):
        for j in range(i+1,4):
            delta=tracks[:,i,:,None,:]-tracks[:,j,None,:,:]
            change=np.diff(delta,axis=0)
            frac=np.clip(-(delta[:-1]*change).sum(-1)/np.maximum((change*change).sum(-1),1e-18),0,1)
            closest=delta[:-1]+frac[...,None]*change
            conflict=((closest*closest).sum(-1)<(5-1e-8)**2).any(axis=0)
            valid &= ~conflict[COMBINATIONS[:,i],COMBINATIONS[:,j]]
    candidates=COMBINATIONS[valid]
    if not len(candidates):raise RuntimeError('No collision-free joint search action')
    changed=candidates!=desired[None]
    deviation=((end[np.arange(4)[None],candidates]-end[np.arange(4),desired][None])**2).sum(axis=(1,2))
    identity=(changed*np.array([4,3,2,1])).sum(axis=1)
    order=np.lexsort((np.arange(len(candidates)),identity,deviation,changed.sum(axis=1)))
    result=candidates[order[0]]
    assert not len(raw_conflicts(tracks[:,np.arange(4),result]))
    return result.copy()


class Carpet:
    def __init__(self,yielding=True):
        self.p=old.Policy()
        self.p.cleanup=self.cleanup
        self.yielding=yielding;self.yield_steps=0;self.yield_changed=0
    def cleanup(self,i,pos,available,people):
        p=self.p;lo_x=0 if i%2==0 else 50;lo_y=0 if i<2 else 50
        uncovered=np.argwhere(~p.seen[lo_y:lo_y+50,lo_x:lo_x+50])
        if not len(uncovered):uncovered=np.argwhere(~p.seen);lo_x=lo_y=0
        if not len(uncovered):return 8
        grid=uncovered+np.array([lo_y,lo_x]);xy=grid[:,::-1]*10+5
        goal=xy[np.linalg.norm(xy-pos,axis=1).argmin()]
        if p.pending[i]:
            expected,a=p.pending[i].pop(0)
            if np.linalg.norm(pos-expected)<.01 and available[a]:return a
            p.pending[i]=[]
        def gain_at(end,a):
            cells=sample_cells(end[None],old.DIRECTIONS[a][None])
            return int((~p.seen.ravel()[cells]).sum())
        ranked=[];recent=p.recent[i]
        for a in range(8):
            if not available[a]:continue
            end=old.short_end(pos,old.DIRECTIONS[a],[])
            if np.linalg.norm(end-pos)<1e-5:continue
            x,y=end;outside=max(0.,lo_x*10-x,x-(lo_x+50)*10)+max(0.,lo_y*10-y,y-(lo_y+50)*10)
            repeated=any(np.linalg.norm(end-v)<1e-3 for v in recent[-8:])
            ranked.append((-gain_at(end,a),round(outside,4),repeated,float(np.linalg.norm(end-goal)),a))
        if not ranked:return 8
        if -min(ranked)[0]==0:
            queue=deque([(pos.copy(),[])]);visited={tuple(np.round(pos,3))};nodes=0
            while queue and nodes<5000:
                here,path=queue.popleft();nodes+=1
                order=sorted(range(8),key=lambda a:(float(np.linalg.norm(old.short_end(here,old.DIRECTIONS[a],[])-goal)),a))
                for a in order:
                    if not path and not available[a]:continue
                    end=old.short_end(here,old.DIRECTIONS[a],[])
                    if np.linalg.norm(end-here)<1e-5:continue
                    key=tuple(np.round(end,3))
                    if key in visited:continue
                    visited.add(key);newpath=path+[(here.copy(),a)]
                    if gain_at(end,a)>0:p.pending[i]=newpath[1:];return newpath[0][1]
                    if len(newpath)<5:queue.append((end,newpath))
        a=min(ranked)[-1];recent.append(pos.copy());del recent[:-8];return a
    def act(self,o):
        # Preserve the old public local+coarse coverage interface, at the precise
        # public positions now provided by the SDK. Empty people => pure search.
        covered=np.asarray(o.last)>=0;obs=np.zeros((4,740),float)
        obs[:,:100]=covered.reshape(10,10,10,10).mean(axis=(1,3)).ravel()
        team=np.zeros((4,4));team[:,:2]=o.pos/1000.;team[:,2:]=o.heading
        obs[:,400:416]=team.ravel();obs[:,416]=o.t/5/300
        padded=np.pad(covered,5,constant_values=True)
        masks=np.ones((4,9),bool)
        for i,pos in enumerate(o.pos):
            row,col=np.clip(np.floor(pos[::-1]/10).astype(int),0,99)
            obs[i,601:722]=padded[row:row+11,col:col+11].ravel()
            for a in range(8):masks[i,a]=np.linalg.norm(old.short_end(pos,old.DIRECTIONS[a],[])-pos)>1e-5
        chosen=self.p.act(obs,masks)
        if (self.p.seen&~covered).any():raise AssertionError('Carpet invented covered cells')
        moves=MAP[chosen]
        if self.yielding:
            resolved=yield_conflicts(np.asarray(o.pos),moves)
            self.yield_steps+=int(np.any(resolved!=moves));self.yield_changed+=int((resolved!=moves).sum())
            moves=resolved
        return moves,np.full(4,-1,int)


def make_policy():return Carpet()
