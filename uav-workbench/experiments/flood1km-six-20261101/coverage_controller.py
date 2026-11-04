"""Observed-history coverage watchdog, optional separate planning baseline."""
import numpy as np
from flood1km.env import MOVES,min_separation

Y,X=np.indices((100,100));CENTERS=np.stack((X*10+5,Y*10+5),-1)

def project(p,d):
    delta=d*50;f=1.
    for k in range(2):
        if delta[k]>0:f=min(f,(1000-p[k])/delta[k])
        elif delta[k]<0:f=min(f,-p[k]/delta[k])
    return np.clip(p+delta*f,0,1000)

def view(p,heading):
    d=CENTERS-p
    # Conservative margins handle float32 public-position rounding.
    return ((d*d).sum(-1)<99.999**2)&(d@heading>.001)

class CoverageWatchdog:
    def __init__(self):
        self.known=np.zeros((100,100),bool);self.stalls=0;self.active=False
        self.activation_step=None;self.calls=0;self.steps=0

    def observe(self,obs):
        previous=int(self.known.sum());state=obs[0,1326:1346].reshape(4,5)
        pos=state[:,:2].astype(float)*1000;heading=state[:,2:4].astype(float)
        for i in range(4):
            self.known |= view(pos[i],heading[i])
            # Skip ambiguous floor indices near internal grid boundaries.
            p=pos[i]
            ambiguous=any(.001<c<999.999 and abs(c/10-round(c/10))<.0001 for c in p)
            if ambiguous:continue
            c,r=np.minimum(99,np.floor(p/10).astype(int))
            patch=obs[i,:726].reshape(6,11,11)[0]>.5
            for dy in range(-5,6):
                for dx in range(-5,6):
                    rr,cc=r+dy,c+dx
                    if 0<=rr<100 and 0<=cc<100 and patch[dy+5,dx+5]:self.known[rr,cc]=True
        full=obs[0,726:826].reshape(10,10)>.999999
        self.known |= np.repeat(np.repeat(full,10,axis=0),10,axis=1)
        new=int(self.known.sum())-previous;self.calls+=1
        self.stalls=self.stalls+1 if new==0 else 0
        if self.stalls>=5 and not self.known.all() and not self.active:
            self.active=True;self.activation_step=self.calls
        return pos,heading,new

    def select(self,obs,mask,actions,phases):
        pos,heading,new=self.observe(obs);a=np.array(actions,copy=True);used=0
        if not self.active:return a,dict(active=False,new_memory_cells=new,planned_agents=0)
        unassigned=~self.known.copy();ends=np.array([project(p,MOVES[int(x)]) for p,x in zip(pos,a)])
        for i in range(4):
            if phases[i]!='learned':unassigned &= ~view(ends[i],heading[i])
        for i in range(4):
            if phases[i]!='learned':continue
            region=np.zeros((100,100),bool);region[(i//2)*50:(i//2+1)*50,(i%2)*50:(i%2+1)*50]=True
            target_mask=unassigned & region
            if not target_mask.any():target_mask=unassigned
            ids=np.argwhere(target_mask)
            if not len(ids):continue
            coords=ids[:,::-1]*10+5;goal=coords[np.argmin(((coords-pos[i])**2).sum(1))]
            candidates=[]
            for act in range(8):
                if not mask[i,act]:continue
                end=project(pos[i],MOVES[act]);delta=end-pos[i]
                if np.linalg.norm(delta)<.001:continue
                h=delta/np.linalg.norm(delta);visible=view(end,h)
                gain=int(np.count_nonzero(visible & target_mask))
                conflict=any(min_separation(pos[i],end,pos[j],ends[j] if phases[j]!='learned' or j<i else pos[j])<5.001 for j in range(4) if i!=j)
                candidates.append(dict(action=act,end=end,heading=h,gain=gain,conflict=conflict,visible=visible))
            if not candidates:continue
            safe=[x for x in candidates if not x['conflict']]
            choices=safe or candidates
            if max(x['gain'] for x in choices)>0:
                chosen=max(choices,key=lambda x:(x['gain'],-np.linalg.norm(x['end']-goal),int(x['action']==a[i]),-x['action']))
            else:
                # A near hole may sit behind every 50m endpoint. Two-step lookahead
                # avoids forward/backward oscillation without changing motion rules.
                for x in choices:
                    best=np.inf
                    for d in MOVES[:8]:
                        end2=project(x['end'],d);delta=end2-x['end'];length=np.linalg.norm(delta)
                        if length<.001:continue
                        rel=goal-end2
                        if np.linalg.norm(rel)<99.999 and rel@(delta/length)>.001:best=min(best,np.linalg.norm(rel))
                    x['two_step']=best
                possible=[x for x in choices if np.isfinite(x['two_step'])]
                chosen=min(possible,key=lambda x:(x['two_step'],x['action'])) if possible else min(choices,key=lambda x:(np.linalg.norm(x['end']-goal),x['action']))
            a[i]=chosen['action'];ends[i]=chosen['end'];unassigned &= ~chosen['visible'];used+=1
        assert np.asarray(mask)[np.arange(4),a].all()
        self.steps+=used
        return a,dict(active=self.active,new_memory_cells=new,planned_agents=used,activation_step=self.activation_step)

def service_mask(boundary_mask,forced_actions,phases):
    mask=np.array(boundary_mask,copy=True)
    for i,phase in enumerate(phases):
        if phase!='learned':mask[i]=False;mask[i,int(forced_actions[i])]=True
    return mask
