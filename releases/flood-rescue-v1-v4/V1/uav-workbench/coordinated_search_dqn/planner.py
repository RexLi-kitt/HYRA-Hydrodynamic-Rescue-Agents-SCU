"""Global search uses only the boolean team measurement history and public geometry."""
from dataclasses import dataclass
import numpy as np
from flood_dqn.env import DIRECTIONS, DX, DZ, EPS
from rescue_assignment_dqn.controller import Controller

GRID=np.arange(10000).reshape(10,10,10,10).transpose(0,2,1,3).reshape(100,100)
X,Z=GRID%100,GRID//100
ZONE=(np.arange(10000)//1000)*10+np.arange(10000)%100//10

def regions(unseen):
    mask=unseen[GRID]
    n=mask.sum(1)
    # Integer distances to unnormalised centroid preserve exact ties.
    ds=(X*n[:,None]-(X*mask).sum(1)[:,None])**2+(Z*n[:,None]-(Z*mask).sum(1)[:,None])**2
    anchor=GRID[np.arange(100),np.where(mask,ds,10**12).argmin(1)]
    return n,np.column_stack((anchor%100,anchor//100))*10+5

def geometry(p,h,d,targets):
    with np.errstate(divide='ignore',invalid='ignore'):
        edge=np.where(d>0,(1000-p)/d,np.where(d<0,-p/d,np.inf)).min(-1)
    length=np.minimum(edge,50.)
    if len(targets):
        delta=targets-p[...,None,:]
        ahead=(delta*d[...,None,:]).sum(-1)
        sq=(delta**2).sum(-1)
        disc=100-sq+ahead**2
        entry=np.where((sq>100+EPS)&(ahead>0)&(disc>=-EPS),ahead-np.sqrt(np.maximum(0,disc)),np.inf)
        length=np.minimum(length,entry.min(-1))
    moved=(length>EPS)&(d!=0).any(-1)
    end=p+d*length[...,None]
    heading=np.where(moved[...,None],d,h)
    x=np.floor(end[...,0,None]/10).astype(int)+DX
    z=np.floor(end[...,1,None]/10).astype(int)+DZ
    dx=x*10+5-end[...,0,None]; dz=z*10+5-end[...,1,None]
    hit=(x>=0)&(x<100)&(z>=0)&(z<100)&(dx*dx+dz*dz<=625+EPS)&(dx*heading[...,0,None]+dz*heading[...,1,None]>=-EPS)
    return end,heading,np.where(hit,z*100+x,-1)

@dataclass
class Route:
    zone:int
    actions:list
    ends:list
    footprints:list
    gain:set

def candidates(p,h,targets,unseen,counts,anchors,budget):
    eta=np.maximum(1,np.ceil(np.linalg.norm(anchors-p,axis=1)/50))
    order=np.argsort(-counts/eta,kind='stable')
    order=order[counts[order]>0]
    for offset in range(0,len(order),8):
        zones=order[offset:offset+8]
        rows=np.arange(len(zones))
        pos=np.tile(p,(len(rows),1)); heading=np.tile(h,(len(rows),1))
        best=np.linalg.norm(pos-anchors[zones],axis=1); stall=np.zeros(len(rows),int)
        plans=[Route(int(z),[],[],[],set()) for z in zones]
        live=np.ones(len(rows),bool); found=[]
        for _ in range(min(32,budget)):
            end,hd,ids=geometry(pos[:,None,:],heading[:,None,:],DIRECTIONS[:8],targets)
            hits=(ids>=0)&unseen[ids]&(ZONE[ids]==zones[:,None,None])
            count=hits.sum(-1)
            dist=np.linalg.norm(end-anchors[zones,None],axis=-1)
            actions=np.where(count==count.max(1)[:,None],dist,np.inf).argmin(1)
            pos=end[rows,actions];heading=hd[rows,actions]
            for r in np.flatnonzero(live):
                a=int(actions[r]); plan=plans[r]; cells=ids[r,a]; cells=cells[cells>=0]
                plan.actions.append(a);plan.ends.append(pos[r].copy());plan.footprints.append(set(cells.tolist()))
                plan.gain.update(cells[unseen[cells]].tolist())
                if count[r,a]:
                    found.append(plan);live[r]=False
                elif dist[r,a]<=best[r]-1:
                    best[r]=dist[r,a];stall[r]=0
                else:
                    stall[r]+=1
                    if stall[r]>=2:live[r]=False
            if not live.any():break
        if found:return found
    return []

def select(routes,published):
    def key(r,discount):
        n=len(r.actions)
        blocked=set().union(*published[:n]) if discount else set()
        return (len(r.gain-blocked)/n,len(r.gain),-n,-r.zone)
    independent=max(routes,key=lambda r:key(r,False))
    joint=max(routes,key=lambda r:key(r,True))
    return independent,joint if key(joint,True)[0]>0 else independent

class Search:
    def __init__(self,group):
        self.group=group;self.rescue=Controller();self.cache={};self.signature=None

    def actions(self,e,q):
        a,reasons=self.rescue.actions(e,q)
        assigned=self.rescue.assigned
        order=[(e.step_no//4+k)%4 for k in range(4)]
        free=[i for i in order if i not in assigned]
        protected=free[0]
        note=dict(protected=protected,zones={},plans=[])
        if self.group=='S0':return a,reasons,note
        known=np.flatnonzero(e.found & ~e.rescued)
        targets=e.targets[known]
        sig=(e.step_no//4,tuple(sorted(assigned.items())),tuple(known))
        if sig!=self.signature:self.cache.clear()
        self.signature=sig
        published=[set() for _ in range(4)]
        def publish_action(i):
            _,_,cells=geometry(e.p[i],e.heading[i],DIRECTIONS[a[i]],targets)
            published[0].update(cells[cells>=0].tolist())
        for i in (*assigned,protected):publish_action(i)
        unseen=e.stamps<0
        stats=None
        # A retained route has already been announced, irrespective of ID order.
        for i,(route,k) in list(self.cache.items()):
            if k>=len(route.actions) or not np.allclose(e.p[i],route.ends[k-1],atol=1e-7,rtol=0):
                del self.cache[i]
            else:
                for j,cells in enumerate(route.footprints[k:k+4-e.step_no%4]):published[j].update(cells)
        for i in free[1:]:
            if i not in self.cache:
                if stats is None:stats=regions(unseen)
                choices=candidates(e.p[i],e.heading[i],targets,unseen,*stats,300-e.step_no)
                if not choices:
                    publish_action(i);note['plans'].append(dict(agent=i,zone=None));continue
                solo,joint=select(choices,published)
                route=joint if self.group=='S2' else solo
                self.cache[i]=(route,0)
                note['plans'].append(dict(agent=i,zone=route.zone,solo=solo.zone,solo_action=solo.actions[0],
                            action=route.actions[0],route=route.actions,options=[(r.zone,len(r.actions),len(r.gain),
                            len(r.gain-set().union(*published[:len(r.actions)]))) for r in choices]))
                for j,cells in enumerate(route.footprints[:4-e.step_no%4]):published[j].update(cells)
            route,k=self.cache[i];a[i]=route.actions[k];reasons[i]='search_route'
            note['zones'][i]=route.zone;self.cache[i]=(route,k+1)
        return a,reasons,note
