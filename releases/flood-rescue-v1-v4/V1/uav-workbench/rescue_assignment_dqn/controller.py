"""Persistent reservations based only on discovered targets and service feedback."""
import math
import numpy as np
from flood_dqn.env import DIRECTIONS,stop_distance,EPS


def endpoint(p,action,targets):
    length,_=stop_distance(p,DIRECTIONS[action],targets)
    return p+DIRECTIONS[action]*length


def navigation(p,target,targets):
    distance=np.linalg.norm(p-target)
    if distance>60:
        return min(range(8),key=lambda a:(np.linalg.norm(endpoint(p,a,targets)-target),a))
    frontier=[(p,())]
    best=None
    for depth in range(1,4):
        nxt=[];goals=[]
        for pos,seq in frontier:
            for a in range(8):
                end=endpoint(pos,a,targets);path=seq+(a,)
                d=np.linalg.norm(end-target)
                key=(d,path)
                if best is None or key<best[0]:best=(key,path)
                if d<=10+EPS:goals.append(path)
                nxt.append((end,path))
        if goals:return min(goals)[0]
        frontier=nxt
    return best[1][0]


class Controller:
    def __init__(self):
        self.assigned={}
        self.discovered={}
        self.cooldown={}
        self.pair_cooldown={}
        self.best={}
        self.stall={}
        self.releases=[]
        self.cache={}

    def release(self,i,reason,t,cool=0):
        if i not in self.assigned:return
        j=self.assigned.pop(i)
        self.releases.append(dict(t=t,agent=i,target=j,reason=reason))
        if cool:self.cooldown[j]=t+cool
        if reason=='stalled':self.pair_cooldown[i,j]=t+3
        self.best.pop(i,None);self.stall.pop(i,None);self.cache.pop(i,None)

    def prepare(self,e):
        t=e.step_no
        for j in np.flatnonzero(e.found):self.discovered.setdefault(int(j),t)
        for i,j in list(self.assigned.items()):
            if e.rescued[j]:self.release(i,'rescued',t)
        # Real service takes priority over an earlier approach reservation.
        for j,i in enumerate(e.owner):
            if i<0:continue
            i=int(i)
            if self.assigned.get(i)==j:continue
            for other,goal in list(self.assigned.items()):
                if other==i or goal==j:self.release(other,'service_handoff',t)
            self.assigned[i]=j;self.best[i]=np.linalg.norm(e.p[i]-e.targets[j]);self.stall[i]=0
        capacity=3-len(self.assigned)
        if capacity<=0:return
        free=[i for i in range(4) if i not in self.assigned]
        goals=[int(j) for j in np.flatnonzero(e.found & ~e.rescued)
               if j not in self.assigned.values() and self.cooldown.get(int(j),0)<=t]
        dp={0:(0.,())}
        for j in goals:
            update=dict(dp)
            for mask,(cost,pairs) in dp.items():
                if mask.bit_count()>=capacity:continue
                for n,i in enumerate(free):
                    if mask&(1<<n) or self.pair_cooldown.get((i,j),0)>t:continue
                    eta=math.ceil(max(0,np.linalg.norm(e.p[i]-e.targets[j])-10)/50)+2
                    if eta>300-t:continue
                    value=(cost+eta/(1+min(t-self.discovered[j],60)/12),tuple(sorted(pairs+((i,j),))))
                    key=mask|(1<<n)
                    if key not in update or value<update[key]:update[key]=value
            dp=update
        mask=min(dp,key=lambda m:(-m.bit_count(),dp[m]))
        for i,j in dp[mask][1]:
            self.assigned[i]=j;self.best[i]=np.linalg.norm(e.p[i]-e.targets[j]);self.stall[i]=0
        assert len(self.assigned)<=3 and len(set(self.assigned.values()))==len(self.assigned)

    def actions(self,e,q):
        self.prepare(e)
        result=q.argmax(1)
        reasons=['network']*4
        targets=e.targets[e.found & ~e.rescued]
        for i in range(4):
            if i in self.assigned:
                j=self.assigned[i]
                if np.sum((e.p[i]-e.targets[j])**2)<=100+EPS:
                    result[i]=8;reasons[i]='service'
                else:
                    key=(tuple(e.p[i]),tuple(np.flatnonzero(e.found & ~e.rescued)),j)
                    if self.cache.get(i,(None,None))[0]!=key:
                        self.cache[i]=(key,navigation(e.p[i],e.targets[j],targets))
                    result[i]=self.cache[i][1];reasons[i]='approach'
            elif result[i]==8 and len(targets) and (np.sum((targets-e.p[i])**2,axis=1)<=100+EPS).any():
                result[i]=q[i,:8].argmax();reasons[i]='explorer_no_service'
        return result,reasons

    def after(self,e,actions):
        for i,j in list(self.assigned.items()):
            if e.rescued[j]:
                self.release(i,'rescued',e.step_no);continue
            actual=np.flatnonzero(e.owner==i)
            if actions[i]==8:
                if len(actual):
                    continue
                self.release(i,'no_service_progress',e.step_no,6);continue
            distance=np.linalg.norm(e.p[i]-e.targets[j])
            if distance<=self.best[i]-1:
                self.best[i]=distance;self.stall[i]=0
            else:self.stall[i]+=1
            if self.stall[i]>=3:self.release(i,'stalled',e.step_no)
