import numpy as np
from scipy.linalg import solve
from collections import deque
from .scenarios import *
Z=np.arange(100);CENTERS=np.c_[Z%10,Z//10]*100+50
I=np.arange(10000);GRID=np.c_[I%100,I//100]*10+5;ZONE=I%100//10+(I//1000)*10
DIST=np.linalg.norm(CENTERS[:,None]-CENTERS[None],axis=-1)/100
A=(DIST==1).astype(float);W=(DIST<=2)/(1+DIST)
Q=np.eye(100)+np.diag(A.sum(1))-A
def norms(x):
    return (x-x.min())/max(1e-8,np.ptp(x))
class Memory:
    def __init__(self):
        self.stamp=np.full(10000,-1);self.last=np.full(40,-1);self.loc=np.zeros((40,2));self.done=set()
        self.frames=deque(maxlen=12);self.history=deque(maxlen=12);self.owner=np.full(4,-1);self.last_zone=np.full(4,-1)
    def observe(self,t,p,r,f,visible,pos):
        # Environment passes only visible IDs and their positions.
        delta=GRID[:,None]-p;mask=(np.square(delta).sum(-1)<=r*r).any(1)
        self.stamp[mask]=t
        ids=np.flatnonzero(mask);counts=np.bincount(ZONE[ids],minlength=100)
        water=f[t//5].reshape(3,-1)[:,ids]
        sums=np.stack([np.bincount(ZONE[ids],weights=a,minlength=100) for a in water],1)
        wet=np.bincount(ZONE[ids],weights=(water[0]>=.01).astype(float),minlength=100)
        new=visible[self.last[visible]==-1];self.last[visible]=t;self.loc[visible]=pos

        # Use physical region of discovery; exact boundary belongs to higher index.
        zi=np.floor(self.loc[new]/100).astype(int).clip(0,9)
        found=np.bincount(zi[:,0]+10*zi[:,1],minlength=100)
        missing=np.flatnonzero((self.last>=0)&~np.isin(np.arange(40),visible)&(self.last<t))
        for j in missing:
            if np.min(np.linalg.norm(p-self.loc[j],axis=1))<=r:self.last[j]=-2
        self.frames.append((counts,sums,wet,found));return mask
    def features(self,t,gm):
        rows=list(self.frames);weights=np.exp(-np.arange(len(rows)-1,-1,-1)/4)
        cnt=sum(w*v[0] for w,v in zip(weights,rows));su=sum(w*v[1] for w,v in zip(weights,rows));wet=sum(w*v[2] for w,v in zip(weights,rows))
        neigh=W@cnt;estimate=(W@su)/np.maximum(1e-10,neigh[:,None])
        stale=np.where(self.stamp<0,1,np.clip((t-self.stamp)/240,0,1))
        means=lambda a:np.bincount(ZONE,weights=a,minlength=100)/100
        age=means(stale);wetr=wet/np.maximum(cnt,1e-8)
        found=sum(v[3] for v in rows)
        feats=[np.c_[age,means(self.stamp>=0),rows[-1][0]/100,cnt/100,wetr,neigh>0,estimate,found,t/1320+np.zeros(100)]]
        def edges(flow):
            h,u,v=flow.T
            result=[]
            for off,valid,vel in [(1,Z%10<9,u),(-1,Z%10>0,-u),(10,Z//10<9,v),(-10,Z//10>0,-v)]:
                dst=(Z+off).clip(0,99);nvel=vel[dst]
                result.append(np.where(valid,.5*(vel+nvel),0))
                result.append(np.where(valid,.5*(vel*h+nvel*h[dst]),0))
            return np.stack(result,1)
        feats+=[edges(estimate),(A@feats[0])/A.sum(1)[:,None]]
        base=np.column_stack(feats)
        if gm:
            precision=cnt/20;mat=Q+np.diag(precision)
            from scipy.linalg import cho_factor,cho_solve
            fac=cho_factor(mat,check_finite=False)
            post=cho_solve(fac,su/20,check_finite=False)
            sd=np.sqrt(np.diag(cho_solve(fac,np.eye(100),check_finite=False)))
            extra=np.c_[post,sd,edges(post)]
        else:extra=np.zeros((100,12))
        self.history.append(np.c_[base,extra].astype('f'))
        return age,wetr
    def sequence(self):
        seq=list(self.history);return np.stack([np.zeros_like(seq[0])]*(12-len(seq))+seq,1)
def collide(p,end):
    """Continuous minimum over each of the five one-second segments, with iterative full-step stops."""
    delta=end-p;length=np.linalg.norm(delta,axis=1);direction=delta/np.maximum(length[:,None],1e-12)
    from flood_dqn.env import close
    blocked=set()
    while True:
        bad=set()
        for i in range(4):
            for j in range(i):
                if close(p[i],direction[i],0 if i in blocked else length[i],p[j],direction[j],0 if j in blocked else length[j]):bad.update((i,j))
        if not (bad-blocked):break
        blocked|=bad
    if blocked:length[list(blocked)]=0
    return np.stack([p+direction*np.minimum(length,10*k)[:,None] for k in range(1,6)]),len(blocked)
def plan(m,p,t,age,wet,demand,alpha):
    targets=[j for j in range(40) if m.last[j]>=0 and t-m.last[j]<=10 and j not in m.done]
    reserved=set();goal=p.copy();assign=np.full(4,-1);zone=np.full(4,-1)
    for i,j in enumerate(m.owner):
        if j in targets and np.linalg.norm(p[i]-m.loc[j])<=10:assign[i]=j;reserved.add(j)
    for dist,i,j in sorted((np.linalg.norm(p[i]-m.loc[j]),i,j) for i in range(4) for j in targets):
        if assign[i]<0 and j not in reserved:assign[i]=j;reserved.add(j)
    order=[(t//20+k)%4 for k in range(4)]
    for i in order:
        if assign[i]>=0:goal[i]=m.loc[assign[i]];continue
        dist=np.linalg.norm(CENTERS-p[i],axis=1)
        # Patrol the interiors of 100m zones, preferring stale observed wet regions.
        score=(.12+age)*(1+3*wet)/(1+dist/180)
        used=zone[zone>=0]
        if len(used):score/=1+2*np.exp(-DIST[:,used]**2).sum(1)
        score[dist<12]=0
        if demand is not None and np.ptp(demand)>1e-8 and alpha>0:
            score=(1-alpha)*norms(score)+alpha*norms(demand)/(1+dist/180)
            score[dist<12]=-1
        zone[i]=score.argmax();goal[i]=CENTERS[zone[i]]
    m.owner=assign;m.last_zone=zone
    d=goal-p;return assign,p+d*np.minimum(1,50/np.maximum(np.linalg.norm(d,axis=1),1e-10))[:,None],zone

