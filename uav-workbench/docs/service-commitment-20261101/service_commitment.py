"""Optional observed-information controller; never changes Env or model weights.

hold: force service only once inside a discovered target's service disk.
commit: reserve discovered target, approach, then hold; learned exploration otherwise.
This is a hybrid controller, not improvement of the trained neural policy itself.
"""
import numpy as np
from flood1km.env import MOVES

class ServiceCommitment:
    def __init__(self,mode):
        if mode not in ('hold','commit'):raise ValueError(mode)
        self.mode=mode;self.jobs={};self.cooldown={};self.previous=None
        self.calls=0;self.overrides=0;self.assignments=0;self.releases=0

    @staticmethod
    def decode(obs):
        obs=np.asarray(obs)
        if obs.shape!=(4,1468):raise ValueError('Frozen observation layout expected')
        pos=obs[0,1326:1346].reshape(4,5)[:,:2].astype(float)*1000
        records={}
        for row in obs[0,1346:1466].reshape(20,6):
            if row[5]>.5 and .25<float(row[2])<.75:
                key=tuple(np.round(row[:2].astype(float)*1000,3))
                records[key]=dict(progress=float(row[3])*10,owner=int(round(float(row[4])*4))-1)
        return pos,records

    @staticmethod
    def observed_unsafe(obs,pos,i,key):
        # Use only the stored measurement and mask from the actor's local patch.
        x,y=np.floor(np.asarray(key)/10).astype(int)-np.minimum(99,np.floor(pos[i]/10).astype(int))+5
        if not(0<=x<11 and 0<=y<11):return False
        patch=obs[i,:726].reshape(6,11,11)
        if patch[0,y,x]<.5:return False # unknown is NOT certified safe
        h,u,v=patch[1:4,y,x]*5
        return bool(h>.08+1e-7 or np.hypot(u,v)>.3+1e-7)

    def select(self,obs,mask,proposed):
        pos,targets=self.decode(obs);self.calls+=1;t=self.calls
        actions=np.array(proposed,dtype=np.int64,copy=True);mask=np.asarray(mask,bool)
        self.cooldown={k:end for k,end in self.cooldown.items() if end>t}
        def drop(i,key):
            self.jobs.pop(i,None);self.cooldown[(i,key)]=t+10;self.releases+=1
        for i,job in list(self.jobs.items()):
            key=job['target']
            if key not in targets:self.jobs.pop(i);continue
            dist=np.linalg.norm(pos[i]-key)
            stationary=self.previous is not None and np.linalg.norm(pos[i]-self.previous[i])<.001
            job['stalls']=job['stalls']+1 if stationary and (dist>10.001 or targets[key]['progress']==0) else 0
            if self.observed_unsafe(obs,pos,i,key) or t-job['start']>=40 or job['stalls']>=3:drop(i,key)
        reserved={job['target'] for job in self.jobs.values()}
        # Respect publicly observed service ownership before assigning new work.
        for key,r in sorted(targets.items()):
            i=r['owner']
            if r['progress']>0 and 0<=i<4 and i not in self.jobs and key not in reserved and not self.observed_unsafe(obs,pos,i,key):
                self.jobs[i]=dict(target=key,start=t,stalls=0);reserved.add(key)
        candidates=[]
        for i in range(4):
            if i in self.jobs:continue
            for key in targets:
                dist=float(np.linalg.norm(pos[i]-key))
                if key in reserved or (i,key) in self.cooldown or self.observed_unsafe(obs,pos,i,key):continue
                if self.mode=='hold' and dist>10.001:continue
                candidates.append((dist,i,key))
        for _,i,key in sorted(candidates):
            if i in self.jobs or key in reserved:continue
            self.jobs[i]=dict(target=key,start=t,stalls=0);reserved.add(key);self.assignments+=1
        phases=['learned']*4
        for i,job in self.jobs.items():
            key=job['target'];d=np.asarray(key)-pos[i]
            if np.linalg.norm(d)<=10.001:actions[i]=8;phases[i]='hold'
            elif self.mode=='commit':
                scores=MOVES[:8]@d;scores[~mask[i,:8]]=-np.inf
                actions[i]=int(np.argmax(scores)) if np.isfinite(scores).any() else 8
                phases[i]='approach'
        assert mask[np.arange(4),actions].all()
        changed=int(np.count_nonzero(actions!=proposed));self.overrides+=changed
        self.previous=pos.copy()
        return actions,dict(phases=phases,overrides=changed,jobs={str(i):list(j['target']) for i,j in self.jobs.items()})
