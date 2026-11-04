"""Privileged information probe; oracle modes are NOT deployable policies."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-variable-v1'))
from variable_dispatch import DispatchEnv

class PredictionEnv(DispatchEnv):
    def __init__(self,water,scenario,mode,trajectory=None):
        self.mode=mode;self.trajectory=trajectory
        super().__init__(water,scenario)

    def predict(self,i,j):
        if self.mode=='original':return np.clip(self.seen_xy[j]+self.seen_v[j]*min(30,self.t-self.seen_at[j]),0,1000)
        if self.mode=='hold':return self.seen_xy[j].copy()
        if self.mode=='oracle_now':return np.clip(self.trajectory[self.t,j],0,1000)
        cap=int(self.mode.rsplit('_',1)[1])
        tau=min(cap,float(np.linalg.norm(self.seen_xy[j]-self.pos[i])/self.cfg.speed))
        if self.mode.startswith('linear'):
            return np.clip(self.seen_xy[j]+self.seen_v[j]*(self.t-self.seen_at[j]+tau),0,1000)
        return np.clip(self.trajectory[min(self.cfg.duration,self.t+int(round(tau))),j],0,1000)

    def choices(self):
            """4 observed-person choices + 4 patrol regions + hold/service.
    
            No true deadlines, undiscovered persons or future water are consulted.
            The same candidate set is available to the coordinated planning baseline.
            """
            mask=np.zeros((self.cfg.uavs,9),bool);expert=np.full(self.cfg.uavs,8,int);options=[];features=np.zeros((self.cfg.uavs,9,10),np.float32)
            stale=np.where(self.known,(self.t-self.last)/120,2.)
            risk=np.clip(self.measured[0]/self.cfg.trigger_depth,0,2)+np.clip(np.linalg.norm(self.measured[1:],axis=0)/self.cfg.trigger_speed,0,2)
            score=stale*(1+risk)
            known=np.flatnonzero((self.belief==1)&self.discovered&(self.t-self.seen_at<=120))
            locked=set();claimed=set()
            for i,p in enumerate(self.pos):
                row=[None]*9;row[8]=dict(xy=p.copy(),person=-1,utility=0.,region=-1);mask[i,8]=True
                close=[j for j in known if self.visible[j] and np.linalg.norm(self.seen_xy[j]-p)<=self.cfg.drop_radius+1e-8]
                if close:
                    j=min(close,key=lambda j:np.linalg.norm(self.seen_xy[j]-p))
                    row[8]=dict(xy=self.seen_xy[j].copy(),person=int(j),utility=1.,region=-1)
                    locked.add(i);claimed.add(int(j))
                else:
                    nearest=sorted(known,key=lambda j:np.linalg.norm(self.seen_xy[j]-p))[:4]
                    for slot,j in enumerate(nearest):
                        goal=self.predict(i,j)
                        age=min(1.,max(0.,self.t-self.first_seen[j])/self.cfg.window)
                        utility=(1+1.5*age)/(10+np.linalg.norm(goal-p)/self.cfg.speed)
                        row[slot]=dict(xy=goal,person=int(j),utility=float(utility),region=-1);mask[i,slot]=True
                    for region in range(4):
                        ys=slice((region//2)*50,(region//2+1)*50);xs=slice((region%2)*50,(region%2+1)*50)
                        utility=score[ys,xs]/(1+np.linalg.norm(self.cells[ys,xs]-p,axis=-1)/250)
                        rr,cc=np.unravel_index(np.argmax(utility),utility.shape)
                        goal=self.cells[ys,xs][rr,cc]
                        row[4+region]=dict(xy=goal,person=-1,utility=float(utility[rr,cc]),region=region);mask[i,4+region]=True
                options.append(row)
                for slot,o in enumerate(row):
                    if o is None:continue
                    j=o['person'];goal=o['xy']
                    features[i,slot]=[1.,float(j>=0),*list(goal/1000),*list((goal-p)/1000),
                        (self.t-self.seen_at[j])/120 if j>=0 else 0.,
                        min(1.,max(0.,self.t-self.first_seen[j])/self.cfg.window) if j>=0 else 0.,
                        float(self.visible[j]) if j>=0 else 0.,min(5.,o['utility'])/5]
            # Global greedy assignment across all offered UAV-person pairs, with an
            # age-since-first-observation urgency proxy, never the hidden fall time.
            pairs=sorted([(o['utility'],i,s,o['person']) for i,row in enumerate(options) if i not in locked
                          for s,o in enumerate(row[:4]) if o is not None],reverse=True)
            assigned=set(locked)
            for _,i,s,j in pairs:
                if i not in assigned and j not in claimed:
                    expert[i]=s;assigned.add(i);claimed.add(j)
            patrol_claimed=set()
            for i,row in enumerate(options):
                if i in assigned:continue
                candidates=[s for s in range(4,8) if mask[i,s]]
                if candidates:
                    best=max(candidates,key=lambda s:row[s]['utility']*(1. if row[s]['region']==i%4 else .5)*(.5 if row[s]['region'] in patrol_claimed else 1.))
                    expert[i]=best;patrol_claimed.add(row[best]['region'])
            assert mask[np.arange(self.cfg.uavs),expert].all()
            return mask,expert,options,features
