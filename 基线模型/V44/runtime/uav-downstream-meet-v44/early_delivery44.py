"""Choose a 120-second fleet policy using only released observations."""
import numpy as np
from continuous44 import Continuous
from destinations import DestinationSearch
from lattice import prepare_plan
from occupancy44 import predictors, heat_from_counts
from windows import features


def state(o):
    x = features(o)
    known = (o.seen_at>=0)&o.active
    if known.any():
        delta = o.seen_xy[known,None,:]-o.pos
        distances = np.linalg.norm(delta,axis=2)
        nearest = distances.min(1)
        extra = [(nearest<60).mean(),(distances<120).sum(1).mean()/4,
                 np.minimum(nearest/300,1).mean(), np.minimum((o.t-o.seen_at[known])/180,1).mean()]
    else:
        extra = [0.]*4
    return np.asarray(x+extra,dtype=float)


class SpeedChoice:
    def configure(self, fit):
        self.speed_fit = fit
        self.force = None
        self.epoch = -1
        self.arm = 0
        self.log = []
        self.dispatch_changes = 0

    def act(self,o):
        period=(o.t-60)//120
        if period>=0 and period!=self.epoch:
            self.epoch=period
            x=state(o)
            arm=0
            if self.force is not None:
                arm=self.force
            elif self.speed_fit is not None:
                z=np.clip((x-self.speed_fit["mean"])/self.speed_fit["scale"],-4,4)
                gain=self.speed_fit["bias"]+z@self.speed_fit["coef"]
                arm=int(np.argmax(np.r_[0.,gain]))
            self.arm=arm
            self.log.append(dict(t=int(o.t),x=x.tolist(),arm=arm))
        return super().act(o)


class QuickSearch(SpeedChoice,Continuous):
    def __init__(self, fit, occupancy_weights):
        super().__init__()
        self.configure(fit)
        self.density_coef=np.asarray(occupancy_weights)
        self.density_time=-100

    def fields(self,o):
        fixed,mobile=super().fields(o)
        if self.arm==2:
            if o.t-self.density_time>=30:
                eta=predictors(o,self.belief)@self.density_coef
                self.density=heat_from_counts(np.exp(np.clip(eta,-16,4)))
                self.density_time=o.t
            fixed=fixed+.5*self.density*10000/len(o.seen_at)
        return fixed,mobile

    def pressure_search(self,o,ends,length,assigned,targets):
        saved=self.easy,self.cross,self.commit
        if self.arm==1:
            self.easy,self.commit=1.15,2
        elif self.arm==2:
            self.cross=max(self.cross,.38)
        try:
            return super().pressure_search(o,ends,length,assigned,targets)
        finally:
            self.easy,self.cross,self.commit=saved


def joint_match(offers):
    # At most four drones. Process each person once so no target is duplicated.
    states={0:(0.,())}
    for j,options in offers.items():
        new=states.copy()
        for mask,(total,pairs) in states.items():
            for i,score in options:
                bit=1<<i
                if mask&bit:
                    continue
                key=mask|bit
                if key not in new or new[key][0]<total+score:
                    new[key]=(total+score,pairs+((i,j),))
        states=new
    return max(states.values(),key=lambda row:row[0])


class QuickDispatch(SpeedChoice,DestinationSearch):
    def __init__(self,fit):
        super().__init__(use_templates=True)
        self.configure(fit)

    def assign(self,o):
        prior,since=self.track.copy(),self.since.copy()
        assigned,goals=super().assign(o)
        if not self.arm or not np.any(assigned>=0):
            return assigned,goals
        locked=set()
        for i,j in enumerate(assigned):
            if j>=0 and (o.owner[j]>=0 or (o.visible[j] and np.linalg.norm(goals[i]-o.pos[i])<=1+1e-9)):
                locked.add(i)
        free=[i for i in range(4) if i not in locked]
        if not free:
            return assigned,goals
        available=set(int(assigned[i]) for i in free if assigned[i]>=0)
        if self.arm==2:
            available.update(np.flatnonzero(o.visible&o.active&(o.owner<0)))
        available.difference_update(assigned[i] for i in locked)
        offers={}
        locs={}
        base_value=0.
        for j in sorted(available):
            age=o.t-o.seen_at[j]
            vel=o.seen_v[j]
            loc=np.clip(o.seen_xy[j]+age*vel,0,1000)
            locs[j]=loc
            for i in free:
                old=assigned[i]==j
                gap=np.linalg.norm(loc-o.pos[i])
                if not old and (gap>140 or self.blocked.get((i,j),-1)>o.t or o.owner[j]>=0):
                    continue
                # Estimate staged intercept using last seen velocity and legal directions.
                approach=None
                if gap<=120:
                    approach=prepare_plan(o.pos[i],o.heading[i],loc,vel,bool(o.visible[j]),6)
                seconds=10+5*len(approach) if approach else 35+gap/10
                seconds+=.20*age + (0 if prior[i]==j else 5)
                value=100*np.exp(-seconds/60)
                offers.setdefault(j,[]).append((i,value))
                if old:
                    base_value+=value
        best,pairs=joint_match(offers)
        if best<=base_value+(7 if self.arm==1 else 3):
            return assigned,goals
        change=assigned.copy()
        for i in free:
            change[i]=-1
            goals[i]=o.pos[i]
        for i,j in pairs:
            change[i]=j
            goals[i]=locs[j]
        self.dispatch_changes+=int((change!=assigned).sum())
        self.track=change.copy()
        self.since=np.where(change==prior,since,o.t)
        return change,goals
