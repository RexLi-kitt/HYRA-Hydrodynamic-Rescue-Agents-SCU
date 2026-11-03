"""Common observation-only reservation layer. Physical collision checks unchanged."""
from pathlib import Path
import sys,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-v2-joint-v1'))
from candidate_env import DispatchEnv as CandidateEnv
from variable_environment import Env
from variable_dispatch import DispatchEnv as NearestEnv

class ReservationMixin:
    def __init__(self,*args,**kwargs):
        self.reservation_changes=0;self.reservation_person_changes=0;self.reservation_patrol_changes=0
        super().__init__(*args,**kwargs)

    def reserve(self,orders,options):
        """Keep nonconflicting requests; never access true fall/deadline/status.

        Person service has precedence over travel. Patrol exclusivity refers to
        four offered regions only; multiple distinct rescue targets in one region
        remain allowed. Unresolved cases retain safe stationary orders.
        """
        orders=copy.deepcopy(orders);groups={}
        for i,o in enumerate(orders):
            key=('person',o['person']) if o['person']>=0 else ('patrol',o['region']) if o['region']>=0 else None
            if key is not None:groups.setdefault(key,[]).append(i)
        claimed_people={o['person'] for o in orders if o['person']>=0}
        claimed_regions={o['region'] for o in orders if o['person']<0 and o['region']>=0}
        changes=[]
        def rank(i):
            o=orders[i];j=o['person'];distance=float(np.linalg.norm(o['xy']-self.pos[i]))
            serving=j>=0 and self.visible[j] and np.linalg.norm(self.seen_xy[j]-self.pos[i])<=self.cfg.drop_radius+1e-8
            return (not serving,distance,(i-int(self.t//self.cfg.step))%self.cfg.uavs)
        losers=[]
        for key,agents in groups.items():
            if len(agents)>1:losers.extend((i,key[0]) for i in sorted(agents,key=rank)[1:])
        for i,kind in sorted(losers,key=lambda x:rank(x[0])):
            alternatives=[]
            if kind=='person':
                alternatives=[o for o in options[i][:4] if o is not None and o['person']>=0 and o['person'] not in claimed_people]
            if not alternatives:
                alternatives=[o for o in options[i] if o is not None and o['person']<0 and o['region']>=0 and o['region'] not in claimed_regions]
            if alternatives:
                chosen=max(alternatives,key=lambda o:(o['utility'],-np.linalg.norm(o['xy']-self.pos[i])))
                orders[i]=copy.deepcopy(chosen)
                if chosen['person']>=0:claimed_people.add(chosen['person'])
                else:claimed_regions.add(chosen['region'])
            else:orders[i]=dict(xy=self.pos[i].copy(),person=-1,region=-1,utility=0.)
            changes.append((i,kind))
        return orders,changes

    def step(self,actions):
        actions=np.asarray(actions);mask,_,options,_=self.choices()
        if actions.shape!=(self.cfg.uavs,) or not np.issubdtype(actions.dtype,np.integer) or ((actions<0)|(actions>8)).any():raise ValueError('Invalid dispatch action')
        if not mask[np.arange(self.cfg.uavs),actions].all():raise ValueError('Unavailable dispatch choice')
        orders=[options[i][actions[i]] for i in range(self.cfg.uavs)]
        self._orders,changes=self.reserve(orders,options)
        self.reservation_changes+=len(changes)
        self.reservation_person_changes+=sum(kind=='person' for _,kind in changes)
        self.reservation_patrol_changes+=sum(kind=='patrol' for _,kind in changes)
        moves=self.control()[1]
        try:return Env.step(self,moves)
        finally:self._orders=None

    def metrics(self):
        m=super().metrics();m.update(reservation_changes=self.reservation_changes,
            reservation_person_changes=self.reservation_person_changes,reservation_patrol_changes=self.reservation_patrol_changes)
        return m

class ReservationEnv(ReservationMixin,CandidateEnv):pass
class NearestReservationEnv(ReservationMixin,NearestEnv):pass

class ResidualEnv(ReservationEnv):
    def choices(self):
        mask,expert,options,features=super().choices()
        for i,a in enumerate(expert):
            options[i][8]=copy.deepcopy(options[i][a]);features[i,8]=features[i,a].copy();mask[i,8]=True
        return mask,np.full(self.cfg.uavs,8,dtype=np.int64),options,features
