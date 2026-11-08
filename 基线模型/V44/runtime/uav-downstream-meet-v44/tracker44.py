import copy
from types import SimpleNamespace
import numpy as np
from rendezvous44 import Combined,projection,prepare,solve,sight,endpoints,FOLLOW


def incumbent_eta(agent,o,i,j,limit):
    """Prediction, not an oracle: the old intercept code receives a constant-velocity world."""
    shadow=copy.copy(agent)
    for name in ("meetings","plans","routes"):
        setattr(shadow,name,getattr(agent,name).copy())
    obs=SimpleNamespace(**vars(o))
    for name in ("pos","heading","seen_xy","seen_v","seen_at","visible","owner"):
        setattr(obs,name,getattr(o,name).copy())
    x=obs.seen_xy[j].copy();v=obs.seen_v[j].copy();progress=0
    for k in range(limit+2):
        p=obs.pos[i].copy();h=obs.heading[i].copy();y=x+5*k*v
        if not ((y>=0)&(y<1000)).all():
            return None
        here=bool(obs.visible[j])
        ends,_=endpoints(p[None])
        a,transit=Combined.intercept_action(shadow,obs,i,j,y,ends[0])
        code=int(a)+(0 if transit else 9)
        nxt,new_head=projection(p,h,y,v,here,code)
        credit=(here and code==FOLLOW and np.linalg.norm(y-p)<=1+1e-9 and
                np.linalg.norm(y+5*v-nxt)<=1+1e-9 and ((y+5*v>=0)&(y+5*v<1000)).all())
        progress=progress+1 if credit else 0
        if progress==2:
            return k+1
        obs.owner[j]=i if credit else -1
        obs.pos[i]=nxt;obs.heading[i]=new_head;obs.t+=5
        obs.visible[j]=sight(nxt,new_head,y+5*v)
        if obs.visible[j]:
            obs.seen_xy[j]=y+5*v;obs.seen_at[j]=obs.t
    return None


class Downstream(Combined):
    def __init__(self,mode,only_person=None):
        super().__init__()
        self.method44=mode;self.only44=only_person
        self.known44=np.zeros(64,bool);self.new44=set()
        self.plans44={};self.events44=[]

    def act(self,o):
        self.new44=set(np.flatnonzero(o.visible&~self.known44).tolist())
        self.known44|=o.visible
        return super().act(o)

    def assign(self,o):
        assigned,targets=super().assign(o)
        for i,entry in list(self.plans44.items()):
            j=entry["person"];step=(o.t-entry["start"])//5
            event=self.events44[entry["event"]]
            if step>=len(entry["steps"]):
                event["end"]="commands_finished";del self.plans44[i];continue
            pose,head,forecast,code=entry["steps"][step]
            y=o.seen_xy[j]+(o.t-o.seen_at[j])*o.seen_v[j]
            other=int(assigned[i])
            ready_elsewhere=(other>=0 and other!=j and o.visible[other] and np.linalg.norm(o.seen_xy[other]-o.pos[i])<=1+1e-9)
            reason=None
            if not o.active[j] or np.any((assigned==j)&(np.arange(4)!=i)) or o.owner[j] not in (-1,i):
                reason="target_inactive_or_handed_over"
            elif ready_elsewhere or np.any((o.owner==i)&(np.arange(len(o.owner))!=j)):
                reason="other_delivery_priority"
            elif np.linalg.norm(o.pos[i]-pose)>.03 or np.linalg.norm(o.heading[i]-head)>.02:
                reason="actual_aircraft_path_changed"
            elif np.linalg.norm(y-forecast)>.30 or np.linalg.norm(o.seen_v[j]-entry["v"])>.04:
                reason="observed_motion_changed"
            elif not o.visible[j] and (code>=9 or sight(o.pos[i],o.heading[i],y)):
                reason="failed_to_see_target"
            elif code==FOLLOW and np.linalg.norm(y-o.pos[i])>1+1e-9:
                reason="outside_delivery_radius"
            if reason:
                event["end"]=reason;del self.plans44[i];continue
            if assigned[i]!=j:
                self.since[i]=entry["start"];self.track[i]=j
            assigned[i]=j;targets[i]=np.clip(y,0,1000)
        return assigned,targets

    def intercept_action(self,o,i,j,target,ends):
        entry=self.plans44.get(i)
        if entry is not None and entry["person"]==j:
            k=(o.t-entry["start"])//5;code=entry["steps"][k][3]
            self.events44[entry["event"]]["commands"]+=1
            return code%9,code<9
        if self.method44=="base" or not self.wet or j not in self.new44 or self.only44 not in (None,j):
            return super().intercept_action(o,i,j,target,ends)
        v=o.seen_v[j];speed=np.linalg.norm(v)
        if speed<=.05 or not o.visible[j]:
            return super().intercept_action(o,i,j,target,ends)
        xy=o.seen_xy[j];ray=xy-o.pos
        observers=((ray*ray).sum(1)<=400+1e-9)&((ray*o.heading).sum(1)>=-1e-9)
        if not observers[i]:
            return super().intercept_action(o,i,j,target,ends)
        cross=np.abs(ray[:,0]*v[1]-ray[:,1]*v[0])/speed
        if np.min(cross[observers])<=1+1e-9:
            return super().intercept_action(o,i,j,target,ends)
        info=dict(t=int(o.t),i=int(i),j=int(j),cross=float(cross[i]),offered=False,used=False,commands=0)
        self.events44.append(info)
        limit=min(10,(2400-o.t-10)//5)
        if limit<0:
            info["end"]="insufficient_mission_time"
            return super().intercept_action(o,i,j,target,ends)
        best,extra=solve(o.pos[i],o.heading[i],xy,v,True,limit,self.method44=="extended")
        plan=None if best is None else prepare(o.pos[i],o.heading[i],xy,v,True,best)
        if not plan:
            info["end"]="no_certified_plan"
            return super().intercept_action(o,i,j,target,ends)
        eta=incumbent_eta(self,o,i,j,limit)
        info.update(offered=True,extra=extra,eta=len(plan),old_eta=eta)
        if eta is not None and eta<=len(plan):
            info["end"]="incumbent_as_fast"
            return super().intercept_action(o,i,j,target,ends)
        self.plans44[i]=dict(person=int(j),start=int(o.t),steps=plan,v=v.copy(),event=len(self.events44)-1)
        info.update(used=True,commands=1,codes=[q[3] for q in plan],end="in_progress")
        code=plan[0][3]
        return code%9,code<9
