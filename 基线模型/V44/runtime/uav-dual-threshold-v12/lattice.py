"""Longer-range interception on the existing eight-direction action set."""
from functools import lru_cache
from collections import defaultdict
from itertools import product
from support import *
# cardinal x, cardinal y, NE/SW, SE/NW. Direction IDs match MOVES.
BASIS=DIRECTIONS[[3,0,6,7]]*50
POS_IDS=(3,0,6,7);NEG_IDS=(2,1,5,4)
@lru_cache(maxsize=1)
def bins():
    result=defaultdict(list)
    for count in product(range(-10,11),repeat=4):
        steps=sum(abs(i) for i in count)
        if steps>10:continue
        pos=np.asarray(count)@BASIS
        grid=tuple(np.floor(pos/24).astype(int))
        result[(steps,*grid)].append((pos,count))
    return result
def near(relative,n):
    x,y=np.floor(relative/24).astype(int);b=bins()
    for a in (-1,0,1):
        for k in (-1,0,1):
            yield from b.get((n,x+a,y+k),())
def directions(count):
    return [POS_IDS[k] if v>0 else NEG_IDS[k] for k,v in enumerate(count) for _ in range(abs(v))]
def route(start,actions,ending):
    left=actions.copy();left.remove(ending)
    dest=start+DIRECTIONS[actions].sum(axis=0)*50
    p=start.copy();flight=[]
    for i in range(len(actions)-1):
        line=start+(dest-start)*(i+1)/len(actions)
        ids=sorted(set(left));ends=p+DIRECTIONS[ids]*50
        ok=((ends>=0)&(ends<=1000)).all(axis=1)
        if not ok.any():return None
        a=ids[np.argmin(np.where(ok,np.square(ends-line).sum(1),np.inf))]
        flight.append((a,True));left.remove(a);p=p+DIRECTIONS[a]*50
    return flight+[(ending,True)]
def prepare_plan(pos,heading,xy,vel,visible,max_steps):
    # The only person data here are the last known location and observed velocity.
    for n in range(max_steps+1):
        goal=xy+5*n*vel
        if np.any(goal<-2) or np.any(goal>1002):continue
        for displacement,count in near(goal-pos,n):
            stage=pos+displacement
            if ((stage<0)|(stage>1000)).any():continue
            delta=goal-stage
            if delta@delta>400+1e-9:continue
            moves=directions(count)
            for last in sorted(set(moves)) if n else (-1,):
                head=DIRECTIONS[last] if n else heading
                seen=delta@head>=-1e-9 if n else visible
                if not seen:continue
                finish=None
                for waits in (0,1,2):
                    offset=delta+waits*5*vel
                    if offset@offset<=1+1e-9 and offset@head>=-1e-9:
                        finish=[(8,True)]*waits;break
                if finish is None:
                    along=DIRECTIONS@delta;side=np.maximum(0,delta@delta-along*along)
                    for a in np.flatnonzero((along>=0)&(side<=.25)):
                        step=max(0.,float(along[a]-np.sqrt(.25-side[a])))
                        endpoint=stage+step*DIRECTIONS[a]
                        if ((endpoint<0)|(endpoint>1000)).any():continue
                        for waits in (0,1,2):
                            diff=goal+5*(1+waits)*vel-endpoint
                            if diff@diff<=1+1e-9 and diff@DIRECTIONS[a]>=-1e-9:
                                finish=[(int(a),False)]+[(8,True)]*waits;break
                        if finish is not None:break
                if finish is None:continue
                prefix=route(pos,moves,last) if n else []
                if prefix is not None and (prefix or finish):return prefix+finish
    return []
class Router(WaitingNavigator):
    def __init__(self,mode):
        super().__init__(np.zeros(49),np.ones((20,20)),"wait")
        self.lookup=np.repeat((H>=.01).astype(float),len(V));self.legacy.original=Planner("water",1)
        self.mode=mode;self.routes={};self.used=0
        with np.load(D/"reliability.npz") as w:self.w=dict(w)
    def intercept_action(self,o,i,j,target,ends):
        pos=o.pos[i];v=o.seen_v[j];visible=o.visible[j];gap=np.linalg.norm(target-pos)
        if visible and gap<=1+1e-9:return 8,False
        old=self.routes.get(i)
        if old is not None and old[0]==j and old[1]==o.t and np.linalg.norm(old[2]-pos)<.03 and np.linalg.norm(old[3]-target)<.55:
            plan=old[4]
        else:
            plan=[]
            if self.mode=="geometry":horizon=8
            else:
                taus=np.arange(1,11)*5
                z=np.clip((feature(o,np.full(len(taus),j),taus)-self.w["mean"])/self.w["std"],-4,4)
                logits=np.tanh(z@self.w["w"]+self.w["b"])@self.w["out"]+self.w["bias"]
                probability=1/(1+np.exp(-logits))
                acceptable=np.flatnonzero(probability>(.55 if self.mode=="balanced" else .82))
                horizon=0 if not len(acceptable) else int(acceptable[-1]+1)
            if gap<170 and horizon:plan=prepare_plan(pos,o.heading[i],target,v,visible,horizon)
        if not plan:
            self.routes.pop(i,None)
            return super().intercept_action(o,i,j,target,ends)
        action,transit=plan[0];self.used+=1
        predicted=pos if action==8 else ends[action]
        self.routes[i]=(j,o.t+5,predicted,target+5*v,plan[1:])
        return action,transit
