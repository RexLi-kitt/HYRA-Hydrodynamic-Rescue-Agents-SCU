"""Predict an intercept with a known contact; all wait steps issue target ID -1."""
from pathlib import Path
import sys
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/"uav-rescue-gate-v16"))
from beam16 import NoveltyBeam,DIRECTIONS
from lattice import near,directions,route,prepare_plan

HOLD=(8,True)
def in_domain(xy):
    return bool(np.all(xy>=0) and np.all(xy<1000))


def meeting_plan(pos,heading,xy,velocity,visible,max_wait,limit):
    """Return fewer than limit approach steps; service steps are common to all plans."""
    r2=1.+1e-9
    best=None
    horizon=min(15,limit-1)
    if horizon<1:return None
    forecast=xy+np.arange(horizon+3)[:,None]*5*velocity
    good=np.array([in_domain(p) for p in forecast])
    for moves in range(min(6,horizon)+1):
        for wait in range(min(max_wait,horizon-moves)+1):
            phase=moves+wait
            if best is not None and phase>=len(best):break
            target=forecast[phase]
            if not good[phase]:continue
            for displacement,counts in near(target-pos,moves):
                stage=pos+displacement
                if (stage<0).any() or (stage>1000).any():continue
                delta=target-stage
                gap_sq=delta@delta
                if gap_sq>400+1e-9:continue
                actions=directions(counts)
                for last in sorted(set(actions)) if actions else (-1,):
                    head=DIRECTIONS[last] if actions else heading
                    if (not visible if phase==0 else delta@head < -1e-9):continue
                    arrivals=[]
                    if gap_sq<=r2 and good[phase+2]:
                        arrivals.append((phase,[],gap_sq))
                    else:
                        along=DIRECTIONS@delta
                        side=np.maximum(0.,gap_sq-along*along)
                        for a in np.flatnonzero((along>=0)&(side<=.25+1e-12)):
                            short=max(0.,along[a]-np.sqrt(max(0.,.25-side[a])))
                            end=stage+DIRECTIONS[a]*short
                            if (end<0).any() or (end>1000).any():continue
                            # A zero-length movement leaves its previous heading intact.
                            final_head=DIRECTIONS[a] if short>1e-8 else head
                            last_t=min(horizon,phase+1+max_wait-wait)
                            for when in range(phase+1,last_t+1):
                                future=forecast[when]-end
                                if future@future<=r2 and future@final_head>=-1e-9 and good[when+2]:
                                    arrivals.append((when,[(int(a),False)]+[HOLD]*(when-phase-1),future@future))
                                    break
                    if not arrivals:continue
                    eta,tail,_=min(arrivals,key=lambda z:(z[0],z[2]))
                    if not eta or eta>=limit or (best is not None and eta>=len(best)):continue
                    prefix=route(pos,actions,last) if actions else []
                    if prefix is not None:best=prefix+[HOLD]*wait+tail
    return best


class Intercept(NoveltyBeam):
    def __init__(self,wait):
        super().__init__("service")
        self.max_wait=wait
        self.meetings={}
        self.proposed=0;self.reused=0;self.passive_waits=0

    def intercept_action(self,o,i,j,target,ends):
        p=o.pos[i];v=o.seen_v[j];vis=bool(o.visible[j]);gap=np.linalg.norm(target-p)
        if not self.wet or np.linalg.norm(v)<.05 or gap>170 or (vis and gap<=1.+1e-9):
            self.meetings.pop(i,None)
            return super().intercept_action(o,i,j,target,ends)
        prior=self.meetings.pop(i,None)
        plan=None
        if prior is not None:
            who,when,nextpos,nextxy,vel,tail=prior
            if who==j and when==o.t and np.linalg.norm(p-nextpos)<.03 and np.linalg.norm(target-nextxy)<.12 and np.linalg.norm(vel-v)<.02 and tail:
                if tail[0][1] or vis:
                    plan=tail;self.reused+=1
        if not plan:
            original=prepare_plan(p,o.heading[i],target,v,vis,8)
            cap=len(original) if original else 16
            plan=meeting_plan(p,o.heading[i],target,v,vis,self.max_wait,cap)
            if not plan:return super().intercept_action(o,i,j,target,ends)
            self.proposed+=1
        a,transit=plan[0]
        if a==8:
            predicted=p;self.passive_waits+=1
        elif transit:predicted=ends[a]
        else:
            along=(target-p)@DIRECTIONS[a]
            side=max(0.,(target-p)@(target-p)-along**2)
            if not vis or side>.25+1e-12:
                return super().intercept_action(o,i,j,target,ends)
            predicted=p+max(0.,along-np.sqrt(max(0.,.25-side)))*DIRECTIONS[a]
        self.meetings[i]=(j,o.t+5,predicted.copy(),target+5*v,v.copy(),plan[1:])
        return a,transit
