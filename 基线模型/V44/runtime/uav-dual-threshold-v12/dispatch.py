from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import *
from lattice import Router,prepare_plan
OPTIONS=("cost_only","probability","risk_averse")
class Dispatch(Router):
    def __init__(self,version):
        super().__init__("geometry" if version=="cost_only" else "balanced")
        self.version=version
        self.match_count=0
    def assign(self,o):
        prior=self.track.copy();since=self.since.copy()
        base,fallback=super().assign(o)
        if not (base>=0).any():return base,fallback
        ids=np.flatnonzero(o.active & (o.seen_at>=0))
        age=o.t-o.seen_at[ids];vel=o.seen_v[ids];moving=np.linalg.norm(vel,axis=1)>.02
        valid=(age<=np.where(moving,self.memory,480))
        valid &= np.array([self.invalid.get(j,-1)!=o.seen_at[j] for j in ids])|o.visible[ids]
        ids=ids[valid]
        if not len(ids):return base,fallback
        age=o.t-o.seen_at[ids];vel=o.seen_v[ids];loc=np.clip(o.seen_xy[ids]+vel*age[:,None],0,1000)
        moving=np.linalg.norm(vel,axis=1)>.02
        offers=[]
        for i in range(4):
            delta=loc-o.pos[i];distance=np.linalg.norm(delta,axis=1)
            nearby=np.flatnonzero(distance<np.where(moving,240,360))
            if not len(nearby):continue
            order=nearby[np.argsort(distance[nearby]-.5*(prior[i]==ids[nearby])*20)]
            considered=list(order[:3])
            for k in nearby:
                if (base[i]==ids[k] or o.owner[ids[k]]==i) and k not in considered:considered.append(k)
            for k in considered:
                j=ids[k];owner=(o.owner[j]==i);ready=o.visible[j] and distance[k]<=1+1e-9
                if self.blocked.get((i,j),-1)>o.t and not owner:continue
                cache=self.routes.get(i)
                if cache and cache[0]==j and cache[1]==o.t and np.linalg.norm(cache[2]-o.pos[i])<.03 and np.linalg.norm(cache[3]-loc[k])<.55 and cache[4]:
                    steps=len(cache[4])
                elif ready:steps=0
                elif distance[k]<165:
                    route=prepare_plan(o.pos[i],o.heading[i],loc[k],vel[k],o.visible[j],7)
                    steps=len(route) if route else 9+distance[k]/50
                else:steps=6+distance[k]/50
                offers.append((i,int(j),loc[k],owner,ready,steps,age[k]))
        if not offers:return base,fallback
        js=np.array([r[1] for r in offers]);times=np.array([min(40,5*r[5]+10) for r in offers])
        if self.version=="cost_only":prob=np.ones(len(js))
        else:
            x=np.clip((feature(o,js,times)-self.w["mean"])/self.w["std"],-4,4)
            prob=1/(1+np.exp(-np.clip(np.tanh(x@self.w["w"]+self.w["b"])@self.w["out"]+self.w["bias"],-20,20)))
        chosen=np.full(4,-1,int);goals=o.pos.copy();claimed=set()
        def score(k):
            i,j,_,owner,ready,steps,age=offers[k]
            wait=(steps+2)/(max(.1,prob[k])**(1.5 if self.version=="risk_averse" else .75))
            return (not owner,not ready,wait+.008*age-.9*(prior[i]==j),i,j)
        for k in sorted(range(len(offers)),key=score):
            i,j,xy,*_=offers[k]
            if chosen[i]>=0 or j in claimed:continue
            chosen[i]=j;goals[i]=xy;claimed.add(j)
        self.match_count+=int((chosen!=base).sum())
        self.since=np.where(chosen==prior,since,o.t);self.track=chosen.copy()
        return chosen,goals
def ejob(task):
    c,i=task
    out=D/"dispatch-rows"/f"{c[1]}-{c[2]}-{i}.json"
    if out.exists():return
    f,p=load_scene(c,i);cls,_=runtime(1,"planning");rows={}
    for mode in OPTIONS:
        a=Dispatch(mode);env=cls(f,p,(64,1200,2400))
        while env.t<2400:
            env.step(*a.act(env.observation()))
            if (env.supplied>=0).all() and (env.last>=0).all():break
        rows[mode]=dict(**env.metrics(),changed=a.match_count)
    save(out,dict(seed=c[1],family=c[2],panel=c[0],rep=i,results=rows))
def main():
    (D/"dispatch-rows").mkdir(exist_ok=True)
    jobs=[(c,i) for c in s.cases("a")+s.cases("b") for i in range(2)]
    s.pool(ejob,jobs,1,workers=6)
    summary={}
    for mode in OPTIONS:
        panels={k:[0,0] for k in ("a","b")};found=0;alls=0
        for c,i in jobs:
            file=f"{c[1]}-{c[2]}-{i}.json";m=read(D/"dispatch-rows"/file)["results"][mode]
            panels[c[0]][0]+=m["delivered"];panels[c[0]][1]+=read(D/file)["planning"]["delivered"];found+=m["discovered"];alls+=m["all"]
        total=sum(a for a,b in panels.values());control=sum(b for a,b in panels.values())
        summary[mode]=dict(total=total,relative=total/control-1,found=found,all=alls,panels=panels,gate=total/control>=1.10 and all(a>b for a,b in panels.values()))
    choice=max(OPTIONS[1:],key=lambda x:summary[x]["total"])
    save(D/"extension-results.json",dict(chosen=choice,candidates=summary))
    print(choice,summary,flush=True)
if __name__=="__main__":main()
