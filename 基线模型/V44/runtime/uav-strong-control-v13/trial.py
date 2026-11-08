from pathlib import Path
import sys,os,copy
from concurrent.futures import ProcessPoolExecutor

OUT=Path(__file__).resolve().parent
OLD=OUT.parent/"uav-dual-threshold-v12"
sys.path.insert(0,str(OLD))
os.environ["HYRA_STUDY_RADIUS"]="1"
from support import load_scene,read,save,sha,np,s,runtime,quota
from dispatch import Dispatch

INTERVAL=90
MODES=("cost","constant","scout","persist")
LEARNED=("switch0","switch15","switch40")
def state(o):
    known=o.seen_at>=0
    now=known&o.active;age=o.t-o.seen_at
    delta=o.pos[:,None]-o.seen_xy[None,:]
    distance=np.linalg.norm(delta,axis=-1).min(0)
    moving=np.linalg.norm(o.seen_v,axis=1)>.02
    near=now&(distance<100)&(age<=45)
    wet=o.measured.reshape(3,-1)[:,o.observed_cells]
    return np.array([o.t/2400,known.mean(),o.visible.sum()/4,now.sum()/8,near.sum()/4,
        (now&moving&(age<30)).sum()/4,(near&~moving).sum()/4,(now&(age>45)).sum()/8,
        (o.owner>=0).sum()/4,np.mean(o.last>=0),np.mean((o.last>=0)&(o.t-o.last<90)),
        np.mean(wet[0]>.01) if wet.size else 0.,np.mean(np.hypot(wet[1],wet[2])) if wet.size else 0.,
        np.linalg.norm(o.pos[:,None]-o.pos[None,:],axis=-1)[np.triu_indices(4,1)].mean()/1000])
class Switcher(Dispatch):
    def __init__(self,policy,head=None):
        super().__init__("cost_only")
        self.policy=policy;self.head=head
        self.variant=0
        self.initial_budget=self.budget;self.initial_memory=self.memory
        self.mode_steps=np.zeros(4,int)
        self.constant_weights={k:v.copy() for k,v in self.w.items()}
        for k in ("w","b","out"):self.constant_weights[k].fill(0.)
        self.constant_weights["bias"]=np.array(40.)
        self.w=self.constant_weights
    def act(self,o):
        if self.policy in MODES:self.variant=MODES.index(self.policy)
        elif o.t%INTERVAL==0:
            if self.policy=="rule":
                a=state(o)
                self.variant=3 if a[4]>.75 else 2 if a[7]>.4 else 0
            elif self.head is not None:
                z=np.clip((state(o)-self.head["mean"])/self.head["std"],-4,4)
                design=np.r_[1.,z,np.tanh(z@self.head["w"]+self.head["b"])]
                values=design@self.head["coef"];values[0]=int(self.policy[6:])/100
                self.variant=int(values.argmax())
        self.mode="balanced" if self.variant==1 else "geometry"
        self.version="probability" if self.variant==1 else "cost_only"
        self.budget=75 if self.variant==3 else self.initial_budget
        self.memory=110 if self.variant==3 else self.initial_memory
        self.mode_steps[self.variant]+=1
        return super().act(o)
    def assign(self,o):
        prior=self.track.copy();since=self.since.copy()
        assigned,xy=super().assign(o)
        if self.variant==2 and sum(assigned>=0)>=3:
            options=[i for i,j in enumerate(assigned) if j>=0 and o.owner[j]!=i and np.linalg.norm(xy[i]-o.pos[i])>65 and not o.visible[j]]
            if options:
                i=max(options,key=lambda i:np.linalg.norm(xy[i]-o.pos[i])+.4*(o.t-o.seen_at[assigned[i]]))
                assigned[i]=-1;xy[i]=o.pos[i]
        elif self.variant==3:
            for i,j in enumerate(prior):
                if j<0 or j in assigned or not o.active[j] or o.seen_at[j]<0:continue
                gap=o.t-o.seen_at[j]
                if gap>20 or o.t-since[i]>75 or self.invalid.get(j,-1)==o.seen_at[j]:continue
                other=assigned[i]
                if other>=0 and (o.owner[other]==i or (o.visible[other] and np.linalg.norm(o.seen_xy[other]-o.pos[i])<=1+1e-9)):continue
                goal=np.clip(o.seen_xy[j]+gap*o.seen_v[j],0,1000)
                if np.linalg.norm(goal-o.pos[i])<80:assigned[i]=j;xy[i]=goal
        self.since=np.where(assigned==prior,since,o.t);self.track=assigned.copy()
        return assigned,xy
def batches(fn,jobs):
    for i in range(0,len(jobs),18):
        quota()
        with ProcessPoolExecutor(max_workers=6) as ex:list(ex.map(fn,jobs[i:i+18]))
        print(fn.__name__,min(len(jobs),i+18),len(jobs),flush=True)
def train_roster(job):
    c,rep=job;path=OUT/"samples"/f"{c[1]}-{c[2]}-{rep}.npz"
    if path.exists():return
    f,p=load_scene(c,rep);cls,_=runtime(1,"planning");env=cls(f,p,(64,720,2400));a=Switcher("behavior")
    rng=np.random.default_rng(np.random.SeedSequence([19017,c[1],rep]));xx=[];yy=[]
    while env.t<2400:
        if env.t%INTERVAL==0 and env.t<=2250:
            start=env.t;xx.append(state(env.observation()));counts=[]
            for choice in range(4):
                shadow=copy.deepcopy(env,{id(env.water):env.water,id(env.truth):env.truth})
                agent=copy.deepcopy(a);agent.variant=choice
                while shadow.t<start+150:
                    if shadow.t==start+90:agent.variant=0
                    shadow.step(*agent.act(shadow.observation()))
                counts.append(int((shadow.supplied>=0).sum()-(env.supplied>=0).sum()))
            yy.append(counts);a.variant=int(rng.integers(4))
        env.step(*a.act(env.observation()))
    np.savez_compressed(path,x=np.array(xx),y=np.array(yy))
def train():
    (OUT/"samples").mkdir(exist_ok=True)
    fit=[c for c in s.cases("test") if c[3]<(24 if c[2]=="ordinary" else 8)]
    batches(train_roster,[(c,i) for c in fit for i in range(2)])
    xs=[];ys=[]
    for path in sorted((OUT/"samples").glob("*.npz")):
        with np.load(path) as f:xs.append(f["x"]);ys.append(f["y"])
    x=np.concatenate(xs);y=np.concatenate(ys)
    assert len(xs)==80 and x.shape==(2080,14) and y.shape==(2080,4)
    mean=x.mean(0);std=np.maximum(.05,x.std(0));z=np.clip((x-mean)/std,-4,4)
    rng=np.random.default_rng(28179);w=rng.normal(0,.4,(14,32));b=rng.normal(0,.2,32)
    matrix=np.c_[np.ones(len(z)),z,np.tanh(z@w+b)]
    reg=np.eye(matrix.shape[1])*45;reg[0,0]=0
    coef=np.linalg.solve(matrix.T@matrix+reg,matrix.T@(y-y[:,[0]]))
    np.savez(OUT/"switch.npz",mean=mean,std=std,w=w,b=b,coef=coef)
    save(OUT/"fit.json",dict(maps=len(fit),n_rosters=len(xs),states=len(y),branches=y.size,vary=int((np.ptp(y,axis=1)>0).sum()),cohort_deliveries=y.sum(0).tolist(),weights=sha(OUT/"switch.npz")))
def validate_roster(job):
    c,i=job;file=f"{c[1]}-{c[2]}-{i}.json";path=OUT/"dev"/file
    if path.exists():return
    f,p=load_scene(c,i);cls,_=runtime(1,"planning");rows={}
    existing=read(OLD/"transfer"/file)["results"]["720-64"]
    for dest,k in (("original","planning"),("cost","geom"),("V12","learning")):rows[dest]=existing[k]
    with np.load(OUT/"switch.npz") as z:head=dict(z)
    for mode in ("constant","scout","persist","rule",*LEARNED):
        a=Switcher(mode,head if mode in LEARNED else None);env=cls(f,p,(64,720,2400))
        while env.t<2400:
            env.step(*a.act(env.observation()))
            if (env.supplied>=0).all() and (env.last>=0).all():break
        rows[mode]=dict(**env.metrics(),mode_steps=a.mode_steps.tolist())
    save(path,dict(seed=c[1],family=c[2],rep=i,panel=c[0],methods=rows))
def validate():
    (OUT/"dev").mkdir(exist_ok=True)
    jobs=[(c,i) for c in s.cases("a")+s.cases("b") for i in range(2)]
    for c in (s.cases("a")[0],s.cases("a")[-1]):
        f,p=load_scene(c,0);cls,_=runtime(1,"planning");env=cls(f,p,(64,720,2400));a=Switcher("cost");b=Dispatch("cost_only")
        for k in range(480):
            o=env.observation();first=a.act(o);assert np.array_equal(first,b.act(o))
            env.step(*first)
    batches(validate_roster,jobs)
    rr=[read(OUT/"dev"/f"{c[1]}-{c[2]}-{i}.json") for c,i in jobs]
    keys=list(rr[0]["methods"]);counts={k:sum(r["methods"][k]["delivered"] for r in rr) for k in keys}
    pure=max(("original","cost","constant","scout","persist","rule"),key=lambda k:counts[k])
    chosen=max(LEARNED,key=lambda k:counts[k])
    outcome=dict(counts=counts,total=5120,strongest=pure,selected=chosen,relative=counts[chosen]/counts[pure]-1,
        passed=20*counts[chosen]>23*counts[pure],both_preferred=counts[chosen]/5120>.85 and counts[pure]/5120<.70)
    save(OUT/"development.json",outcome);print("DEVELOPMENT",outcome,flush=True)
if __name__=="__main__":
    train();validate()
