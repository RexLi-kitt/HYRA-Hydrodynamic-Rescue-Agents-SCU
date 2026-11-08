from pathlib import Path
import os,sys
D=Path(__file__).resolve().parent
V10=D.parent/"uav-priority-confirm-v10"
sys.path.insert(0,str(V10))
os.environ["HYRA_STUDY_RADIUS"]="1"
from campaign10 import runtime
import scenario10 as s
from scenario10 import np,read,save,quota,water,sha
from navigation import DIRECTIONS
from extra import WaitingNavigator
from planner import H,V,Planner
def load_scene(c,rep):
    f=water(read(s.inputs(c)/"input.json")["water"])
    with np.load(s.history(c,rep)) as z:p={k:(z[k][:64] if k=="fall" else z[k][:,:64]) for k in ("points","fall","available","velocity")}
    return f,p
def feature(o,j,tau):
    j=np.asarray(j);t=np.broadcast_to(tau,j.shape)
    age=o.t-o.seen_at[j]
    xy=np.clip(o.seen_xy[j],0,999.999);ij=(xy/10).astype(int)
    h,u,v=o.measured[:,ij[...,1],ij[...,0]]
    velocity=o.seen_v[j];speed=np.linalg.norm(velocity,axis=-1)
    flow=np.hypot(u,v);moving=speed>.02
    horizon=(age+t)/60
    future=xy+(age+t)[...,None]*velocity
    edge=np.minimum(future,1000-future).min(-1)/25
    return np.stack([horizon,horizon**2,speed,np.log1p(np.maximum(0,h)),flow,abs(flow-speed),moving,
        np.clip(edge,-3,3),age/60,moving*horizon,o.t/2400+0*t],-1)
def fit_job(job):
    c,rep=job
    f,p=load_scene(c,rep);cls,policy=runtime(1,"planning");env=cls(f,p,(64,1200,2400))
    xs=[];ys=[]
    while env.t<2400:
        o=env.observation()
        if o.t%15==0:
            ids=np.flatnonzero(o.active&(o.seen_at>=0)&(o.t-o.seen_at<=60))
            if len(ids):
                who=np.repeat(ids,8);dt=np.tile(np.arange(5,45,5),len(ids));t=dt+o.t;ok=t<=2390
                who=who[ok];dt=dt[ok];t=t[ok]
                if len(who):
                    pred=o.seen_xy[who]+(t-o.seen_at[who])[:,None]*o.seen_v[who]
                    good=p["available"][t,who]&(t+10<=env.deadline[who])&(np.linalg.norm(pred-p["points"][t,who],axis=1)<=1.)
                    xs.append(feature(o,who,dt));ys.append(good)
        env.step(*policy.act(o))
    assert xs
    np.savez_compressed(D/"sample"/f"{c[1]}-{c[2]}-{rep}.npz",x=np.concatenate(xs),y=np.concatenate(ys))
def train():
    (D/"sample").mkdir(exist_ok=True)
    cases=[c for c in s.cases("test") if c[3]<(24 if c[2]=="ordinary" else 8)]
    jobs=[(c,i) for c in cases for i in range(2)]
    s.pool(fit_job,jobs,1,workers=4)
    x=[];y=[]
    for c,i in jobs:
        with np.load(D/"sample"/f"{c[1]}-{c[2]}-{i}.npz") as a:x.append(a["x"]);y.append(a["y"])
    x=np.concatenate(x);y=np.concatenate(y);mean=x.mean(0);std=np.maximum(.07,x.std(0));x=np.clip((x-mean)/std,-4,4)
    rng=np.random.default_rng(81604);w=rng.normal(0,.2,(x.shape[1],24));b=np.zeros(24);out=rng.normal(0,.15,24);bias=np.array(0.)
    pars=[w,b,out,bias];m=[np.zeros_like(k) for k in pars];v=[k.copy() for k in m]
    for step in range(1,701):
        q=rng.integers(len(x),size=768);z=x[q];hidden=np.tanh(z@w+b);pred=1/(1+np.exp(-np.clip(hidden@out+bias,-20,20)))
        err=(pred-y[q])/len(q);dh=err[:,None]*out*(1-hidden**2)
        grads=[z.T@dh+.0002*w,dh.sum(0),hidden.T@err+.0002*out,err.sum()]
        for i,(p,g) in enumerate(zip(pars,grads)):
            m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
            p-=.006*m[i]/(1-.9**step)/(np.sqrt(v[i]/(1-.999**step))+1e-8)
    np.savez(D/"reliability.npz",w=w,b=b,out=out,bias=bias,mean=mean,std=std)
    save(D/"fit.json",dict(maps=len(cases),rosters=len(jobs),examples=len(x),positive=int(y.sum()),updates=700,weights=sha(D/"reliability.npz")))
if __name__=="__main__":train()
