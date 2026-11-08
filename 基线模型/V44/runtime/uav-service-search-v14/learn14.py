from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service14 import *
from concurrent.futures import ProcessPoolExecutor
def fit_cases():
    out=[]
    for root,ordinary,pressure in [(s.ROOT/"uav-priority-confirm-v10",97151001,97161001),(s.ROOT/"uav-short-deadline-learning-v11",98751001,98761001)]:
        out.extend([(str(root),ordinary+i,"ordinary",i) for i in range(24)]+[(str(root),pressure+i,f,i) for i in range(8) for f in ("continuous","rewet")])
    return out
def material(c,rep):
    root,seed,fam,_=c;root=Path(root)
    field=s.water(read(root/"scenes"/f"{seed}-{fam}"/"input.json")["water"])
    with np.load(root/"population"/f"{seed}-{fam}-{rep}.npz") as z:
        p={k:(z[k][:64] if k=="fall" else z[k][:,:64]) for k in ("points","fall","available","velocity")}
    return field,p
def sample(job):
    c,rep=job;path=ROOT/"samples"/f"{c[1]}-{c[2]}-{rep}.npz"
    if path.exists():return
    f,p=material(c,rep);cls,_=runtime(1,"planning");env=cls(f,p,(64,720,2400))
    a=Switcher("persist" if rep==0 else "constant");xs=[];ys=[];legal=[];checks=0
    while env.t<2400:
        o=env.observation();moves,claims=a.act(o)
        if o.t%15==0 and o.t+15<=2400:
            free=np.flatnonzero((claims<0)&(a.track<0))
            if len(free):
                positions,length=endpoints(o.pos[free])
                t=o.t+5;eligible=np.flatnonzero((o.seen_at<0)&p["available"][t]&(t+10<=env.deadline))
                target=p["points"][t,eligible]
                rel=target[None,None,:,:]-positions[:,:,None,:]
                ok=(np.square(rel).sum(-1)<=400+1e-9)&((rel*DIRECTIONS[None,:,None,:]).sum(-1)>=-1e-9)&(length>1)[:,:,None]
                labels=np.zeros((len(free),8,2))
                labels[...,0]=ok.any(-1)
                for aircraft,action,which in zip(*np.nonzero(ok)):
                    j=eligible[which];checks+=1
                    value=short_service(p,j,t,positions[aircraft,action],DIRECTIONS[action],int(env.deadline[j]))
                    labels[aircraft,action,1]=max(labels[aircraft,action,1],value)
                xs.append(covariates(o,positions,DIRECTIONS).astype("f4"));ys.append(labels.astype("f4"));legal.append(length>1)
        env.step(moves,claims)
    np.savez_compressed(path,x=np.concatenate(xs),y=np.concatenate(ys),legal=np.concatenate(legal),n=checks)
def batch(fn,jobs,workers=6):
    for start in range(0,len(jobs),workers*4):
        quota()
        with ProcessPoolExecutor(max_workers=workers) as ex:list(ex.map(fn,jobs[start:start+workers*4]))
        print(fn.__name__,min(start+workers*4,len(jobs)),len(jobs),flush=True)
def assemble(validation):
    xx=[];yy=[];ll=[];count=0
    for c in fit_cases():
        held=c[3]>=(20 if c[2]=="ordinary" else 6)
        if held!=validation:continue
        for i in range(2):
            with np.load(ROOT/"samples"/f"{c[1]}-{c[2]}-{i}.npz") as rec:
                xx.append(rec["x"]);yy.append(rec["y"]);ll.append(rec["legal"]);count+=int(rec["n"])
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(ll),count
def train():
    (ROOT/"samples").mkdir(exist_ok=True)
    batch(sample,[(c,i) for c in fit_cases() for i in range(2)])
    x,y,legal,events=assemble(False)
    mean=x.mean((0,1),dtype=float);std=np.maximum(.05,x.std((0,1),dtype=float))
    z=np.clip((x-mean)/std,-4,4).astype("f4");del x
    groups=[np.flatnonzero(y[...,k].sum(1)>0) for k in (0,1)]
    assert len(groups[1])>200 and np.all(y[...,1]<=y[...,0])
    rng=np.random.default_rng(71106)
    w=rng.normal(0,.18,(len(mean),48));b=np.zeros(48);out=rng.normal(0,.1,(48,2));bias=np.zeros(2)
    params=[w,b,out,bias];m=[np.zeros_like(a) for a in params];v=[a.copy() for a in m]
    for step in range(1,1801):
        if step%200==0:quota()
        grads=[np.zeros_like(a) for a in params]
        for k in (0,1):
            idx=rng.choice(groups[k],256)
            zz=z[idx];h=np.tanh(zz@w+b);score=h@out[:,k]+bias[k]
            score=np.where(legal[idx],score,-1e6);soft=np.exp(score-score.max(axis=1,keepdims=True));soft/=soft.sum(1,keepdims=True)
            label=y[idx,:,k];label=label/label.sum(1,keepdims=True)
            err=(soft-label)/256
            dh=err[:,:,None]*out[:,k]*(1-h*h)
            grads[0]+=zz.reshape(-1,len(mean)).T@dh.reshape(-1,48)
            grads[1]+=dh.sum((0,1));grads[2][:,k]+=np.einsum("bmh,bm->h",h,err)
        grads[0]+=.002*w;grads[2]+=.002*out
        for k,(a,g) in enumerate(zip(params,grads)):
            m[k]=.9*m[k]+.1*g;v[k]=.999*v[k]+.001*g*g
            a-=.004*m[k]/(1-.9**step)/(np.sqrt(v[k]/(1-.999**step))+1e-8)
    assert all(np.isfinite(a).all() for a in params)
    np.savez(ROOT/"service.npz",mean=mean,std=std,w=w,b=b,out=out,bias=bias)
    tx,ty,tl,ec=assemble(True)
    diagnostics={}
    for k in (0,1):
        good=ty[:,:,k].sum(1)>0
        inp=np.clip((tx[good]-mean)/std,-4,4)
        score=np.tanh(inp@w+b)@out[:,k]
        choices=np.where(tl[good],score,-1e6).argmax(1)
        labels=ty[good,:,k]
        diagnostics[str(k)]=dict(n=int(good.sum()),chosen_mean=float(labels[np.arange(len(labels)),choices].mean()),uniform_mean=float((labels.sum(1)/tl[good].sum(1)).mean()))
    save(ROOT/"training.json",dict(rosters=160,train_rosters=128,held_rosters=32,train_choices=len(y),valid_train_directions=int(legal.sum()),queried_targets=events,held_target_queries=ec,
        positive_choice_groups=[len(q) for q in groups],epochs_description="1800 minibatch updates; two listwise heads",updates=1800,held=diagnostics,weights_sha=sha(ROOT/"service.npz")))
    print("FIT",read(ROOT/"training.json"),flush=True)
if __name__=="__main__":train()
