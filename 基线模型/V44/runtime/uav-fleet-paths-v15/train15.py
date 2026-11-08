from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
from paths15 import *
from learn14 import material,batch
from service14 import short_service
from support import runtime,s,quota,read,save,sha
def cases():
    out=[]
    for root,ordinary,pressure in [(s.ROOT/"uav-priority-confirm-v10",97151001,97161001),(s.ROOT/"uav-short-deadline-learning-v11",98751001,98761001)]:
        out += [(str(root),ordinary+i,"ordinary",i) for i in range(18)]
        out += [(str(root),pressure+i,f,i) for i in range(6) for f in ("continuous","rewet")]
    return out

def job(args):
    c,rep=args;path=OUT/"features"/f"{c[1]}-{c[2]}-{rep}.npz"
    if path.exists():return
    f,p=material(c,rep);cls,_=runtime(1,"planning");env=cls(f,p,(64,720,2400))
    a=Switcher("persist" if rep==0 else "constant")
    belief=Cohorts(8);belief.personal_window_seconds=720
    xs=[];ys=[];valids=[];seenlabels=0;geometrychecks=0
    while env.t<2400:
        o=env.observation();belief.observe(o);moves,claims=a.act(o)
        if o.t%60==0 and o.t+70<=2400:
            free=np.flatnonzero((a.track<0)&(claims<0))
            if len(free):
                point,valid,pos=path_inputs(o,free,belief.mass(o.t,1.))
                label=np.zeros((*valid.shape,64,2),dtype="f4")
                for step in range(6):
                    t=o.t+5*(step+1)
                    idx,reverse=np.unique(INDEX[:,step],return_inverse=True)
                    pts=pos[:,idx];hd=HEADS[idx]
                    who=np.flatnonzero((o.seen_at<0)&p["available"][t]&(t+10<=env.deadline))
                    delta=p["points"][t,who][None,None,:,:]-pts[:,:,None,:]
                    visible=((delta*delta).sum(-1)<=400+1e-9)&((delta*hd[None,:,None,:]).sum(-1)>=-1e-9)
                    for ai,k,jj in zip(*np.nonzero(visible)):
                        route_ids=np.flatnonzero((reverse==k)&valid[ai])
                        if not len(route_ids):continue
                        j=who[jj];geometrychecks+=1
                        quality=short_service(p,j,t,pts[ai,k],hd[k],int(env.deadline[j]))
                        label[ai,route_ids,j,0]=np.maximum(label[ai,route_ids,j,0],np.exp(-step/10))
                        label[ai,route_ids,j,1]=np.maximum(label[ai,route_ids,j,1],quality*np.exp(-step/10))
                xs.append(compress(point).astype("f2"));ys.append(label.sum(2));valids.append(valid);seenlabels+=len(free)
        env.step(moves,claims)
    np.savez_compressed(path,x=np.concatenate(xs),y=np.concatenate(ys),valid=np.concatenate(valids),n=geometrychecks)
def combine(held):
    xx=[];yy=[];ll=[];queries=0
    for c in cases():
        if (c[3]>=(14 if c[2]=="ordinary" else 4))!=held:continue
        for i in (0,1):
            with np.load(OUT/"features"/f"{c[1]}-{c[2]}-{i}.npz") as z:
                xx.append(z["x"]);yy.append(z["y"]);ll.append(z["valid"]);queries+=int(z["n"])
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(ll),queries
def train():
    (OUT/"features").mkdir(exist_ok=True)
    batch(job,[(c,i) for c in cases() for i in (0,1)],5)
    x,labels,valid,q=combine(False)
    y=labels[:,:,1]+.2*labels[:,:,0]
    # Train a shared nonlinear path utility using complete-route, deduplicated opportunities.
    pos=np.flatnonzero(y.sum(1)>0)
    samples=x[::4].astype("f4")
    mean=samples.mean((0,1),dtype=float);std=np.maximum(.06,samples.std((0,1),dtype=float))
    rng=np.random.default_rng(150691);w=rng.normal(0,.11,(len(mean),40));b=np.zeros(40);out=rng.normal(0,.1,40)
    pars=[w,b,out];mom=[np.zeros_like(k) for k in pars];var=[np.zeros_like(k) for k in pars]
    for step in range(1,1101):
        if step%200==0:quota()
        ii=rng.choice(pos,72);z=np.clip((x[ii]-mean)/std,-4,4)
        hidden=np.tanh(z@w+b);score=np.where(valid[ii],hidden@out,-1.e6)
        prob=np.exp(score-score.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        tar=y[ii]/y[ii].sum(1,keepdims=True);error=(prob-tar)/len(ii)
        dh=error[:,:,None]*out*(1-hidden**2)
        grad=[z.reshape(-1,len(mean)).T@dh.reshape(-1,40)+.001*w,dh.sum((0,1)),np.einsum("bkh,bk->h",hidden,error)+.001*out]
        for i,(p,g) in enumerate(zip(pars,grad)):
            if not np.isfinite(g).all():raise ValueError(f"Nonfinite route gradient at update {step}")
            mom[i]=.9*mom[i]+.1*g;var[i]=.999*var[i]+.001*g*g
            p-=.003*mom[i]/(1-.9**step)/(np.sqrt(var[i]/(1-.999**step))+1e-8)
    assert all(np.isfinite(k).all() for k in pars)
    np.savez(OUT/"routes.npz",mean=mean,std=std,w=w,b=b,out=out)
    audit_fit(len(x),int(valid.sum()),len(pos),q)
def audit_fit(states,routes,positive,q):
    with np.load(OUT/"routes.npz") as model:
        mean,std,w,b,out=[model[k] for k in ("mean","std","w","b","out")]
        assert all(np.isfinite(k).all() for k in (mean,std,w,b,out))
    tx,ty,legal,tq=combine(True);vals=[];uniform=[]
    for a in range(0,len(tx),128):
        z=np.clip((tx[a:a+128]-mean)/std,-4,4);scores=np.tanh(z@w+b)@out
        mask=legal[a:a+128];reward=ty[a:a+128,:,1]
        have=mask.any(1)
        vals.extend(reward[np.arange(len(z)),np.where(mask,scores,-1.e6).argmax(1)][have])
        uniform.extend((np.sum(np.where(mask,reward,0),1)/np.maximum(1,mask.sum(1)))[have])
    save(OUT/"fit.json",dict(fit_rosters=88,held_rosters=32,train_aircraft_states=states,train_routes=routes,states_with_positive_target=positive,queried_encounters=q,held_queried=tq,
        held_selected=float(np.mean(vals)),held_uniform=float(np.mean(uniform)),held_states=len(vals),updates=1100,weights=sha(OUT/"routes.npz")))
    print("FIT",read(OUT/"fit.json"),flush=True)
if __name__=="__main__":train()
