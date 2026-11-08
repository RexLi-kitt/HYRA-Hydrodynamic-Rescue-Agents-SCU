"""Upper bound for legal discovery where a camera cannot observe two river reaches at once."""
import sys
from pathlib import Path
import numpy as np
from rendezvous44 import HERE,ROOT,s,sha,read,save,quota
sys.path.insert(0,str(ROOT/"uav-prefall-zones-v43"))
from common_view import make_envelope,check,R
from joint_capacity import TAU,LAST,trajectory,water_seconds

B=10;C=40;N=9933;K=481
CUT=np.r_[np.arange(150,1501,150),1900,2401]
G=len(CUT)
GROUP=np.searchsorted(CUT,TAU)
CENTRE=5+10*np.arange(100)
REACH=np.minimum(B-1,(CENTRE//100).astype(int))
EDGES=np.arange(C)*25.


def trap_counts(traps,axis,sign,distance):
    result=np.zeros((len(TAU),B))
    for centre in traps:
        for line in CENTRE[np.abs(CENTRE-centre[1-axis])<=4.0001]:
            w=np.sqrt(max(0.,4.0001**2-(line-centre[1-axis])**2))
            positions=CENTRE if sign>0 else (1000-CENTRE)
            q=centre[axis] if sign>0 else 1000-centre[axis]
            for b in range(B):
                eligible=positions[(positions//100==b)&(positions<=q+w+1e-5)]
                result[:,b]+=np.searchsorted(np.sort(q-w-eligible-1e-5),distance,side="right")
    return result


def span(displacement,left):
    lo=left-20.0001-displacement;hi=left+45.0001-displacement
    # Only grid centres 5,15,...,995 can carry a standing or untrapped origin.
    first=np.ceil((lo-5)/10).astype(int).clip(0,100)
    last=np.floor((hi-5)/10).astype(int).clip(-1,99)
    a=(first//10).clip(0,B-1);b=(last//10).clip(0,B-1)
    assert np.all(first>last) or np.max((b-a)[first<=last])<=1
    return a,b,first<=last


def training_arrays(offset,prob):
    times=np.arange(0,K,6);nk=len(times);q=offset[times]
    live=times[:,None]<=LAST[None]
    # The area of the last block is one time sample, not six.
    multiplicity=np.minimum(6,K-times)
    row=np.arange(nk)[:,None];single=np.empty((nk,C,G*B));both=np.empty_like(single)
    for i,left in enumerate(EDGES):
        a,b,ok=span(q,left);idx=(row*G+GROUP)*B+a
        for mask,array in ((live&ok&(a==b),single),(live&ok&(a!=b),both)):
            array[:,i]=np.bincount(idx[mask],weights=np.broadcast_to(prob,mask.shape)[mask],
                        minlength=nk*G*B).reshape(nk,G*B)
    return single,both,multiplicity


def optimise(offset,prob,free,trap,indiv):
    single,both,mult=training_arrays(offset,prob)
    weight=np.array([np.sum(prob[(GROUP==g)&(indiv<1)])/np.sum(prob[GROUP==g])
                     if np.sum(prob[GROUP==g]) else 0 for g in range(G)])[:,None]*np.ones((1,B))
    best=None
    for iteration in range(101):
        nxt=np.c_[weight[:,1:],weight[:,-1:]]
        two=np.maximum(weight,nxt)
        utility=single@weight.ravel()+both@two.ravel()
        value=float((free*(1-weight)+trap*weight).sum()+36/N*(utility.max(1)@mult))
        if iteration%25==0 and (best is None or value<best[0]):
            best=value,weight.copy(),iteration
        if iteration==100:
            break
        # A subgradient chooses any maximizing camera bin; average tied bins to avoid
        # dependence on which river bank is enumerated first.
        tied=np.isclose(utility,utility.max(1)[:,None],rtol=0,atol=1e-13)
        sel=mult[:,None]*tied/tied.sum(1)[:,None]
        one=(sel.ravel()@single.reshape(-1,G*B)).reshape(G,B)
        amb=(sel.ravel()@both.reshape(-1,G*B)).reshape(G,B)
        share=np.where(weight>nxt,1.,np.where(weight<nxt,0.,.5))
        grad=trap-free+36/N*(one+share*amb)
        grad[:,1:]+=36/N*(amb*(1-share))[:,:-1]
        weight=np.clip(weight-2./np.sqrt(iteration+1)*grad/np.maximum(free,.001),0,1)
    return best


def certify(offset,prob,weight,free,trap,frontier):
    top=np.zeros(K);whole=np.zeros(K);time=np.arange(K)[:,None]
    # Reversing a longitudinal coordinate is an isometry of the lattice and the
    # set of allowed orientations. Both offset and camera bin use the downstream axis.
    phase=np.floor(np.remainder(offset/10,1.)*R).astype(int)%R
    key=time*R+phase;live=time<=LAST
    gids=np.broadcast_to(GROUP,offset.shape)
    for left in EDGES:
        a,b,ok=span(offset,left)
        w=prob[None]*np.maximum(weight[gids,a],weight[gids,b])*(live&ok)
        hist=np.bincount(key.ravel(),weights=w.ravel(),minlength=K*R).reshape(K,R)
        upper=(hist@frontier.T).max(1)
        assert (upper<=9*hist.sum(1)+1e-10).all()
        top=np.maximum(top,upper);whole=np.maximum(whole,hist.sum(1)*9)
    base=float((free*(1-weight)+trap*weight).sum())
    return min(1,base+4/N*top.sum()+1e-8),min(1,base+4/N*whole.sum()+1e-8)


def calculate(c,record,frontier):
    meta=read(s.inputs(c)/"input.json")
    assert meta["water_sha"]==record["water"]==sha(meta["water"])
    field=s.water(meta["water"]);point,cum=water_seconds(field[:,:,0,0])
    axis=np.abs(point[:,1:]).max(0).argmax();sign=np.sign(point[:,axis+1].sum())
    assert np.abs(point[:,2-axis]).sum()<1e-5 and (sign*point[:,axis+1]>=-1e-10).all()
    shift,weak=trajectory(point,axis);assert weak==record["weak_seconds"]
    offset=sign*shift;assert offset.min()>-1e-7
    with np.load(s.inputs(c)/"eligible.npz") as z:
        ids=z["ids"];loc=10*(ids%100 if axis==0 else ids//100)+5
        bins=np.bincount(((loc if sign>0 else 1000-loc)//100).astype(int),minlength=B)
    assert bins.sum()==9996 and np.isclose(bins.min(),998)
    survival=np.exp(-cum);prob=np.r_[survival[:-1]-survival[1:],survival[-1]]
    exc=trap_counts(np.array(meta["traps"]),axis,sign,offset[LAST,np.arange(len(TAU))])
    trap=np.stack([(exc[GROUP==g]*prob[GROUP==g,None]).sum(0)/N for g in range(G)])
    free=np.bincount(GROUP,weights=prob,minlength=G)[:,None]*bins/N
    indiv=(4*9*(1+LAST)+exc.sum(1))/N
    val,weight,iteration=optimise(offset,prob,free,trap,indiv)
    bound,coarse=certify(offset,prob,weight,free,trap,frontier)
    use=min(bound,record["generic"])
    origin=sum(rep["shared_people"] for rep in record["reps"])
    count=sum(len(rep["exceptions"]) for rep in record["reps"])
    refined=count+(128-count)*use+2e-8
    return dict(seed=c[1],family=c[2],water_sha=meta["water_sha"],exceptional_people=count,
              previous_people=origin,refined_people=refined,weights=weight.tolist(),iteration=iteration,approximate_training_value=val,
              verified_coarse=coarse,verified_shared=bound,old_bound=record["generic"])


def main():
    prior=read(ROOT/"uav-prefall-zones-v43"/"joint-capacity-result.json")
    for p in (ROOT/"uav-prefall-zones-v43"/"common_view.py",ROOT/"uav-prefall-zones-v43"/"joint_capacity.py"):
        assert sha(p)==prior["source"][p.name]
    table,front=make_envelope();check(table,front);front=front.astype(float)
    cs=[c for c in s.cases("a")+s.cases("b") if c[2]!="ordinary"]
    cases=[];dest=HERE/"reach";dest.mkdir(exist_ok=True)
    code={p.name:sha(p) for p in (HERE/"river_bound.py",HERE/"reach-bound-protocol.md",ROOT/"uav-prefall-zones-v43"/"joint-capacity-result.json")}
    for i,c in enumerate(cs):
        quota();p=dest/f"{c[1]}-{c[2]}.json"
        if p.is_file():
            row=read(p);assert row["source"]==code
        else:
            old=next(r for r in prior["rows"] if (r["seed"],r["family"])==(c[1],c[2]))
            row=calculate(c,old,front);row["source"]=code;save(p,row)
        cases.append(row)
        print(i+1,len(cs),round(row["previous_people"]-row["refined_people"],3),flush=True)
    groups={}
    for f in ("all","continuous","rewet"):
        rows=[r for r in cases if f=="all" or r["family"]==f]
        groups[f]={k:sum(r[k] for r in rows) for k in ("refined_people","previous_people","exceptional_people")}
    save(HERE/"reach-bound.json",dict(groups=groups,pressure_required=1574,source=code,
                                      files={p.name:sha(p) for p in dest.glob("*.json")}))
    print(groups,flush=True)


if __name__=="__main__":
    main()
