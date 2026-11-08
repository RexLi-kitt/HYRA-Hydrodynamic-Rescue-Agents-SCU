"""Cross-check pass-by, boundary, short approach and downstream meetings against the real judge."""
import numpy as np
from rendezvous44 import HERE,read,save,sha,quota,runtime,DIR,solve,prepare,projection


def mission(p,h,x,v):
    cls,_=runtime(1,"planning")
    field=np.broadcast_to(np.array([2.4,*v])[None,:,None,None],(481,3,100,100))
    origin=np.column_stack([40.+np.arange(64),100.+np.arange(64)])
    pts=np.broadcast_to(origin[None],(2401,64,2)).copy()
    pts[:,0]=x+np.arange(2401)[:,None]*v
    vel=np.zeros((481,64,2));vel[:,0]=v
    avail=((pts>=0)&(pts<1000)).all(2)
    pop=dict(fall=np.zeros(64,int),points=pts,velocity=vel,available=avail)
    env=cls(field,pop,(64,720,2400))
    env.pos[0]=p;env.heading[0]=h;env.sample()
    return env


def demo(p,h,x,v):
    plans=[]
    for extended in (False,True):
        initial=mission(p,h,x,v)
        assert initial.visible[0]
        codes,fractional=solve(p,h,x,v,True,10,extended)
        if codes is None:
            plans.append(None);continue
        path=prepare(p,h,x,v,True,codes)
        assert path is not None
        for pos,heading,target,code in path:
            assert np.allclose(initial.pos[0],pos,rtol=0,atol=2e-7)
            assert np.allclose(initial.heading[0],heading,rtol=0,atol=2e-7)
            assert np.linalg.norm(initial.truth["points"][initial.t,0]-target)<1e-7
            moves=np.full(4,8);claims=np.full(4,-1);moves[0]=code%9
            if code>=9:
                assert initial.visible[0]
                claims[0]=0
            initial.step(moves,claims)
        assert initial.supplied[0]==len(path)*5 and initial.conflicts==0
        plans.append(dict(codes=codes,extra=fractional,delivery_s=int(initial.supplied[0])))
    assert plans[0] is None or plans[1] is not None and plans[1]["delivery_s"]<=plans[0]["delivery_s"]
    return plans


def main():
    quota();p=np.array([500.,500.]);h=np.array([1.,0.]);rng=np.random.default_rng(4406)
    cases=[(np.array([15.,0]),np.array([.7,0.])),(np.array([1.,0]),np.array([0.,0.]))]
    # Explicit diagonal and cardinal visible positions test target-relative stopping.
    for i in range(44):
        ang=rng.uniform(-1.5,1.5) if i%3 else (i%3-1)*np.pi/4
        cases.append((rng.uniform(1.01,19.)*np.array([np.cos(ang),np.sin(ang)]),
                      rng.uniform(.25,1.60)*DIR[i%8]))
    records=[dict(offset=q.tolist(),v=v.tolist(),plans=demo(p,h,p+q,v)) for q,v in cases]
    code_checks=0
    for pos,heading,q,v in ((p,h,np.array([15.,0]),np.array([.75,0])),
             (np.array([998.2,710.]),np.array([-1.,0.]),np.array([-2.2,.3]),np.array([.8,0.]))):
        for code in range(18):
            env=mission(pos,heading,pos+q,v)
            e,g=projection(pos,heading,pos+q,v,env.visible[0],code)
            m=np.full(4,8);m[0]=code%9;cl=np.full(4,-1)
            if code>=9:
                cl[0]=0
            env.step(m,cl)
            assert np.allclose(e,env.pos[0],atol=1e-8,rtol=0) and np.allclose(g,env.heading[0],atol=1e-8,rtol=0)
            code_checks+=1
    n=[sum(r["plans"][k] is not None for r in records) for k in (0,1)]
    assert min(n)>0
    save(HERE/"motion-checks.json",dict(cases=records,verified_meetings=n,action_checks=code_checks,
        program=sha(HERE/"verify44.py"),geometry=sha(HERE/"rendezvous44.py")))
    print("certified",n,"motion",code_checks)


if __name__=="__main__":
    main()
