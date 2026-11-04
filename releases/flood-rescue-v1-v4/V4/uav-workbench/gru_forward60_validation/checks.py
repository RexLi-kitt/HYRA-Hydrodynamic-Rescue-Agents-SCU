from unittest.mock import patch
import numpy as np
from . import env
from .scenario import make
from .study import OUT,maps


def verify():
    mass=np.arange(1,101,dtype=float);flow=np.ones((100,2))*.7
    out,lost=env.transport(mass,flow)
    assert lost>0 and np.isclose(out.sum()+lost,mass.sum()) and np.array_equal(env.transport(mass,flow*0)[0],mass)
    p=env.START.copy();p[0]=[65,5]
    assert 0 in env.footprint(p)
    p[0,0]+=.001
    assert 0 not in env.footprint(p)
    base=make(31001,0);b=np.zeros(40,int)
    xy=np.full((1501,40,2),490.);xy[:,0]=env.START[0]
    st=np.ones((1501,40),np.uint8);observed=[]
    def fixed(m,p,t,x,d):
        a=np.full(4,-1)
        if 170<=t<180:a[0]=0
        observed.append(x.copy())
        return a,p.copy(),a.copy(),0
    with patch.object(env,'movement',fixed),patch.object(env,'N',181):
        row,persons,*_=env.run((base[0],b,xy,st,{}),'P0')
        assert persons[0]['saved']==180
        before=observed[0];observed.clear()
        changed=xy.copy();changed[:,1:]=700
        f=base[0].copy();f[1:]=0
        env.run((f,b,changed,st,{}),'P0')
        assert np.array_equal(before,observed[0])
    maps.save_json(OUT/'checks.json',dict(mass_conservation=True,radius_equality=True,deadline_equality=True,hidden_future_isolation=True))

