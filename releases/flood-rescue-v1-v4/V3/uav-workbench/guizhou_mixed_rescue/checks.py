from unittest.mock import patch
import numpy as np
from .study import decide,planner,sim,policy,OUT,maps
from dynamic_drop_rebuild.scenarios import contact
from flood_dqn.env import START


def checks():
    p=START.copy();m=planner.Memory()
    f=np.broadcast_to(np.zeros((1,3,1,1)),(61,3,100,100))
    m.observe(0,p,40,f,np.array([0]),p[:1])
    age,wet=m.features(0,False);dm=np.arange(100.)
    a,e,z=planner.plan(m,p,0,age,wet,dm,.5);end,_=policy.select('K1',m.stamp,p,0,z,e,40)
    for arm in ('G','T'):
        aa,ee,zz=decide(m,p,0,age,wet,dm,arm,dict(reports=[],corridor=[]))
        assert np.array_equal(aa,a) and np.array_equal(end,ee) and np.array_equal(zz,z)
    assert a[0]==0 and np.isclose(contact(np.array([[0.,0]]),np.array([[10.,0]]),np.array([[6.,0]]))[0],.2)
    b=np.zeros(40,int);xy=np.broadcast_to([950.,900.],(301,40,2)).copy();xy[:,0]=p[0]
    state=np.ones((301,40),np.uint8);state[30:,1]=4
    def fixed(m,p,t,*args):
        assign=np.full(4,-1)
        if t>=230:assign[0]=0
        return assign,p.copy(),np.full(4,-1)
    with patch.object(sim,'T',300),patch.object(sim,'fields',lambda _:f),patch.object(sim,'scenario',lambda *args:(b,xy,state,'fixture',0)),patch.object(sim,'plan',fixed):
        rr=sim.run(0,0,40)[0]['persons']
    assert rr[0]['saved']==240 and rr[1]['reason']=='left'
    maps.save_json(OUT/'checks.json',dict(passed=True,checks=['no-alarm K1 equivalence','shared discovery','obstacle intersection','departure','deadline equality']))

