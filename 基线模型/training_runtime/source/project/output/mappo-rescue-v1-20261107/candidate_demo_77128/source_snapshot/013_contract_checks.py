"""Executable boundaries that catch common independent-implementation discrepancies."""
from types import SimpleNamespace
import numpy as np
from judge import within_radius
from population import sector
from delivery import DeliveryMission
from supply import intentions
from long_environment import Config,Env as Drift,sample_water


def check():
    pos=np.array([[100.,90.],[90.,100.],[75.,100.],[800.,800.]])
    velocity=np.array([[0.,5.],[5.,0.],[4.,0.],[0.,0.]])
    track,stopped,count=intentions(pos,velocity,np.full(4,5.))
    assert count==1 and stopped.tolist()==[True,True,True,False] and np.all(track==pos)
    pos=np.array([[100.,100.],[105.,100.],[200.,200.],[400.,400.]])
    assert intentions(pos,np.zeros((4,2)),np.full(4,5.))[2]==0
    assert sector(np.array([[0.,20.],[-.01,0.],[20.,0.],[20.01,0.]]),np.zeros(2),np.array([1.,0.]),20.).tolist()==[True,False,True,False]
    a=np.zeros((6,2));b=a.copy();b[:,0]=1.
    assert within_radius(a,b)
    b[3,0]=1.00001
    assert not within_radius(a,b)

    def mission(v,offset,fall=0,window=10):
        xy=np.vstack(([500.,500.],np.array([500.,500.])+np.cumsum(v,axis=0)))
        slopes=np.vstack((v,v[-1]))[::5,None,:]
        pop=dict(points=xy[:,None,:],available=np.ones((21,1),bool),velocity=slopes,fall=np.array([fall]))
        field=np.zeros((5,3,100,100),dtype=np.float32)
        env=DeliveryMission(field,pop,(1,window,20))
        env.pos[0]=xy[0]-[offset,0.];env.heading[0]=[1.,0.];env.sample()
        return env
    hold=np.full(4,8);claimed=np.array([0,-1,-1,-1]);none=np.full(4,-1)
    flow=np.zeros((20,2));flow[:,0]=.4
    e=mission(flow,.3)
    e.step(hold,claimed);assert e.progress[0]==5 and abs(e.pos[0,0]-501.7)<1e-8
    e.step(hold,claimed);assert e.supplied[0]==10 and e.metrics()["FRT"]==2
    e.step(hold,none);assert e.supplied[0]==10
    e=mission(flow,.3,window=9)
    e.step(hold,claimed);e.step(hold,claimed);assert e.supplied[0]==-1
    e=mission(flow,.3,fall=21);e.step(hold,claimed);e.step(hold,none)
    assert e.progress[0]==0 and e.supplied[0]==-1
    turn=flow.copy();turn[2:]=[0,.4]
    e=mission(turn,.3);e.step(hold,claimed)
    assert e.progress[0]==0 and abs(e.pos[0,0]-501.7)<1e-8
    e=mission(np.zeros_like(flow),.5,fall=21)
    # Exercise owner reset using two distinct aircraft across decision boundaries.
    e.pos[1]=[950.,950.];e.sample();e.step(hold,claimed)
    e.pos[[0,1]]=e.pos[[1,0]];e.heading[1]=[1,0];e.sample()
    e.step(hold,np.array([-1,0,-1,-1]))
    assert e.progress[0]==5 and e.supplied[0]==-1

    water=np.zeros((2,3,100,100));water[1,0]=2.
    water[:,1]=np.arange(100)[None,None,:]
    val=sample_water(water,2.5,np.array([[10.,5.]]))[0]
    assert np.allclose(val,[1.,.5,0.])
    def mover(location=(500.,500.)):
        return SimpleNamespace(cfg=Config(trigger_depth=2.,trigger_speed=1.),water=np.zeros((3,3,100,100)),
                  person=np.array([location],float),status=np.ones(1,int),stuck=np.zeros(1,bool),
                  stop_kind=np.zeros(1,int),refloat_enabled=True,refloat_counts=np.zeros(1,int),
                  traps=np.empty((0,2)),events=[],refloat_draw=lambda j,t:1.)
    m=mover();Drift.move_people(m,0);assert m.stop_kind[0]==1
    m.water[:,0]=.5;m.water[:,1]=.4;Drift.move_people(m,1);assert m.stuck[0]
    m.water[:,0]=2.;Drift.move_people(m,2);assert not m.stuck[0] and m.person[0,0]>500
    m=mover();Drift.move_people(m,0);m.water[:,0]=.5;m.water[:,1]=.2
    m.refloat_draw=lambda j,t:0.;Drift.move_people(m,1);assert not m.stuck[0]
    m=mover();m.water[:,0]=2.;m.water[:,1]=2.;m.traps=np.array([[505.,500.]])
    Drift.move_people(m,0);assert m.stop_kind[0]==2 and np.isclose(m.person[0,0],501.)
    Drift.move_people(m,1);assert m.stuck[0] and np.isclose(m.person[0,0],501.)
    m=mover((999.5,500.));m.water[:,0]=2.;m.water[:,1]=1.;Drift.move_people(m,0)
    assert m.status[0]==4
