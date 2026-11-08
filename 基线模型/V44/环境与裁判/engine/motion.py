"""Archived population replay implementation, isolated from training imports."""
from types import SimpleNamespace
import numpy as np
from long_environment import Config,Env as Drift,sample_water
from fall_risk import response
K=0.005962180452265221

def shape(h,v):
    return h/(h+.05)*response(h,v)+.02*(h/(h+1.))**4

def simulate(field,ids,fall,seed,rep,traps):
    count=len(ids)
    yy,xx=np.unravel_index(ids,(100,100))
    pos=np.column_stack([xx,yy])*10.+5
    cfg=Config(duration=2400,window=1320,people=count,sensor=20.,trigger_depth=2.,trigger_speed=1.)
    salt=int(np.random.SeedSequence([7812,seed,rep]).generate_state(1)[0])
    motion=SimpleNamespace(cfg=cfg,water=field,person=pos.copy(),traps=traps,status=np.zeros(count,int),
        stuck=np.zeros(count,bool),stop_kind=np.zeros(count,int),refloat_counts=np.zeros(count,int),
        scenario={"seed":salt,"events":[{} for _ in range(count)]},refloat_enabled=True,events=[])
    motion.refloat_draw=lambda j,now:Drift.refloat_draw(motion,j,now)
    points=np.empty((2401,count,2));avail=np.empty((2401,count),bool);velocity=np.zeros((481,count,2))
    for t in range(2401):
        motion.status[(motion.status==0)&(fall<=t)]=1
        points[t]=motion.person;avail[t]=motion.status!=4
        if t%5==0:
            active=(motion.status==1)&~motion.stuck
            if active.any():
                values=sample_water(field,t,motion.person[active])
                velocity[t//5,active]=np.where(values[:,:1]>=.01,values[:,1:],0.)
        if t<2400:Drift.move_people(motion,t)
    assert np.isfinite(points).all() and np.isfinite(velocity).all() and avail[0].all()
    return dict(points=points,available=avail,velocity=velocity,fall=fall)
