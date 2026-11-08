"""Unchanged V4 per-second drift/refloating mechanics with supplied first-fall times."""
from common import np, Config, Drift, sample_water, SimpleNamespace
def simulate(water,ids,fall,seed,rep,traps):
    yy,xx=np.unravel_index(ids,(100,100))
    pos=np.column_stack([xx,yy])*10.+5
    cfg=Config(duration=2400,window=1320,people=48,sensor=20.,trigger_depth=2.,trigger_speed=1.)
    salt=int(np.random.SeedSequence([7812,seed,rep]).generate_state(1)[0])
    motion=SimpleNamespace(cfg=cfg,water=water,person=pos.copy(),traps=traps,status=np.zeros(48,int),
             stuck=np.zeros(48,bool),stop_kind=np.zeros(48,int),refloat_counts=np.zeros(48,int),
             scenario={"seed":salt,"events":[{} for _ in range(48)]},refloat_enabled=True,events=[])
    motion.refloat_draw=lambda j,now: Drift.refloat_draw(motion,j,now)
    points=np.empty((2401,48,2));avail=np.empty((2401,48),bool);velocity=np.zeros((481,48,2))
    for t in range(2401):
        motion.status[(motion.status==0)&(fall<=t)]=1
        points[t]=motion.person;avail[t]=motion.status!=4
        if t%5==0:
            active=(motion.status==1)&~motion.stuck
            if active.any():
                value=sample_water(water,t,motion.person[active])
                velocity[t//5,active]=np.where(value[:,:1]>=.01,value[:,1:],0.)
        if t<2400:Drift.move_people(motion,t)
    assert np.isfinite(points).all() and np.isfinite(velocity).all() and avail[0].all()
    return dict(points=points,available=avail,velocity=velocity,fall=fall)
