"""Synthetic official-referee fixtures and an explicitly training-only teacher.

Fixture construction changes initial conditions/truth, never referee code.
The teacher receives only public observations and predicts constant velocity.
It labels only near hover or directly reachable endpoints; it abstains on hard
unreachable geometry, whose outcomes must be reported separately.
"""
import numpy as np
import core as c

s = c.search


def fixture(seed, kind):
    rng = np.random.default_rng(seed)
    goal, distractor = rng.choice(64, 2, replace=False)
    agent = int(rng.integers(4))
    pos = rng.uniform(220.,780.,2)
    angle = rng.uniform(0., 2*np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    if kind == 'near':
        gap = rng.uniform(.05, .9) * direction
    elif kind == 'aligned':
        move = rng.integers(8); unit = s.geometry.MOVES[move]
        gap = rng.uniform(2., 18.)*unit + rng.uniform(-.35,.35)*np.array([-unit[1],unit[0]])
    elif kind == 'unaligned':
        gap = rng.uniform(2., 19.)*direction
    else: raise ValueError(kind)
    direction = gap/np.linalg.norm(gap)
    angle = np.arctan2(direction[1], direction[0])
    coords = rng.uniform(40.,960.,(64,2))
    # Prevent accidental third-person visibility in the synthetic fixture.
    bad=np.linalg.norm(coords-pos,axis=1)<40.
    coords[bad]=np.array([20.,20.])
    coords[goal] = pos + gap
    # Distant enough to avoid ambiguous duplicate service labels, close enough
    # to exercise the target-scoring network when visible.
    coords[distractor] = pos + 14.*np.array([np.cos(angle+.25), np.sin(angle+.25)])
    velocities = np.zeros((64,2), dtype=np.float64)
    velocities[goal] = rng.uniform(-.10,.10,2)
    velocities[distractor] = rng.uniform(-.10,.10,2)
    seconds = np.arange(2401, dtype=np.float64)[:,None,None]
    points = coords[None,:,:] + seconds*velocities[None,:,:]
    water = np.broadcast_to(np.array([.2,0.,0.], np.float32)[None,:,None,None], (481,3,100,100))
    truth = dict(points=points, available=np.broadcast_to(np.ones((1,64),bool),(2401,64)),
                 velocity=np.broadcast_to(velocities[None,:,:],(481,64,2)), fall=np.full(64,2401,dtype=int))
    b = s.RefereeBridge(s.sdk.DeliveryMission(water,truth,(64,720,2400)))
    positions=[]
    while len(positions)<3:
        candidate=rng.uniform(40.,960.,2)
        if np.linalg.norm(candidate-pos)>100. and all(np.linalg.norm(candidate-q)>30. for q in positions):
            positions.append(candidate)
    team=np.empty((4,2));team[agent]=pos;team[np.arange(4)!=agent]=positions
    theta=rng.uniform(-np.pi,np.pi,4)
    headings=np.column_stack((np.cos(theta),np.sin(theta)))
    heading_angle=angle+rng.uniform(-1.35,1.35)
    headings[agent]=[np.cos(heading_angle),np.sin(heading_angle)]
    b.env.pos[:] = team; b.env.heading[:] = headings
    b.env.last[:] = -1; b.env.measured[:] = 0.; b.env.seen_at[:] = -1
    b.env.seen_xy[:] = 0.; b.env.seen_v[:] = 0.; b.env.visible[:] = False; b.env.active[:] = False
    b.env.sample()
    if not b.observation().visible[goal]: raise AssertionError('Fixture target not visible')
    return b, int(goal), agent


def teacher(o, agent=0):
    """Return one labelled packed action or None. Never called in inference."""
    mask = s.geometry.action_mask(o)
    speed, duration, _, _ = s.geometry.intent_geometry(o)
    eligible = np.flatnonzero(np.asarray(o.visible) & np.asarray(o.active) &
                              ((np.asarray(o.owner) == -1) | (np.asarray(o.owner) == agent)))
    choices = []
    for j in eligible:
        delta = o.seen_xy[j] - o.pos[agent]; distance = np.linalg.norm(delta)
        if distance <= 1.+1e-9 and mask[agent, 8*65+j+1]:
            choices.append((o.owner[j] != agent, 0, distance, int(j), 8*65+int(j)+1))
        elif distance <= 20.:
            for move in range(8):
                a = move*65+j+1
                if not mask[agent,a]: continue
                end = o.pos[agent]+speed[agent,a]*duration[agent,a]
                predicted = o.seen_xy[j]+5.*o.seen_v[j]
                error = np.linalg.norm(end-predicted)
                if error <= .9:
                    choices.append((o.owner[j] != agent, 1, float(error), int(j), int(a)))
    if choices:return min(choices)[-1]
    # Approximate training-only beam search on fresh public position/velocity.
    # No teacher output is included in the neural features or evaluator.
    import teacher_multistep
    fresh=(np.asarray(o.seen_at)>=0)&np.asarray(o.active)&((o.t-np.asarray(o.seen_at))<=30)
    fresh &= (np.asarray(o.owner)==-1)|(np.asarray(o.owner)==agent)
    ids=np.flatnonzero(fresh)
    ids=sorted(ids,key=lambda j:np.linalg.norm(o.seen_xy[j]-o.pos[agent]))[:2]
    plans=[]
    for j in ids:
        if np.linalg.norm(o.seen_xy[j]-o.pos[agent])>120.:continue
        path=teacher_multistep.plan(o,agent,int(j))
        if path and mask[agent,path[0]*65+j+1]:plans.append((len(path),int(j),path[0]*65+int(j)+1))
    return None if not plans else min(plans)[-1]


def collect(seed, kind, horizon=9):
    b, goal, agent = fixture(seed,kind); rows=[]
    for _ in range(horizon):
        o=b.observation(); action=teacher(o,agent)
        rows.append((c.encode(o)[agent],-1 if action is None else action))
        if action is None: break
        moves=np.full(4,8);targets=np.full(4,-1)
        moves[agent]=action//65;targets[agent]=action%65-1
        b.step(moves,targets,local_action_order=False)
        if b.env.supplied[goal] >= 0: break
    return rows, dict(kind=kind,seed=seed,goal=goal,agent=agent,labels=sum(a>=0 for _,a in rows),
                      abstentions=sum(a<0 for _,a in rows),goal_delivered=bool(b.env.supplied[goal]>=0))


def evaluate(policy, seeds, kinds=('near','aligned','unaligned'), mode='greedy', horizon=8):
    import torch
    rows=[]
    for seed in seeds:
      for kind in kinds:
        b,goal,agent=fixture(seed,kind); events=[]
        s.seed_all(850000+seed)
        for _ in range(horizon):
            o=b.observation()
            with torch.no_grad(): a,_,_=policy.act(c.encode(o), deterministic=mode=='greedy')
            a=a.cpu().numpy()
            # Fixture evaluates agent 0 in isolation; teammates stay stationary.
            moves=np.full(4,8);targets=np.full(4,-1)
            moves[agent]=a[agent,0];targets[agent]=a[agent,1]-1
            before=int(o.owner[goal]); b.step(moves,targets,local_action_order=False)
            events.append(dict(move=int(moves[agent]),target=int(targets[agent]),owner_before=before,
                               owner_after=int(b.observation().owner[goal]),delivered=bool(b.env.supplied[goal]>=0)))
            if b.env.supplied[goal]>=0: break
        rows.append(dict(seed=seed,kind=kind,goal=goal,agent=agent,first_five=bool(b.env.first_five[goal]),
                         delivered=bool(b.env.supplied[goal]>=0),metrics=b.env.metrics(),events=events))
    return rows
