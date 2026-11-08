"""Training-only public-state demonstrations; never imported by inference."""
import importlib.util
import itertools
import sys
from pathlib import Path
import numpy as np
import policy_core as p
import replay

MAPPO_SEARCH=p.HERE.parent.parent/'mappo-search-v1-20261107'
MAPPO_RESCUE=p.HERE.parent.parent/'mappo-rescue-v1-20261107'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def load_search_teacher():
    # carpet.py imports `core`; bind its documented search core only while the
    # teacher module loads. This never changes files or deploys the teacher.
    previous=sys.modules.get('core');sys.modules['core']=p.geometry.search
    try:return load('dqn_training_search_teacher',MAPPO_SEARCH/'carpet.py')
    finally:
        if previous is None:sys.modules.pop('core',None)
        else:sys.modules['core']=previous


def load_micro_teacher():
    previous=sys.modules.get('core');sys.modules['core']=p.geometry
    try:
        teacher=load('teacher_multistep',MAPPO_RESCUE/'teacher_multistep.py')
        sys.modules['teacher_multistep']=teacher
        return load('dqn_training_micro_teacher',MAPPO_RESCUE/'micro.py')
    finally:
        if previous is None:sys.modules.pop('core',None)
        else:sys.modules['core']=previous


def legal_search_action(module,teacher,o):
    """Previous carpet teacher, with the exact no-target boundary mask.

    This is a training-only teacher. The joint yielding uses public UAV
    positions and straight-line intended trajectories; it is not deployed.
    """
    covered=np.asarray(o.last)>=0;obs=np.zeros((4,740),float)
    obs[:,:100]=covered.reshape(10,10,10,10).mean(axis=(1,3)).ravel()
    team=np.zeros((4,4));team[:,:2]=o.pos/1000.;team[:,2:]=o.heading
    obs[:,400:416]=team.ravel();obs[:,416]=o.t/5/300
    padded=np.pad(covered,5,constant_values=True)
    exact=p.geometry.search.geometry.action_mask(o)
    referee_legal=exact[:,np.arange(9)*65].astype(bool)
    old_legal=referee_legal[:,module.MAP]
    for i,pos in enumerate(o.pos):
        row,col=np.clip(np.floor(pos[::-1]/10).astype(int),0,99)
        obs[i,601:722]=padded[row:row+11,col:col+11].ravel()
    desired=module.MAP[teacher.p.act(obs,old_legal)]
    if (teacher.p.seen&~covered).any():raise AssertionError('Teacher invented coverage')
    if not referee_legal[np.arange(4),desired].all():raise AssertionError('Teacher ignored boundary mask')
    return resolve_joint(o,desired,referee_legal),desired


def resolve_joint(o,desired,referee_legal):
    positions=np.asarray(o.pos);velocity=p.geometry.search.geometry.MOVES*10.
    conflict=False
    for i in range(4):
      for j in range(i+1,4):
        offset=positions[i]-positions[j];delta=(velocity[desired[i]]-velocity[desired[j]])*5.
        fraction=np.clip(-float(offset@delta)/max(float(delta@delta),1e-18),0.,1.)
        if np.linalg.norm(offset+fraction*delta)<5.-1e-8:conflict=True
    if not conflict:return desired.copy()
    combinations=np.array(list(itertools.product(range(9),repeat=4)),dtype=np.int64)
    valid=np.ones(len(combinations),bool)
    for i in range(4):valid &= referee_legal[i,combinations[:,i]]
    for i in range(4):
      for j in range(i+1,4):
        offset=positions[i]-positions[j]
        delta=(velocity[:,None,:]-velocity[None,:,:])*5.
        fraction=np.clip(-(delta*offset).sum(-1)/np.maximum((delta*delta).sum(-1),1e-18),0.,1.)
        distance=np.linalg.norm(offset+fraction[...,None]*delta,axis=-1)
        valid &= (distance>=5.-1e-8)[combinations[:,i],combinations[:,j]]
    candidates=combinations[valid]
    if not len(candidates):raise ValueError('No legal conflict-free search-teacher joint action')
    changed=candidates!=desired[None,:]
    deviation=((velocity[candidates]-velocity[desired][None])**2).sum((1,2))
    identity=(changed*np.array([4,3,2,1])).sum(1)
    order=np.lexsort((np.arange(len(candidates)),identity,deviation,changed.sum(1)))
    return candidates[order[0]]


class Frontier:
    """Training-only bounded frontier lookahead on public observed coverage.

    All hypothetical movements have full 50 m endpoints inside the square;
    every route step is checked, and joint first-step conflicts are resolved.
    """
    def __init__(self):self.recent=[[] for _ in range(4)]

    def act(self,o):
        from supply import sample_cells
        covered=np.asarray(o.last)>=0;flat=covered.ravel()
        legal=p.geometry.search.geometry.action_mask(o)[:,np.arange(9)*65].astype(bool)
        directions=np.asarray(p.geometry.search.geometry.MOVES,dtype=float)
        desired=np.empty(4,dtype=np.int64)
        for agent in range(4):
            pos=np.asarray(o.pos[agent],dtype=float);lo_x=0 if agent%2==0 else 50;lo_y=0 if agent<2 else 50
            xy=np.argwhere(~covered[lo_y:lo_y+50,lo_x:lo_x+50])+[lo_y,lo_x]
            if not len(xy):xy=np.argwhere(~covered)
            if not len(xy):desired[agent]=8;continue
            goals=xy[:,::-1]*10+5
            goal=goals[np.square(goals-pos).sum(axis=1).argmin()]
            recent=self.recent[agent][-8:]
            frontier=[(pos,(),0.)];visited={tuple(np.round(pos,3))};best=None
            for depth in range(1,6):
                following=[]
                for here,path,_ in frontier:
                    for action in range(8):
                        if not path and not legal[agent,action]:continue
                        end=here+directions[action]*50.
                        if np.any(end < -1e-9) or np.any(end > 1000.+1e-9):continue
                        key=tuple(np.round(end,3))
                        if key in visited:continue
                        visited.add(key);newpath=path+(action,)
                        cells=sample_cells(end[None],directions[action][None])
                        gain=int((~flat[cells]).sum());distance=float(np.linalg.norm(end-goal))
                        outside=max(0.,lo_x*10-end[0],end[0]-(lo_x+50)*10)+max(0.,lo_y*10-end[1],end[1]-(lo_y+50)*10)
                        repeated=any(np.linalg.norm(end-v)<.01 for v in recent)
                        if gain:
                            score=(depth,-gain,round(outside,3),repeated,round(distance,3),newpath)
                            if best is None or score<best:best=score
                        following.append((end,newpath,distance+outside*2+float(repeated)*25))
                if best is not None:break
                frontier=sorted(following,key=lambda x:(x[2],x[1]))[:32]
                if not frontier:break
            if best is not None:desired[agent]=best[-1][0]
            else:
                valid=np.flatnonzero(legal[agent,:8]);desired[agent]=int(min(valid,key=lambda a:np.linalg.norm(pos+directions[a]*50-goal))) if len(valid) else 8
            self.recent[agent].append(pos.copy());self.recent[agent]=self.recent[agent][-8:]
        if not legal[np.arange(4),desired].all():raise AssertionError('Illegal public frontier action')
        return resolve_joint(o,desired,legal),desired


def search_examples(model,steps_per_map,device):
    if not 0<steps_per_map<=480:raise ValueError('Invalid training-map horizon')
    ds=p.geometry.search.Dataset();examples=[];records=[]
    for parent in p.geometry.search.TRAIN:
        b=ds.bridge(parent);teacher=Frontier();hidden=model.init_hidden(4,device)
        previous=np.full(4,-1,np.int64);invalid=0;steps=0;yielded=0
        while not b.done and steps<steps_per_map:
            o=b.observation();state,hidden=replay.capture(model,o,previous,hidden)
            moves,desired=teacher.act(o);packed=moves*65
            mask=np.unpackbits(state['mask_bits'],axis=1,count=585).astype(bool)
            accepted=mask[np.arange(4),packed]
            if not accepted.all():raise AssertionError('Search teacher emitted an illegal label')
            applied=packed.copy();yielded+=int((moves!=desired).sum())
            examples.append((state,moves.copy(),accepted.copy()))
            invalid+=int((~accepted).sum())
            b.step(applied//65,applied%65-1,local_action_order=False)
            previous=applied;steps+=1
        records.append(dict(parent=parent,roster=0,joint_steps=steps,accepted_labels=4*steps-invalid,
                            rejected_illegal_teacher_labels=invalid,yielded_agent_actions=yielded,
                            teacher='public frontier strict full-step legal lookahead and joint collision-free yielding',metrics=b.env.metrics()))
        print('SEARCH_DEMO',parent,steps,'accepted',4*steps-invalid,'rejected',invalid,flush=True)
    return examples,records


def rescue_examples(model,cases,device):
    micro=load_micro_teacher();examples=[];records=[]
    for i in range(cases):
        kind=('near','aligned','unaligned')[i%3];seed=100000+i
        b,goal,agent=micro.fixture(seed,kind);hidden=model.init_hidden(4,device)
        previous=np.full(4,-1,np.int64);positive=0;abstain=0
        for _ in range(9):
            o=b.observation();state,hidden=replay.capture(model,o,previous,hidden)
            action=micro.teacher(o,agent)
            if action is None:
                examples.append((state,agent,-1));abstain+=1;break
            mask=np.unpackbits(state['mask_bits'],axis=1,count=585).astype(bool)
            if not mask[agent,action]:raise ValueError('Teacher label is not legal')
            examples.append((state,agent,int(action)));positive+=1
            actions=np.full(4,520,np.int64);actions[agent]=action
            b.step(actions//65,actions%65-1,local_action_order=False);previous=actions
            if b.env.supplied[goal]>=0:break
        records.append(dict(seed=seed,kind=kind,agent=agent,goal=goal,labels=positive,abstentions=abstain,
                            goal_delivered=bool(b.env.supplied[goal]>=0)))
    return examples,records
