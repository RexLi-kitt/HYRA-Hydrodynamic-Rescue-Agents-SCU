from pathlib import Path
import argparse,copy
import numpy as np
import torch
from train_coordination import read,write,seed,model_for,METHODS,RECIPES,ResidualEnv,DispatchEnv

def run(out):
    cases=read(out/'scenarios.json');case=cases['train'][0]
    water=np.load(case['field']['path'],mmap_mode='r',allow_pickle=False)
    for group in cases['development'].values():
        e=ResidualEnv(water,group[0]['scenario']);assert e.observe().shape==(4,1240)
    s=copy.deepcopy(case['scenario']);e=DispatchEnv(water,s)
    # Fabricated observations solely for candidate-selection and permutation tests.
    e.t=200;e.belief[:6]=1;e.discovered[:6]=True;e.seen_at[:6]=200;e.first_seen[:6]=200
    e.seen_xy[:6]=np.array([[25,5],[35,5],[45,5],[55,5],[65,5],[75,5]])
    e.first_seen[5]=0
    offered=[x['person'] for x in e.choices()[2][0][:4] if x is not None]
    assert 5 in offered and 0 in offered and 1 in offered,offered
    order=np.random.default_rng(123).permutation(e.cfg.people);permuted=copy.deepcopy(s)
    permuted['events']=[s['events'][j] for j in order];other=DispatchEnv(water,permuted);other.t=e.t
    for attr in ('belief','discovered','seen_at','first_seen','seen_xy','seen_v','visible'):
        setattr(other,attr,getattr(e,attr)[order].copy())
    assert np.array_equal(e.observe(),other.observe()),'public-ID permutation changed input'
    a=e.choices();b=other.choices();assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert np.array_equal(a[3],b[3])
    obs=e.observe().copy();mask=e.choices()[0].copy()
    e.person[6:]=999.;e.fall[6:]+=1;e.deadline[6:]+=1
    assert np.array_equal(obs,e.observe()) and np.array_equal(mask,e.choices()[0]),'hidden roster leak'
    torch.set_num_threads(2);checks={}
    for recipe in RECIPES:
        for method in METHODS:
            seed(8301);model=model_for(method,recipe);env=ResidualEnv(water,s);ref=DispatchEnv(water,s);rows=[]
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            for _ in range(8):
                obs=env.observe();mask=env.choices()[0];action,lp,v=model.act(obs,mask,True);assert (action==8).all()
                nxt,r,done,_=env.step(action);ref.step(ref.choices()[1]);assert np.array_equal(env.pos,ref.pos)
                rows.append(dict(obs=obs,mask=mask,action=action,reward=r,done=float(done),next_obs=nxt,next_mask=env.choices()[0],logp=lp,value=v,next_action=action))
            if method=='maac':model.update_off_policy(rows)
            else:model.update_on_policy(rows)
            model.check_finite();assert model.updates>0
            assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items()),'no parameter update'
            restored=model_for(method,recipe);restored.restore(model.checkpoint())
            assert np.array_equal(model.act(nxt,env.choices()[0],True)[0],restored.act(nxt,env.choices()[0],True)[0])
            checks[recipe['id']+'-'+method]=dict(finite_update=True,parameters_changed=True,restore=True,initial_planning_trajectory=True)
    write(out/'readiness.json',dict(status='passed',candidate_far_urgent_included=True,observed_ID_permutation_invariant=True,
        hidden_roster_isolated=True,all_profile_input_dims=1240,methods=checks))
    print('Readiness passed: candidates, ID permutation, hidden truth,3 learners, checkpoint restore.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
