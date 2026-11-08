"""Meaningful disposable checks before any registered comparison is launched."""
import copy
import json
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
import core as c
from train_round import Run,evaluate,audit_evaluations
from carpet import Carpet,MAP,old
ARTIFACTS=c.HERE/f'validation_artifacts_{time.time_ns()}'


def geometry_tests():
    sys.path.insert(0,str(c.ALIGN/'dqn_boundary_mask_v1'))
    existing=c.module('existing_boundary_acceptance',c.ALIGN/'dqn_boundary_mask_v1'/'validate_mask.py')
    return existing.geometry_tests(SimpleNamespace(sdk=c.sdk))


def public_and_local(ds):
    o=ds.bridge(41001).observation()
    allowed={'t','pos','heading','last','measured','seen_xy','seen_v','seen_at','visible','active','owner'}
    class Guard:
        def __getattr__(self,k):
            if k not in allowed:raise AssertionError('Hidden attribute '+k)
            return getattr(o,k)
    for arm in c.ARMS:assert np.array_equal(c.encode(o,arm),c.encode(Guard(),arm))
    patch=c.local_coverage(o);assert patch.shape==(4,452)
    assert np.all(patch[:,:225]<=patch[:,225:450])
    # Opposite local known cells in the same coarse block have equal old encoding.
    a=copy.deepcopy(o);a.pos[0]=[455,455];a.last[:]=-1;a.measured[:]=0
    a.last[45,46]=0;b=copy.deepcopy(a);b.last[45,46]=-1;b.last[45,44]=0
    assert np.array_equal(c.original_model.encode(a),c.original_model.encode(b))
    assert not np.array_equal(c.local_coverage(a)[0],c.local_coverage(b)[0])
    a.pos[0]=[1000,1000];patch=c.local_coverage(a)[0]
    assert patch[225+7*15+7]==0 and patch[7*15+7]==0
    return dict(public_only=True,local_shape=[4,452],coarse_alias_distinguished=True,outside_distinguished_from_unexplored=True)


def probabilities(ds):
    p,_,ck=c.initialize('mask',77126);q,_,_=c.initialize('local_mask',77126)
    o=ds.bridge(41001).observation();x=c.encode(o,'mask');y=c.encode(o,'local_mask')
    with torch.no_grad():
        d=p.joint(p.tensor(x));e=q.joint(q.tensor(y))
        assert torch.equal(d.logits,e.logits) and torch.equal(p.critic(p.tensor(x)),q.critic(q.tensor(y)))
        m,t=p.actor.distributions(p.tensor(x));prod=(m.probs[:,:,None]*t.probs[:,None,:]).reshape(4,585)
        mask=p.tensor(x[:,-585:])>.5;prod=prod*mask;prod/=prod.sum(-1,keepdim=True)
        assert torch.allclose(prod,d.probs,atol=2e-7,rtol=2e-6)
        assert bool((d.probs[~mask]==0).all())
        analytical=-(prod*prod.clamp_min(1e-30).log()).sum(-1)
        assert torch.allclose(analytical,d.entropy(),atol=1e-6)
        for greedy in (False,True):
            for _ in range(50):
                a,lp,val=p.act(x,greedy);packed=a[:,0]*65+a[:,1]
                assert bool(mask.gather(1,packed[:,None]).all())
                _,computed,_=p.evaluate_actions(x,x,None,None,a,None,None,np.ones((4,1)))
                assert torch.equal(lp,computed)
    # PPO must reject a stored action excluded by its exact mask.
    bad=np.zeros((4,2),int);bad[:,0]=1
    bad[0]=[1,0]
    try:p.evaluate_actions(x,x,None,None,bad,None,None,np.ones((4,1)))
    except ValueError:pass
    else:raise AssertionError('Illegal training action accepted')
    return dict(initial_zero_residual_exact=True,conditional_probabilities=True,entropy=True,behavior_ppo_logp_exact=True,illegal_actions_rejected=True)


def original_equivalence(ds):
    ref=Run(ds,'original',77126)
    _,p,tr=c.original_trainer.make(torch.device('cuda:0'))
    ck=torch.load(c.source_checkpoint(77126),map_location='cuda:0',weights_only=False)
    for n in ('actor','critic','actor_optimizer','critic_optimizer'):getattr(p,n).load_state_dict(ck[n])
    c.original_trainer.load_norm_state(tr,ck['value_norm']);ref.policy=p;ref.trainer=tr
    c.set_rng({k:ck[k+'_rng'] for k in ('python','numpy','torch','cuda')})
    data_ref,trace_ref=ref.collect(reference_original=True);ref.update(data_ref);state_ref=ref.snapshot()
    run=Run(ds,'original',77126);data,trace=run.collect();run.update(data);state=run.snapshot()
    assert c.same(trace_ref,trace)
    assert np.array_equal(data_ref[0],data[0][:,:998])
    assert all(np.array_equal(a,b) for a,b in zip(data_ref[1:],data[1:]))
    assert c.same(state_ref,state),'Original model/PPO/optimizer/environment/RNG mismatch'
    return dict(exact_rollout_steps=128,exact_five_epoch_update=True,actor_critic_optimizer_valuenorm_env_rng_equal=True)


def restore_test(ds,arm):
    run=Run(ds,arm,77126);data,_=run.collect();run.update(data)
    saved=run.snapshot();path=ARTIFACTS/f'{arm}.pt'
    c.checkpoint_write(path,saved)
    data,trace=run.collect();run.update(data);expected=run.snapshot()
    other=Run(ds,arm,77126);other.restore(torch.load(path,map_location='cuda:0',weights_only=False))
    rows=evaluate(other,[41004]);audit=audit_evaluations(ds,rows)
    # Audit itself does not use RNG; restore anyway to explicitly isolate it.
    other.restore(saved)
    again,again_trace=other.collect();other.update(again)
    assert c.same(trace,again_trace) and all(np.array_equal(a,b) for a,b in zip(data,again))
    assert c.same(expected,other.snapshot()),'Resume mismatch '+arm
    if arm=='local_mask':
        assert float(other.policy.actor.local[2].weight.abs().sum())>0
        assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in other.policy.actor.local.parameters())
    return dict(arm=arm,exact_resume=True,evaluation_isolated=True,official_evaluation_replay=audit,
                disposable_rollout_steps=384,disposable_ppo_updates=3,local_parameters_learn=arm=='local_mask')


def carpet_test(ds):
    assert np.array_equal(old.DIRECTIONS,c.geometry.MOVES[MAP[:8]])
    b=ds.bridge(41004);p=Carpet();actions=[];zero=0
    allowed={'t','pos','heading','last'}
    class Guard:
        def __getattr__(self,k):
            if k not in allowed:raise AssertionError('Carpet hidden attribute '+k)
            return getattr(b.observation(),k)
    while not b.done:
        pos=b.env.pos.copy();a,t=p.act(Guard());assert (t==-1).all()
        b.step(a,t,local_action_order=False);actions.append(a.tolist()+t.tolist())
        zero+=int((np.linalg.norm(b.env.pos-pos,axis=1)<=1e-8).sum())
    row=dict(parent=41004,metrics=b.env.metrics(),actions_and_targets=actions)
    replay=audit_evaluations(ds,[row])
    assert b.env.conflicts==0,'Public yielding failed'
    return dict(metrics=b.env.metrics(),zero_motion=zero,public_only=True,official_replay=replay,
                yield_steps=p.yield_steps,yield_changed_actions=p.yield_changed,
                source_sha256=c.sha(c.RULE),route_index=p.p.route_index,scan_complete=p.p.scan_complete)


def main():
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ARTIFACTS.mkdir(exist_ok=False)
    ds=c.Dataset();results=[];began=time.perf_counter()
    checks=[('official_sdk_regression',lambda:c.sdk.verify()),('boundary_geometry',geometry_tests),
            ('public_local_features',lambda:public_and_local(ds)),('masked_probabilities',lambda:probabilities(ds)),
            ('original_equivalence',lambda:original_equivalence(ds))]
    checks += [(f'restore_{arm}',lambda arm=arm:restore_test(ds,arm)) for arm in c.ARMS]
    checks += [('carpet_current_referee',lambda:carpet_test(ds))]
    try:
        for name,fn in checks:
            at=time.perf_counter();details=fn()
            results.append(dict(name=name,status='PASS',details=details,seconds=time.perf_counter()-at))
            print(name,'PASS',json.dumps(details,ensure_ascii=False),flush=True)
    except BaseException:
        c.json_write(c.HERE/'validation.json',dict(status='FAIL',checks=results,error=traceback.format_exc()))
        raise
    c.json_write(c.HERE/'validation.json',dict(status='PASS',checks=results,seconds=time.perf_counter()-began,
        source_lock=c.source_lock(),scope='Disposable engineering trajectories and updates; no registered training arm started'))


if __name__=='__main__':main()
