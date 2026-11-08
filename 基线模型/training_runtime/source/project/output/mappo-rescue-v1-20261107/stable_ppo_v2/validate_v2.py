"""Disposable real-referee microfixture checks, not the registered training run."""
import json
import argparse
import traceback
import numpy as np
import torch
import train_v2 as v
import micro

s=v.s


class FixtureDataset:
    def bridge(self,parent):return micro.fixture(300003,'near')[0]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='')
    args=parser.parse_args()
    if args.tag and not args.tag.replace('_','').isalnum():parser.error('tag must contain only letters, digits and underscores')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    suffix='_'+args.tag if args.tag else ''
    out=v.HERE/f'validation{suffix}.json';artifacts=v.HERE/f'validation_artifacts{suffix}'
    if out.exists() or artifacts.exists():raise FileExistsError('Validation outputs already exist')
    artifacts.mkdir();result=dict(status='RUNNING',scope='CPU official-referee synthetic fixture; 12 disposable joint steps and three one-epoch PPO updates, not formal training',checks={})
    try:
        seed=77128;lock=v.source_lock(seed);ds=FixtureDataset();run=v.StableRun(ds,seed,device='cpu')
        run.run.bridge=ds.bridge(run.run.parent)
        p=run.run.policy
        before=v.actor_hash(p.actor);head_before=v.actor_hash(p.actor,rescue=True)
        frozen={n:q.detach().clone() for n,q in p.actor.named_parameters() if not n.startswith('rescue.')}
        assert all(not q.requires_grad for n,q in p.actor.named_parameters() if not n.startswith('rescue.'))
        assert all(q.requires_grad for n,q in p.actor.named_parameters() if n.startswith('rescue.'))
        assert all(g['lr']==.0001 for g in p.actor_optimizer.param_groups)
        assert all(g['lr']==.0005 for g in p.critic_optimizer.param_groups)
        data,trace=run.collect(4);x,a,old_lp,*_=data
        replay=ds.bridge(run.run.parent)
        for step in range(4):
            features=v.c.encode(replay.observation());ix=slice(4*step,4*(step+1))
            assert np.array_equal(x[ix],features),'stored observation or boundary mask differs'
            actions=a[ix];packed=actions[:,0]*65+actions[:,1]
            assert features[np.arange(4),-585+packed].all(),'stored action violates stored mask'
            with torch.no_grad():
                _,lp,_=p.evaluate_actions(features,features,None,None,actions,None,None,np.ones((4,1),np.float32))
            assert np.array_equal(old_lp[ix],lp.cpu().numpy()),'stored behavior/PPO log probabilities differ'
            replay.step(actions[:,0],actions[:,1]-1,local_action_order=False)
        result['checks']['all_16_stored_masks_and_behavior_ppo_logp_exact']='PASS'
        run.update(data,epochs=1)
        assert v.actor_hash(p.actor)==before
        assert all(torch.equal(q.detach(),frozen[n]) for n,q in p.actor.named_parameters() if n in frozen)
        assert v.actor_hash(p.actor,rescue=True)!=head_before,'Rescue head failed to update'
        assert any(q.grad is not None and bool(q.grad.abs().sum()>0) for q in p.actor.rescue.parameters())
        result['checks']['backbone_bitwise_fixed_rescue_head_learns']='PASS'
        saved=run.snapshot(lock);checkpoint=artifacts/'resume.pt';s.checkpoint_write(checkpoint,saved)
        next_data,next_trace=run.collect(4);run.update(next_data,epochs=1);expected=run.snapshot(lock)
        other=v.StableRun(ds,seed,device='cpu');other.restore(torch.load(checkpoint,map_location='cpu',weights_only=False))
        again_data,again_trace=other.collect(4);other.update(again_data,epochs=1);actual=other.snapshot(lock)
        assert s.same(next_trace,again_trace)
        assert all(np.array_equal(x,y) for x,y in zip(next_data,again_data))
        assert s.same(expected,actual),'resume state differs'
        result['checks']['disk_resume_model_optimizers_valuenorm_env_rng_exact']='PASS'
        assert actual['schema']=='MAPPO_RESCUE_V1' and actual['stable_ppo_v2']['parent_bc_sha256']==s.sha(v.bc_checkpoint(seed))
        assert actual['stable_ppo_v2']['frozen_actor_current_sha256']==before
        s.check_lock(lock)
        result.update(status='PASS',backbone_before_sha256=before,backbone_after_sha256=v.actor_hash(other.run.policy.actor),
            rescue_before_sha256=head_before,rescue_after_sha256=v.actor_hash(other.run.policy.actor,rescue=True),
            parent_bc_sha256=s.sha(v.bc_checkpoint(seed)),checkpoint_sha256=s.sha(checkpoint),source_lock=lock)
    except BaseException:
        result.update(status='FAIL',error=traceback.format_exc());s.json_write(out,result);raise
    s.json_write(out,result);print(json.dumps({k:v for k,v in result.items() if k!='source_lock'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
