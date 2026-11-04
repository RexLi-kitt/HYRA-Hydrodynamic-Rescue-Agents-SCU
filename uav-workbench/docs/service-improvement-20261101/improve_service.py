"""Bounded MAAC intervention: preserved baseline, extra-RL control, imitation+RL.

The teacher uses observed target positions and accumulated sensed coverage only.
No target assignment/teacher action is injected at evaluation. This is an adapted
MAAC+BC/DAgger candidate, not the original algorithm or a six-method ranking.
"""
import argparse,copy,json,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from flood1km.env import Env,rule_action
from flood1km.learners import Learner,optimize,distribution
from flood1km.train import evaluate,seed_all,sources
from flood1km.data import sha

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def teacher_check(env):
    """Changing all water and hidden persons cannot change the teacher decision."""
    before=rule_action(env)
    other=copy.copy(env);other.people=env.people.copy()
    other.people[env.status==0]=[123.,456.]
    other.water=None
    assert np.array_equal(before,rule_action(other)), 'Teacher leaked hidden information'

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    protocol=dict(candidate='maac+BC+DAgger+32RL',control='maac+32RL',
        baseline=str(args.baseline/'maac/latest.pt'),seed=802,
        demo_episodes=48,demo_layout_start=820000,initial_supervised_epochs=12,
        dagger_rounds=2,dagger_episodes_per_round=12,dagger_epochs_per_round=4,
        dagger_layout_start=830000,batch=128,imitation_lr=.0003,
        rl_episodes_each=32,rl_layout_start=840000,rl_seed=803,
        evaluation='unchanged 2 validation parents x 4 layouts; final 50-scene test untouched',
        selection='MAAC chosen for prior highest coverage and zero rescue, not to rank six algorithms',
        teacher='existing rule_action; actually observed targets/coverage/public UAV positions only; no water or hidden people',
        action_mode='deterministic learner during DAgger; stochastic during RL; deterministic at evaluation',
        success='final candidate RCR > control and saved baseline, with coverage >= control; all results retained',
        fail='no rescue gain => do not promote or expand to six; nonfinite/hash mismatch => abort',
        compute='CPU2, one process, maximum 900 seconds including data and evaluations',
        cost_caveat='candidate has extra teacher trajectories and supervised updates; not equal total compute',
        changes='training procedure only; original environment, reward, observation, network and validation fixed',
        source='https://proceedings.mlr.press/v15/ross11a.html',
        script_hash=sha(Path(__file__)),source_hashes=sources(),
        baseline_hash=sha(args.baseline/'maac/latest.pt'),data_hash=sha(args.baseline/'data/dataset.json'))
    write(out/'protocol.json',protocol)
    def stage(name,**kw):
        elapsed=time.perf_counter()-started
        write(out/'progress.json',dict(stage=name,elapsed_s=elapsed,**kw))
        if elapsed>900: raise TimeoutError('Frozen 900s budget exceeded')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed_all(802)
    base=torch.load(args.baseline/'maac/latest.pt',map_location='cpu',weights_only=False)
    for name,h in base['identity']['source_hashes'].items():
        assert sources()[name]==h, 'Original source changed: '+name
    assert base['identity']['data_hash']==protocol['data_hash']
    assert sha(Path(__file__).parent/'protocol.json')==base['identity']['protocol_hash']
    cases={}
    for item in json.loads((args.baseline/'data/dataset.json').read_text())['cases']:
        p=Path(item['path']);assert sha(p)==item['sha256']
        cases.setdefault(item['split'],[]).append((p.name,np.load(p,mmap_mode='r')))
    assert not ({x[0] for x in cases['train']} & {x[0] for x in cases['validation']})
    def fresh():
        m=Learner('maac',base['learner']['dim']);m.restore(base['learner']);return m
    def rl(model,label):
        seed_all(803);replay=list(base['replay']);steps=base['steps'];records=[]
        for episode in range(32):
            stage(label,episode=episode)
            name,water=cases['train'][episode%len(cases['train'])]
            env=Env(water,840000+episode);obs=env.observe();done=False;returns=0.;loss={}
            while not done:
                mask=env.available();a,logs,val=model.act(obs,mask)
                nxt,reward,done,_=env.step(a);returns+=reward;steps+=1
                replay.append(dict(obs=obs.copy(),mask=mask.copy(),action=a.copy(),reward=reward,done=float(done),
                    next_obs=nxt.copy(),next_mask=env.available().copy(),logp=logs.copy(),value=val))
                if len(replay)>4096:replay.pop(0)
                if len(replay)>=1024 and steps%32==0:
                    for _ in range(4):loss=model.update_off_policy(random.sample(replay,64))
                obs=nxt
            record=dict(episode=episode+1,case=name,layout=840000+episode,return_sum=returns,loss=loss,**env.metrics())
            records.append(record)
            with (out/(label+'-episodes.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            if (episode+1)%8==0:
                torch.save(dict(learner=model.checkpoint(),episode=episode+1,protocol=protocol),out/(label+'-latest.pt'))
                print(label,episode+1,record['RCR'],flush=True)
        result=evaluate(model,cases['validation']);write(out/(label+'-validation.json'),result)
        return result
    stage('control')
    control=rl(fresh(),'control')
    seed_all(802);model=fresh();dataset=[];demo_rows=[]
    def collect(count,start,teacher_execute):
        for ep in range(count):
            stage('teacher_rollouts' if teacher_execute else 'dagger_rollouts',layout=start+ep)
            name,water=cases['train'][ep%len(cases['train'])]
            env=Env(water,start+ep);obs=env.observe();done=False;teacher_check(env)
            while not done:
                mask=env.available();label=rule_action(env)
                assert mask[np.arange(4),label].all()
                dataset.append((obs.copy(),mask.copy(),label.copy()))
                a=label if teacher_execute else model.act(obs,mask,True)[0]
                obs,_,done,_=env.step(a)
            demo_rows.append(dict(case=name,layout=start+ep,teacher_execute=teacher_execute,**env.metrics()))
        write(out/'teacher-rollouts.json',demo_rows)
    optimizers=[torch.optim.Adam(actor.parameters(),lr=.0003) for actor in model.actors]
    def imitate(epochs,round_id):
        obs=torch.from_numpy(np.stack([x[0] for x in dataset]));masks=torch.from_numpy(np.stack([x[1] for x in dataset]));labels=torch.from_numpy(np.stack([x[2] for x in dataset]))
        for epoch in range(epochs):
            stage('supervised',round=round_id,epoch=epoch,samples=len(dataset))
            perm=torch.randperm(len(dataset));total=0.;correct=0
            for batch in perm.split(128):
                for i,actor in enumerate(model.actors):
                    dist=distribution(actor(obs[batch,i]),masks[batch,i])
                    loss=-dist.log_prob(labels[batch,i]).mean()
                    total+=float(loss.detach())*len(batch)
                    correct+=int((dist.probs.argmax(-1)==labels[batch,i]).sum())
                    optimize(loss,optimizers[i],actor.parameters())
            model.check_finite()
            record=dict(round=round_id,epoch=epoch+1,samples=len(dataset),loss=total/(len(dataset)*4),teacher_action_agreement=correct/(len(dataset)*4))
            with (out/'imitation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('imitation',record,flush=True)
    collect(48,820000,True);imitate(12,0)
    for r in range(2):
        collect(12,830000+r*1000,False);imitate(4,r+1)
    stage('save_demonstrations')
    np.savez(out/'demonstrations.npz',obs=np.stack([x[0] for x in dataset]),mask=np.stack([x[1] for x in dataset]),action=np.stack([x[2] for x in dataset]))
    torch.save(dict(learner=model.checkpoint(),protocol=protocol),out/'imitation-final.pt')
    before=evaluate(model,cases['validation']);write(out/'imitation-validation.json',before)
    del dataset[:]
    candidate=rl(model,'candidate')
    original=json.loads((args.baseline/'maac/final-validation.json').read_text())
    report=dict(baseline=original['summary'],control=control['summary'],imitation_only=before['summary'],candidate=candidate['summary'],
        successful=candidate['summary']['RCR']>max(control['summary']['RCR'],original['summary']['RCR']) and candidate['summary']['coverage']>=control['summary']['coverage'],
        elapsed_s=time.perf_counter()-started,formal_test_executed=False,
        limitation='one seed, small reused validation set, adapted implementation, extra demonstration cost; no six-method ranking')
    write(out/'result.json',report)
    write(out/'hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('progress.json','hashes.json')})
    stage('completed',successful=report['successful'])
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args)
    except Exception:
        if args.output.exists():write(args.output/'failure.json',dict(traceback=traceback.format_exc()))
        raise
