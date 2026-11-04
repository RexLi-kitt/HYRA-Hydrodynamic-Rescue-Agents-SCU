"""Frozen six-method controller comparison, then monitored top-two continuation.

Run only with --execute. No change to the original learners, task or rewards.
"""
import argparse, gc, hashlib, json, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

EXPERIMENT = Path(os.environ.get('HYRA_UAV_EXPERIMENT', r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\experiments\flood1km-six-20261101'))
sys.path.insert(0, str(EXPERIMENT))
from flood1km.env import Env
from flood1km.learners import Learner, ALGORITHMS
import flood1km.learners as learning
from flood1km.train import aggregate, seed_all, rng_state, restore_rng, sources
from flood1km.data import sha
from service_commitment import ServiceCommitment
from coverage_controller import service_mask
from retry_guard import RetryGuard

LAYOUTS = (9001, 9002, 9003, 9004)
SAMPLE_SEEDS = (3101, 3102)


def write(path, obj):
    tmp = path.with_suffix('.partial-json')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


class LoggedLearner(Learner):
    def measured_update(self, rows, off_policy):
        records = []
        original = learning.optimize
        def measured(loss, optimizer, parameters):
            value = float(loss.detach())
            norm = original(loss, optimizer, parameters)
            records.append(dict(loss=value, gradient_norm=norm))
            return norm
        learning.optimize = measured
        try:
            metrics = self.update_off_policy(rows) if off_policy else self.update_on_policy(rows)
        finally:
            learning.optimize = original
        metrics['optimizer_steps'] = records
        return metrics


def decide(model, service, guard, obs, boundary, deterministic):
    forced, detail = service.select(obs, boundary, np.full(4, 8, dtype=np.int64))
    mask = service_mask(boundary, forced, detail['phases'])
    gd = {}
    if guard is not None:
        mask, gd = guard.mask(obs, mask, detail['phases'])
    action, logs, value = model.act(obs, mask, deterministic)
    controlled = np.array([p != 'learned' for p in detail['phases']])
    assert np.array_equal(action[controlled], forced[controlled])
    assert np.all(np.abs(logs[controlled]) < 1e-6)
    assert mask[np.arange(4), action].all()
    if guard is not None:
        guard.remember(action)
    return action, mask, logs, value, detail, gd


def rollout(model, water, layout, guard_enabled, deterministic=False, consume=None, trace=None):
    e = Env(water, layout)
    obs = e.observe()
    service = ServiceCommitment('commit')
    guard = RetryGuard() if guard_enabled else None
    previous = None
    done = False
    reward_sum = 0.
    nfree = nhover = nforced = 0
    surprise_sum = 0.
    while not done:
        action, mask, logs, value, detail, gd = decide(model, service, guard, obs, e.available(), deterministic)
        if previous is not None and consume is not None:
            previous['next_mask'] = mask.copy()
            previous['next_action'] = action.copy()
            consume(previous)
        free = mask.sum(1) > 1
        nfree += int(free.sum())
        nhover += int(((action == 8) & free).sum())
        surprise_sum -= float(logs[free].sum())
        nforced += sum(p != 'learned' for p in detail['phases'])
        nxt, reward, done, info = e.step(action)
        reward_sum += reward
        if trace is not None:
            trace.write(json.dumps(dict(step=e.t, actions=action.tolist(), service=detail, guard=gd,
                                       info={k:v.item() if isinstance(v,np.generic) else v for k,v in info.items()}))+'\n')
        if consume is not None:
            previous = dict(obs=obs.copy(), mask=mask.copy(), action=action.copy(), reward=reward,
                            done=float(done), next_obs=nxt.copy(), next_mask=e.available().copy(),
                            next_action=np.full(4,8,dtype=np.int64), logp=logs.copy(), value=value)
        obs = nxt
    if consume is not None:
        consume(previous)
    ids = [x['target'] for x in e.events if x['type']=='rescued']
    assert len(ids)==len(set(ids))==e.metrics()['rescued']
    return dict(**e.metrics(), return_sum=reward_sum, free_decisions=nfree,
                free_hover_fraction=nhover/max(1,nfree), sampled_action_surprise=surprise_sum/max(1,nfree),
                forced_agent_steps=nforced, guard_agent_steps=guard.constrained_steps if guard else 0)


def evaluate(model, cases, out, label, guard, deterministic, seeds=None, traces=False):
    saved = rng_state()
    rows = []
    seeds = (0,) if deterministic else (SAMPLE_SEEDS if seeds is None else seeds)
    try:
        for seed in seeds:
            for name, water in cases:
                for layout in LAYOUTS:
                    seed_all(seed+layout)
                    trace = (out/f'{label}-{seed}-{name}-{layout}.jsonl').open('w',encoding='utf-8') if traces else None
                    try:
                        record = rollout(model, water, layout, guard, deterministic, trace=trace)
                    finally:
                        if trace: trace.close()
                    rows.append(dict(case=name, layout=layout, action_seed=seed, **record))
        # Equal repeats for every scene; the mean is scene-equivalent. Report
        # seed-specific summaries as repeated evaluation, not training seeds.
        result = dict(episodes=rows, summary=aggregate(rows), guard=guard, deterministic=deterministic,
                      by_action_seed={str(s):aggregate([r for r in rows if r['action_seed']==s]) for s in seeds},
                      scope='Reused development validation, not sealed50 independent test')
        write(out/(label+'.json'), result)
        return result
    finally:
        restore_rng(saved)


def score(result):
    s=result['summary']
    worst=min(x['RCR'] for x in result['by_action_seed'].values())
    return (worst, s['RCR'], s['coverage'], -s['ACC'])


def guard_admissible(base, candidate):
    b,c=base['summary'],candidate['summary']
    return (c['RCR']>=b['RCR']-1e-12 and c['coverage']>=b['coverage']-1e-12
            and c['ACC']<=b['ACC']+1e-12
            and any(abs(c[k]-b[k])>1e-12 for k in ('RCR','coverage','ACC')))


def significant_improvement(candidate, best):
    # Primary: no loss in worst action-seed rescue rate. Avoid choosing an
    # apparent rescue gain that comes with a large coverage or conflict loss.
    c,b=candidate['summary'],best['summary']
    return (score(candidate)[0]>=score(best)[0]-1e-12 and
            c['RCR']>=b['RCR']-1e-12 and c['coverage']>=b['coverage']-.01-1e-12 and
            c['ACC']<=max(.02,b['ACC']+.005)+1e-12 and
            (c['RCR']>=b['RCR']+.00625-1e-12 or
             (abs(c['RCR']-b['RCR'])<1e-12 and c['coverage']>=b['coverage']+.005-1e-12)))


def train_selected(a, guard_enabled, base_result, base_checkpoint, cases, out, protocol_hash, target=1024):
    out.mkdir()
    ck = torch.load(base_checkpoint,map_location='cpu',weights_only=False)
    model = LoggedLearner(a,ck['learner']['dim']); model.restore(ck['learner'])
    # The same reward and learner are retained. Historical masks remain valid
    # observed constraints in off-policy replay; no reward relabeling occurs.
    replay = ck.get('replay',[])
    steps = ck['phase_steps']; restore_rng(ck['rng'])
    episode = ck['total_episode']; assert episode==256
    pending=[]; last_loss={}; stale=0; best_episode=256; best=base_result
    best_checkpoint=str(base_checkpoint); begin=time.perf_counter()
    identity=dict(algorithm=a, parent_checkpoint=str(base_checkpoint), parent_hash=sha(base_checkpoint),
                  protocol_hash=protocol_hash, guard=guard_enabled, target=target, reward='unchanged',
                  training_layout='920000+total_episode-65 (same sequence continued)',
                  replay='retained original valid transitions and masks, no reward relabeling')
    write(out/'identity.json',identity)
    del ck; gc.collect()
    def consume(row):
        nonlocal steps,last_loss
        assert row['mask'][np.arange(4),row['action']].all()
        assert row['next_mask'][np.arange(4),row['next_action']].all()
        steps+=1
        if a in ('hasac','maac'):
            replay.append(row)
            if len(replay)>4096:replay.pop(0)
            if len(replay)>=1024 and steps%32==0:
                for _ in range(4):last_loss=model.measured_update(random.sample(replay,64),True)
        else:pending.append(row)
    while episode<target:
        total=episode+1
        write(out/'status.json',dict(status='training',completed_episode=episode,target=target,
                                    best_episode=best_episode,stale_validations=stale,elapsed_s=time.perf_counter()-begin))
        name,water=cases['train'][(total-65)%len(cases['train'])]
        record=rollout(model,water,920000+total-65,guard_enabled,consume=consume)
        episode=total
        if a not in ('hasac','maac') and episode%2==0:
            last_loss=model.measured_update(pending,False);pending=[]
        with (out/'episodes.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(total_episode=episode,case=name,layout=920000+episode-65,
                                   phase_steps=steps,loss=last_loss,elapsed_s=time.perf_counter()-begin,**record))+'\n')
        if episode%32==0:
            assert not pending
            checkpoint=out/f'checkpoint-{episode:04d}.pt'
            assert not checkpoint.exists()
            temp=checkpoint.with_suffix('.partial')
            torch.save(dict(learner=model.checkpoint(),total_episode=episode,phase_steps=steps,
                            replay=replay,rng=rng_state(),identity=identity),temp)
            os.replace(temp,checkpoint)
            write(out/'latest.json',dict(path=str(checkpoint),sha256=sha(checkpoint),episode=episode))
            print(a,'trained',episode,'elapsed',round(time.perf_counter()-begin,1),flush=True)
            stop_requested=(out.parent/'STOP_AFTER_CHECKPOINT').exists()
            if stop_requested or time.perf_counter()-begin>3600:
                reason='manual_stop' if stop_requested else 'walltime_limit'
                break
        if episode%64==0:
            sample=evaluate(model,cases['validation'],out,f'validation-{episode:04d}-sample',guard_enabled,False)
            greedy=evaluate(model,cases['validation'],out,f'validation-{episode:04d}-greedy',guard_enabled,True)
            if significant_improvement(sample,best):
                best=sample;best_episode=episode;best_checkpoint=str(checkpoint);stale=0
            else:stale+=1
            write(out/'best.json',dict(episode=best_episode,path=best_checkpoint,validation=best,
                                      selected_on='predeclared stochastic validation criterion'))
            print(a,'validated',episode,json.dumps(sample['summary']),'stale',stale,flush=True)
            if stale>=3:
                reason='three_validations_without_improvement';break
    else: reason='target_reached'
    assert sha(base_checkpoint)==identity['parent_hash']
    result=dict(status='completed',reason=reason,final_episode=episode,target=target,best_episode=best_episode,
                best_checkpoint=best_checkpoint,best_validation=best,guard=guard_enabled,
                added_episodes=episode-256,elapsed_s=time.perf_counter()-begin,
                formal_test_executed=False)
    write(out/'status.json',result)
    return result


def run(args):
    if not args.execute:raise RuntimeError('Explicit --execute required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed_all(1001)
    begin=time.perf_counter()
    manifest=args.pilot/'data/dataset.json'
    items=json.loads(manifest.read_text(encoding='utf-8'))['cases'];cases={}
    for item in items:
        p=Path(item['path']);assert sha(p)==item['sha256']
        cases.setdefault(item['split'],[]).append((p.name,np.load(p,mmap_mode='r',allow_pickle=False)))
    parent_hashes={a:sha(args.baseline/a/'latest.pt') for a in ALGORITHMS}
    protocol=dict(date='2026-11-01',user_request='Improve first, prioritize stronger models for training',
        original_checkpoint_episodes=256,algorithms=list(ALGORITHMS),forbidden=['MAPPO','QMIX','VDN','IPPO','DQN','DDPG'],
        changes=['optional observed-motion RetryGuard: three failed moves, five-step cooldown; service always wins',
                 'same sampled execution for six methods; paired greedy results retained',
                 'independent checkpoints every32, validation every64, early stopping'],
        unchanged=['physics','water','service conditions','reward','learner architecture and optimization','observation rights'],
        validation=dict(parents=[x[0] for x in cases['validation']],layouts=list(LAYOUTS),action_seeds=list(SAMPLE_SEEDS),
            experiment='each256 model: unguarded sampled, guarded sampled, guarded greedy; reuse archived unguarded greedy',
            selection='guard accepted per algorithm only if sample RCR and coverage nondecrease, ACC nonincrease, one strict improvement; otherwise original controller retained',
            rank='minimum of two action-seed RCR means, then overall RCR, coverage, negative ACC; exact ties retain registry order',
            scope='development validation reuse; not independent confirmation or final50 test'),
        training=dict(number_selected=2,target=1024,additional_cap_per_selected=768,
            seed='restore checkpoint RNG; not independent training replication',checkpoint_every=32,evaluate_every=64,
            early_stop='three consecutive evaluations without predeclared significant improvement',
            improvement='worst seed RCR nondecrease; mean RCR nondecrease; coverage loss<=.01; ACC<=max(.02,bestACC+.005); RCR gain>=.00625 or equal RCR and coverage gain>=.005',
            hard_limit_seconds_per_model=3600,simultaneous_heavy_jobs=1,reward='unchanged',
            stop_file='STOP_AFTER_CHECKPOINT in this output directory'),
        source_hashes=sources(),new_code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        input_hashes={str(manifest):sha(manifest)},parent_hashes=parent_hashes)
    write(args.output/'protocol.json',protocol)
    protocol_hash=sha(args.output/'protocol.json')
    all_eval={};ranking=[]
    write(args.output/'queue-status.json',dict(status='comparing',pid=os.getpid()))
    for a in ALGORITHMS:
        d=args.output/('compare-'+a);d.mkdir()
        ck=torch.load(args.baseline/a/'latest.pt',map_location='cpu',weights_only=False)
        for k,h in ck['identity']['source_hashes'].items():assert sources()[k]==h
        m=LoggedLearner(a,ck['learner']['dim']);m.restore(ck['learner']);del ck;gc.collect()
        original=evaluate(m,cases['validation'],d,'sample-original',False,False)
        guarded=evaluate(m,cases['validation'],d,'sample-guarded',True,False)
        fixed=evaluate(m,cases['validation'],d,'greedy-guarded',True,True,traces=True)
        accepted=guard_admissible(original,guarded)
        chosen=guarded if accepted else original
        all_eval[a]=dict(original=original,guarded=guarded,greedy_guarded=fixed,guard_accepted=accepted)
        ranking.append(dict(algorithm=a,guard=accepted,score=list(score(chosen)),validation=chosen))
        print('comparison',a,'guard',accepted,'sample',json.dumps(chosen['summary']),flush=True)
        write(args.output/'comparison.json',all_eval)
        del m;gc.collect()
    ranking.sort(key=lambda x:x['score'],reverse=True)
    selected=ranking[:2]
    write(args.output/'selection.json',dict(ranking=ranking,selected=[x['algorithm'] for x in selected],
        reference='Old greedy comparison is preserved at baseline; new ranking uses fixed common sampled protocol.'))
    results={}
    for row in selected:
        a=row['algorithm']
        if (args.output/'STOP_AFTER_CHECKPOINT').exists():break
        write(args.output/'queue-status.json',dict(status='training',algorithm=a,pid=os.getpid(),
            selected=[x['algorithm'] for x in selected],completed=list(results)))
        results[a]=train_selected(a,row['guard'],row['validation'],args.baseline/a/'latest.pt',cases,
                                 args.output/a,protocol_hash)
        write(args.output/'results.json',results)
        gc.collect()
    for a,h in parent_hashes.items():assert sha(args.baseline/a/'latest.pt')==h
    write(args.output/'queue-status.json',dict(status='completed' if len(results)==2 else 'stopped',
        completed=list(results),elapsed_s=time.perf_counter()-begin,formal_test_executed=False))
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    try:sys.exit(run(args))
    except Exception:
        if args.output.exists():
            write(args.output/'failure.json',dict(traceback=traceback.format_exc()))
            write(args.output/'queue-status.json',dict(status='failed',pid=os.getpid()))
        raise
