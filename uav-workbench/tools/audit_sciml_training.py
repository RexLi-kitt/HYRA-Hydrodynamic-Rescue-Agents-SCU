"""Audit actual completed jobs, initialization, all replay shards and selection.

This verifies provenance/budget and training tensors. It does not substitute
for independent physical trajectory/forecast reconstruction or efficacy tests.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from hyra_learning.sciml_data import read,write,digest
from hyra_learning.sciml_study import verify
from hyra_learning.world_model import WorldLearner


def equal_state(a,b):
    assert set(a)==set(b)
    for name in a:torch.testing.assert_close(a[name],b[name],atol=0,rtol=0)


def audit_job(root,plan,method,seed):
    folder=root/'formal'/(method.replace('+','-')+f'-s{seed}')
    summary=read(folder/'summary.json');history=read(folder/'history.json');validation=read(folder/'validation.json')
    assert summary['status']=='completed' and summary['method']==method and summary['seed']==seed
    assert summary['episodes']==len(history)==2000 and summary['training_joint_steps']==120000
    assert summary['updates']==1993 and summary['validation_opportunities']==10 and summary['all_losses_finite']
    initial=folder/'initial-weights.pt'
    assert digest(initial)==read(folder/'initial-weights.json')['sha256']
    state=torch.load(initial,map_location='cpu',weights_only=True)
    # Reconstruct seeded fresh weights; no pilot parameters are accepted.
    torch.manual_seed(seed)
    fresh=WorldLearner(2,137,plan,'cpu',method,None,seed)
    equal_state(state['actor'],fresh.actor.state_dict());equal_state(state['mixer'],fresh.mixer.state_dict())
    if fresh.world is None:assert state['world'] is None
    else:equal_state(state['world'],fresh.world.state_dict())
    del fresh
    fit=[e for e in plan['inputs'] if e['split']=='fit']
    selection={e['id'] for e in plan['inputs'] if e['split']=='selection'}
    identity=sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    shard_records=[]
    assert len(list((folder/'episodes').glob('episode-*.pt')))==2000
    for i,row in enumerate(history):
        assert row['episode']==i+1 and row['event']==fit[i%9]['id'] and row['metrics']['steps']==60
        if i<7:assert row['update'] is None
        else:
            assert row['update']['updates']==i-6 and all(np.isfinite(v) for v in row['update'].values())
        path=folder/'episodes'/f'episode-{i+1:04d}.pt'
        shard=torch.load(path,map_location='cpu',weights_only=True)
        assert shard['case']==row['event'] and shard['metrics']==row['metrics'] and shard['plan_identity']==identity
        data=shard['data']
        assert tuple(data['images'].shape)==(61,2,16,7,7) and tuple(data['poses'].shape)==(61,2,6)
        assert tuple(data['actions'].shape)==(60,2) and tuple(data['states'].shape)==(61,137)
        assert data['actions'].min()>=0 and data['actions'].max()<4
        assert data['case_index'].shape==(61,) and torch.all(data['case_index']==i%9)
        for value in data.values():
            if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
        assert abs(float(data['rewards'].sum())-row['metrics']['return_'])<1e-6
        shard_records.append(dict(episode=i+1,path=str(path.relative_to(root)),sha256=digest(path),predecessor=shard['predecessor']))
    assert {v['episode'] for v in validation}==set(range(200,2001,200))
    milestone_hashes={}
    for val in validation:
        assert len(val['rows'])==3 and {v['case'] for v in val['rows']}==selection
        assert np.isclose(val['mean_return'],np.mean([v['return_'] for v in val['rows']]),rtol=0,atol=1e-12)
        episode=val['episode'];path=folder/f'checkpoint-{episode:04d}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['method']==method and payload['episode']==episode and payload['updates']==episode-7
        assert len(payload['history'])==episode and payload['validation'][-1]==val
        assert len(payload['replay_refs'])==128
        for ref in payload['replay_refs']:
            assert digest(folder/ref['path'])==ref['sha256']
        assert bool(payload['world'] is not None)==(method in ('Sur-QMIX+Data','Sur-QMIX+Physics'))
        milestone_hashes[episode]=digest(path)
    best=max(validation,key=lambda v:(v['mean_return'],-v['episode']))
    chosen=summary['selected']
    assert chosen['episode']==best['episode'] and chosen['mean_validation_return']==best['mean_return']
    assert chosen['sha256']==milestone_hashes[best['episode']]==digest(root/chosen['path'])
    return dict(method=method,seed=seed,episodes=2000,steps=120000,updates=1993,validation_opportunities=10,
                initialization='exact fresh seed reconstruction; no pilot warm start',initial_sha256=digest(initial),
                selected=chosen,shards=shard_records,milestones=milestone_hashes,status='passed')


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--completed-only',action='store_true')
    args=parser.parse_args();root=args.output.resolve();plan=read(root/'formal-plan.json');verify(plan)
    torch.set_num_threads(1)
    jobs=[]
    for method in plan['methods']:
        for seed in plan['seeds']:
            folder=root/'formal'/(method.replace('+','-')+f'-s{seed}')
            if not (folder/'summary.json').exists():
                if args.completed_only:continue
                raise ValueError('Formal job incomplete: '+str(folder))
            jobs.append(audit_job(root,plan,method,seed));print('audited',method,seed,flush=True)
    if not args.completed_only:
        assert len(jobs)==12
        freeze=read(root/'model-freeze.json')
        assert sorted(freeze['selected_models'],key=lambda r:(r['method'],r['seed']))==sorted([r['selected'] for r in jobs],key=lambda r:(r['method'],r['seed']))
    evidence=ROOT/'docs/evidence/20261025-sciml-development'
    name='training-audit-progress.json' if args.completed_only else 'training-audit-final.json'
    write(evidence/name,dict(status='passed_completed_jobs_only' if args.completed_only else 'passed',completed_jobs=len(jobs),
        formal_plan_sha256=digest(root/'formal-plan.json'),auditor_sha256=digest(Path(__file__)),jobs=jobs,
        limitation='Budget/provenance/tensor audit, not a substitute for independent raw-frame trajectory and forecast reconstruction'))


if __name__=='__main__':main()
