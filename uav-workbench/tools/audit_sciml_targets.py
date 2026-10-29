"""Audit actual target-copy snapshots,optimizer settings and auxiliary RNG."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from hyra_learning.sciml_data import read,write,digest
from hyra_learning.sciml_study import verify


def same(a,b):
    assert set(a)==set(b)
    for k in a:torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)


def recovery(folder,episode):
    paths=list((folder/'recoveries').glob(f'recovery-{episode:04d}-*.pt'));assert paths
    data=torch.load(paths[0],map_location='cpu',weights_only=True)
    for path in paths[1:]:
        other=torch.load(path,map_location='cpu',weights_only=True)
        same(data['actor'],other['actor']);same(data['mixer'],other['mixer'])
    return data,dict(path=str(paths[0]),sha256=digest(paths[0]),episode=episode)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--completed-only',action='store_true')
    args=parser.parse_args();root=args.output.resolve();plan=read(root/'formal-plan.json');verify(plan);torch.set_num_threads(1)
    jobs=[];initials={}
    for method in plan['methods']:
        for seed in plan['seeds']:
            folder=root/'formal'/(method.replace('+','-')+f'-s{seed}')
            if not (folder/'summary.json').exists():
                if args.completed_only:continue
                raise ValueError('Incomplete job: '+folder.name)
            initial=torch.load(folder/'initial-weights.pt',map_location='cpu',weights_only=True);initials[method,seed]=initial
            first,_=recovery(folder,8);assert first['updates']==1
            same(initial['actor'],first['target_actor']);same(initial['mixer'],first['target_mixer'])
            copies={};references=[]
            for update in range(100,1901,100):
                actual,reference=recovery(folder,update+7);assert actual['updates']==update
                same(actual['actor'],actual['target_actor']);same(actual['mixer'],actual['target_mixer'])
                copies[update]=(actual['actor'],actual['mixer']);references.append(reference)
            aux_rng=np.random.default_rng(seed+419);expected_aux={}
            milestones={episode-7 for episode in range(200,2001,200)}
            for update in range(1,1994):
                if method in ('Sur-QMIX+Data','Sur-QMIX+Physics'):
                    for _ in range(8):aux_rng.choice(60,4,replace=False)
                if update in milestones:expected_aux[update]=aux_rng.bit_generator.state
            validation_states=[]
            for episode in range(200,2001,200):
                data=torch.load(folder/f'checkpoint-{episode:04d}.pt',map_location='cpu',weights_only=True)
                update=episode-7;last_copy=(update//100)*100
                same(data['target_actor'],copies[last_copy][0]);same(data['target_mixer'],copies[last_copy][1])
                assert data['auxiliary_rng']==expected_aux[update]
                groups=data['optimizer']['param_groups'];assert len(groups)==(2 if method in plan['methods'][2:] else 1)
                for group in groups:
                    assert group['lr']==plan['training']['lr'] and group['alpha']==plan['training']['alpha'] and group['eps']==plan['training']['eps']
                for value in data['optimizer']['state'].values():
                    assert int(value['step'])==update
                    for item in value.values():
                        if isinstance(item,torch.Tensor):assert torch.isfinite(item).all()
                validation_states.append(dict(episode=episode,last_target_copy_update=last_copy,auxiliary_rng_exact=True))
            history=read(folder/'history.json')
            for row in history[7:]:
                if method in plan['methods'][2:]:
                    assert row['update']['auxiliary_anchors']==32
                    assert row['update']['auxiliary_data_loss']>=0 and row['update']['auxiliary_physics_loss']>=0
                    if method=='Sur-QMIX+Data':assert row['update']['auxiliary_physics_loss']==0
            jobs.append(dict(method=method,seed=seed,status='passed',target_copy_updates=19,copy_references=references,
                actual_milestones=validation_states,optimizer='all current parameters at expected update count and frozen RMSprop settings'))
            print('audited targets/optimizer/anchor RNG',method,seed,flush=True)
    parity=[]
    for seed in plan['seeds']:
        peers=[m for m in plan['methods'] if (m,seed) in initials]
        for method in peers[1:]:
            same(initials[peers[0],seed]['actor'],initials[method,seed]['actor'])
            same(initials[peers[0],seed]['mixer'],initials[method,seed]['mixer'])
        if all((m,seed) in initials for m in plan['methods'][2:]):same(initials['Sur-QMIX+Data',seed]['world'],initials['Sur-QMIX+Physics',seed]['world'])
        parity.append(dict(seed=seed,available_methods=peers,initial_core_weights_identical=True))
    if not args.completed_only:assert len(jobs)==12
    write(ROOT/'docs/evidence/20261025-sciml-development'/('targets-audit-progress.json' if args.completed_only else 'targets-audit-final.json'),
        dict(status='passed_completed_jobs_only' if args.completed_only else 'passed',completed_jobs=len(jobs),jobs=jobs,
            seed_initial_weight_parity=parity,auditor_sha256=digest(Path(__file__)),formal_plan_sha256=digest(root/'formal-plan.json')))


if __name__=='__main__':main()
