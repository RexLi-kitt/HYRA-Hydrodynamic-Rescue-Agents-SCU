"""All confirmation actions and offline metrics rebuilt from actual ZIP frames."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from torch.nn import functional as F
from hyra_learning.networks import SearchAgent
from hyra_learning.observations import encode
from hyra_learning.team_rules import TeamRules
from hyra_uav.core import Config,FloodProcess,ACTIONS
from hyra_learning.sciml_data import read,write,digest
from hyra_learning.sciml_study import verify
from audit_team_replays import audit_replay,assert_close


def audit_file(path,entry,plan,actor=None,rule=None):
    envelope=read(path);replay=envelope['replay'];process=FloodProcess(entry['path'])
    assert envelope['case']==entry['id'] and envelope['source_sha256']==entry['source_sha256']==process.digest
    assert replay['config']==plan['config'] and replay['team']==plan['team']
    result=audit_replay(replay,process);views=result.pop('views')
    for field in result['metrics']:
        if field=='final':
            for key in result['metrics'][field]:assert_close(envelope['metrics'][field][key],result['metrics'][field][key])
        else:assert_close(envelope['metrics'][field],result['metrics'][field])
    policy=TeamRules(rule,views[0],Config(**plan['config']),plan['team']['separation_m']) if rule else None
    hidden=actor.initial_hidden(2) if actor is not None else None
    previous=torch.zeros(2,4,device=hidden.device) if actor is not None else None
    for t,row in enumerate(replay['actions']):
        actual=[ACTIONS.index(a) for a in row['actions']]
        if policy:np.testing.assert_array_equal(actual,policy.choose(views[t]))
        else:
            encoded=encode(views[t])
            with torch.no_grad():
                q,hidden,_,_=actor(torch.as_tensor(encoded['images'],device=hidden.device),
                    torch.as_tensor(encoded['poses'],device=hidden.device),previous,t,hidden,enhance=False)
            np.testing.assert_array_equal(actual,q.argmax(dim=-1).cpu().numpy())
            previous=F.one_hot(torch.tensor(actual,device=hidden.device),4).float()
    return dict(**result,case=entry['id'],family=entry['family'],forcing_family=entry['forcing_family'],method=envelope['method'],
        seed=envelope.get('seed'),input_sha256=digest(path),source_sha256=process.digest,
        model_sha256=envelope.get('model_sha256'),all_actions_reproduced_from_independent_legal_views=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output.resolve();plan=read(root/'formal-plan.json');verify(plan)
    freeze=read(root/'model-freeze.json');assert len(freeze['selected_models'])==12
    held={e['id']:e for e in read(root/'index.json') if e['split']=='holdout'};assert len(held)==9
    folder=root/'evaluation';out=root/'audits/trajectories';out.mkdir(parents=True,exist_ok=True)
    policy=list((folder/'policies').glob('*.json'));rules=list((folder/'rules').glob('*.json'))
    assert len(policy)==108 and len(rules)==27
    assert {(read(p)['case'],read(p)['method'],read(p)['seed']) for p in policy}=={
        (case,method,seed) for case in held for method in plan['methods'] for seed in plan['seeds']}
    assert {(read(p)['case'],read(p)['method']) for p in rules}=={(case,rule) for case in held for rule in ('R0','R1','R2')}
    audit_sources={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'tools/audit_team_replays.py')}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    results=[]
    def checked(path,actor=None,rule=None):
        target=out/path.name;envelope=read(path)
        if target.exists():
            result=read(target);assert result['input_sha256']==digest(path) and result['audit_sources']==audit_sources
        else:
            result=audit_file(path,held[envelope['case']],plan,actor,rule);result['audit_sources']=audit_sources;write(target,result)
        results.append(result)
    for path in sorted(rules):checked(path,rule=read(path)['method'])
    for model in freeze['selected_models']:
        assert digest(root/model['path'])==model['sha256']
        payload=torch.load(root/model['path'],map_location='cpu',weights_only=True)
        actor=SearchAgent().to(plan['runtime']['device']);actor.load_state_dict(payload['actor']);actor.eval()
        deployment=root/'deployment'/(model['method'].replace('+','-')+f"-s{model['seed']}")
        exported=torch.load(deployment/'actor.pt',map_location='cpu',weights_only=True)
        assert set(exported)==set(payload['actor'])
        for name in exported:torch.testing.assert_close(exported[name],payload['actor'][name],atol=0,rtol=0)
        metadata=read(deployment/'actor.json');assert digest(deployment/'actor.pt')==metadata['actor_sha256']
        assert metadata['selected_checkpoint_sha256']==model['sha256']
        for path in sorted(policy):
            row=read(path)
            if row['method']==model['method'] and row['seed']==model['seed']:
                assert row['model_sha256']==model['sha256'];checked(path,actor=actor)
        print('independently audited policy',model['method'],model['seed'],flush=True)
    assert len(results)==135
    write(ROOT/'docs/evidence/20261025-sciml-development/trajectory-audit-final.json',dict(status='passed',
        policy_episodes=108,rule_episodes=27,frames=135*121,observations=135*242,actions=135*60,
        results=results,audit_sources=audit_sources,
        note='Union visit gaps deduplicate simultaneous agent observations; original per-agent continuous-gap statistic retained and separately disclosed. No formal reward/primary metric altered.'))


if __name__=='__main__':main()
