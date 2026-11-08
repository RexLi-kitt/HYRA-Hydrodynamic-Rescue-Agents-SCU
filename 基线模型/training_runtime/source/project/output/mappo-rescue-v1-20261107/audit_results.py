"""Independently replay recorded actions with the official SDK DeliveryMission.

Optionally reproduce the first development map by executing the exported policy
adapter and comparing every action. No search/rescue training code is imported
for the default CPU replay; no private referee state enters a policy.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SDK=Path(r'C:\Users\李\Desktop\hyra\HYRA-referee-20261106.1')
PANEL=Path(r'C:\Users\李\Documents\Codex\2026-11-05\task\hyra_20261106_alignment\local_training_candidate_panel_0_2400')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def replay(sdk,case,row,policy=None):
    water,people=sdk.load_trial(PANEL,case)
    env=sdk.DeliveryMission(water,people,(64,720,2400));trace=row['trace']
    comparisons=0
    for step,r in enumerate(trace):
        o=sdk.view(env);moves=np.asarray(r['moves'],dtype=np.int64);targets=np.asarray(r['targets'],dtype=np.int64)
        sdk.action_pair((moves,targets))
        if policy is not None:
            actual_moves,actual_targets=sdk.action_pair(policy.act(o))
            if not (np.array_equal(moves,actual_moves) and np.array_equal(targets,actual_targets)):
                raise AssertionError(f'Policy differs parent={row["parent"]} mode={row["mode"]} step={step}')
        if int(o.t)!=r['t'] or not np.array_equal(o.pos,np.asarray(r['position'])):
            raise AssertionError(f'Position/time differs step={step}')
        near=(np.linalg.norm(o.seen_xy[None,:,:]-o.pos[:,None,:],axis=-1)<=1.+1e-9)&o.visible[None,:]&o.active[None,:]
        old_sup=env.supplied.copy();old_progress=env.progress.copy()
        old_conf,old_boundary=env.conflicts,env.boundary
        old_seen=int((o.seen_at>=0).sum());owners=o.owner.copy()
        close=int(near.any(axis=1).sum())
        hover=sum(moves[i]==8 and targets[i]>=0 and near[i,targets[i]] for i in range(4))
        continuation=sum(moves[i]==8 and targets[i]>=0 and owners[targets[i]]==i for i in range(4))
        env.step(moves,targets);after=sdk.view(env)
        observed=dict(near=close,own_near_agents=close,correct_hover=int(hover),correct_near_hover=int(hover),
            public_owner_continuations=int(continuation),
            service_starts=int(((env.progress>0)&(old_progress==0)).sum()),
            delivered=int(((env.supplied>=0)&(old_sup<0)).sum()),
            conflicts=int(env.conflicts-old_conf),boundary=int(env.boundary-old_boundary),
            discoveries=int((after.seen_at>=0).sum())-old_seen,new_cells=len(after.new_cells),
            selected=int((targets>=0).sum()),
            selected_visible=int(sum(j>=0 and o.visible[j] for j in targets)))
        displacement=np.linalg.norm(env.pos-o.pos,axis=1)
        if not np.array_equal(displacement,np.asarray(r['displacement'])):
            raise AssertionError(f'Displacement differs step={step}')
        for name,value in observed.items():
            if name in r and r[name]!=value:
                raise AssertionError(f'Event {name} differs parent={row["parent"]} mode={row["mode"]} step={step}: {value}!={r[name]}')
            if name in r:comparisons+=1
    if not (env.t>=env.duration or ((env.supplied>=0).all() and (env.last>=0).all())):
        raise AssertionError('Action tape ended before mission completion')
    if env.metrics()!=row['metrics']:raise AssertionError('Final official metrics differ')
    return dict(parent=row['parent'],roster=row['roster'],mode=row['mode'],steps=len(trace),
                event_comparisons=comparisons,all_metrics_exact=True,policy_actions_exact=policy is not None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--verify-policy',action='store_true')
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(HERE) or output.exists():ap.error('output must be a new file in experiment directory')
    result=read(args.result);sdk=load_module('independent_official_sdk_audit',SDK/'run.py')
    panel=read(PANEL/'panel.json');manifest=read(PANEL/'SHA256.json')
    cases={(r['parent'],r['roster']):r for r in panel}
    records=[]
    for row in result['full']:
        key=(row['parent'],row['roster']);case=cases[key]
        for field in ('water','population'):
            if sha(PANEL/case[field])!=manifest[case[field]]:raise ValueError('Input SHA mismatch '+case[field])
        records.append(replay(sdk,case,row))
        print('REPLAY PASS',key,row['mode'],flush=True)
    reproduced=[]
    if args.verify_policy:
        if result.get('zero_rescue') or not result.get('checkpoint'):
            raise ValueError('Policy reproduction requires a non-ablated explicit checkpoint')
        cp=Path(result['checkpoint'])
        if sha(cp)!=result['checkpoint_sha256']:raise ValueError('Checkpoint hash differs')
        adapter=load_module('independent_rescue_policy_adapter',HERE/'policy_adapter.py')
        first=41004
        selected=[row for row in result['full'] if row['parent']==first and row['roster']==0]
        if not selected:raise ValueError('First development map 41004 absent')
        for row in selected:
            p=adapter.Frozen(cp,mode=row['mode'],device=args.device,seed=row['evaluation_seed'])
            reproduced.append(replay(sdk,cases[(first,0)],row,p))
            print('POLICY PASS',first,row['mode'],flush=True)
    out=dict(status='PASS',source_result=str(args.result.resolve()),source_sha256=sha(args.result),
        official_entry_sha256=sha(SDK/'run.py'),panel_sha256=sha(PANEL/'panel.json'),
        panel_manifest_sha256=sha(PANEL/'SHA256.json'),replays=records,policy_reproductions=reproduced)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':main()
