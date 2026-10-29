"""Rebuild every formal training transition from raw actual-frame events.

Independent motion/footprint/shared records from audit_team_replays; no TeamEnv
or production rollout. Centralized mixer truth is checked separately from actor
inputs. This is read-only evidence, never additional policy training.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from hyra_uav.core import FloodProcess,ACTIONS
from hyra_learning.observations import encode,pool_mean
from hyra_learning.sciml_data import read,write,digest
from hyra_learning.sciml_study import verify
from audit_team_replays import rebuild


def training_state(view,depth,duration):
    pooled_bed=pool_mean(view.bed[None],8).ravel();pooled_bed=(pooled_bed-pooled_bed.mean())/max(float(pooled_bed.std()),.05)
    positions=[]
    for agent in view.agents:
        radians=np.radians(agent.pose.heading_deg)
        positions.extend([(agent.pose.x_m+agent.dx/2)/(agent.shape[1]*agent.dx),
                          (agent.pose.z_m+agent.dx/2)/(agent.shape[0]*agent.dx),np.sin(radians),np.cos(radians)])
    return np.concatenate((pool_mean(depth[None],8).ravel(),pooled_bed,positions,[view.agents[0].time_s/duration])).astype(np.float32)


def audit_job(root,folder,plan,processes,sources):
    history=read(folder/'history.json');assert len(history)==2000
    provenance=dict(history_sha256=digest(folder/'history.json'),formal_plan_sha256=digest(root/'formal-plan.json'),audit_sources=sources)
    target=root/'audits/transitions'/f'{folder.name}.json'
    if target.exists():
        result=read(target)
        for k,v in provenance.items():assert result[k]==v
        return result
    records=[]
    initial=[dict(x_m=0.,z_m=0.,heading_deg=0.),dict(x_m=0.,z_m=155.,heading_deg=0.)]
    for row in history:
        path=folder/'episodes'/f"episode-{row['episode']:04d}.pt";shard=torch.load(path,map_location='cpu',weights_only=True)
        process=processes[row['event']];data=shard['data']
        actions=[dict(time_s=t*10,actions=[ACTIONS[int(a)] for a in values]) for t,values in enumerate(data['actions'])]
        result=rebuild(process,plan['config'],plan['team'],actions,initial);views=result['views'];samples=result['snapshots']
        total=0.
        for t,view in enumerate(views):
            encoded=encode(view)
            np.testing.assert_array_equal(encoded['images'],data['images'][t].numpy())
            np.testing.assert_array_equal(encoded['poses'],data['poses'][t].numpy())
            np.testing.assert_array_equal(training_state(view,process.depths[2*t],600),data['states'][t].numpy())
            if t==60:break
            a,b,end=samples[2*t:2*t+3]
            reward=(a['metrics']['correct_coverage']+b['metrics']['correct_coverage'])*5/600
            reward-=plan['team']['distance_cost_per_m']*(end['distance_m']-a['distance_m'])
            reward-=plan['team']['turn_cost_per_s']*(end['turn_s']-a['turn_s'])
            reward-=plan['team']['collision_cost']*(end['collision_events']-a['collision_events'])
            np.testing.assert_allclose(reward,float(data['rewards'][t]),rtol=1e-6,atol=1e-9)
            assert bool(data['terminal'][t])==(t==59)
            total+=reward
        assert abs(total-row['metrics']['return_'])<1e-10
        assert abs(samples[-1]['distance_m']-row['metrics']['distance_m'])<1e-8
        assert samples[-1]['turn_s']==row['metrics']['turn_s'] and samples[-1]['collision_events']==row['metrics']['collision_events']
        records.append(dict(episode=row['episode'],case=row['event'],sha256=digest(path)))
        if row['episode']%100==0:print(folder.name,'raw transitions audited through episode',row['episode'],flush=True)
    result=dict(status='passed',job=folder.name,episodes=2000,transitions=120000,actor_inputs_rebuilt=122000,
                checks=['all legal images and poses','all separately rebuilt CTDE state tensors','all rewards/terminals',
                        'physical distances, paid turns, collisions and actual5s observations'],records=records,**provenance)
    write(target,result);return result


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--completed-only',action='store_true')
    args=parser.parse_args();root=args.output.resolve();plan=read(root/'formal-plan.json');verify(plan);torch.set_num_threads(1)
    processes={e['id']:FloodProcess(e['path']) for e in plan['inputs'] if e['split']=='fit'}
    sources={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'tools/audit_team_replays.py')}
    results=[]
    for method in plan['methods']:
        for seed in plan['seeds']:
            folder=root/'formal'/(method.replace('+','-')+f'-s{seed}')
            if not (folder/'summary.json').exists():
                if args.completed_only:continue
                raise ValueError('Incomplete formal job: '+folder.name)
            results.append(audit_job(root,folder,plan,processes,sources))
    if not args.completed_only:assert len(results)==12 and sum(r['transitions'] for r in results)==1440000
    write(ROOT/'docs/evidence/20261025-sciml-development'/('transition-audit-progress.json' if args.completed_only else 'transition-audit-final.json'),
        dict(status='passed_completed_jobs_only' if args.completed_only else 'passed',completed_jobs=len(results),
            training_transitions=sum(r['transitions'] for r in results),jobs=[dict(job=r['job'],episodes=r['episodes'],transitions=r['transitions']) for r in results],audit_sources=sources))


if __name__=='__main__':main()
