"""Paired development difficulty probe. Keeps the V2 planner and weights intact.

This diagnoses transfer to harder conditions; it does not train, select a winning
difficulty, or provide an independent final test of learned-planner superiority.
"""
from pathlib import Path
from dataclasses import replace
import argparse,copy,json,sys,time,os
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
V2=HERE.parent/'flood-drop-v2'
sys.path.insert(0,str(V2))
from environment import Config,Env,make_scenario
from pilot import read,write,sha,seed,rollout,Learner,summary

# Frozen before any outcomes from this probe; all five get equal primary weight.
LEVELS=(('reference',360,100.),('deadline',300,100.),
        ('visibility',360,75.),('combined',300,75.),('stress',240,75.))
METHODS=('planner','maac','mat','happo')

def variant(case,window,sensor):
    result=copy.deepcopy(case)
    cfg=replace(Config(**case['scenario']['config']),window=window,sensor=sensor)
    result['scenario']['config']=cfg.__dict__
    for e in result['scenario']['events']:
        e['deadline']=e['fall']+window
        if e['deadline']>cfg.duration:raise ValueError('Truncated deadline; reject entire probe')
    return result

def paired_summary(records,method):
    refs={(r['level'],r['case']):r for r in records if r['method']=='planner'}
    selected=[r for r in records if r['method']==method]
    groups={}
    for r in selected:
        ref=refs[(r['level'],r['case'])]
        groups.setdefault(r['parent'],[]).append(float(r['scene_success'])-float(ref['scene_success']))
    means=np.array([np.mean(values) for values in groups.values()])
    rng=np.random.default_rng(75023)
    boot=means[rng.integers(0,len(means),(5000,len(means)))].mean(1)*100
    return dict(difference_percentage_points=float(means.mean()*100),
        parent_cluster_bootstrap_95_interval=[float(x) for x in np.quantile(boot,[.025,.975])],
        parent_count=len(groups),formal_claim=False,
        limitation='Six development parents and one training seed; interval does not include training-seed uncertainty')

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def status(state,**kw):write(out/'status.json',dict(status=state,pid=os.getpid(),elapsed_s=time.monotonic()-started,**kw))
    try:
        status('preparing');old=read(args.pilot/'protocol.json');leaders=read(args.pilot/'leaders.json')
        assert read(args.pilot/'status.json')['status']=='completed'
        for p,h in old['source_hashes'].items():
            if sha(p)!=h:raise ValueError('Original V2 source changed')
        fields=[];seen=set()
        for c in read(args.pilot/'scenarios.json')['development']:
            f=c['field']
            if f['parent'] not in seen:seen.add(f['parent']);fields.append(f)
        assert len(fields)==6
        for f in fields:assert sha(f['path'])==f['sha256']
        cases=[]
        for i in range(24):
            f=fields[i%6];water=np.load(f['path'],mmap_mode='r',allow_pickle=False)
            cases.append(dict(id=f'pressure-dev-{i:03}',field=f,scenario=make_scenario(water,941000+i,Config())))
        configs={name:[variant(c,w,s) for c in cases] for name,w,s in LEVELS}
        for name,cs in configs.items():
            for original,c in zip(cases,cs):
                assert original['scenario']['traps']==c['scenario']['traps']
                for a,b in zip(original['scenario']['events'],c['scenario']['events']):
                    assert all(a[k]==b[k] for k in ('fall','xy','trigger','field'))
        hashes={str(p):sha(p) for p in HERE.glob('*.py')}
        hashes.update(old['source_hashes'])
        selected={a:leaders[a] for a in METHODS if a!='planner'}
        for b in selected.values():assert sha(b['path'])==b['sha256']
        protocol=dict(version='paired-pressure-development-01',date='2026-11-02',
            targets_percentage_points=dict(minimum=15,satisfactory=25,aspirational=50),
            learner_absolute_scene_success_target=[.90,.94],
            levels=[dict(name=n,window=w,sensor=s) for n,w,s in LEVELS],methods=list(METHODS),
            roster=20,uav_count=4,duration=1500,delivery_seconds=10,
            common_fall_interval=[0,1140],scenarios_per_level=24,parent_count=6,
            primary='Unweighted mean scene-success over all five preregistered levels; paired difference vs unmodified V2 planner',
            keep_all_levels=True,select_level_by_gap=False,select_final_checkpoint_on_this_probe=False,
            scope='DEVELOPMENT DIAGNOSTIC ONLY. New person layouts on old development hydraulic parents. No formal holdout.',
            neural_candidates='Top three positive-update V2 development checkpoints selected before this probe; other three retained in original pilot, no claim excluded models improve.',
            checkpoints=selected,source_hashes=hashes,original_protocol_sha256=sha(args.pilot/'protocol.json'),
            fairness='Identical water, fall times, persons, traps, sensor, deadlines, movement/service and observations for all methods within a case; unchanged planner source.',
            no_training=True,no_test_used=True,decision='Report every tier. Do not tune environment to hit a performance gap. Inspect high-level task allocation before more training.',
            future_validation='Freeze scenario distribution, tune planning baselines on development too, compare against strongest registered planning baseline. Use disjoint hydraulic parents and event seeds, multiple training seeds, paired confidence intervals.',
            sources=[dict(url='https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html',
                use='Use interval estimates and report uncertainty with few RL runs; does not predict any 15/25/50 point gain.')])
        write(out/'protocol.json',protocol);write(out/'scenarios.json',configs)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed(8701)
        records=[]
        for method in METHODS:
            model=None
            if method!='planner':
                ck=torch.load(selected[method]['path'],map_location='cpu',weights_only=False)
                model=Learner(method,ck['learner']['dim']);model.restore(ck['learner']);model.eval();del ck
            for name,_,_ in LEVELS:
                group=[]
                for case in configs[name]:
                    if (out/'STOP_AFTER_EPISODE').exists():status('stopped',completed=len(records));return
                    if time.monotonic()-started>1800:raise TimeoutError('Development probe cap; keep partial records')
                    result,_=rollout(model,case)
                    assert result['supplied']+result['expired']+result['outside']==20
                    result.update(method=method,level=name,parent=case['field']['parent'])
                    result['expired_seen_at_deadline']=result['expired']-result['expired_unseen']
                    records.append(result);group.append(result)
                    with (out/'episodes.jsonl').open('a',encoding='utf8') as fp:fp.write(json.dumps(result,allow_nan=False)+'\n')
                    status('evaluating',method=method,level=name,completed=len(records),total=480)
                write(out/f'{method}-{name}.json',dict(rows=group,summary=summary(group)))
                print(json.dumps(dict(method=method,level=name,summary=summary(group))),flush=True)
        assert len(records)==480 and len({(r['method'],r['level'],r['case']) for r in records})==480
        for p,h in hashes.items():assert sha(p)==h
        for b in selected.values():assert sha(b['path'])==b['sha256']
        for f in fields:assert sha(f['path'])==f['sha256']
        result=dict(status='completed',episodes=480,parent_count=6,development_only=True,
            totals={m:summary([r for r in records if r['method']==m]) for m in METHODS},
            paired_differences={m:paired_summary(records,m) for m in METHODS if m!='planner'},
            frozen_weights_unchanged=True,planner_source_unchanged=True,no_formal_success_claim=True)
        write(out/'summary.json',result)
        status('completed',completed=480,total=480,no_training=True,no_test_used=True)
    except BaseException as ex:
        import traceback
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()))
        status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:p.error('Explicit --execute required')
    run(args)
