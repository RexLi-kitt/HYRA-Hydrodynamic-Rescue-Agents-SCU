"""Freeze all selected weights before opening the two holdout policy evaluations.

Evaluates 216 learned-policy and 36 rule trials, retaining deterministic raw
replays and paired parent-level outcomes. No model selection or tuning here.
"""
import argparse,json,time,gzip,hashlib
from pathlib import Path
from scale100_train import *
from scale100_formal import validate_plan,check_selected,selection_key,process_alive
from scale100_backend import make_actor
from scale100_run_audit import audit,require
from scale100_development import legal_rule
from scale100_rules import water_aware

RULES=('frontier','sweep_refresh','water_aware')
METRICS=('first_coverage_auc','balanced_current_correct_coverage_auc','pixel_current_correct_coverage_auc','rescue_fraction','stale_error_area_auc','distance_m','collision_attempt_events')

def rule_sources():return {name:digest(Path(__file__).with_name(name)) for name in ('scale100_rules.py','scale100_development.py')}

def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported replay type: {type(value).__name__}')

def replay_bytes(payload):
    return json.dumps(payload,ensure_ascii=False,separators=(',',':'),default=json_scalar).encode('utf-8')

def save_replay(path,payload):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as archive:archive.write(replay_bytes(payload))
    temp.replace(path)

def seal_models(formal,plan,output):
    require(json.loads((formal/'queue.json').read_text())['status']=='completed','All36 formal jobs must finish before holdout scoring')
    expected={f'{task}-{method}-s{seed}' for task in ('E1','E2') for method in METHODS for seed in (11,29,47)}
    require(len(plan['jobs'])==36 and {j['id'] for j in plan['jobs']}==expected,'Formal matrix is incomplete/duplicated')
    models=[]
    for job in plan['jobs']:
        model=formal/job['id'];meta=check_selected(model,job,plan,digest(formal/'plan.json'))
        reports=[json.loads((model/f'development-ep{ep:04d}.json').read_text()) for ep in plan['development_checkpoints']]
        chosen=max(reports,key=lambda r:selection_key(r,job['task']))
        require(chosen['episode']==meta['selected_episode'] and chosen['actor_sha256']==meta['actor_sha256'],'Selected weight no longer matches the frozen development rule')
        models.append(dict(**job,folder=str((model/'selected').resolve()),actor_sha256=meta['actor_sha256'],selected_episode=meta['selected_episode']))
    payload=dict(schema='hyra-scale100-sealed-models-v1',formal_plan_sha256=digest(formal/'plan.json'),evaluation_source_sha256=digest(__file__),rule_sources=rule_sources(),training_sources=plan['training_sources'],training_backend=plan.get('training_backend','absolute_head_v1'),models=models,holdout_parents=[r['parent_terrain'] for r in plan['inputs'] if r['split']=='holdout'],scenario_seeds=plan['evaluation_scenario_seeds'],rule_baselines=list(RULES),selection='Frozen for all36 models before any holdout policy score; no replacement based on outcomes')
    require(len(payload['holdout_parents'])==2,'Expected two frozen holdout parents')
    seal=output/'sealed-models.json'
    if seal.exists():require(json.loads(seal.read_text())==payload,'Model/evaluation seal changed; retain original results and inspect before proceeding')
    else:write(seal,payload)
    return payload

def run_trial(output,trial_id,case,task,method,scenario_seed,actor=None,actor_sha256=None,training_seed=None,split='holdout'):
    record=output/(trial_id+'.json');replay=output/(trial_id+'.json.gz')
    if record.exists():
        saved=json.loads(record.read_text());require(saved['actor_sha256']==actor_sha256 and saved['case_sha256']==case.digest and saved['audit_source_sha256']==digest(Path(__file__).with_name('scale100_run_audit.py')),'Saved trial provenance differs')
        require((saved['trial_id'],saved['task'],saved['method'],saved['scenario_seed'],saved['training_seed'],saved['split'])==(trial_id,task,method,scenario_seed,training_seed,split),'Saved trial has wrong task/method/seed/split attribution')
        require(saved['evaluation_source_sha256']==digest(__file__) and saved['rule_sources']==rule_sources(),'Saved evaluator/rule sources changed')
        require(replay.exists() and digest(replay)==saved['replay_sha256'],'Saved replay missing/corrupted');return saved
    env=ScaleEnv(case,config(task,method if actor is not None else 'B3',scenario_seed))
    if actor is not None:total,data,_=rollout(actor,env,np.random.default_rng(scenario_seed),0.,True);images=data['images'].numpy()
    else:
        view=env.reset();images=[view.images.copy()];total=0.;done=False
        while not done:
            actions=water_aware(env) if method=='water_aware' else legal_rule(env,method)
            view,reward,done=env.step(actions);total+=reward;images.append(view.images.copy())
        images=np.asarray(images)
    result=audit(env,images)
    payload=dict(schema='hyra-scale100-evaluation-replay-v1',split=split,task=task,method=method,scenario_seed=scenario_seed,training_seed=training_seed,actor_sha256=actor_sha256,source_sha256=case.digest,grid=case.process['grid'],config=asdict(env.config),snapshots=env.snapshots,observations=env.observations,trajectory=env.trajectory,actions=env.actions,events=env.events,scenario_truth=dict(obstacles=env._blocked.astype(int).ravel().tolist(),targets=[list(p) for p in env._targets],scope='offline evaluator/replay only; not actor input'))
    save_replay(replay,payload)
    report=dict(trial_id=trial_id,split=split,parent=case.case['parent_terrain'],task=task,method=method,scenario_seed=scenario_seed,training_seed=training_seed,actor_sha256=actor_sha256,case_sha256=case.digest,replay_path=replay.name,replay_sha256=digest(replay),evaluation_source_sha256=digest(__file__),rule_sources=rule_sources(),audit_source_sha256=digest(Path(__file__).with_name('scale100_run_audit.py')),return_=total,metrics=result)
    write(record,report);return report

def average(rows,key):
    values=[r['metrics'][key] for r in rows if r['metrics'][key] is not None]
    return float(np.mean(values)) if values else None

def aggregate(rows):
    parent_tables=[];summary=[];contrasts=[]
    parents=sorted({r['parent'] for r in rows})
    for task in ('E1','E2'):
        for method in (*METHODS,*RULES):
            selected=[r for r in rows if r['task']==task and r['method']==method]
            if not selected:continue
            per_parent=[]
            for parent in parents:
                group=[r for r in selected if r['parent']==parent]
                if not group:continue
                values={k:average(group,k) for k in METRICS};parent_tables.append(dict(task=task,method=method,parent=parent,trials=len(group),**values));per_parent.append(values)
            summary.append(dict(task=task,method=method,independent_parents=len(per_parent),trials=len(selected),means={k:float(np.mean([r[k] for r in per_parent if r[k] is not None])) if any(r[k] is not None for r in per_parent) else None for k in METRICS}))
        # Each difference is paired by terrain, TRAINING seed and scenario seed.
        for name,weights in [('physics_increment',{'P':1.,'D':-1.}),('full_auxiliary',{'P':1.,'B3':-1.}),('gmrf_main',{'B1':.5,'B0':-.5,'B3':.5,'B2':-.5}),('sur_main',{'B2':.5,'B0':-.5,'B3':.5,'B1':-.5}),('gmrf_sur_interaction',{'B3':1.,'B1':-1.,'B2':-1.,'B0':1.})]:
            base=[r for r in rows if r['task']==task and r['method'] in weights];index={(r['parent'],r['training_seed'],r['scenario_seed'],r['method']):r for r in base};paired=[]
            for parent,train_seed,scenario_seed in sorted({k[:3] for k in index}):
                if any((parent,train_seed,scenario_seed,m) not in index for m in weights):continue
                values={}
                for metric in METRICS:
                    vals={m:index[(parent,train_seed,scenario_seed,m)]['metrics'][metric] for m in weights}
                    values[metric]=sum(weights[m]*vals[m] for m in weights) if all(v is not None for v in vals.values()) else None
                paired.append(dict(parent=parent,training_seed=train_seed,scenario_seed=scenario_seed,**values))
            per_parent=[]
            for parent in parents:
                group=[r for r in paired if r['parent']==parent]
                if group:per_parent.append(dict(parent=parent,pairs=len(group),**{k:float(np.mean([r[k] for r in group if r[k] is not None])) if any(r[k] is not None for r in group) else None for k in METRICS}))
            if per_parent:contrasts.append(dict(task=task,name=name,method_weights=weights,parent_differences=per_parent,mean_difference={k:float(np.mean([r[k] for r in per_parent if r[k] is not None])) if any(r[k] is not None for r in per_parent) else None for k in METRICS},inference='descriptive paired differences; two independent parents are insufficient for broad significance claims'))
    return dict(parent_tables=parent_tables,summary=summary,contrasts=contrasts,independent_parent_count=len(parents),uncertainty_scope='Parent means are equally weighted. Training/scenario seeds, agents, cells and times are not extra independent flood events; no p-values or confidence claims are produced.')

def execute(formal,output):
    torch.set_num_threads(1);plan=json.loads((formal/'plan.json').read_text());validate_plan(plan);output.mkdir(parents=True,exist_ok=True)
    previous=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
    require(not process_alive(previous.get('pid')),'Another holdout evaluator is live; refusing duplicate evaluation')
    seal=seal_models(formal,plan,output);status=dict(status='running',pid=os.getpid(),started_at=time.time(),model_seal_sha256=digest(output/'sealed-models.json'),completed=0,expected=252,current=None);write(output/'status.json',status);rows=[]
    try:
        cases=[FloodCase(r['path']) for r in plan['inputs'] if r['split']=='holdout']
        for model in seal['models']:
            actor=make_actor(plan.get('training_backend','absolute_head_v1'));actor.load_state_dict(torch.load(Path(model['folder'])/'actor.pt',weights_only=True));actor.eval()
            for case in cases:
                for seed in plan['evaluation_scenario_seeds']:
                    identity=f"{model['id']}__{case.case['parent_terrain']}__s{seed}";status['current']=identity;write(output/'status.json',status)
                    rows.append(run_trial(output,identity,case,model['task'],model['method'],seed,actor,model['actor_sha256'],model['seed']));status['completed']=len(rows);write(output/'status.json',status)
        for task in ('E1','E2'):
            for rule in RULES:
                for case in cases:
                    for seed in plan['evaluation_scenario_seeds']:
                        identity=f"{task}-{rule}__{case.case['parent_terrain']}__s{seed}";status['current']=identity;write(output/'status.json',status)
                        rows.append(run_trial(output,identity,case,task,rule,seed));status['completed']=len(rows);write(output/'status.json',status)
        require(len(rows)==252 and len({r['trial_id'] for r in rows})==252,'Evaluation matrix incomplete/duplicated')
        report=dict(schema='hyra-scale100-sealed-evaluation-v1',sealed_models_sha256=digest(output/'sealed-models.json'),formal_plan_sha256=digest(formal/'plan.json'),evaluation_source_sha256=digest(__file__),complete=True,learned_trials=216,rule_trials=36,all_audits_passed=all(r['metrics']['audit_passed'] for r in rows),rows=rows,**aggregate(rows));write(output/'results.json',report)
        status.update(status='completed',finished_at=time.time(),current=None);write(output/'status.json',status);return report
    except Exception as error:
        with (output/'failed-attempts.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(trial_id=status['current'],error=repr(error),time=time.time()))+'\n')
        status.update(status='failed',error=repr(error),finished_at=time.time());write(output/'status.json',status);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--formal-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();execute(a.formal_root,a.output)
