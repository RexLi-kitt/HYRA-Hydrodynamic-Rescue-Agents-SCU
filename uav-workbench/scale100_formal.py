"""Freeze and execute the paired 36-job study after a complete development review.

No formal budget is inferred here. Preparation requires an explicit reviewed
budget; running requires an immutable plan and verified source/data hashes.
"""
import os,sys,time,json,subprocess,shutil,ctypes
from pathlib import Path
from scale100_train import *
from scale100_run_audit import audit,require
import scale100_run_audit as audit_module
from scale100_backend import BACKENDS,source_hashes,validate_meta,make_actor

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'
PILOT=ROOT/'generated/scale100-budget-pilot-20261026'

def runtime():return dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__),device='cpu',threads=1,deterministic_algorithms=True)

def process_alive(pid):
    if not pid:return False
    if os.name!='nt':
        try:os.kill(int(pid),0);return True
        except ProcessLookupError:return False
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:
        if ctypes.get_last_error()==87:return False
        raise RuntimeError('Cannot establish process liveness; inspect PID before restarting')
    try:
        code=wintypes.DWORD();require(kernel.GetExitCodeProcess(handle,ctypes.byref(code)),'Cannot read process exit state');return code.value==259
    finally:kernel.CloseHandle(handle)

def validate_review(review):
    expected={f'{task}-{method}-s11' for task in ('E1','E2') for method in METHODS}
    require(review['complete'] and review['completed']==12 and not review['missing_jobs'],'Development review is incomplete')
    require({r['job'] for r in review['rows']}==expected and len(review['rows'])==12,'Wrong or duplicated development groups')
    require(all(r['episodes']==256 and r['updates']==253 and r['full_trajectory_audits_passed'] for r in review['rows']),'Development artifacts failed auditing')
    require(review['matched_initial_tensors'] and not review['holdout_policy_evaluated'],'Unmatched initialization or holdout contamination')
    require(all([p['episode'] for p in r['all_development_points']]==[64,128,256] for r in review['rows']),'Unequal development selection opportunities')

def validate_method_decision(decision,episodes):
    require(decision.get('status')=='ready_for_formal','Method readiness is unresolved; do not formalize an underfit auxiliary model')
    require(decision.get('episodes_per_job')==episodes,'Formal budget differs from the reviewed method decision')
    backend=decision.get('selected_training_backend');require(backend in BACKENDS,'Selected backend needs a verified formal adapter before execution')
    require(decision.get('training_sources')==source_hashes(backend),'Method decision refers to different training sources')

def validate_aux_review(review):
    expected={f'{task}-{method}-s11' for task in ('E1','E2') for method in ('D','P')}
    require(review['complete'] and review['completed']==4 and not review['missing_jobs'],'Auxiliary optimization review is incomplete')
    require(len(review['rows'])==4 and {r['job'] for r in review['rows']}==expected,'Wrong auxiliary optimization task matrix')
    require(review['matched_initial_tensors'] and not review['holdout_policy_evaluated'],'Auxiliary controls/holdout integrity failed')
    require(all(r['full_trajectory_audits_passed'] and r['episodes']==256 and r['updates']==253 for r in review['rows']),'Auxiliary optimization artifacts failed auditing')
    require(all(r.get('shared_auxiliary_mechanism_live_at_end',False) for r in review['rows']),'Auxiliary gradient path collapsed during training')
    require(set(review['matched_forecast_paths'])=={'E1','E2'},'Both task-specific matched-input diagnostics are required')

def prepare(output,episodes,rationale):
    decision_path=ROOT/'research/scale100_training_decision_20261026.json'
    require(decision_path.exists(),'A reviewed method/budget decision is required before formal preparation')
    decision=json.loads(decision_path.read_text(encoding='utf-8'));validate_method_decision(decision,episodes);backend=decision['selected_training_backend']
    from scale100_review_pilot import review as refresh_review
    review_path=PILOT/'review.json';review=refresh_review(PILOT);validate_review(review)
    require(review['training_sources']==sources() and review['dataset_sha256']==digest(DATA),'Development evidence no longer matches training sources/data')
    require(review['audit_source_sha256']==digest(audit_module.__file__),'Replay audit changed since review')
    require(type(episodes) is int and 256<=episodes<=8192 and episodes%16==0,'Use a reviewed episode budget divisible by16, in [256,8192]')
    require(bool(rationale.strip()),'Record the common-budget rationale before freezing')
    dataset=json.loads(DATA.read_text());inputs=[]
    for row in dataset['cases']:
        require(digest(row['path'])==row['sha256'],'Hydraulic source changed')
        inputs.append(dict(path=row['path'],sha256=row['sha256'],split=row['split'],parent_terrain=row['parent_terrain']))
    require(len({r['parent_terrain'] for r in inputs})==len(inputs),'Parent terrain split overlap')
    checkpoints=[episodes//4,episodes//2,episodes]
    plan=dict(schema='hyra-scale100-formal-plan-v1',frozen_at=time.time(),budget_frozen=True,formal_training_started=False,episodes_per_job=episodes,updates_per_job=episodes-3,maximum_joint_decisions=36*episodes*180,methods=list(METHODS),tasks=['E1','E2'],training_seeds=[11,29,47],evaluation_scenario_seeds=[11,29,47],development_checkpoints=checkpoints,inputs=inputs,dataset_path=str(DATA.resolve()),dataset_sha256=digest(DATA),training_sources=sources(),controller_source_sha256=digest(__file__),audit_source_sha256=digest(audit_module.__file__),pilot_review_path=str(review_path.resolve()),pilot_review_sha256=digest(review_path),budget_rationale=rationale,selection=dict(E1=['first_coverage_auc','balanced_current_correct_coverage_auc','earlier_checkpoint'],E2=['rescue_fraction','first_coverage_auc','balanced_current_correct_coverage_auc','earlier_checkpoint'],scope='greedy development only; all methods and seeds have the same three opportunities'),primary_metrics=dict(E1='first_coverage_auc',E2='rescue_fraction'),physics_secondary_metric='balanced_current_correct_coverage_auc',comparison_order=['P versus D: physical-constraint increment','P versus B3: full auxiliary increment','B0/B1/B2/B3 factorial contrasts: structural encoding and Sur'],holdout_policy_evaluated=False,scope='paired synthetic 100m/900s task study; 2 holdout parents do not establish broad real-flood generalization',jobs=[dict(id=f'{task}-{method}-s{seed}',task=task,method=method,seed=seed) for task in ('E1','E2') for method in METHODS for seed in (11,29,47)])
    plan['runtime']=runtime();plan['worker_source_sha256']=digest(ROOT/'scale100_formal_worker.py')
    plan['max_concurrent_workers']=3;plan['parallel_controller_sha256']=digest(ROOT/'scale100_formal_parallel.py')
    plan['scheduling']='independent one-thread learners; at most3 concurrent workers, memory guard reserves1.5GiB plus0.8GiB per worker; identical data/steps/selection'
    plan['jobs']=[dict(id=f'{task}-{method}-s{seed}',task=task,method=method,seed=seed) for seed in (11,29,47) for method in METHODS for task in ('E1','E2')]
    plan['training_backend']=backend;plan['training_sources']=source_hashes(backend);plan['backend_adapter_sha256']=digest(ROOT/'scale100_backend.py')
    if backend!='absolute_head_v1':
        if backend=='aux_resilient_v4':
            from scale100_review_resilient import review as refresh_aux_review
            aux_folder=ROOT/'generated/scale100-aux-resilient-20261026'
        elif backend=='aux_precision_v3':
            from scale100_review_precision import review as refresh_aux_review
            aux_folder=ROOT/'generated/scale100-aux-precision-20261026'
        else:
            from scale100_review_aux import review as refresh_aux_review
            aux_folder=ROOT/'generated/scale100-aux-tuning-20261026'
        aux_path=aux_folder/'review.json';aux=refresh_aux_review();validate_aux_review(aux)
        require(aux['training_sources']==source_hashes(backend) and aux['dataset_sha256']==digest(DATA),'Auxiliary source/data changed')
        plan['auxiliary_review_path']=str(aux_path.resolve());plan['auxiliary_review_sha256']=digest(aux_path)
        plan['matched_forecast_evidence']={task:dict(path=path,sha256=digest(path)) for task,path in aux['matched_forecast_paths'].items()}
    plan['method_decision_path']=str(decision_path.resolve());plan['method_decision_sha256']=digest(decision_path)
    inherited=json.loads((ROOT/'generated/sciml-round1/formal-plan.json').read_text(encoding='utf-8'))['sources']
    for path,sha in inherited.items():require(digest(path)==sha,'Original registered scientific source changed')
    plan['inherited_sources']=inherited
    evidence={'pilot-review.json':review_path,'method-decision.json':decision_path}
    if backend!='absolute_head_v1':
        recovery=aux_folder/'recovery-verification.json';proof=json.loads(recovery.read_text())
        require(proof['actor_tensors_bitwise_equal'] and proof['history_equal'] and proof['sources']==source_hashes(backend),'Auxiliary recovery has not been verified')
        evidence.update({'auxiliary-review.json':aux_path,'auxiliary-recovery.json':recovery})
        for task,path in aux['matched_forecast_paths'].items():
            diagnostic=json.loads(Path(path).read_text());require(diagnostic['task']==task and diagnostic['trained_episodes']==256 and not diagnostic['holdout_policy_evaluated'],'Wrong matched forecast evidence')
            diagnostic_code=ROOT/({'aux_resilient_v4':'scale100_resilient_forecast.py','aux_precision_v3':'scale100_precision_forecast.py'}.get(backend,'scale100_forecast_audit.py'))
            require(diagnostic['diagnostic_source_sha256']==digest(diagnostic_code) and diagnostic['rule_source_sha256']==digest(ROOT/'scale100_rules.py'),'Matched diagnostic code/observations changed')
            for method in ('D','P'):
                model=next(r for r in aux['rows'] if r['task']==task and r['method']==method)
                require(diagnostic['actor_sha256'][method]==model['actor_sha256'],'Forecast diagnostic used another model')
            evidence[f'{task}-matched-forecast.json']=Path(path)
    payloads={name:Path(path).read_bytes() for name,path in evidence.items()}
    output.mkdir(parents=True,exist_ok=False);(output/'evidence').mkdir()
    for name,payload in payloads.items():(output/'evidence'/name).write_bytes(payload)
    plan['evidence_files']={name:dict(path=str((output/'evidence'/name).resolve()),sha256=digest(output/'evidence'/name)) for name in payloads}
    plan['pilot_review_path']=plan['evidence_files']['pilot-review.json']['path'];plan['pilot_review_sha256']=plan['evidence_files']['pilot-review.json']['sha256']
    plan['method_decision_path']=plan['evidence_files']['method-decision.json']['path'];plan['method_decision_sha256']=plan['evidence_files']['method-decision.json']['sha256']
    if backend!='absolute_head_v1':
        plan['auxiliary_review_path']=plan['evidence_files']['auxiliary-review.json']['path'];plan['auxiliary_review_sha256']=plan['evidence_files']['auxiliary-review.json']['sha256']
        plan['matched_forecast_evidence']={task:plan['evidence_files'][f'{task}-matched-forecast.json'] for task in ('E1','E2')}
    write(output/'plan.json',plan);write(output/'queue.json',dict(status='frozen_not_started',formal_training_started=False,completed=[]));return plan

def validate_plan(plan):
    require(plan['training_sources']==source_hashes(plan.get('training_backend','absolute_head_v1')) and plan['dataset_sha256']==digest(DATA),'Frozen training source/data changed')
    require(plan['backend_adapter_sha256']==digest(ROOT/'scale100_backend.py'),'Frozen backend adapter changed')
    require(plan['controller_source_sha256']==digest(__file__) and plan['audit_source_sha256']==digest(audit_module.__file__),'Frozen execution/audit code changed')
    require(plan['runtime']==runtime() and plan['worker_source_sha256']==digest(ROOT/'scale100_formal_worker.py'),'Frozen runtime/worker changed')
    require(plan['parallel_controller_sha256']==digest(ROOT/'scale100_formal_parallel.py'),'Frozen parallel scheduler changed')
    for path,sha in plan['inherited_sources'].items():require(digest(path)==sha,'Original registered scientific source changed')
    for row in plan['inputs']:require(digest(row['path'])==row['sha256'],'Frozen hydraulic case changed')
    for row in plan['evidence_files'].values():require(digest(row['path'])==row['sha256'],'Frozen development evidence changed')

def evaluate_checkpoint(model,episode,job,plan):
    pointer=json.loads((model/'latest.json').read_text());checkpoint=model/pointer['path'];require(pointer['episode']==episode and digest(checkpoint)==pointer['sha256'],'Wrong recovery checkpoint for selection')
    state=torch.load(checkpoint,weights_only=False,map_location='cpu');actor=make_actor(plan.get('training_backend','absolute_head_v1'));actor.load_state_dict(state['actor']);actor.eval()
    row=next(r for r in plan['inputs'] if r['split']=='development');case=FloodCase(row['path']);rows=[]
    for seed in plan['evaluation_scenario_seeds']:
        env=ScaleEnv(case,config(job['task'],job['method'],seed));total,data,_=rollout(actor,env,np.random.default_rng(seed),0.,True)
        rows.append(dict(scenario_seed=seed,return_=total,**audit(env,data['images'].numpy())))
    actor_path=model/f'actor-ep{episode:04d}.pt';torch.save(actor.state_dict(),actor_path)
    report=dict(episode=episode,updates=state['updates'],actor_path=actor_path.name,actor_sha256=digest(actor_path),checkpoint_sha256=pointer['sha256'],parent=row['parent_terrain'],split='development',rows=rows,first_coverage_auc=float(np.mean([r['first_coverage_auc'] for r in rows])),balanced_current_correct_coverage_auc=float(np.mean([r['balanced_current_correct_coverage_auc'] for r in rows])),rescue_fraction=float(np.mean([r['rescue_fraction'] for r in rows])) if job['task']=='E2' else None)
    write(model/f'development-ep{episode:04d}.json',report);return report

def selection_key(report,task):
    if task=='E1':return report['first_coverage_auc'],report['balanced_current_correct_coverage_auc'],-report['episode']
    return report['rescue_fraction'],report['first_coverage_auc'],report['balanced_current_correct_coverage_auc'],-report['episode']

def select_model(model,job,plan):
    reports=[json.loads((model/f'development-ep{ep:04d}.json').read_text()) for ep in plan['development_checkpoints']]
    chosen=max(reports,key=lambda r:selection_key(r,job['task']));source=model/chosen['actor_path'];require(digest(source)==chosen['actor_sha256'],'Selected checkpoint checksum mismatch')
    selected=model/'selected';selected.mkdir(exist_ok=True)
    if (selected/'actor.pt').exists():require(digest(selected/'actor.pt')==chosen['actor_sha256'],'Partial selection contains a different actor')
    else:
        shutil.copyfile(source,selected/'actor.pt.tmp');(selected/'actor.pt.tmp').replace(selected/'actor.pt')
    meta=json.loads((model/'actor.json').read_text());meta.update(trained_episodes=chosen['episode'],updates=chosen['updates'],total_training_episodes_executed=plan['episodes_per_job'],selected_episode=chosen['episode'],actor_sha256=digest(selected/'actor.pt'),selection_data_split='development',selection_rule=plan['selection'][job['task']],formal_plan_sha256=digest(model.parent/'plan.json'))
    write(selected/'actor.json',meta);write(model/'selection.json',dict(selected_episode=chosen['episode'],actor_sha256=meta['actor_sha256'],all_scores=[{k:r[k] for k in ('episode','first_coverage_auc','balanced_current_correct_coverage_auc','rescue_fraction')} for r in reports],holdout_evaluated=False))
    return meta

def check_selected(model,job,plan,plan_digest):
    meta=json.loads((model/'selected/actor.json').read_text());require(digest(model/'selected/actor.pt')==meta['actor_sha256'],'Selected actor changed')
    validate_meta(meta,plan.get('training_backend','absolute_head_v1'))
    require(meta['formal_plan_sha256']==plan_digest and meta['sources']==plan['training_sources'],'Selected actor belongs to another formal plan')
    require((meta['task'],meta['method'],meta['seed'],meta['total_training_episodes_executed'],meta['formal'])==(job['task'],job['method'],job['seed'],plan['episodes_per_job'],True),'Selected actor belongs to another task/budget/phase')
    require(meta['config']==asdict(config(job['task'],job['method'],job['seed'])),'Selected environment contract changed')
    require(meta['selected_episode'] in plan['development_checkpoints'] and meta['trained_episodes']==meta['selected_episode'] and meta['updates']==meta['selected_episode']-3,'Unregistered selected checkpoint')
    return meta

def run(output):
    plan=json.loads((output/'plan.json').read_text());validate_plan(plan);torch.set_num_threads(1)
    previous=json.loads((output/'queue.json').read_text())
    if previous.get('status')=='completed':
        for job in plan['jobs']:check_selected(output/job['id'],job,plan,digest(output/'plan.json'))
        return previous
    for pid in (previous.get('pid'),previous.get('current',{}).get('child_pid')):
        require(not process_alive(pid),'Existing controller/child is still live; refusing duplicate training')
    require(shutil.disk_usage(output).free>4*1024**3,'Less than4GiB free; inspect before starting the paired study')
    status=dict(schema='hyra-scale100-formal-queue-v1',status='running',pid=os.getpid(),started_at=previous.get('started_at',time.time()),formal_training_started=previous.get('formal_training_started',False),completed=[],current={});write(output/'queue.json',status)
    try:
        for job in plan['jobs']:
            model=output/job['id']
            if (model/'selection.json').exists():
                check_selected(model,job,plan,digest(output/'plan.json'));status['formal_training_started']=True;status['completed'].append(job['id']);continue
            for stop in plan['development_checkpoints']:
                evaluation=model/f'development-ep{stop:04d}.json'
                if evaluation.exists():continue
                current=json.loads((model/'latest.json').read_text())['episode'] if (model/'latest.json').exists() else 0
                require(current<=stop,'Checkpoint ahead of missing evaluation; manual audit needed')
                if current<stop:
                    validate_plan(plan)
                    args=[sys.executable,'-B','scale100_formal_worker.py','--plan',str(output/'plan.json'),'--job',job['id'],'--stop-after',str(stop)]
                    if current:args.append('--resume')
                    with (output/(job['id']+'.log')).open('a',encoding='utf-8') as log:
                        child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));status.update(formal_training_started=True,current=dict(job=job['id'],segment_end=stop,child_pid=child.pid));write(output/'queue.json',status);code=child.wait()
                    require(code==0,f'Formal training failed: {job["id"]}, exit={code}')
                status['current']=dict(job=job['id'],stage='development_evaluation',episode=stop);write(output/'queue.json',status);evaluate_checkpoint(model,stop,job,plan)
            select_model(model,job,plan);status['completed'].append(job['id']);write(output/'queue.json',status)
        status.update(status='completed',finished_at=time.time(),current={});write(output/'queue.json',status);return status
    except Exception as error:
        status.update(status='failed',error=repr(error),finished_at=time.time());write(output/'queue.json',status);raise

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    freeze=sub.add_parser('prepare');freeze.add_argument('--output',type=Path,required=True);freeze.add_argument('--episodes',type=int,required=True);freeze.add_argument('--rationale',required=True)
    execute=sub.add_parser('run');execute.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.command=='prepare':prepare(a.output,a.episodes,a.rationale)
    else:run(a.output)
if __name__=='__main__':main()
