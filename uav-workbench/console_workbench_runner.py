"""Run the exact chosen task and engine; never substitute a different group."""
import argparse,json,time
from pathlib import Path
from datetime import datetime,timezone
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
from sur_rescue.hydraulics import HydraulicFrames
from sur_rescue.environment import rule_actions
from workbench_tasks import make_env
from workbench_api import ROOT,validate,group_record,TASKS,LABELS
from hyra_uav.experiments import save_run

def execute_job(job):
    checked=validate(dict(scene=job['scene']['id'],task=job['task'],group=job['requested_group'],mode=job['mode'],config=job['config']))
    if checked['source_sha256']!=job['source_sha256']:raise ValueError('洪水过程摘要已变化')
    source=HydraulicFrames(checked['scene']['path']);env=make_env(source,checked['task'],checked['config']);start=time.perf_counter();meta=None
    if checked['mode']=='rule':
        view=env.reset();total=0.;done=getattr(env,'_done',False)
        while not done:view,reward,done=env.step(rule_actions(view));total+=reward
    else:
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        from sur_rescue.learning import load_actor,execute
        folder,meta=group_record(checked['group']);actor,_=load_actor(folder);total,_=execute(actor,env)
    result=env.result();result.update(schema='hyra-eight-workbench-v1',task=checked['task'],task_label=TASKS[checked['task']],group=checked['group'],requested_group=checked['requested_group'],
        scene={k:v for k,v in checked['scene'].items() if k!='path'},method=('八向规则核查' if meta is None else LABELS[checked['group']]),status='completed',elapsed_s=time.perf_counter()-start,
        created_at=datetime.now(timezone.utc).isoformat(),data_split='interactive_engineering_only_not_formal_results',independent_event_claim=False,
        execution=dict(mode=checked['mode'],kind='eight_action_checkpoint' if meta else 'eight_action_rule',actual_group=checked['group'],
            actor_sha256=meta['actor_sha256'] if meta else None,seed=meta['seed'] if meta else checked['config']['seed'],trained_episodes=meta['trained_episodes'] if meta else None,return_=total,
            scope='规则核查不属于六组模型结果；现有模型仅12回合工程检查点'))
    summary=result['summary'];summary['discovered_targets']=len(env.view().targets)
    if checked['task']=='exploration_rescue':
        summary['time_average_first_coverage']=sum(sum(s['explored'])/len(s['explored']) for s in result['snapshots'][:-1])/max(1,len(result['snapshots'])-1)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args();job=json.loads(a.job.read_text(encoding='utf-8'));id_=job['run_id']
    if len(id_)!=12 or any(c not in '0123456789abcdef' for c in id_):raise ValueError('无效运行编号')
    result=execute_job(job);save_run(result,ROOT/'generated/web'/id_);print(json.dumps(dict(run_id=id_,task=result['task'],group=result['group'],mode=result['execution']['mode'],summary=result['summary'])))
