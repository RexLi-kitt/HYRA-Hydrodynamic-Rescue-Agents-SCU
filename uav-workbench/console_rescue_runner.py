import argparse,json,sys,time
from pathlib import Path
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
from sur_rescue.environment import RescueConfig,RescueEnv,rule_actions
from sur_rescue.hydraulics import HydraulicFrames
from hyra_uav.experiments import save_run
from rescue_console import validate,model_record,ROOT

def execute_job(job):
    checked=validate(dict(job,scene=job['scene']['id']))
    if checked['source_sha256']!=job['source_sha256']:raise ValueError('Hydraulic scene changed')
    source=HydraulicFrames(checked['scene']['path']);config=RescueConfig(**checked['rescue_config']);env=RescueEnv(source,config);start=time.perf_counter();meta=None
    if checked['method']=='rescue:rule':
        view=env.reset();total=0.;done=False
        while not done:view,reward,done=env.step(rule_actions(view));total+=reward
    else:
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        from sur_rescue.learning import load_actor,execute
        folder,meta=model_record(checked['method']);actor,_=load_actor(folder);total,_=execute(actor,env)
    result=env.result();result.update(scene={k:v for k,v in checked['scene'].items() if k!='path'},method=checked['method'],status='completed',elapsed_s=time.perf_counter()-start,
        data_split='engineering_interactive_only_not_confirmation',independent_event_claim=False,
        execution=dict(kind='eight_action_clean_actor' if meta else 'eight_action_rule',method=checked['method'],seed=meta['seed'] if meta else config.seed,
            actor_sha256=meta['actor_sha256'] if meta else None,trained_episodes=meta['trained_episodes'] if meta else None,return_=total,scope='dual-GMRF/eight-action engineering implementation; not paper performance reproduction'))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args();job=json.loads(a.job.read_text(encoding='utf-8'));id_=job['run_id']
    if len(id_)!=12 or any(c not in '0123456789abcdef' for c in id_):raise ValueError('Invalid run id')
    replay=execute_job(job);save_run(replay,ROOT/'generated/web'/id_);print(json.dumps(dict(run_id=id_,method=replay['method'],summary=replay['summary'])))
