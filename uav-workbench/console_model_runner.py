"""Inference-only worker for the model console; no trainer/optimizer/world head."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import torch
from hyra_learning.networks import SearchAgent
from hyra_learning.team_env import TeamEnv,TeamConfig
from hyra_learning.team_rules import TeamRules
from hyra_uav.core import Config,FloodProcess
from hyra_uav.experiments import save_run
from console_app import model_record,validate_team_request,read,digest,STUDY,MODELS,TEAM_RULES
from tools.run_team_policy import execute

def execute_job(job):
    checked=validate_team_request(dict(job,scene=job['scene']['id']))
    if checked['source_sha256']!=job['source_sha256']:raise ValueError('洪水输入摘要已变化')
    config=Config(**checked['config']);process=FloodProcess(checked['scene']['path']);team=TeamConfig()
    method=checked['method'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=None;start=time.perf_counter()
    if method in MODELS:
        plan=read(STUDY/'formal-plan.json')
        for path,h in plan['sources'].items():
            if digest(Path(path))!=h:raise ValueError('冻结的训练/执行源码已变化：'+path)
        model=model_record(method,checked['seed']);weights=torch.load(model['path'],map_location='cpu',weights_only=True)
        if not weights or not all(isinstance(v,torch.Tensor) for v in weights.values()):raise ValueError('不是actor-only导出文件')
        actor=SearchAgent();actor.load_state_dict(weights,strict=True)
        env,total,_=execute(actor,process,config,team)
    else:
        env=TeamEnv(process,config,team);view,_=env.reset();policy=TeamRules(TEAM_RULES[method],view,config,team.separation_m)
        total=0.;done=False
        while not done:
            view,reward,terminated,truncated,_=env.step(policy.choose(view));total+=reward;done=terminated or truncated
    replay=env.result();scene={k:v for k,v in checked['scene'].items() if k!='path'}
    replay.update(scene=scene,method=method,status='completed',data_split='interactive_demonstration_not_new_independent_evidence',independent_event_claim=False,
        elapsed_s=time.perf_counter()-start,
        execution=dict(kind='frozen_clean_actor' if model else 'paired_rule',method=method,seed=checked['seed'] if model else None,
            actor_sha256=model['actor_sha256'] if model else None,selected_checkpoint_sha256=model['selected_checkpoint_sha256'] if model else None,
            selected_episode=model['selected_episode'] if model else None,device='cpu',return_=total,
            permissions='Actor receives only legal current TeamView encoding/poses,previous action and recurrent state. No truth,source/future labels,mixer,world head or optimizer inputs.'),
        console_version=dict(sources={name:digest(ROOT/name) for name in ['console_app.py','console_model_runner.py','console-static/app.js','console-static/index.html']},python=sys.version,torch=str(torch.__version__)))
    return replay

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args();job=read(args.job)
    run_id=job['run_id']
    if len(run_id)!=12 or any(c not in '0123456789abcdef' for c in run_id):raise ValueError('无效运行编号')
    replay=execute_job(job);save_run(replay,ROOT/'generated/web'/run_id)
    print(json.dumps(dict(run_id=run_id,method=replay['method'],agents=2,actions=len(replay['actions']),actor_sha256=replay['execution']['actor_sha256']),ensure_ascii=False))
if __name__=='__main__':main()
