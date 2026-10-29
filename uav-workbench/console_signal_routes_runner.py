"""UI-only adapter; frozen study sources, parameters and checkpoints stay intact."""
from runtime_assets import canonical_sources
import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime-assets/frozen-sources/uav-workbench'))
import signal_routes
import numpy as np
import torch
from console_signal_routes_contract import STUDY,ROOT,LABELS,read,digest,require,validate,model_record
from console_scale100_runner import assemble,scalar
from signal_routes.environment import RouteEnv
from signal_routes.belief import Parameters
from signal_routes.common import sources
from scale100_train import FloodCase,config
from scale100_backend import make_actor
from hyra_uav.experiments import save_run


class RecordedRouteEnv(RouteEnv):
    def reset(self):
        self.ui_signal_estimates={}
        return super().reset()

    def view(self):
        value=super().view()
        self.ui_signal_estimates[self.time_s]=(self.signal if self.estimator is None else self.signal_belief).copy()
        return value


def execute_job(job):
    checked=validate(job)
    if checked['mode']=='rule':
        result=assemble(checked,contract_id='signal-routes-900s-v1')
        result['execution']['scope']='三路径场景上的独立规则核查；不执行所选路径模型'
        return result
    path,meta,selected=model_record(checked['group'],checked['training_seed'])
    require(canonical_sources(meta['sources'])==canonical_sources(sources()), '冻结训练源码与当前估计器不匹配')
    p=Parameters(**read(STUDY/'parameters.json')['parameters'][checked['group']])
    source=FloodCase(checked['scene']['path'])
    env=RecordedRouteEnv(source,replace(config('E2','B3',checked['config']['seed']),duration_s=900.),checked['group'],p)
    actor=make_actor('aux_resilient_v4'); actor.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')); actor.eval()
    ui_meta={**meta,'formal':True,'selected_episode':selected['episode'],'total_training_episodes_executed':1024}
    result=assemble(checked,actor,ui_meta,environment=env,method_label=LABELS[checked['group']],contract_id='signal-routes-900s-v1')
    # Independently replay the actual actions through the unmodified study environment.
    from sur_rescue.environment import ACTIONS
    reference=RouteEnv(source,env.config,checked['group'],p); reference.reset(); total=0.
    for action in env.actions:
        _,reward,_=reference.step([ACTIONS.index(a) for a in action['actions']]); total+=reward
    require(reference.observations==env.observations and reference.events==env.events and total==result['execution']['return_'],
            '工作台适配改变了原始记录、事件或收益')
    require(np.array_equal(reference.view().images,env.view().images), '工作台适配改变了模型输入')
    for snapshot in result['snapshots']:
        snapshot['signal_posterior']=env.ui_signal_estimates[snapshot['time_s']].ravel().tolist()
    result['audit']['frozen_route_replay_equal']=True
    result['execution'].update(kind='signal_routes_checkpoint',signal_route=checked['group'],
        parameters_sha256=digest(STUDY/'parameters.json'),model_seal_sha256=digest(STUDY/'sealed-models.json'),
        adapter_sources={name:digest(ROOT/name) for name in ('console_signal_routes_contract.py',
            'console_signal_routes_runner.py','console_scale100_runner.py')},
        scope='冻结三路径模型的交互回放；本轮仅作描述性研究，未证实物理耦合收益')
    return json.loads(json.dumps(result,ensure_ascii=False,allow_nan=False,default=scalar))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--job',required=True); parser.add_argument('--run-id',required=True)
    args=parser.parse_args(); require(len(args.run_id)==12 and all(c in '0123456789abcdef' for c in args.run_id),'无效运行编号')
    result=execute_job(read(args.job)); save_run(result,ROOT/'generated/web'/args.run_id)
    print(json.dumps(dict(run_id=args.run_id,group=result['group'],audit_passed=result['audit']['audit_passed'])))
