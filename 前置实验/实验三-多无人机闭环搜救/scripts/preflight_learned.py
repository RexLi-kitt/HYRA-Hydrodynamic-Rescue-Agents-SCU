"""Test neural interface timing on an actual training scene; never model evidence."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
import numpy as np
import torch
from hyra_exp3.artifacts import new_run,finish_run,write_json
from hyra_exp3.protocol import load_protocol
from hyra_exp3.diagnostics import load_field
from hyra_exp3.models import SharedForecastBackbone
from hyra_exp3.simulation import EpisodeConfig
from hyra_exp3.model_execution import execute_network_episode
from hyra_exp3.formal_audit import audit_saved_episode
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-run',type=Path,required=True)
    parser.add_argument('--method',choices=['H','E','P','P_wrong_links','P_uniform_time'],default='H')
    parser.add_argument('--noise-mode',choices=['legacy_stream','counter_keyed'],default='counter_keyed')
    args=parser.parse_args()
    case=load_protocol(next((args.data_run/'jobs').glob('*side70/case.json')))
    run_path=new_run(ROOT,'learned_interface_preflight',dict(data_run=str(args.data_run),trained_model=False,method=args.method,noise_mode=args.noise_mode))
    print(f'Interface preflight directory: {run_path}',flush=True)
    try:
        truth=load_field(case['truth']['manifest']['field_path'])
        nominal=load_field(case['nominal']['manifest']['field_path'])
        z=np.zeros((16,16)).tolist()
        bank=dict(banks={'70':dict(train_gain=z,held_out_train_gain=z,lag_steps=z,held_out_train_improving_families=z)})
        torch.manual_seed(101)
        model=SharedForecastBackbone()
        device='cuda' if torch.cuda.is_available() else 'cpu'
        config=EpisodeConfig(map_shape=(70,70),n_drones=20,n_targets=60,seed=case['panels'][0]['seed'],noise_mode=args.noise_mode)
        result=execute_network_episode(run_path,truth,nominal,config,model,bank,args.method,device=device,
                                      trained=False,provenance=dict(label='random_weights_not_a_trained_H_result'))
        write_json(run_path/'interface_preflight_summary.json',dict(summary=result['summary'],audit=result['audit'],
            trained_model=False,device_warmup_s=result['published_forecast_log']['warmup_wall_s']))
        write_json(run_path/'serialized_integrity_audit.json',dict(status='passed',**audit_saved_episode(run_path/'episode.json',truth),
            trained_model=False,not_model_effect_evidence=True))
        finish_run(ROOT,run_path)
        print(result['summary'],flush=True)
    except BaseException as error:
        import traceback
        (run_path/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(ROOT,run_path,status='failed',error=repr(error))
        raise
