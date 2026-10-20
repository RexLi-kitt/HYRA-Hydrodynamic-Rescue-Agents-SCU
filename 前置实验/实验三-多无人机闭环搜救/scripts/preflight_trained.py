"""Exercise all frozen training outputs on one training scene, without test access."""
from pathlib import Path
import argparse
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.artifacts import new_run,finish_run,write_json,sha256
from hyra_exp3.protocol import load_protocol
from hyra_exp3.execution_plan import register
from hyra_exp3.diagnostics import load_field
from hyra_exp3.evaluation import checkpoint_registry,load_models,configure_device,config_for
from hyra_exp3.model_execution import execute_network_episode
from hyra_exp3.formal_audit import audit_saved_episode

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-run',type=Path,required=True)
    parser.add_argument('--training-run',type=Path,required=True)
    args=parser.parse_args()
    candidates=[(p,load_protocol(p)) for p in sorted((args.data_run/'jobs').glob('*side70/case.json'))]
    case_path,case=next((p,c) for p,c in candidates if c['family']['split']=='train')
    registry,bank=checkpoint_registry(args.training_run)
    device=configure_device();models=load_models(registry,device)
    target=new_run(ROOT,'trained_interface_preflight',dict(data_run=str(args.data_run.resolve()),
        training_run=str(args.training_run.resolve()),case=str(case_path.resolve()),
        training_scene_only=True,test_accessed=False,not_model_effect_evidence=True))
    print(f'Trained interface preflight directory: {target}',flush=True)
    try:
        truth=load_field(case['truth']['manifest']['field_path'])
        nominal=load_field(case['nominal']['manifest']['field_path'])
        config=config_for(register(ROOT),case['family']['family_id'],70,0)
        rows=[]
        for key,model in models.items():
            directory=target/f'{key[0]}_seed{key[1]}'
            provenance=dict(**registry[key],training_case_path=str(case_path.resolve()),
                training_case_sha256=sha256(case_path),training_scene_only=True,test_accessed=False)
            result=execute_network_episode(directory,truth,nominal,config,model,bank,key[0],device=device,
                trained=True,provenance=provenance,experiment_role='training_scene_interface_preflight')
            audit=dict(status='passed',**audit_saved_episode(directory/'episode.json',truth,config))
            write_json(directory/'serialized_integrity_audit.json',audit)
            rows.append(dict(method=key[0],training_seed=key[1],summary=result['summary'],audit=audit,
                episode=dict(path=str(directory/'episode.json'),sha256=sha256(directory/'episode.json'))))
            write_json(target/'progress.json',dict(episodes=len(rows),planned_episodes=15))
            print(f'Trained interface preflight {len(rows)}/15: {key[0]} seed{key[1]}, independent audit passed',flush=True)
        write_json(target/'trained_interface_preflight_summary.json',dict(status='passed',episodes=15,
            family_id=case['family']['family_id'],split='train',side=70,test_accessed=False,
            training_scene_only=True,not_model_effect_evidence=True,rows=rows))
        finish_run(ROOT,target)
    except BaseException as error:
        (target/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(ROOT,target,status='failed',error=repr(error))
        raise
