"""Portable entry point. All subprocesses use this interpreter and package root."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def execute(script, *args):
    subprocess.run([sys.executable, '-u', str(ROOT/'scripts'/script), *map(str,args)], cwd=ROOT, check=True)

def verify():
    checks = read(ROOT/'SHA256SUMS.json')
    bad = [name for name, sha in checks.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha]
    if bad: raise RuntimeError('Checksum mismatch: '+repr(bad))
    print(f'OK: {len(checks)} checked files', flush=True)

def equal(actual, expected, path='', tolerance=1e-9):
    import numpy as np
    if isinstance(expected, dict):
        if set(actual) != set(expected): raise AssertionError(f'{path}: keys differ')
        for key in expected: equal(actual[key], expected[key], path+'/'+key, tolerance)
    elif isinstance(expected, list):
        if len(actual)!=len(expected): raise AssertionError(f'{path}: lengths differ')
        for i,(a,b) in enumerate(zip(actual,expected)): equal(a,b,path+'/'+str(i),tolerance)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not np.isclose(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True):
            raise AssertionError(f'{path}: {actual} != {expected}')
    elif actual != expected: raise AssertionError(f'{path}: values differ')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['verify','test','replay','train','generate','timing','development-summary','development-full'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--smoke-only',action='store_true',help='Training interface check; not the complete five-group experiment')
    parser.add_argument('--check-reference',action='store_true',help='Assert numerical agreement with bundled historical results')
    args=parser.parse_args()
    os.chdir(ROOT)
    (ROOT/'results').mkdir(exist_ok=True)
    experiment=read(ROOT/'EXPERIMENT.json')['id']
    if args.mode=='verify': verify(); return
    if args.mode=='test':
        files=['test_data.py','test_representation.py','test_env.py','test_evaluation.py','test_agent.py','test_training.py'] if experiment==1 else ['test_lead.py','test_lead_diagnostic.py','test_lead_direction.py','test_metric_family.py','test_nominal_response.py','test_metric_vector.py','test_metric_tasks.py']
        import uuid
        temp=ROOT/'results'/'test_tmp'/uuid.uuid4().hex
        temp.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,'-m','pytest','-q','--basetemp',str(temp),*[str(ROOT/'tests'/p) for p in files]],cwd=ROOT,check=True)
        return
    from hyra_validation.artifacts import new_run, write_json
    from hyra_validation.data import load_events
    if experiment==1:
        if args.mode=='replay':
            from hyra_validation.pipeline import run_replay
            run=run_replay(ROOT/'configs/pilot.v2.json')
            if args.check_reference:
                for f in ['replay_scores.json','replay_paired.json','frozen_structure.json']:
                    actual,expected=read(run/f),read(ROOT/'reference/replay'/f)
                    if f=='frozen_structure.json':
                        # Historical text said "events"; current implementation correctly says "families".
                        # Compare all graph values and fitted IDs, retaining original reference text.
                        actual.pop('validation_rule');expected.pop('validation_rule')
                    equal(actual,expected)
            write_json(ROOT/'results/latest_reproduction.json',dict(run=str(run),mode='replay',checked_reference=args.check_reference))
        elif args.mode=='train':
            command=['--device',args.device]
            if args.smoke_only: command+=['--smoke-only']
            execute('run_closed_loop.py',*command)
        elif args.mode=='generate': execute('prepare_events.py')
        else: parser.error('This mode belongs to experiment 2')
    else:
        if args.mode in ('replay','timing'):
            source=ROOT/'data/lead'
            if args.mode=='replay':
                from run_lead_stage1 import information,report
                run=new_run('reproduce_lead',dict(source='data/lead',frozen_graph=True,formal_test=False))
                scores=information(load_events(source/'manifest.json'),read(source/'physical_graph.json')['graph'],run)
                report(run,scores)
                folder='lead'; names=['information_scores.json','comparisons.json','stage_gate.json']
            else:
                from run_lead_timing_diagnostic import coverage,information,comparisons
                run=new_run('reproduce_timing',dict(source='data/lead',formal_test=False))
                write_json(run/'input_source.json',dict(run=str(source)))
                graph=read(source/'physical_graph.json')['graph']; events=load_events(source/'manifest.json')
                coverage(events,graph,run);scores=information(events,graph,run);comparisons(scores,run)
                folder='timing'; names=['information_scores.json','comparisons.json','diagnostic_gate.json']
            if args.check_reference:
                for f in names: equal(read(run/f),read(ROOT/'reference'/folder/f))
            write_json(run/'reproduction_check.json',dict(completed=True,checked_reference=args.check_reference,tolerance=1e-9))
            write_json(ROOT/'results/latest_reproduction.json',dict(run=str(run),mode=args.mode,checked_reference=args.check_reference))
        elif args.mode=='generate': execute('run_lead_stage1.py')
        elif args.mode=='development-summary':
            run=new_run('reproduce_development_summary',dict(source='reference/development',raw_hydraulic_replay=False))
            for p in (ROOT/'reference/development').glob('*.json'):
                if p.name.endswith('_information_action.json'): continue
                shutil.copy2(p,run/p.name)
            execute('summarize_metric_development.py','--run',run)
            equal(read(run/'paired_comparisons.json'),read(ROOT/'reference/development/paired_comparisons.json'))
            print('OK: 12-family development summary reproduced; raw solver/model rerun not included in this mode.',flush=True)
        elif args.mode=='development-full':
            # Sequential execution avoids hidden wait loops and stale latest pointers.
            execute('generate_metric_development.py')
            execute('audit_metric_development_data.py')
            execute('cache_metric_features.py')
            execute('audit_metric_feature_inputs.py')
            execute('train_metric_models.py','--train-count','24','--epochs','120')
            modelrun=read(ROOT/'results/latest_metric_model_training.json')['run']
            execute('evaluate_metric_models.py','--models',modelrun,'--split','validation','--count','12')
            selection=read(ROOT/'results/latest_metric_development_selection.json')['run']
            execute('summarize_metric_development.py','--run',selection)
            execute('check_metric_information_action.py','--selection',selection)
        else: parser.error('This mode belongs to experiment 1')

if __name__=='__main__': main()
