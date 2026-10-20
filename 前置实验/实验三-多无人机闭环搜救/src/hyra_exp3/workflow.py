"""Persistent evidence-gated sequential workflow; each heavy stage has a fresh process."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

from .artifacts import new_run,finish_run,write_json
from .protocol import load_protocol


def wait_for_prerequisites(paths,progress_path):
    expected={'data':'common_data_summary.json','training':'training_summary.json',
              'refinement':'refinement_summary.json','sensitivity':'sensitivity_summary.json'}
    previous=None
    while True:
        status={}
        for name,path in paths.items():
            directory=Path(path);state=load_protocol(directory/'state.json')
            if state['status']=='failed':raise RuntimeError(f'Prerequisite {name} failed: {directory}')
            ready=(directory/expected[name]).exists()
            if ready and load_protocol(directory/expected[name])['status']!='completed':ready=False
            progress=load_protocol(directory/'progress.json') if (directory/'progress.json').exists() else {}
            status[name]=dict(ready=ready,progress=progress)
        encoded=json.dumps(status,sort_keys=True)
        if encoded!=previous:
            write_json(progress_path,dict(stage='waiting_for_prerequisites',sources=status))
            print('Full workflow prerequisites: '+', '.join(f'{k}={"ready" if v["ready"] else "running"}' for k,v in status.items()),flush=True)
            previous=encoded
        if all(v['ready'] for v in status.values()):return
        time.sleep(10)


def execute_stage(root,action,arguments):
    """Called by the one-stage subprocess, so CUDA and large arrays are released afterwards."""
    if action=='data_audit':
        from .data_audit import audit_common_data
        from .artifacts import sha256
        data=Path(arguments['data']);proof=data/'common_data_integrity_audit.json'
        if proof.exists():
            audit=load_protocol(proof)
            if audit['status']=='passed' and audit['cases']==108 and not audit['test_accessed'] and audit['summary_sha256']==sha256(data/'common_data_summary.json'):
                from .export import verify_record_references
                verify_record_references(root,{data/'common_data_summary.json',*(data/'jobs').glob('*/case.json')})
                return str(data)
        audit_common_data(root,arguments['data'],workers=3);return arguments['data']
    if action=='refinement_audit':
        from .refinement_audit import audit_refinement
        result=audit_refinement(root,arguments['refinement'],workers=3)
        if not result['numerical_tolerances_passed']:raise RuntimeError('Design numerical tolerance gate failed; formal testing stays closed')
        return arguments['refinement']
    if action=='fixed':
        from .evaluation import evaluate_fixed
        return evaluate_fixed(root,arguments['training'],arguments['data'],resume=arguments.get('resume'))
    if action=='validation':
        from .evaluation import validate_closed_loop
        return validate_closed_loop(root,arguments['training'],arguments['data'],arguments['fixed'],resume=arguments.get('resume'))
    if action=='freeze':
        from .freeze import freeze_configuration
        return freeze_configuration(root,arguments['training'],arguments['data'],arguments['fixed'],arguments['validation'],arguments['refinement'],arguments['sensitivity'])
    if action=='primary':
        from .formal import run_formal
        return run_formal(root,arguments['frozen'],workers=6,resume=arguments.get('resume'))
    if action in ('reassignment_counterfactual','numerical_direction'):
        from .posttest import run_posttest
        return run_posttest(root,arguments['frozen'],action,workers=4,resume=arguments.get('resume'))
    if action=='formal_audit':
        from .formal_audit import audit_formal
        audit_formal(root,arguments['primary'],arguments['frozen'],workers=3)
        return arguments['primary']
    if action=='final_package':
        from .final_report import build_final_package
        return build_final_package(root,arguments)
    raise ValueError(f'Unknown workflow stage {action}')


def run_workflow(root,sources,resume=None):
    root=Path(root).resolve();sources={k:str(Path(v).resolve()) for k,v in sources.items()}
    if resume is None:
        run=new_run(root,'full_experiment_workflow',sources)
    else:
        run=Path(resume).resolve()
        if not run.is_relative_to(root/'results/runs') or load_protocol(run/'arguments.json')!=sources:
            raise ValueError('Workflow resume sources differ')
    print(f'Full experiment workflow directory: {run}',flush=True)
    try:
        wait_for_prerequisites(sources,run/'progress.json')
        arguments=dict(sources)
        for stage in ('data_audit','refinement_audit','fixed','validation','freeze','primary',
                      'formal_audit','reassignment_counterfactual','numerical_direction','final_package'):
            marker=run/'stages'/f'{stage}.json'
            if marker.exists():
                record=load_protocol(marker)
                if record['status']=='completed':
                    if stage=='freeze':arguments['frozen']=record['result']
                    elif stage not in ('data_audit','refinement_audit','formal_audit'):arguments[stage]=record['result']
                    continue
            request=run/'requests'/f'{stage}.json'
            stage_args=dict(arguments)
            # Recover a uniquely registered/previously announced run rather than starting another test.
            if stage=='primary' and (root/'registrations/formal_test.v1.json').exists():
                stage_args['resume']=load_protocol(root/'registrations/formal_test.v1.json')['run']
            elif stage in ('reassignment_counterfactual','numerical_direction') and (root/'registrations'/f'{stage}.v1.json').exists():
                stage_args['resume']=load_protocol(root/'registrations'/f'{stage}.v1.json')['run']
            elif stage in ('fixed','validation'):
                announced=run/'stages'/f'{stage}_announced.json'
                if announced.exists():stage_args['resume']=load_protocol(announced)['run']
            write_json(request,dict(root=str(root),action=stage,arguments=stage_args,result_path=str(marker)))
            write_json(run/'progress.json',dict(stage=stage,status='running',all_prerequisites_complete=True))
            print(f'Full experiment workflow starting: {stage}',flush=True)
            environment=dict(os.environ)
            environment['HYRA3_STAGE_ANNOUNCEMENT']=str(run/'stages'/f'{stage}_announced.json')
            process=subprocess.run([sys.executable,str(root/'scripts/execute_stage.py'),str(request)],cwd=root,check=False,env=environment,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if process.returncode!=0:raise RuntimeError(f'Workflow stage {stage} exited with code {process.returncode}')
            record=load_protocol(marker)
            if record['status']!='completed':raise RuntimeError(f'Workflow stage {stage} did not finish')
            if stage=='freeze':arguments['frozen']=record['result']
            elif stage not in ('data_audit','refinement_audit','formal_audit'):arguments[stage]=record['result']
        write_json(run/'workflow_summary.json',dict(status='completed',results=arguments,objective_completion_requires_package_audit=True))
        finish_run(root,run)
        return run
    except BaseException as error:
        (run/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(root,run,status='failed',error=repr(error));raise
