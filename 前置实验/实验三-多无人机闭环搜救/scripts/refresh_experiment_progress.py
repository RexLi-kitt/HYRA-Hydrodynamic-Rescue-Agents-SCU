"""Snapshot artifact counts; execution handles, not this file, establish job liveness."""
from pathlib import Path
import sys
from datetime import datetime
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.artifacts import write_json,TZ,sha256
from hyra_exp3.protocol import load_protocol
if __name__=='__main__':
    status=load_protocol(ROOT/'RUNNING.json')
    workflow_handle=status['active_work'].get('full_workflow',{}).get('session_id',69315)
    for kind in ('reassignment_counterfactual','numerical_direction'):
        registration=ROOT/'registrations'/f'{kind}.v1.json'
        if registration.exists():
            status['active_work'][kind]=dict(run=load_protocol(registration)['run'],session_id=workflow_handle,supervised_by_workflow=True)
    for name,entry in status['active_work'].items():
        directory=Path(entry['run']);entry['state']=load_protocol(directory/'state.json')
        if entry['state']['status'] in ('completed','failed'):entry['execution_handle_status']='terminal'
        if name=='design_refinement':
            rows=[load_protocol(p) for p in (directory/'jobs').glob('*/condition_*/refinement_case.json')]
            entry.update(completed_cases=len(rows),planned_cases=216,tolerance_failures_so_far=sum(not r['tolerances_passed'] for r in rows))
        elif name=='common_records':entry.update(completed_cases=len(list((directory/'jobs').glob('*/case.json'))),planned_cases=108)
        elif name=='model_training':
            entry['completed_models']=len(list((directory/'models').glob('*/training_result.json')))
            status['models_trained']=entry['completed_models']
            if (directory/'progress.json').exists():entry['progress']=load_protocol(directory/'progress.json')
            entry['per_model_progress']=[dict(**load_protocol(p),completed=(p.parent/'training_result.json').exists()) for p in sorted((directory/'models').glob('*/progress.json'))]
            entry['training_link_bank_exists']=(directory/'frozen_training_links.json').exists()
        elif name=='planner_sensitivity':
            entry.update(completed_episodes=sum(load_protocol(p)['episodes'] for p in (directory/'jobs').glob('*/progress.json')),planned_episodes=10368)
        elif name=='fixed_route_validation':
            progress=directory/'progress.json'
            entry.update(completed_records=load_protocol(progress)['records'] if progress.exists() else 0,planned_records=720)
        elif name=='closed_loop_validation':
            progress=directory/'progress.json'
            entry.update(completed_episodes=load_protocol(progress)['episodes'] if progress.exists() else 0,planned_episodes=360)
        elif name=='primary_test':
            progress=directory/'progress.json'
            fields=directory/'field_progress.json'
            entry.update(completed_field_cases=load_protocol(fields)['cases'] if fields.exists() else 0,planned_field_cases=108,
                completed_episodes=load_protocol(progress)['episodes'] if progress.exists() else 0,planned_episodes=3024)
            entry['stage']='preparing_independent_test_fields' if entry['completed_field_cases']<108 else (
                'running_frozen_primary_episodes' if entry['completed_episodes']<3024 else 'primary_complete_awaiting_audit_and_diagnostics')
            proof=directory/'formal_integrity_audit.json'
            if entry['completed_episodes']==3024 and proof.exists() and load_protocol(proof)['status']=='passed':
                entry['stage']='primary_audited_diagnostics_pending'
        elif name in ('reassignment_counterfactual','numerical_direction'):
            progress=directory/'progress.json'
            entry.update(completed_episodes=load_protocol(progress)['episodes'] if progress.exists() else 0,planned_episodes=1728)
            if name=='numerical_direction':
                fields=directory/'field_progress.json'
                entry.update(completed_field_cases=load_protocol(fields)['cases'] if fields.exists() else 0,planned_field_cases=108)
    workflow=ROOT/'results/runs/20261018_235201_35018c2a_full_experiment_workflow'
    status['active_work']['full_workflow']=dict(run=str(workflow),session_id=workflow_handle,state=load_protocol(workflow/'state.json'),
        progress=load_protocol(workflow/'progress.json'))
    tests=sorted((ROOT/'results/runs').glob('*_test/test_result.json'),reverse=True)
    if tests:
        test_run=tests[0].parent
        suite=ET.parse(test_run/'tests.xml').getroot().find('testsuite')
        status['latest_tests']=int(suite.attrib['tests'])
        status['latest_tests_passed']=load_protocol(test_run/'test_result.json')['passed']
        status['latest_test_run']=str(test_run)
    status['pending_execution']=['complete_training_and_pretest_gates','fixed_record_and_closed_loop_validation',
        'freeze_once_then_3024_independent_test','paired_reassignment_and_numerical_diagnostics','independent_audits_report_and_verified_export']
    status.pop('next_implementation',None)
    status['recorded_at']=datetime.now(TZ).isoformat()
    status['formal_test_started']=(ROOT/'registrations/formal_test.v1.json').exists()
    if status['formal_test_started']:
        status['pending_execution']=['complete_3024_frozen_independent_test','independent_primary_audit',
            'paired_reassignment_and_numerical_diagnostics','independent_audits_report_and_verified_export']
        primary=status['active_work'].get('primary_test',{})
        if primary.get('completed_episodes')==3024:
            status['pending_execution'].remove('complete_3024_frozen_independent_test')
            proof=Path(primary['run'])/'formal_integrity_audit.json'
            if proof.exists() and load_protocol(proof)['status']=='passed':status['pending_execution'].remove('independent_primary_audit')
    status['learned_interface_preflight']=dict(run=str(ROOT/'results/runs/20261018_233627_7cd1d01b_learned_interface_preflight'),
        trained_model=False,successful_services=60,planning_calls=137,planning_timeouts=2,median_planning_ms=33.1944,
        p95_planning_ms=67.5762,max_planning_ms=104.3105,service_and_curve_audits_passed=True)
    latest_preflight=sorted((ROOT/'results/runs').glob('*_learned_interface_preflight/serialized_integrity_audit.json'),reverse=True)
    if latest_preflight:
        directory=latest_preflight[0].parent
        status['latest_serialized_interface_preflight']=dict(run=str(directory),arguments=load_protocol(directory/'arguments.json'),
            audit=load_protocol(latest_preflight[0]),summary=load_protocol(directory/'interface_preflight_summary.json')['summary'])
    gpu_checks=sorted((ROOT/'results/runs').glob('*_gpu_training_preflight/gpu_training_preflight_summary.json'),reverse=True)
    if gpu_checks:
        status['gpu_training_preflight']=dict(run=str(gpu_checks[0].parent),summary=load_protocol(gpu_checks[0]))
    write_json(ROOT/'RUNNING.json',status)
    write_json(ROOT/'NEXT_STEP.json',dict(stage='full_experiment_execution_in_progress',objective='完成实验三',
        workflow=str(workflow),workflow_session_id=workflow_handle,current_work=status['active_work'],pending_execution=status['pending_execution'],
        formal_test_started=status['formal_test_started'],reselect_by_P_gain=False,recorded_at=status['recorded_at']))
    ledger=load_protocol(ROOT/'COMPLETION.json')
    if not ledger.get('complete') and ledger.get('export_status') not in ('building','verified'):
        ledger['automated_workflow']=str(workflow)
        training=status['active_work']['model_training']
        training_run=Path(training['run'])
        proof=training_run/'training_integrity_audit.json'
        if training.get('per_model_progress'):
            ledger['requirements']['five_parameter_matched_models_three_training_seeds']['status']='training_in_progress'
        if training.get('training_link_bank_exists'):
            ledger['requirements']['training_only_statistical_and_validated_physical_links_and_lags']['status']='fitted_awaiting_independent_training_audit'
        if proof.exists():
            audit=load_protocol(proof)
            if (audit['status']=='passed' and audit['trained_models']==15 and audit['checkpoint_files']==45
                and not audit['test_accessed'] and audit['summary_sha256']==sha256(training_run/'training_summary.json')
                and audit['recipe_sha256']==sha256(training_run/'training_recipe.json')
                and audit['link_bank_sha256']==sha256(training_run/'frozen_training_links.json')):
                evidence=[dict(path=str(proof.relative_to(ROOT)),sha256=sha256(proof))]
                for key in ('five_parameter_matched_models_three_training_seeds',
                    'training_only_statistical_and_validated_physical_links_and_lags'):
                    ledger['requirements'][key]=dict(status='verified',evidence=evidence)
                ledger['models_trained']=True
        ledger['latest_clock_interface_tests']=dict(passed=status['latest_tests'],run=status['latest_test_run'])
        for key,work,proof_name,summary_name,count,expected in (
            ('fixed_route_identical_record_prediction_and_window_evaluation','fixed_route_validation',
             'fixed_record_integrity_audit.json','fixed_validation_summary.json','records',720),
            ('planner_candidate_and_budget_sensitivity','planner_sensitivity',
             'sensitivity_integrity_audit.json','sensitivity_summary.json','episodes',10368)):
            if work in status['active_work']:
                directory=Path(status['active_work'][work]['run']);proof=directory/proof_name
                if proof.exists():
                    audit=load_protocol(proof)
                    if audit['status']=='passed' and audit[count]==expected and audit['summary_sha256']==sha256(directory/summary_name):
                        ledger['requirements'][key]=dict(status='verified',evidence=[dict(path=proof.relative_to(ROOT).as_posix(),sha256=sha256(proof))])
        for key in ('fixed_route_identical_record_prediction_and_window_evaluation','validation_B_star_and_configuration_freeze',
            'single_independent_3024_episode_test','paired_family_and_training_seed_95_percent_intervals_and_3pp_target',
            'causal_reassignment_counterfactuals','numerical_direction_stability_of_final_gain',
            'independent_reproducible_package_frozen_configuration_raw_statistics_final_report'):
            if ledger['requirements'][key]['status'] not in ('verified','requires_numerical_investigation'):
                ledger['requirements'][key]['status']='implemented_awaiting_execution_and_independent_verification'
        if 'primary_test' in status['active_work']:
            primary=status['active_work']['primary_test']
            ledger['formal_test_episodes']=primary['completed_episodes']
            if ledger['requirements']['single_independent_3024_episode_test']['status']!='verified':
                ledger['requirements']['single_independent_3024_episode_test']['status']=primary['stage']
        write_json(ROOT/'COMPLETION.json',ledger)
    print({k:{f:v for f,v in item.items() if f.startswith('completed') or f.startswith('tolerance')} for k,item in status['active_work'].items()})
