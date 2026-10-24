"""Evidence-gated freeze and outcome-independent formal test registration."""
from pathlib import Path
from datetime import datetime
import json
from .artifacts import sha256,write_json,TZ
from .execution_plan import register
from .protocol import load_protocol

EXECUTION_FILES=('simulation.py','noise.py','planning.py','windows.py','models.py','model_features.py',
    'learned_predictor.py','links.py','model_execution.py','outcomes.py','records.py','hydraulics.py',
    'common_data.py','diagnostics.py','execution_plan.py','evaluation.py','formal.py','freeze.py','posttest.py',
    'protocol.py','artifacts.py','design.py','statistics.py')


def execution_hashes(root):
    root=Path(root)
    return {name:sha256(root/'src/hyra_exp3'/name) for name in EXECUTION_FILES}


def formal_matrix(plan):
    rows=[]
    for family in sorted((f for f in plan['allocation'] if f['split']=='test'),key=lambda f:f['family_id']):
        for side in (32,50,70):
            methods=plan['methods'] if side==50 else plan['endpoint_methods']
            for delay in ((0.,10.) if side==50 else (0.,)):
                group=methods if delay==0 else plan['delay_methods']
                for seed in plan['training_seeds']:
                    for layout in (0,1):
                        for method in group:
                            identity=f"{family['family_id']}_s{side}_d{delay:g}_seed{seed}_layout{layout}_{method}"
                            rows.append(dict(id=identity,family_id=family['family_id'],side=side,delay_s=delay,
                                training_seed=seed,layout=layout,method=method,hydrology=family['hydrology']))
    if len(rows)!=3024 or len({r['id'] for r in rows})!=3024:
        raise AssertionError('Formal test count differs from original plan')
    return rows


def freeze_configuration(root,training_run,data_run,fixed_run,validation_run,refinement_run,sensitivity_run):
    from .evaluation import checkpoint_registry
    root=Path(root).resolve();plan=register(root)
    paths={k:Path(v).resolve() for k,v in dict(training=training_run,data=data_run,fixed=fixed_run,
        validation=validation_run,refinement=refinement_run,sensitivity=sensitivity_run).items()}
    registry,bank=checkpoint_registry(paths['training'])
    provenance=load_protocol(paths['training']/'data_provenance.json')
    if Path(provenance['summary_path']).resolve()!=paths['data']/'common_data_summary.json' or provenance['sha256']!=sha256(paths['data']/'common_data_summary.json'):
        raise AssertionError('Models were trained on different common records')
    training_sources=load_protocol(paths['training']/'source_sha256.json')
    for name in ('models.py','model_features.py','links.py','training.py'):
        if training_sources[f'src/hyra_exp3/{name}']!=sha256(root/'src/hyra_exp3'/name):
            raise AssertionError('Loaded training code differs from the code being frozen')
    train_ids={f['family_id'] for f in plan['allocation'] if f['split']=='train'}
    fit=set(bank['fit_families']);held=set(bank['held_out_TRAIN_families'])
    if fit&held or fit|held!=train_ids or len(fit)!=18 or len(held)!=6:
        raise AssertionError('Physical/statistical link training-family isolation differs')
    data_audit=load_protocol(paths['data']/'common_data_integrity_audit.json')
    numerical=load_protocol(paths['refinement']/'refinement_integrity_audit.json')
    fixed=load_protocol(paths['fixed']/'fixed_validation_summary.json')
    val=load_protocol(paths['validation']/'validation_summary.json')
    sensitivity=load_protocol(paths['sensitivity']/'sensitivity_summary.json')
    sensitivity_audit_path=paths['sensitivity']/'sensitivity_integrity_audit.json'
    if sensitivity_audit_path.exists():
        sensitivity_audit=load_protocol(sensitivity_audit_path)
    else:
        from .sensitivity_audit import audit_sensitivity
        sensitivity_audit=audit_sensitivity(root,paths['sensitivity'],workers=3)
    if sensitivity_audit['status']!='passed' or sensitivity_audit['episodes']!=10368 or sensitivity_audit['summary_sha256']!=sha256(paths['sensitivity']/'sensitivity_summary.json'):
        raise ValueError('Full sensitivity read-back audit not passed')
    if data_audit['status']!='passed' or data_audit['cases']!=108 or data_audit['test_accessed'] or data_audit['summary_sha256']!=sha256(paths['data']/'common_data_summary.json'):
        raise ValueError('Shared data audit not passed')
    if numerical['status']!='passed' or not numerical['numerical_tolerances_passed'] or numerical['cases']!=216:
        raise ValueError('Full design numerical tolerance gate not passed')
    if fixed['status']!='completed' or len(fixed['records'])!=720 or fixed['test_accessed']:
        raise ValueError('Shared fixed-route validation incomplete')
    if val['status']!='completed' or val['episodes']!=360 or val['test_accessed']:
        raise ValueError('Complete five-method closed-loop validation required')
    val_ids={f['family_id'] for f in plan['allocation'] if f['split']=='validation'}
    if {r['family_id'] for r in val['rows']}!=val_ids:
        raise AssertionError('Wrong validation families')
    if sensitivity['status']!='completed' or sensitivity['episodes']!=10368 or sensitivity['default_changed_by_outcomes']:
        raise ValueError('Prespecified planner sensitivity not completed')
    selected=val['B_star']
    if load_protocol(paths['validation']/'B_star_selection.json')!=selected:
        raise AssertionError('Saved B* selection differs from validation summary')
    if load_protocol(paths['fixed']/'arguments.json')!=dict(training_run=str(paths['training']),data_run=str(paths['data'])):
        raise AssertionError('Fixed-route replay used different training/data sources')
    if load_protocol(paths['validation']/'arguments.json')!=dict(training_run=str(paths['training']),data_run=str(paths['data']),fixed_run=str(paths['fixed'])):
        raise AssertionError('Closed-loop validation used different training/data sources')
    if selected['B_star'] not in ('H','E') or selected['P_or_test_results_used']:
        raise AssertionError('Invalid B* selection')
    source_files={
        'training':paths['training']/'training_summary.json','links':paths['training']/'frozen_training_links.json',
        'training_audit':paths['training']/'training_integrity_audit.json',
        'data_audit':paths['data']/'common_data_integrity_audit.json',
        'numerical_audit':paths['refinement']/'refinement_integrity_audit.json',
        'fixed_validation':paths['fixed']/'fixed_validation_summary.json',
        'closed_loop_validation':paths['validation']/'validation_summary.json',
        'B_star':paths['validation']/'B_star_selection.json',
        'planner_sensitivity':paths['sensitivity']/'sensitivity_summary.json',
        'planner_sensitivity_audit':sensitivity_audit_path}
    frozen=dict(schema='hyra3-frozen-formal-v1',configuration_frozen=True,formal_test_enabled=True,
        plan_path=str(root/'manifests/full_execution_plan.v1.json'),plan_sha256=sha256(root/'manifests/full_execution_plan.v1.json'),
        B_star=selected['B_star'],B_star_selection=selected,checkpoints=[registry[k] for k in sorted(registry)],
        parameter_count=37070,hydraulic_mesh_m=5.,hydraulic_output_interval_s=1.,
        primary_confirmatory_group=dict(side=50,delay_s=0.),
        mission_start_s=0.,mission_duration_s=1200.,forecast_horizon_s=60.,history_s=60.,noise_mode='counter_keyed',
        planner=dict(beam_width=6,frontier_limit=16,solver_budget_s=.02,planner_budget_s=.1),
        formal_matrix=formal_matrix(plan),planned_main_episodes=2376,planned_delayed_episodes=648,planned_total_episodes=3024,
        evidence={k:dict(path=str(p),sha256=sha256(p)) for k,p in source_files.items()},
        source_hashes=execution_hashes(root),test_results_used_for_freeze=False,
        statistics=dict(bootstrap_unit=['family','training_seed'],layouts_averaged_first=True,replicates=10000,seed=2026101803),
        counterfactual='defer_only_feasible_nonperiodic_new_evidence_task_switches_until_next_periodic_replan',
        numerical_gain_recheck='same_frozen_P_and_B_star_on_independent_coarse_10m2s_fields_with_identical_world_noise')
    path=root/'configs/frozen_experiment3.v1.json'
    if path.exists():
        if load_protocol(path)!=frozen:raise ValueError('A different configuration is already frozen')
    else:write_json(path,frozen)
    seal_path=root/'registrations/frozen_configuration.v1.json'
    if not seal_path.exists():
        if (root/'registrations/formal_test.v1.json').exists():
            raise ValueError('Cannot create a first freeze seal after independent test registration')
        seal_path.parent.mkdir(parents=True,exist_ok=True)
        seal=dict(schema='hyra3-pretest-freeze-seal-v1',path=str(path),sha256=sha256(path),
            sealed_at=datetime.now(TZ).isoformat(),before_independent_test_registration=True)
        with seal_path.open('x',encoding='utf-8') as stream:
            json.dump(seal,stream,ensure_ascii=False,indent=2);stream.write('\n')
    sealed=load_protocol(seal_path)
    if sealed['sha256']!=sha256(path) or Path(sealed['path']).resolve()!=path.resolve():
        raise AssertionError('Configuration differs from the original pre-test freeze seal')
    return path


def verify_frozen(root,path):
    root,path=Path(root),Path(path)
    frozen=load_protocol(path)
    seal=load_protocol(root/'registrations/frozen_configuration.v1.json')
    if not seal['before_independent_test_registration'] or seal['sha256']!=sha256(path) or Path(seal['path']).resolve()!=path.resolve():
        raise AssertionError('Frozen configuration changed after its pre-test seal')
    if not frozen.get('configuration_frozen') or not frozen.get('formal_test_enabled') or frozen['test_results_used_for_freeze']:
        raise ValueError('Configuration is not eligible for independent test')
    if sha256(frozen['plan_path'])!=frozen['plan_sha256'] or execution_hashes(root)!=frozen['source_hashes']:
        raise AssertionError('Frozen plan or execution code changed')
    for record in frozen['evidence'].values():
        if sha256(record['path'])!=record['sha256']:raise AssertionError('Pre-test evidence changed')
    for record in frozen['checkpoints']:
        if sha256(record['checkpoint_path'])!=record['checkpoint_sha256']:raise AssertionError('Frozen model changed')
    selected=load_protocol(frozen['evidence']['B_star']['path'])
    if frozen['B_star_selection']!=selected or frozen['B_star']!=selected['B_star']:
        raise AssertionError('Frozen B* differs from the single validation selection')
    if frozen['formal_matrix']!=formal_matrix(register(root)):
        raise AssertionError('Independent test matrix changed')
    if (frozen['hydraulic_mesh_m'],frozen['hydraulic_output_interval_s'],frozen['mission_start_s'],
        frozen['mission_duration_s'],frozen['noise_mode'])!=(5.,1.,0.,1200.,'counter_keyed'):
        raise AssertionError('Original mission clock or registered numerical settings changed')
    return frozen
