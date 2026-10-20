"""A single registered independent test; restart only through exact resume."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime
import contextlib
import json
import time

from .artifacts import new_run,finish_run,write_json,sha256,TZ
from .design import _cached_field
from .execution_plan import register
from .freeze import verify_frozen
from .protocol import load_protocol


def reserve_test(root,frozen_path,resume=None):
    root,frozen_path=Path(root).resolve(),Path(frozen_path).resolve()
    frozen=verify_frozen(root,frozen_path)
    registration=root/'registrations/formal_test.v1.json'
    if resume is not None:
        run=Path(resume).resolve();registered=load_protocol(registration)
        if not run.is_relative_to(root/'results/runs') or registered['run']!=str(run) or registered['frozen_sha256']!=sha256(frozen_path):
            raise ValueError('Resume is not the originally registered frozen test')
        if load_protocol(run/'formal_registration.json')!=registered:
            raise AssertionError('Formal registration copy differs')
        return run,frozen
    if registration.exists():
        raise ValueError('Independent test already registered; use its exact --resume path')
    run=new_run(root,'formal_test',dict(frozen_config=str(frozen_path)),formal_test=True)
    record=dict(schema='hyra3-single-independent-test-v1',run=str(run),frozen_path=str(frozen_path),
        frozen_sha256=sha256(frozen_path),registered_at=datetime.now(TZ).isoformat(),planned_primary_episodes=3024,
        registered_before_any_test_field=True)
    registration.parent.mkdir(parents=True,exist_ok=True)
    with registration.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    write_json(run/'formal_registration.json',record)
    write_json(run/'formal_matrix.json',frozen['formal_matrix'])
    return run,frozen


def _field_guard(root,run,frozen_path,role):
    root,run=Path(root),Path(run)
    frozen=verify_frozen(root,frozen_path)
    registered=load_protocol(root/'registrations/formal_test.v1.json')
    if sha256(frozen_path)!=registered['frozen_sha256']:
        raise AssertionError('Test configuration differs from prior registration')
    if role=='primary':
        if str(run.resolve())!=registered['run']:
            raise ValueError('Primary test fields must use the single registered run')
    elif role=='numerical_replay':
        source=load_protocol(Path(registered['run'])/'formal_summary.json')
        if source['status']!='completed' or source['episodes']!=3024:
            raise ValueError('Numerical test-data replay is allowed only after the complete primary test')
    else:
        raise ValueError('Unknown held-out field role')
    return frozen


def prepare_field(root,run,frozen_path,family,side,*,role='primary',mesh_m=5.,interval_s=1.):
    _field_guard(root,run,frozen_path,role)
    if family['split']!='test':raise ValueError('Wrong split in independent test field generator')
    if role=='primary' and (mesh_m!=5. or interval_s!=1.):
        raise ValueError('Primary hydraulic mesh/output differ from frozen settings')
    if role=='numerical_replay' and (mesh_m!=10. or interval_s!=2.):
        raise ValueError('Only the preregistered 10m/2s numerical contrast is allowed')
    directory=Path(run)/'fields'/family['family_id']/f'side{side}'
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'field_case.json'
    if marker.exists():
        case=load_protocol(marker)
        if case['family']!=family or case['side']!=side or case['role']!=role:
            raise AssertionError('Test field identity changed')
        for kind in ('truth','nominal'):
            record=case[kind]
            if sha256(record['manifest']['field_path'])!=record['manifest']['field_sha256']:
                raise AssertionError('Test hydraulic field changed')
        return case
    with (directory/'solver.log').open('a',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        common=dict(duration_s=1200.,output_interval_s=interval_s,mesh_spacing_m=mesh_m,
            family_seed=family['family_seed'],formal_test=role=='primary',experiment_role=role,**family['hydrology'])
        truth,_=_cached_field(directory/'truth',side,nominal=False,**common)
        nominal,_=_cached_field(directory/'nominal',side,nominal=True,**{**common,'perturbation_m':0.})
    case=dict(family=family,side=side,truth=truth,nominal=nominal,role=role,mission_start_s=0.,
        frozen_sha256=sha256(frozen_path),formal_registration_sha256=sha256(Path(root)/'registrations/formal_test.v1.json'))
    write_json(marker,case)
    return case


def prepare_fields(root,run,frozen_path,workers=6,*,role='primary',mesh_m=5.,interval_s=1.):
    plan=register(root)
    families=[f for f in plan['allocation'] if f['split']=='test']
    cases=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(prepare_field,str(root),str(run),str(frozen_path),family,side,
            role=role,mesh_m=mesh_m,interval_s=interval_s) for family in families for side in (32,50,70)]
        for future in as_completed(futures):
            cases.append(future.result())
            write_json(Path(run)/'field_progress.json',dict(cases=len(cases),planned_cases=108,role=role))
            print(f'{role} hydraulic cases {len(cases)}/108',flush=True)
    if len(cases)!=108:raise AssertionError('Incomplete held-out hydraulic matrix')
    write_json(Path(run)/'field_summary.json',dict(status='completed',cases=cases,role=role,independent_field_count=216))
    return cases


def summarize_primary(rows,frozen):
    from .statistics import paired_effect
    identities=[r['id'] for r in rows]
    planned={r['id'] for r in frozen['formal_matrix']}
    if len(set(identities))!=len(identities) or not set(identities).issubset(planned):
        raise AssertionError('Duplicate or unregistered primary episode')
    groups=[]
    for side,delay in ((32,0.),(50,0.),(50,10.),(70,0.)):
        group=[r for r in rows if r['side']==side and r['delay_s']==delay]
        methods=frozen['formal_matrix']
        names=['H','E','P','P_wrong_links','P_uniform_time'] if side==50 and delay==0 else ['H','E','P']
        if len(group)!=36*2*3*len(names):continue
        rates={}
        for method in names:
            values=[r for r in group if r['method']==method]
            rates[method]=dict(success_rate=sum(r['summary']['success_rate'] for r in values)/len(values),
                episodes=len(values),planning_timeouts=sum(r['summary']['planning_timeouts'] for r in values),
                failure_targets={key:sum(r['outcome_analysis']['categories'][key] for r in values) for key in values[0]['outcome_analysis']['categories']},
                new_evidence_reassignments=sum(r['outcome_analysis']['new_evidence_reassignments'] for r in values),
                coverage=sum(r['summary']['coverage_fraction'] for r in values)/len(values),
                mean_flight_m=sum(r['summary']['distance_m'] for r in values)/len(values),
                repeated_sensor_visits=sum(r['summary']['repeated_sensor_cell_visits'] for r in values),
                max_planning_ms=max(r['summary']['planning_latency_ms']['maximum'] for r in values))
        effect=paired_effect(group,'P',frozen['B_star'])
        ablations={m:paired_effect(group,'P',m) for m in ('P_wrong_links','P_uniform_time') if m in names}
        groups.append(dict(side=side,delay_s=delay,families=36,training_seeds=3,layouts=2,rates=rates,
            primary_effect=effect,mechanism_contrasts=ablations))
    return dict(status='completed' if set(identities)==planned else 'partial',episodes=len(rows),planned_episodes=3024,
        main_episodes=sum(r['delay_s']==0 for r in rows),delayed_episodes=sum(r['delay_s']==10 for r in rows),
        B_star=frozen['B_star'],groups=groups,rows=rows,all_hidden_targets_in_denominator=True,
        filtering_by_method_advantage=False,configuration_reselected_after_test=False)


def run_formal(root,frozen_path,workers=6,resume=None):
    root,frozen_path=Path(root).resolve(),Path(frozen_path).resolve()
    run,frozen=reserve_test(root,frozen_path,resume)
    print(f'Independent registered test directory: {run}',flush=True)
    started=time.perf_counter()
    try:
        cases=prepare_fields(root,run,frozen_path,workers)
        # Load CUDA only in the parent after field workers terminate.
        from .evaluation import configure_device,load_models,config_for,load_or_execute
        from .diagnostics import load_field
        registry={(r['method'],r['seed']):r for r in frozen['checkpoints']}
        bank=load_protocol(frozen['evidence']['links']['path']);device=configure_device();models=load_models(registry,device)
        plan=register(root);case_map={(c['family']['family_id'],c['side']):c for c in cases}
        rows=[];current_key=None
        for spec in frozen['formal_matrix']:
            key=spec['family_id'],spec['side']
            if key!=current_key:
                verify_frozen(root,frozen_path)
                case=case_map[key];truth=load_field(case['truth']['manifest']['field_path']);nominal=load_field(case['nominal']['manifest']['field_path'])
                current_key=key
            model_key=spec['method'],spec['training_seed']
            config=config_for(plan,spec['family_id'],spec['side'],spec['layout'],spec['delay_s'])
            provenance=dict(**registry[model_key],family_id=spec['family_id'],side=spec['side'],layout=spec['layout'],delay_s=spec['delay_s'],
                truth_sha256=case['truth']['manifest']['field_sha256'],nominal_sha256=case['nominal']['manifest']['field_sha256'],
                frozen_config_sha256=sha256(frozen_path),formal_registration_path=str(run/'formal_registration.json'))
            result,record=load_or_execute(run/'episodes'/spec['id'],truth,nominal,config,models[model_key],bank,spec['method'],provenance,device,
                formal_test=True,experiment_role='primary_independent_test')
            rows.append(dict(**spec,summary=result['summary'],outcome_analysis=result['outcome_analysis'],episode=record,
                execution_memory=dict(cpu_peak_bytes=result['execution_record']['peak_process_memory_bytes'],
                    gpu_peak_allocated_bytes=result['execution_record']['gpu_peak_allocated_bytes'])))
            write_json(run/'progress.json',dict(episodes=len(rows),planned_episodes=3024,formal_test=True))
            if len(rows)%36==0:
                print(f'Independent test {len(rows)}/3024 episodes',flush=True)
                write_json(run/'formal_summary.json',summarize_primary(rows,frozen))
        summary=summarize_primary(rows,frozen)
        if summary['status']!='completed' or summary['main_episodes']!=2376 or summary['delayed_episodes']!=648:
            raise AssertionError('Independent formal test matrix incomplete')
        write_json(run/'formal_summary.json',summary)
        finish_run(root,run,formal_test=True,elapsed_wall_s=time.perf_counter()-started)
        write_json(root/'results/latest_formal_test.json',dict(run=str(run),status='completed',episodes=3024))
        return run
    except BaseException as error:
        import traceback
        (run/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(root,run,status='failed',formal_test=True,error=repr(error))
        raise
