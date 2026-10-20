"""Preregistered paired interventions and numerical replay after the primary test."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np

from .artifacts import new_run,finish_run,write_json,sha256
from .protocol import load_protocol
from .freeze import verify_frozen
from .execution_plan import register
from .statistics import paired_effect


def source_primary(root,frozen_path):
    frozen=verify_frozen(root,frozen_path)
    registration=load_protocol(Path(root)/'registrations/formal_test.v1.json')
    if registration['frozen_sha256']!=sha256(frozen_path):
        raise AssertionError('Post-test evidence uses a different frozen configuration')
    primary=Path(registration['run']);summary=load_protocol(primary/'formal_summary.json')
    if summary['status']!='completed' or summary['episodes']!=3024:
        raise ValueError('Complete the single primary independent test first')
    return frozen,primary,summary


def reserve_diagnostic(root,frozen_path,kind,resume=None):
    frozen,primary,summary=source_primary(root,frozen_path)
    marker=Path(root)/'registrations'/f'{kind}.v1.json'
    if resume is None:
        if marker.exists():raise ValueError('Diagnostic already registered; resume its exact run')
        run=new_run(root,kind,dict(frozen_path=str(Path(frozen_path).resolve()),primary=str(primary)))
        record=dict(run=str(run),kind=kind,primary=str(primary),primary_sha256=sha256(primary/'formal_summary.json'),
            frozen_sha256=sha256(frozen_path),planned_episodes=1728,outside_primary_3024=True)
        with marker.open('x',encoding='utf-8') as stream:
            json.dump(record,stream,ensure_ascii=False,indent=2)
        write_json(run/'diagnostic_registration.json',record)
    else:
        run=Path(resume).resolve();record=load_protocol(marker)
        if str(run)!=record['run'] or record!=load_protocol(run/'diagnostic_registration.json'):
            raise ValueError('Diagnostic must resume its exact registered directory')
        if record['frozen_sha256']!=sha256(frozen_path) or record['primary_sha256']!=sha256(primary/'formal_summary.json'):
            raise AssertionError('Diagnostic provenance changed')
    return run,frozen,primary,summary


def prefix_comparison(full,deferred):
    """A changed wall-clock timeout before intervention is a confound, not a policy effect."""
    first=min((e['time_s'] for e in deferred['events'] if e.get('policy_interventions')),default=None)
    limit=float('inf') if first is None else first
    def signature(result):
        rows=[]
        for e in result['events']:
            if e['time_s']>=limit-1e-8:continue
            if e['event']=='plan':
                rows.append(dict(event='plan',time_s=e['time_s'],planning_state=e.get('planning_state'),
                    routes=e.get('planner_diagnostics',{}).get('selected_routes',[])))
            elif e['event'] in ('planning_timeout','service_start','service_complete','target_discovered'):
                rows.append({k:v for k,v in e.items() if k not in ('elapsed_ms','planner_diagnostics')})
        return rows
    a,b=signature(full),signature(deferred)
    # Canonical numeric rounding ignores irrelevant floating arithmetic at simultaneous events.
    def canonical(value):
        if isinstance(value,float):return round(value,7)
        if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
        if isinstance(value,list):return [canonical(v) for v in value]
        return value
    same=canonical(a)==canonical(b)
    if first is not None:
        starts=[e for e in deferred['events'] if e['time_s']==first and e.get('policy_interventions')]
        baseline=[e for e in full['events'] if abs(e['time_s']-first)<1e-7 and e['event']=='plan']
        same=same and bool(baseline) and canonical(starts[0].get('planning_state'))==canonical(baseline[0].get('planning_state'))
    return dict(first_intervention_s=first,prefix_decisions_equal=same,
        uncontaminated_before_first_intervention=same,
        no_intervention=first is None,
        all_episode_decisions_equal_if_no_intervention=same if first is None else None,
        caution=None if same else 'Execution differs before policy intervention; paired difference is not an isolated causal policy gain')


def diagnostic_summary(rows,primary_summary,frozen,kind):
    groups=[]
    for side,delay in ((32,0.),(50,0.),(50,10.),(70,0.)):
        group=[r for r in rows if r['side']==side and r['delay_s']==delay]
        if len(group)!=432:continue
        original=[r for r in primary_summary['rows'] if r['side']==side and r['delay_s']==delay and r['method'] in ('P',frozen['B_star'])]
        if kind=='reassignment_counterfactual':
            effects={}
            for method in ('P',frozen['B_star']):
                paired=[dict(r,method='full') for r in original if r['method']==method]
                paired += [dict(r,method='deferred') for r in group if r['method']==method]
                effects[method]=paired_effect(paired,'full','deferred')
                effects[method]['pre_intervention_contaminated_pairs']=sum(not r['prefix_check']['uncontaminated_before_first_intervention'] for r in group if r['method']==method)
                effects[method]['policy_interventions']=sum(r['outcome_analysis']['policy_interventions'] for r in group if r['method']==method)
                effects[method]['additional_successful_services_across_paired_episodes']=sum(r['actual_additional_successful_services_with_full_policy'] for r in group if r['method']==method)
                effects[method]['full_policy_nonperiodic_reassignments']=sum(r['outcome_analysis']['new_evidence_reassignments'] for r in original if r['method']==method)
                effects[method]['full_policy_feasible_switches_eligible_for_intervention']=sum(len(r['outcome_analysis']['eligible_defer_interventions']) for r in original if r['method']==method)
                effects[method]['isolated_causal_interpretation_supported']=effects[method]['pre_intervention_contaminated_pairs']==0
            record=dict(side=side,delay_s=delay,actual_paired_service_gain_from_reassignment=effects,
                per_event_gains_not_summed=True,no_gain_based_selection=True)
        else:
            fine=paired_effect(original,'P',frozen['B_star']);coarse=paired_effect(group,'P',frozen['B_star'])
            sign=lambda x: int(np.sign(x)) if abs(x)>1e-10 else 0
            record=dict(side=side,delay_s=delay,fine_5m1s=fine,coarse_10m2s=coarse,
                gain_direction_unchanged=sign(fine['delta_percentage_points'])==sign(coarse['delta_percentage_points']),
                gain_change_percentage_points=coarse['delta_percentage_points']-fine['delta_percentage_points'],
                positive_effect_claim_robust_to_numerics=fine['primary_positive'] and coarse['primary_positive'],
                both_have_no_positive_effect_evidence=not fine['primary_positive'] and not coarse['primary_positive'])
        groups.append(record)
    result=dict(status='completed' if len(rows)==1728 else 'partial',kind=kind,episodes=len(rows),planned_episodes=1728,
        outside_primary_3024=True,frozen_models_retrained=False,configuration_reselected=False,groups=groups,rows=rows)
    if kind=='numerical_direction':
        result['all_gain_directions_unchanged']=len(groups)==4 and all(g['gain_direction_unchanged'] for g in groups)
        result['interpretation']='A sign change is reported as numerical sensitivity; it never triggers selection of a favorable result'
    return result


def run_posttest(root,frozen_path,kind,workers=4,resume=None):
    if kind not in ('reassignment_counterfactual','numerical_direction'):raise ValueError('Unknown post-test diagnostic')
    root,frozen_path=Path(root).resolve(),Path(frozen_path).resolve()
    run,frozen,primary,source=reserve_diagnostic(root,frozen_path,kind,resume)
    print(f'{kind} directory: {run}',flush=True)
    try:
        if kind=='numerical_direction':
            from .formal import prepare_fields
            cases=prepare_fields(root,run,frozen_path,workers,role='numerical_replay',mesh_m=10.,interval_s=2.)
        else:
            cases=load_protocol(primary/'field_summary.json')['cases']
        from .evaluation import configure_device,load_models,config_for,load_or_execute
        from .diagnostics import load_field
        plan=register(root);device=configure_device()
        registry={(r['method'],r['seed']):r for r in frozen['checkpoints'] if r['method'] in ('P',frozen['B_star'])}
        models=load_models(registry,device);bank=load_protocol(frozen['evidence']['links']['path'])
        cases={(c['family']['family_id'],c['side']):c for c in cases}
        source_rows={r['id']:r for r in source['rows']};rows=[];current=None
        for spec in frozen['formal_matrix']:
            if spec['method'] not in ('P',frozen['B_star']):continue
            key=spec['family_id'],spec['side']
            if current!=key:
                verify_frozen(root,frozen_path);case=cases[key]
                for field in ('truth','nominal'):
                    m=case[field]['manifest']
                    if sha256(m['field_path'])!=m['field_sha256']:raise AssertionError('Post-test field changed')
                truth=load_field(case['truth']['manifest']['field_path']);nominal=load_field(case['nominal']['manifest']['field_path']);current=key
            config=config_for(plan,spec['family_id'],spec['side'],spec['layout'],spec['delay_s'])
            if kind=='reassignment_counterfactual':config=replace(config,reassignment_policy='defer_new_evidence')
            model_key=spec['method'],spec['training_seed']
            provenance=dict(**registry[model_key],source_primary_episode=source_rows[spec['id']]['episode'],
                truth_sha256=case['truth']['manifest']['field_sha256'],nominal_sha256=case['nominal']['manifest']['field_sha256'],
                frozen_config_sha256=sha256(frozen_path),formal_registration_path=str(primary/'formal_registration.json'),
                diagnostic_kind=kind,world_noise_and_layout_shared=True)
            result,record=load_or_execute(run/'episodes'/spec['id'],truth,nominal,config,models[model_key],bank,spec['method'],provenance,device,
                experiment_role=kind)
            row=dict(**spec,summary=result['summary'],outcome_analysis=result['outcome_analysis'],episode=record)
            if kind=='reassignment_counterfactual':
                origin=source_rows[spec['id']]['episode']
                if sha256(origin['path'])!=origin['sha256']:raise AssertionError('Primary paired episode changed')
                full=load_protocol(origin['path'])
                if [t['cell'] for t in full['targets']]!=[t['cell'] for t in result['targets']]:raise AssertionError('Paired targets changed')
                row['prefix_check']=prefix_comparison(full,result)
                row['actual_additional_successful_services_with_full_policy']=full['summary']['successful_services']-result['summary']['successful_services']
            rows.append(row)
            write_json(run/'progress.json',dict(episodes=len(rows),planned_episodes=1728,kind=kind))
            if len(rows)%36==0:print(f'{kind} {len(rows)}/1728 episodes',flush=True)
        summary=diagnostic_summary(rows,source,frozen,kind)
        if summary['status']!='completed' or len(summary['groups'])!=4:raise AssertionError('Post-test diagnostic incomplete')
        write_json(run/'diagnostic_summary.json',summary);finish_run(root,run)
        write_json(root/'results'/f'latest_{kind}.json',dict(run=str(run),status='completed',episodes=1728))
        return run
    except BaseException as error:
        import traceback
        (run/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(root,run,status='failed',error=repr(error));raise
