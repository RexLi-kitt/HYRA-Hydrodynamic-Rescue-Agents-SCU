"""Preregistered one-factor planner stress sweep, never select by P gains."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import time
import numpy as np

from .artifacts import finish_run, new_run, sha256, write_json
from .diagnostics import load_field
from .frontier_control import NeutralFrontierPredictor
from .phase_probe import mission_view
from .planner_replay import QueryPredictor,audit_episode
from .planning import RollingPlanner
from .protocol import load_protocol
from .simulation import EpisodeConfig,run_episode

SETTINGS={
    'default':dict(beam_width=6,frontier_limit=16,solver_budget_s=.02,planner_budget_s=.1),
    'beam3':dict(beam_width=3,frontier_limit=16,solver_budget_s=.02,planner_budget_s=.1),
    'beam12':dict(beam_width=12,frontier_limit=16,solver_budget_s=.02,planner_budget_s=.1),
    'frontier8':dict(beam_width=6,frontier_limit=8,solver_budget_s=.02,planner_budget_s=.1),
    'frontier32':dict(beam_width=6,frontier_limit=32,solver_budget_s=.02,planner_budget_s=.1),
    'solver5ms':dict(beam_width=6,frontier_limit=16,solver_budget_s=.005,planner_budget_s=.1),
    'solver50ms':dict(beam_width=6,frontier_limit=16,solver_budget_s=.05,planner_budget_s=.1),
    'planning50ms':dict(beam_width=6,frontier_limit=16,solver_budget_s=.02,planner_budget_s=.05),
    'planning200ms':dict(beam_width=6,frontier_limit=16,solver_budget_s=.02,planner_budget_s=.2)}


def _job(run_path,source_job):
    root=Path(run_path)/'jobs'/source_job.name
    root.mkdir(parents=True,exist_ok=True)
    records=[];nominal_cache={}
    for physical in sorted(source_job.glob('condition_*')):
        case=load_protocol(physical/'phase600/phase_case.json')
        if case['condition']['perturbation_m']!=.2:
            continue
        ti,ni=case['truth']['manifest'],case['nominal']['manifest']
        if sha256(ti['field_path'])!=ti['field_sha256']:
            raise AssertionError('Coarse source truth changed')
        full_truth=load_field(ti['field_path'])
        if ni['field_path'] not in nominal_cache:
            if sha256(ni['field_path'])!=ni['field_sha256']:
                raise AssertionError('Independent nominal changed')
            nominal_cache[ni['field_path']]=load_field(ni['field_path'])
        for phase in (0.,600.):
            truth=mission_view(full_truth,phase);nominal=mission_view(nominal_cache[ni['field_path']],phase)
            for pair in case['pairs']:
                old=load_protocol(pair['nominal']['episode_path'])
                original=EpisodeConfig(**old['execution_record']['config'])
                for name,setting in SETTINGS.items():
                    for mode in ('nominal','neutral_frontier','truth_diagnostic'):
                        directory=root/physical.name/f'phase{phase:g}'/f"layout{pair['layout_index']}_delay{pair['delay_s']:g}"/name/mode
                        directory.mkdir(parents=True)
                        config=replace(original,planner_budget_s=setting['planner_budget_s'],retain_raw_observations=False)
                        field=truth if mode=='truth_diagnostic' else nominal
                        klass=NeutralFrontierPredictor if mode=='neutral_frontier' else QueryPredictor
                        predictor=klass(field['times_s'],field['depths_m'].reshape(-1,*config.map_shape),
                            horizon_s=config.forecast_horizon_s,service_duration_s=config.service_duration_s,
                            threshold_m=config.max_service_depth_m,privileged=mode=='truth_diagnostic')
                        planner=RollingPlanner(**{k:v for k,v in setting.items() if k!='planner_budget_s'})
                        result=run_episode(truth['depths_m'].reshape(-1,*config.map_shape),config,times=truth['times_s'],
                                           predictor=predictor,planner=planner)
                        audit=audit_episode(result,predictor,truth,config)
                        result.pop('raw_observations')
                        uncertified=sum(e.get('planner_diagnostics',{}).get('candidate_pool_optimal') is False for e in result['events'])
                        result['execution_record']=dict(setting=setting,mode=mode,formal_test=False,trained_model=False,
                            source_case=str(physical),physical_phase_s=phase,query_count=len(predictor.queries),
                            raw_observations_retained=False,training_data_eligible=False,
                            observation_causality_validated_by_forecast_issue=True,complete_service_audit=audit)
                        path=directory/'episode.json';write_json(path,result)
                        records.append(dict(family_id=case['family']['family_id'],side=case['side'],condition=case['condition'],
                            phase_s=phase,layout=pair['layout_index'],delay_s=pair['delay_s'],setting=name,mode=mode,
                            summary=result['summary'],uncertified_candidate_pool_calls=uncertified,
                            path=str(path),sha256=sha256(path),audit=audit))
                    write_json(root/'progress.json',dict(episodes=len(records)))
    result=dict(job_id=source_job.name,status='completed',records=records)
    write_json(root/'job_result.json',result)
    return result


def summarize(run,jobs):
    records=[r for j in jobs for r in j['records']]
    groups=[]
    for phase in (0.,600.):
        for side,delay in ((32,0.),(50,0.),(50,10.),(70,0.)):
            for setting in SETTINGS:
                subset=[r for r in records if (r['phase_s'],r['side'],r['delay_s'],r['setting'])==(phase,side,delay,setting)]
                if not subset:
                    continue
                group=dict(phase_s=phase,side=side,delay_s=delay,setting=setting,modes={})
                for mode in ('nominal','neutral_frontier','truth_diagnostic'):
                    rows=[r for r in subset if r['mode']==mode]
                    group['modes'][mode]=dict(episodes=len(rows),families=len({r['family_id'] for r in rows}),
                        mean_success_rate=float(np.mean([r['summary']['success_rate'] for r in rows])),
                        timeouts=sum(r['summary']['planning_timeouts'] for r in rows),
                        uncertified_candidate_pool_calls=sum(r['uncertified_candidate_pool_calls'] for r in rows),
                        max_planning_ms=max(r['summary']['planning_latency_ms']['maximum'] for r in rows))
                group['nominal_frontier_gain_pp']=100*(group['modes']['nominal']['mean_success_rate']-group['modes']['neutral_frontier']['mean_success_rate'])
                group['truth_information_gain_pp']=100*(group['modes']['truth_diagnostic']['mean_success_rate']-group['modes']['nominal']['mean_success_rate'])
                groups.append(group)
    result=dict(status='completed' if len(records)==10368 else 'partial',episodes=len(records),planned_episodes=10368,
        groups=groups,formal_test=False,trained_model=False,default_changed_by_outcomes=False,
        selection='all six design families, three scales, four rise combinations, maximum perturbation, both layouts, both phases, main-scale delays')
    write_json(Path(run)/'sensitivity_summary.json',result)
    return result


def run_sensitivity(root,source,workers=2):
    root,source=Path(root).resolve(),Path(source).resolve()
    run=new_run(root,'planner_sensitivity',dict(workers=workers,source=str(source)))
    write_json(run/'sensitivity_spec.json',dict(settings=SETTINGS,phases_s=[0.,600.],perturbation_m=.2,
        planned_pairs=384,planned_episodes=10368,scales=[32,50,70],families=6,rise_combinations=4,
        outcome_selection=False,default_config_not_selected_by_HyRA_gain=True,
        raw_observations_retained=False,labels='engineering_sensitivity_not_model_or_formal_test',
        concurrent_hydraulics_possible=True,final_learned_model_budget_must_be_rechecked=True))
    print(f'Sensitivity directory: {run}',flush=True)
    completed=[];started=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(_job,str(run),job) for job in sorted((source/'jobs').glob('*_side*'))]
            for future in as_completed(futures):
                completed.append(future.result())
                summary=summarize(run,completed)
                print(f'Sensitivity {len(completed)}/18 jobs, {summary["episodes"]} episodes',flush=True)
        summary=summarize(run,completed)
        if summary['status']!='completed':
            raise AssertionError('Planner sensitivity incomplete')
        finish_run(root,run,elapsed_wall_s=time.perf_counter()-started)
        write_json(root/'results/latest_planner_sensitivity.json',dict(run=str(run),status='completed'))
        return run
    except BaseException as error:
        import traceback
        (run/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(root,run,status='failed',error=repr(error))
        raise
