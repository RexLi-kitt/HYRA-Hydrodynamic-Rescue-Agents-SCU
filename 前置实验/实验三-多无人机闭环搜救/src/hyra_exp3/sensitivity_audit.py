"""Read back every prespecified planner stress episode before configuration freeze."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from .artifacts import sha256,write_json
from .protocol import load_protocol
from .diagnostics import load_field
from .phase_probe import mission_view
from .planner_sensitivity import SETTINGS
from .planner_replay import audit_episode
from .simulation import EpisodeConfig


def _audit_job(path):
    job=load_protocol(path);totals=dict(episodes=0,services=0,planned_visits=0,trajectory_segments=0)
    current=None;fields=None;case=None;identities=[];rows=[]
    for row in job['records']:
        if sha256(row['path'])!=row['sha256']:raise AssertionError('Planner sensitivity episode changed')
        episode=load_protocol(row['path']);execution=episode['execution_record'];physical=Path(execution['source_case'])
        key=str(physical),row['phase_s']
        if key!=current:
            case=load_protocol(physical/'phase600/phase_case.json');fields={}
            for kind in ('truth','nominal'):
                record=case[kind]['manifest']
                if sha256(record['field_path'])!=record['field_sha256']:raise AssertionError('Sensitivity source hydraulic field changed')
                fields[kind]=mission_view(load_field(record['field_path']),row['phase_s'])
            current=key
        if (row['family_id'],row['side'],row['condition'])!=(case['family']['family_id'],case['side'],case['condition']):
            raise AssertionError('Sensitivity family/scale/hydrology identity differs')
        pair=next(p for p in case['pairs'] if p['layout_index']==row['layout'] and p['delay_s']==row['delay_s'])
        origin=pair['nominal']
        if sha256(origin['episode_path'])!=origin['episode_sha256']:raise AssertionError('Sensitivity target/configuration source changed')
        old=load_protocol(origin['episode_path']);setting=SETTINGS[row['setting']]
        config=replace(EpisodeConfig(**old['execution_record']['config']),planner_budget_s=setting['planner_budget_s'],retain_raw_observations=False)
        if execution['setting']!=setting or execution['mode']!=row['mode'] or execution['formal_test'] or execution['trained_model'] or execution['training_data_eligible']:
            raise AssertionError('Sensitivity controller/budget/engineering-only identity differs')
        if row['summary']!=episode['summary'] or execution['complete_service_audit']!=row['audit']:
            raise AssertionError('Sensitivity row summary differs from raw execution')
        if [t['cell'] for t in old['targets']]!=[t['cell'] for t in episode['targets']]:raise AssertionError('Sensitivity paired targets changed')
        publications=[]
        for event in episode['events']:
            if event['event'] in ('plan','planning_timeout'):
                version=len(publications)+1
                if 'version' in event and event['version']!=version:raise AssertionError('Sensitivity publication sequence differs')
                now=event['time_s'];publications.append((version,now,min(config.duration_s,now+config.forecast_horizon_s)))
        if len(publications)!=episode['summary']['planning_calls']:raise AssertionError('Sensitivity planning count differs')
        plan_events=[e for e in episode['events'] if e['event'] in ('plan','planning_timeout')]
        latencies=np.asarray([e['elapsed_ms'] for e in plan_events])
        calculated_latency=dict(median=float(np.median(latencies)),p95=float(np.quantile(latencies,.95)),maximum=float(latencies.max()))
        if sum(e['event']=='planning_timeout' for e in plan_events)!=episode['summary']['planning_timeouts'] or any(not np.isclose(episode['summary']['planning_latency_ms'][k],v,rtol=0,atol=1e-10) for k,v in calculated_latency.items()):
            raise AssertionError('Sensitivity deadline/latency statistics differ from actual planning events')
        chosen=fields['truth' if row['mode']=='truth_diagnostic' else 'nominal']
        predictor=SimpleNamespace(queries=[],publications=publications,times=chosen['times_s'],depths=chosen['depths_m'].reshape(-1,*config.map_shape))
        episode['raw_observations']=[]
        audit=audit_episode(episode,predictor,fields['truth'],config)
        for item,stored in (('complete_services_rescored','complete_services_rescored'),('planned_visits_checked','planned_visits_checked')):
            if audit[item]!=row['audit'][stored]:raise AssertionError('Sensitivity service/route audit count differs')
        payload={d:config.payload for d in range(config.n_drones)}
        for event in episode['events']:
            if event['event']=='service_complete' and event['success']:payload[event['drone']]-=1
        if any(robot['payload']!=payload[robot['id']] for robot in episode['drones']):raise AssertionError('Sensitivity final payload differs')
        latest={};segments=0
        for point in episode['trajectories']:
            d=point['drone'];xy=np.asarray(point['position_m']);now=point['time_s']
            if d in latest:
                before,previous=latest[d]
                if np.linalg.norm(xy-before)>config.speed_mps*(now-previous)+1e-5:raise AssertionError('Sensitivity trajectory exceeds speed')
                segments+=1
            latest[d]=(xy,now)
        served=sum(t['served_at'] is not None for t in episode['targets'])
        if served!=episode['summary']['successful_services'] or len(episode['targets'])!=config.n_targets or episode['summary']['final_time_s']!=1200.:
            raise AssertionError('Sensitivity service denominator or full mission clock differs')
        uncertified=sum(e.get('planner_diagnostics',{}).get('candidate_pool_optimal') is False for e in episode['events'])
        if uncertified!=row['uncertified_candidate_pool_calls']:raise AssertionError('Sensitivity solver completion statistic differs')
        identities.append((row['family_id'],row['side'],row['condition']['rise_m'],row['condition']['rise_duration_s'],row['condition']['perturbation_m'],
                           row['phase_s'],row['layout'],row['delay_s'],row['setting'],row['mode']))
        rows.append(row);totals['episodes']+=1;totals['services']+=audit['complete_services_rescored'];totals['planned_visits']+=audit['planned_visits_checked'];totals['trajectory_segments']+=segments
    return dict(totals=totals,identities=identities,rows=rows)


def audit_sensitivity(root,run,workers=3):
    root,run=Path(root),Path(run);summary=load_protocol(run/'sensitivity_summary.json');spec=load_protocol(run/'sensitivity_spec.json')
    if summary['status']!='completed' or summary['episodes']!=10368 or summary['default_changed_by_outcomes']:raise ValueError('Complete prespecified sensitivity matrix required')
    if spec['settings']!=SETTINGS or spec['outcome_selection']:raise AssertionError('Planner settings were changed or selected by outcomes')
    for relative,digest in load_protocol(run/'source_sha256.json').items():
        if sha256(run/'source_snapshot'/relative)!=digest:raise AssertionError('Sensitivity execution source snapshot changed')
    families={f['family_id'] for f in load_protocol(root/'manifests/family_split.json')['families'] if f['split']=='design'}
    expected={(f,s,r,d,.2,phase,l,delay,setting,mode) for f in families for s in (32,50,70)
        for r in (.6,.9) for d in (600.,900.) for phase in (0.,600.) for l in (0,1)
        for delay in ((0.,10.) if s==50 else (0.,)) for setting in SETTINGS for mode in ('nominal','neutral_frontier','truth_diagnostic')}
    totals={};seen=[];rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_audit_job,path) for path in sorted((run/'jobs').glob('*/job_result.json'))]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result();seen.extend(result['identities']);rows.extend(result['rows'])
            for key,value in result['totals'].items():totals[key]=totals.get(key,0)+value
            print(f'Planner sensitivity read-back audit {i}/18 jobs',flush=True)
    if set(seen)!=expected or len(seen)!=len(expected) or len(expected)!=10368:raise AssertionError('Sensitivity full factorial matrix incomplete or duplicated')
    group_ids=set()
    for group in summary['groups']:
        identity=group['phase_s'],group['side'],group['delay_s'],group['setting']
        if identity in group_ids:raise AssertionError('Duplicate sensitivity summary group')
        group_ids.add(identity)
        chosen=[r for r in rows if (r['phase_s'],r['side'],r['delay_s'],r['setting'])==identity]
        for mode in ('nominal','neutral_frontier','truth_diagnostic'):
            values=[r for r in chosen if r['mode']==mode];stored=group['modes'][mode]
            calculated=dict(episodes=len(values),families=len({r['family_id'] for r in values}),
                mean_success_rate=float(np.mean([r['summary']['success_rate'] for r in values])),
                timeouts=sum(r['summary']['planning_timeouts'] for r in values),
                uncertified_candidate_pool_calls=sum(r['uncertified_candidate_pool_calls'] for r in values),
                max_planning_ms=max(r['summary']['planning_latency_ms']['maximum'] for r in values))
            if any(not np.isclose(stored[k],v,atol=1e-12,rtol=0) for k,v in calculated.items()):raise AssertionError('Sensitivity summary aggregation differs')
        nominal=group['modes']['nominal']['mean_success_rate'];neutral=group['modes']['neutral_frontier']['mean_success_rate'];truth=group['modes']['truth_diagnostic']['mean_success_rate']
        if not np.isclose(group['nominal_frontier_gain_pp'],100*(nominal-neutral),atol=1e-10,rtol=0) or not np.isclose(group['truth_information_gain_pp'],100*(truth-nominal),atol=1e-10,rtol=0):
            raise AssertionError('Sensitivity gain arithmetic differs')
    if len(group_ids)!=72:raise AssertionError('Sensitivity aggregate group count differs')
    result=dict(status='passed',**totals,summary_sha256=sha256(run/'sensitivity_summary.json'),all_10368_prespecified_cells=True,
        full_service_scores_routes_payloads_and_trajectories_rechecked=True,
        raw_observations_and_queries_not_retained=True,causal_query_checks_retained_from_online_audit=True,
        formal_test=False,trained_models=False)
    write_json(run/'sensitivity_integrity_audit.json',result)
    return result
