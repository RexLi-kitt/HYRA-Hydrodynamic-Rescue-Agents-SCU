"""Independent read-back of saved model curves, sensing, services and test identities."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import numpy as np

from .artifacts import sha256,write_json
from .protocol import load_protocol
from .diagnostics import load_field
from .freeze import verify_frozen
from .execution_plan import register
from .simulation import EpisodeConfig,stratified_starts
from .windows import WindowEstimate,service_start_intervals
from .planner_replay import audit_episode
from .outcomes import analyze_outcomes
from .noise import world_noise


class SavedWindows:
    def __init__(self,data,config):
        self.data,self.config=data,config
        self.index={(int(v),(int(r),int(c))):i for i,(v,r,c) in enumerate(zip(data['version'],data['row'],data['col']))}
        if len(self.index)!=len(data['version']):raise AssertionError('Duplicate published window')
        self.pubs={int(v):(float(a),float(b)) for v,a,b in data['publications']}
        self.cache={}

    def __getitem__(self,key):
        if key not in self.cache:
            i=self.index[key];n=int(self.data['length'][i]);times=self.data['future_times_s'][i,:n].astype(float)
            depths=self.data['predicted_depth_m'][i,:n].astype(float);a,b=self.pubs[key[0]]
            if not len(times) or not np.isfinite(depths).all() or abs(times[0]-a)>1e-7 or abs(times[-1]-b)>1e-7:
                raise AssertionError('Published curve clock/content differs')
            intervals=service_start_intervals(times,depths,self.config.service_duration_s,self.config.max_service_depth_m) if n>=2 else ()
            support=bool(self.data['evidence_supported'][i]) if 'evidence_supported' in self.data else True
            self.cache[key]=WindowEstimate(key[1],a,key[0],tuple(times),tuple(depths),intervals,support,b,
                evidence='serialized_frozen_forecast',service_duration_s=self.config.service_duration_s)
        return self.cache[key]

    def get(self,key,default=None):
        return self[key] if key in self.index else default


def audit_saved_episode(path,truth,expected_config=None):
    result=load_protocol(path);record=result['execution_record'];raw_config=record['config']
    config=EpisodeConfig(**dict(raw_config,map_shape=tuple(raw_config['map_shape'])))
    if expected_config is not None and raw_config!=json.loads(json.dumps(asdict(expected_config))):
        raise AssertionError('Serialized execution configuration differs')
    for name in ('observation_log','published_forecast_log'):
        if sha256(result[name]['path'])!=result[name]['sha256']:raise AssertionError('Serialized model raw log changed')
    obs=load_field(result['observation_log']['path']);forecasts=load_field(result['published_forecast_log']['path'])
    pubs=forecasts['publications'];q=forecasts['queries']
    if np.any(np.diff(pubs[:,0])!=1) or np.any(np.diff(pubs[:,1])<-1e-9):raise AssertionError('Nonmonotone publications')
    if np.any(pubs[:,2]>np.minimum(pubs[:,1]+config.forecast_horizon_s,config.duration_s)+1e-8):raise AssertionError('Publication horizon exceeded')
    qi=np.searchsorted(pubs[:,0],q[:,3])
    if np.any(qi>=len(pubs)) or np.any(pubs[qi,0]!=q[:,3]) or not np.allclose(q[:,0],pubs[qi,1],atol=1e-8,rtol=0):
        raise AssertionError('Query publication identity differs')
    if np.any(~np.isin(q[:,4],(0.,1.))):raise AssertionError('Unknown query role')
    sample=obs['sample_time_s'];arrival=obs['arrival_time_s'];tick=np.rint(sample/config.sensor_period_s).astype(int)
    if np.any(abs(sample-tick*config.sensor_period_s)>1e-7) or not np.allclose(arrival-sample,config.observation_delay_s,atol=1e-8,rtol=0):
        raise AssertionError('Observation sampling/communication clock differs')
    if np.any(arrival>config.duration_s+1e-7):raise AssertionError('Unarrived observations were retained as arrived')
    side=config.map_shape[0];positions=np.full((241,config.n_drones,2),np.nan);latest={};segments=0
    for point in result['trajectories']:
        d=point['drone'];t=point['time_s'];xy=np.asarray(point['position_m'],float)
        if np.any(xy<0) or np.any(xy>=side*config.cell_size_m):raise AssertionError('Flight outside public map')
        if d in latest:
            before,previous=latest[d]
            if np.linalg.norm(xy-before)>config.speed_mps*(t-previous)+1e-5:raise AssertionError('Serialized trajectory exceeds speed')
            segments+=1
        latest[d]=(xy,t);positions[int(round(t/config.sensor_period_s)),d]=xy
    if not np.isfinite(positions).all():raise AssertionError('Incomplete full-duration sensor trajectory')
    if not np.allclose(positions[0],np.asarray([(c+.5,r+.5) for r,c in stratified_starts(config.map_shape,config.n_drones)])*config.cell_size_m):
        raise AssertionError('Stratified fleet starts differ')
    centers=np.floor(positions/config.cell_size_m).astype(int)
    if np.any(abs(obs['row']-centers[tick,obs['drone'],1])>config.sensor_radius_cells) or np.any(abs(obs['col']-centers[tick,obs['drone'],0])>config.sensor_radius_cells):
        raise AssertionError('An observation lies outside the actual-position footprint')
    if np.any(obs['row']<0) or np.any(obs['col']<0) or np.any(obs['row']>=side) or np.any(obs['col']>=side):raise AssertionError('Observed cell outside map')
    flat=obs['row']*side+obs['col'];counts=np.bincount(flat,minlength=side*side)
    groups=tick*config.n_drones+obs['drone'];group_counts=np.bincount(groups,minlength=241*config.n_drones).reshape(241,config.n_drones)
    radius=config.sensor_radius_cells
    expected_counts=(np.minimum(side,centers[:,:,0]+radius+1)-np.maximum(0,centers[:,:,0]-radius))*(np.minimum(side,centers[:,:,1]+radius+1)-np.maximum(0,centers[:,:,1]-radius))
    expected_counts[np.arange(241)*config.sensor_period_s+config.observation_delay_s>config.duration_s+1e-7]=0
    if not np.array_equal(group_counts,expected_counts) or len(np.unique(groups*(side*side)+flat))!=len(flat):
        raise AssertionError('Incomplete or duplicated sensor footprint records')
    all_visits=np.zeros(side*side,int)
    for x,y in centers.reshape(-1,2):
        cells=np.asarray([r*side+c for r in range(max(0,y-radius),min(side,y+radius+1)) for c in range(max(0,x-radius),min(side,x+radius+1))])
        all_visits[cells]+=1
    if int(np.maximum(all_visits-1,0).sum())!=result['summary']['repeated_sensor_cell_visits']:
        raise AssertionError('Repeated search statistic differs from actual sensor footprints')
    for target in result['targets']:
        r,c=target['cell'];seen=(abs(centers[:,:,0]-c)<=radius)&(abs(centers[:,:,1]-r)<=radius)
        ticks_seen=np.flatnonzero(seen.any(1));first=float(ticks_seen[0]*config.sensor_period_s) if len(ticks_seen) else None
        discovered=None if first is None or first+config.observation_delay_s>config.duration_s+1e-7 else first+config.observation_delay_s
        same_clock=lambda a,b: a is None and b is None or a is not None and b is not None and abs(a-b)<1e-7
        if not same_clock(target['first_sensed_at'],first) or not same_clock(target['discovered_at'],discovered):
            raise AssertionError('Target discovery was not determined by an arrived actual-position observation')
    if abs(float((counts>0).mean())-result['summary']['coverage_fraction'])>1e-10:raise AssertionError('Coverage statistic differs')
    # A deterministic stratified subset verifies the keyed noise implementation, not just its distribution.
    noise_checked=0
    if config.noise_mode=='counter_keyed':
        for i in np.unique(np.linspace(0,len(flat)-1,min(256,len(flat))).astype(int)):
            now=sample[i];upper=min(max(int(np.searchsorted(truth['times_s'],now,side='right')),1),len(truth['times_s'])-1);lower=upper-1
            f=(now-truth['times_s'][lower])/(truth['times_s'][upper]-truth['times_s'][lower])
            depth=(1-f)*float(truth['depths_m'][lower,flat[i]])+f*float(truth['depths_m'][upper,flat[i]])
            noise=world_noise(config.seed,tick[i],obs['drone'][i],[flat[i]],config.observation_noise_m)[0]
            if abs(obs['depth_m'][i]-depth-noise)>1e-10:raise AssertionError('World-keyed sensor noise differs')
            noise_checked+=1
    result['raw_observations']=[dict(sample_time=float(t),arrival_time=float(a),cell=(int(r),int(c)),drone=int(d),depth_m=float(v))
        for t,a,r,c,d,v in zip(sample,arrival,obs['row'],obs['col'],obs['drone'],obs['depth_m'])]
    predictor=SimpleNamespace(queries=q,publications=pubs,published_windows=SavedWindows(forecasts,config))
    audit=audit_episode(result,predictor,truth,config)
    recomputed=analyze_outcomes(truth,result,predictor,config)
    if json.loads(json.dumps(recomputed))!=result['outcome_analysis']:raise AssertionError('Serialized failure/intervention analysis differs')
    if sum(t['served_at'] is not None for t in result['targets'])!=result['summary']['successful_services']:
        raise AssertionError('Served-target denominator differs')
    if result['summary']['final_time_s']!=config.duration_s:raise AssertionError('Mission ended early')
    for robot in result['drones']:
        if robot['payload']!=config.payload-sum(e['event']=='service_complete' and e['success'] and e['drone']==robot['id'] for e in result['events']):
            raise AssertionError('Final payload accounting differs')
    return dict(episodes=1,services=audit['complete_services_rescored'],observations=len(flat),queries=len(q),
        route_visits=audit['planned_visits_checked'],trajectory_segments=segments,independently_checked_keyed_noise_samples=noise_checked)


def _audit_group(case,rows,plan,registry):
    side=case['side'];family=case['family']['family_id'];totals={}
    for kind in ('truth','nominal'):
        manifest=case[kind]['manifest']
        if sha256(manifest['field_path'])!=manifest['field_sha256'] or manifest['kind']!=kind or not manifest['independent_initial_condition'] or not manifest['formal_test']:
            raise AssertionError('Held-out independent field provenance differs')
    truth=load_field(case['truth']['manifest']['field_path'])
    if not np.array_equal(truth['times_s'],np.arange(1201.)):raise AssertionError('Formal hydraulic output is not 1s over 1200s')
    for row in rows:
        checkpoint=registry[row['method'],row['training_seed']]
        episode=row['episode']
        if sha256(episode['path'])!=episode['sha256']:raise AssertionError('Formal episode file changed')
        result=load_protocol(episode['path']);record=result['execution_record'];provenance=record['provenance']
        if not record['trained_model'] or not record['formal_test'] or record['model_parameter_count']!=37070:
            raise AssertionError('Untrained/nonformal/capacity mismatch in primary test')
        if any(provenance[k]!=checkpoint[k] for k in checkpoint):raise AssertionError('Formal checkpoint provenance differs')
        if provenance['truth_sha256']!=case['truth']['manifest']['field_sha256'] or provenance['nominal_sha256']!=case['nominal']['manifest']['field_sha256']:
            raise AssertionError('Formal field pairing differs')
        seed=int.from_bytes(hashlib.sha256(f"{plan['study_id']}:{family}:layout_noise:{row['layout']}".encode()).digest()[:4],'big')
        n={32:4,50:10,70:20}[side]
        expected=EpisodeConfig(map_shape=(side,side),n_drones=n,n_targets=n*3,seed=seed,observation_delay_s=row['delay_s'],noise_mode='counter_keyed')
        candidates=np.argwhere(np.ones((side,side),bool));rng=np.random.default_rng(np.random.SeedSequence([seed,1]))
        targets=candidates[rng.choice(len(candidates),n*3,replace=False)].tolist()
        if targets!=[t['cell'] for t in result['targets']]:raise AssertionError('Hidden target layout was not purely geometry/random-seed based')
        if row['summary']!=result['summary'] or row['outcome_analysis']!=result['outcome_analysis']:raise AssertionError('Primary statistics differ from raw result')
        for key,value in audit_saved_episode(episode['path'],truth,expected).items():totals[key]=totals.get(key,0)+value
    return totals


def audit_formal(root,run,frozen_path,workers=3):
    root,run=Path(root),Path(run);frozen=verify_frozen(root,frozen_path);plan=register(root)
    registered=load_protocol(root/'registrations/formal_test.v1.json')
    if registered!=load_protocol(run/'formal_registration.json') or registered['run']!=str(run.resolve()):raise AssertionError('Single test registration differs')
    if registered['frozen_sha256']!=sha256(frozen_path):raise AssertionError('Frozen configuration registration changed')
    summary=load_protocol(run/'formal_summary.json')
    if summary['status']!='completed' or summary['episodes']!=3024:raise ValueError('Complete independent test required')
    expected={r['id']:r for r in frozen['formal_matrix']};seen=set()
    for row in summary['rows']:
        if row['id'] in seen or row['id'] not in expected or any(row[k]!=v for k,v in expected[row['id']].items()):raise AssertionError('Formal test identity/matrix changed')
        seen.add(row['id'])
    if seen!=set(expected):raise AssertionError('Primary test matrix incomplete')
    for relative,digest in load_protocol(run/'source_sha256.json').items():
        if sha256(run/'source_snapshot'/relative)!=digest:raise AssertionError('Formal execution source snapshot changed')
    registry={(r['method'],r['seed']):r for r in frozen['checkpoints']};cases=load_protocol(run/'field_summary.json')['cases'];totals={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_audit_group,case,[r for r in summary['rows'] if (r['family_id'],r['side'])==(case['family']['family_id'],case['side'])],plan,registry) for case in cases]
        for i,future in enumerate(as_completed(futures),1):
            for key,value in future.result().items():totals[key]=totals.get(key,0)+value
            print(f'Independent serialized formal audit {i}/108 hydraulic cases',flush=True)
    if totals.get('episodes')!=3024:raise AssertionError('Formal read-back audit episode count differs')
    result=dict(status='passed',**totals,formal_summary_sha256=sha256(run/'formal_summary.json'),
        frozen_sha256=sha256(frozen_path),full_service_scores_and_payloads_checked=True,all_failure_denominators_preserved=True,
        sampled_noise_checks_explicit=True,observed_footprints_and_flight_checked_for_all_records=True)
    write_json(run/'formal_integrity_audit.json',result)
    return result


def _audit_diagnostic_group(case,rows,kind,registry,source_rows):
    totals={}
    for name in ('truth','nominal'):
        manifest=case[name]['manifest']
        if sha256(manifest['field_path'])!=manifest['field_sha256']:raise AssertionError('Diagnostic field changed')
    truth=load_field(case['truth']['manifest']['field_path'])
    for row in rows:
        record=row['episode']
        if sha256(record['path'])!=record['sha256']:raise AssertionError('Diagnostic episode changed')
        episode=load_protocol(record['path']);execution=episode['execution_record'];origin=source_rows[row['id']]['episode']
        primary=load_protocol(origin['path'])
        if sha256(origin['path'])!=origin['sha256']:raise AssertionError('Diagnostic primary pair changed')
        expected=dict(primary['execution_record']['config'])
        if kind=='reassignment_counterfactual':expected['reassignment_policy']='defer_new_evidence'
        if execution['config']!=expected or execution['experiment_role']!=kind or execution['formal_test']:
            raise AssertionError('Diagnostic controller/settings differ from registered contrast')
        checkpoint=registry[row['method'],row['training_seed']]
        if any(execution['provenance'][k]!=v for k,v in checkpoint.items()):raise AssertionError('Diagnostic checkpoint differs')
        if [t['cell'] for t in primary['targets']]!=[t['cell'] for t in episode['targets']]:raise AssertionError('Diagnostic targets differ')
        if kind=='reassignment_counterfactual':
            from .posttest import prefix_comparison
            if row['prefix_check']!=prefix_comparison(primary,episode):raise AssertionError('Intervention prefix contamination check differs')
            if row['actual_additional_successful_services_with_full_policy']!=primary['summary']['successful_services']-episode['summary']['successful_services']:
                raise AssertionError('Actual counterfactual service gain differs')
        if row['summary']!=episode['summary'] or row['outcome_analysis']!=episode['outcome_analysis']:raise AssertionError('Diagnostic summary differs')
        for key,value in audit_saved_episode(record['path'],truth).items():totals[key]=totals.get(key,0)+value
    return totals


def audit_diagnostic(root,run,frozen_path,workers=3):
    root,run=Path(root),Path(run);frozen=verify_frozen(root,frozen_path)
    summary=load_protocol(run/'diagnostic_summary.json');kind=summary['kind']
    if summary['status']!='completed' or summary['episodes']!=1728:raise ValueError('Incomplete paired diagnostic')
    registration=load_protocol(root/'registrations'/f'{kind}.v1.json')
    if registration!=load_protocol(run/'diagnostic_registration.json') or registration['run']!=str(run.resolve()):raise AssertionError('Diagnostic registration changed')
    primary=Path(registration['primary']);source=load_protocol(primary/'formal_summary.json')
    if sha256(primary/'formal_summary.json')!=registration['primary_sha256']:raise AssertionError('Diagnostic source primary summary changed')
    source_rows={r['id']:r for r in source['rows']};expected={r['id'] for r in frozen['formal_matrix'] if r['method'] in ('P',frozen['B_star'])}
    seen={r['id'] for r in summary['rows']}
    if seen!=expected or len(summary['rows'])!=len(seen):raise AssertionError('Diagnostic matrix differs')
    cases=load_protocol((run if kind=='numerical_direction' else primary)/'field_summary.json')['cases']
    registry={(r['method'],r['seed']):r for r in frozen['checkpoints']};totals={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_audit_diagnostic_group,case,[r for r in summary['rows'] if (r['family_id'],r['side'])==(case['family']['family_id'],case['side'])],kind,registry,source_rows) for case in cases]
        for index,future in enumerate(as_completed(futures),1):
            for key,value in future.result().items():totals[key]=totals.get(key,0)+value
            if index%12==0:print(f'{kind} independent audit {index}/108 cases',flush=True)
    if totals.get('episodes')!=1728:raise AssertionError('Paired diagnostic audit incomplete')
    result=dict(status='passed',kind=kind,**totals,summary_sha256=sha256(run/'diagnostic_summary.json'))
    write_json(run/'diagnostic_integrity_audit.json',result)
    return result
