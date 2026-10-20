"""Independent checks of every shared raw observation and model-input panel."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from .artifacts import sha256,write_json
from .diagnostics import load_field
from .execution_plan import register
from .model_features import full_raw_frame
from .protocol import load_protocol


def _audit_case(case):
    side=case['side'];n=side*side
    fields={}
    for kind in ('truth','nominal'):
        info=case[kind];manifest=info['manifest']
        if sha256(manifest['field_path'])!=manifest['field_sha256']:
            raise AssertionError('Shared hydraulic field changed')
        if manifest['kind']!=kind or not manifest['independent_initial_condition']:
            raise AssertionError('Hydraulic independence marker differs')
        fields[kind]=load_field(manifest['field_path'])
        if not np.array_equal(fields[kind]['times_s'],np.arange(1201.)):
            raise AssertionError('Expected independent 1s hydraulic output')
    totals=dict(panels=0,observations=0,verified_raw_feature_entries=0)
    for info in case['panels']:
        if sha256(info['path'])!=info['sha256'] or sha256(info['raw_observations']['path'])!=info['raw_observations']['sha256']:
            raise AssertionError('Shared observation/panel log changed')
        if sha256(info['trajectory_path'])!=info['trajectory_sha256']:
            raise AssertionError('Patrol trajectory changed')
        panel=load_field(info['path']);obs=load_field(info['raw_observations']['path'])
        times=np.arange(0.,1201.,5.)
        if not np.array_equal(panel['times_s'],times) or not np.array_equal(obs['sample_time_s'],obs['arrival_time_s']):
            raise AssertionError('Collection clock or zero-delay transport differs')
        if not np.array_equal(panel['label_truth_sample_m'],fields['truth']['depths_m'][::5]) or not np.array_equal(panel['nominal_sample_m'],fields['nominal']['depths_m'][::5]):
            raise AssertionError('Stored labels/public forecast do not match their own independent fields')
        trajectory=load_protocol(info['trajectory_path']);positions={}
        previous={}
        for row in trajectory:
            key=(row['time_s'],row['drone'])
            if key in positions:raise AssertionError('Duplicate trajectory sample')
            position=np.asarray(row['position_m'])
            if row['drone'] in previous:
                before,t=previous[row['drone']]
                if np.linalg.norm(position-before)>5*(row['time_s']-t)+1e-6:
                    raise AssertionError('Patrol exceeds physical speed')
            positions[key]=position;previous[row['drone']]=(position,row['time_s'])
        tick=np.rint(obs['sample_time_s']/5).astype(int)
        if np.any(np.abs(tick*5-obs['sample_time_s'])>1e-8):raise AssertionError('Off-grid sample clock')
        cell=obs['row']*side+obs['col'];flat=tick*n+cell
        expected_truth=fields['truth']['depths_m'][np.rint(obs['sample_time_s']).astype(int),cell].astype(float)
        error=obs['depth_m']-expected_truth
        if abs(error.mean())>.001 or not .018<error.std()<.022:
            raise AssertionError('Raw sensor noise scale/bias differs from 2cm contract')
        for t,d,r,c in zip(obs['sample_time_s'],obs['drone'],obs['row'],obs['col']):
            col,row=np.floor(positions[t,int(d)]/10).astype(int)
            if abs(int(r)-row)>2 or abs(int(c)-col)>2:
                raise AssertionError('Observation outside actual-position footprint')
        count=np.zeros((241,n),int);total=np.zeros((241,n));squared=np.zeros((241,n))
        np.add.at(count.ravel(),flat,1);np.add.at(total.ravel(),flat,obs['depth_m']);np.add.at(squared.ravel(),flat,obs['depth_m']**2)
        mean=total/count.clip(1);std=np.sqrt(np.maximum(0.,squared/count.clip(1)-mean**2))
        if not np.array_equal(panel['count'],count) or not np.allclose(panel['mean_depth_m'],mean,atol=1e-7,rtol=0) or not np.allclose(panel['std_depth_m'],std,atol=1e-7,rtol=0):
            raise AssertionError('Raw observations are not conserved in fused history statistics')
        for now,delay in ((0.,0.),(15.,10.),(600.,0.),(1195.,10.)):
            features=full_raw_frame(panel,obs,now,delay)
            legal=(obs['sample_time_s']>=now-60)&(obs['sample_time_s']<=now)&(obs['arrival_time_s']+delay<=now)
            if int(features[:,:,26:].sum())!=int(legal.sum()):
                raise AssertionError('Complete raw history lost samples or included unarrived data')
            chosen=np.flatnonzero(legal);slot=(12-(now-obs['sample_time_s'][chosen])/5).astype(int)
            values=features[cell[chosen],slot,6+obs['drone'][chosen]]
            if not np.array_equal(values,obs['depth_m'][chosen].astype(np.float32)):
                raise AssertionError('An individual raw sample changed before the shared model')
            totals['verified_raw_feature_entries']+=len(chosen)
        totals['panels']+=1;totals['observations']+=len(cell)
    return totals


def audit_common_data(root,data_run,workers=3):
    root,data_run=Path(root),Path(data_run);plan=register(root)
    summary=load_protocol(data_run/'common_data_summary.json')
    if summary['status']!='completed' or summary['cases']!=108 or summary['panels']!=216:
        raise ValueError('Complete shared records required')
    expected={(f['family_id'],side):f for f in plan['allocation'] if f['split'] in ('train','validation') for side in (32,50,70)}
    seen=set()
    for case in summary['records']:
        key=case['family']['family_id'],case['side']
        if key not in expected or key in seen or case['family']!=expected[key]:
            raise AssertionError('Shared family/scale/condition allocation differs')
        seen.add(key)
        path=data_run/'jobs'/f'{key[0]}_side{key[1]}'/'case.json'
        if load_protocol(path)!=case:raise AssertionError('Case marker differs from data summary')
    if seen!=set(expected):raise AssertionError('Shared family matrix incomplete')
    totals={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_audit_case,case) for case in summary['records']]
        for index,future in enumerate(as_completed(futures),1):
            for key,value in future.result().items():totals[key]=totals.get(key,0)+value
            if index%12==0:print(f'Shared-data audit {index}/108 cases',flush=True)
    result=dict(status='passed',cases=108,**totals,test_accessed=False,all_training_and_validation_families=True,
                summary_sha256=sha256(data_run/'common_data_summary.json'))
    write_json(data_run/'common_data_integrity_audit.json',result)
    return result
