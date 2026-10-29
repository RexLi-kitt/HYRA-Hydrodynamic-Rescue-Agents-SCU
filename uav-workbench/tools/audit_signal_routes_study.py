"""Independent completion evidence, outside the frozen training source package.

Preflight covers actual E1 trajectories and fit-only feature normalization.
Final mode refuses partial studies and inspects every named study artifact.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import zipfile
import csv
import signal_routes
import numpy as np
import torch
from scale100_env import ScaleEnv,config
from sur_rescue.environment import rule_actions
from hyra_learning.physics import SWEOperator
from signal_routes import METHODS,SCHEMA
from signal_routes.belief import Parameters,SignalBeliefEstimator
from signal_routes.environment import RouteEnv
from signal_routes.offline import load_records,estimator
from signal_routes.data import cases
from signal_routes.common import DEFAULT,ROOT,read,write,digest,sources,registration


def require(condition,message):
    if not condition: raise AssertionError(message)


def write_evidence(path,value):
    """Never replace an earlier independent evidence artifact."""
    path=Path(path)
    if path.exists():
        require(read(path)==value,f'Refusing to replace existing evidence: {path.name}')
        return
    write(path,value)


def frozen_contract(study):
    pre=read(study/'independent-preflight.json')
    current=sources()
    require(pre['training_sources']==current,'Source changed since independent preflight')
    require(pre['feature_fit']['parameters_sha256']==digest(study/'parameters.json'),'Parameters changed since independent preflight')
    require(read(study/'registration.json')==registration(),'Registered contract changed')
    return dict(passed=True,sources=current,parameters_sha256=digest(study/'parameters.json'),
                dataset_sha256=digest(study/'data/dataset.json'),
                registration_sha256=digest(study/'registration.json'),
                task_selection_sha256=digest(study/'task-selection.json'))


def independent_statistics(parent_values):
    """Recompute all nine comparisons without calling the study reporter."""
    values={key:np.asarray(value,dtype=float) for key,value in parent_values.items()}
    require(set(values)==set(METHODS)|{'frontier','sweep_refresh','water_aware'},'Wrong report method set')
    require(all(v.shape==(8,) and np.isfinite(v).all() for v in values.values()),'Invalid parent-level scores')
    indices=np.random.default_rng(20261027).integers(0,8,size=(10000,8))
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=8)))
    output={}
    for i,method in enumerate(('R1','R2','R3'),1):
        for baseline in ('Bpost',f'C{i}','Braw'):
            difference=values[method]-values[baseline]
            estimate=float(difference.mean())
            samples=np.take(difference,indices).mean(axis=1)
            ci=np.quantile(samples,(.025,.975))
            probability=float(np.count_nonzero(np.abs(signs@difference/8)>=abs(estimate)-1e-15)/256)
            output[(method,baseline)]=dict(delta=estimate,ci95=ci.tolist(),
                paired_parent_deltas=difference.tolist(),exact_sign_flip_p=probability)
    primary=sorted((key for key in output if key[1]=='Bpost'),key=lambda key:output[key]['exact_sign_flip_p'])
    previous=0.
    for index,key in enumerate(primary):
        previous=max(previous,min(1.,(3-index)*output[key]['exact_sign_flip_p']))
        output[key]['holm_p']=previous
    return output


def report_audit(rows,report):
    parents=sorted({r['parent'] for r in rows})
    require(len(parents)==8,'Wrong independent parent count')
    expected_methods=set(METHODS)|{'frontier','sweep_refresh','water_aware'}
    require(set(report['means'])==expected_methods and set(report['parent_values'])==expected_methods,'Report omitted a method')
    values={}
    for method in expected_methods:
        # First average the fixed repeated conditions within each independent parent.
        values[method]=[float(np.mean([r['metrics']['rescue_fraction'] for r in rows
                                      if r['method']==method and r['parent']==parent])) for parent in parents]
        np.testing.assert_allclose(report['parent_values'][method],values[method],rtol=0,atol=1e-14)
        require(abs(report['means'][method]-np.mean(values[method]))<1e-14,'Wrong parent-level averaging')
    recalculated=independent_statistics(values)
    recorded={}
    for comparison in report['comparisons']:
        key=(comparison['method'],comparison['baseline'])
        require(key not in recorded,'Duplicate report comparison')
        recorded[key]=comparison
    require(set(recorded)==set(recalculated),'Missing matched-control or baseline comparison')
    for key,expected in recalculated.items():
        for field,value in expected.items():
            np.testing.assert_allclose(recorded[key][field],value,rtol=0,atol=1e-14,
                                       err_msg=f'Incorrect {key} {field}')
    primary=report['primary_holm']
    require(len(primary)==3 and {(r['method'],r['baseline']) for r in primary}==
            {(r,'Bpost') for r in ('R1','R2','R3')},'Wrong Holm family')
    for row in primary:
        require(row==recorded[(row['method'],row['baseline'])],'Holm table differs from comparison table')
    return dict(passed=True,independent_parents=parents,comparisons=9,
                matched_controls=['R1-C1','R2-C2','R3-C3'],bootstrap_replicates=10000,
                sign_flip_patterns=256,holm_family=['R1-Bpost','R2-Bpost','R3-Bpost'])


def permission_and_null_audit(study):
    """Adversarial checks use fit records only and include nonzero engineering coefficients."""
    parameters=read(study/'parameters.json')['parameters']; records=load_records(study/'fit-records.npz')
    subset=[r for r in records if r['seed']==11 and r['step']==12]
    require(len(subset)==12 and {r['parent'] for r in subset}==
            {f'signal-parent-{p}' for p in range(7101,7107)},'Unexpected fit-only permission records')
    for route,control in (('R1','C1'),('R2','C2'),('R3','C3')):
        for name in ('mean','scale','feature_mean','feature_scale'):
            require(parameters[route][name]==parameters[control][name],'Unmatched control normalization')
    effect={}
    for method in ('R1','R2','R3'):
        p=parameters[method]
        is_zero=(p['alpha']==0 or p['kappa']==0) if method=='R1' else (
                all(value==0 for value in p['beta']) if method=='R2' else p['strength']==0)
        errors=[]
        for record in subset:
            baseline=estimator('Bpost',parameters['Bpost'],record).posterior()['mean']
            actual=estimator(method,p,record).posterior()['mean']
            errors.append(float(np.max(np.abs(actual-baseline))))
        if is_zero: require(max(errors)<1e-12,'Zero physical coefficients do not recover baseline')
        effect[method]=dict(fitted_zero_effect=is_zero,max_fit_mean_difference_from_Bpost=max(errors),
            coefficients={key:p[key] for key in ('alpha','kappa','beta','strength','cross')},
            interpretation='zero-effect result; do not claim active physics' if is_zero else 'nonzero fitted candidate; benefit unproven')
    rng=np.random.default_rng(891); shape=(3,4); mask=np.zeros(shape,bool); mask[0,:]=True
    signal=rng.normal(size=shape); signal[0,0]=0.; water=rng.normal(size=(3,*shape))*.03
    water[0]=np.abs(water[0]); stamps=np.where(mask,0.,-1.); obstacles=np.zeros(shape,bool)
    engineering=Parameters(alpha=.7,kappa=2,beta=(.2,.1,.2,.1),strength=2,cross=.6)
    checks=[]
    for method in METHODS[1:]:
        originals=[array.copy() for array in (signal,water,mask,stamps,obstacles)]
        model=SignalBeliefEstimator(method,engineering).reset(shape)
        model.update(signal,water,mask,stamps,5.,obstacles)
        expected=model.posterior(True)
        poisoned_signal=signal.copy(); poisoned_water=water.copy(); poisoned_stamps=stamps.copy()
        poisoned_signal[~mask]=1e100; poisoned_water[:,~mask]=np.nan; poisoned_stamps[~mask]=1e100
        other=SignalBeliefEstimator(method,engineering).reset(shape)
        other.update(poisoned_signal,poisoned_water,mask,poisoned_stamps,5.,obstacles)
        actual=other.posterior(True)
        np.testing.assert_array_equal(actual['mean'],expected['mean'])
        np.testing.assert_array_equal(actual['variance'],expected['variance'])
        np.testing.assert_allclose(expected['mean'][mask],signal[mask],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(expected['variance'][mask],0.)
        for before,after in zip(originals,(signal,water,mask,stamps,obstacles)):
            np.testing.assert_array_equal(before,after)
        future=stamps.copy(); future[mask]=6.
        try: model.update(signal,water,mask,future,5.,obstacles)
        except ValueError: pass
        else: raise AssertionError('Estimator accepted future records')
        require(not any(hasattr(model,key) for key in ('source','targets','_targets','truth','water_truth')),
                'Estimator holds a privileged environment handle')
        checks.append(method)
    return dict(passed=True,fit_records=len(subset),matched_control_normalization=True,
                unknown_poison_future_and_exact_zero_checks=checks,engineering_nonzero_coefficients=True,
                selected_effects=effect,confirmation_content_read=False)


def e2_record_regression(study):
    """Run actual development truth with shared actions, independently of every actor."""
    parameters=read(study/'parameters.json')['parameters']; duration=read(study/'task-selection.json')['duration_s']
    rows=[]; unchanged=[i for i in range(21) if i not in (3,7)]
    for row,case in cases(study/'data/dataset.json','development'):
        cfg=replace(config('E2','B3',11),duration_s=float(duration))
        reference=ScaleEnv(case,cfg); view=reference.reset()
        routes={method:RouteEnv(case,cfg,method,Parameters(**parameters[method])) for method in METHODS}
        for env in routes.values(): env.reset()
        done=False; count=0; trace=hashlib.sha256()
        while not done:
            actions=rule_actions(view); view,reward,done=reference.step(actions); count+=1
            for method,env in routes.items():
                actual,r,d=env.step(actions)
                require(r==reward and d==done,'Signal estimator changed actual task/reward')
                np.testing.assert_array_equal(actual.images[:,unchanged],view.images[:,unchanged])
                np.testing.assert_array_equal(actual.available,view.available)
                require(actual.targets==view.targets and actual.busy==view.busy,'Estimator changed known target/service state')
                for attribute in ('signal','water','observed_s','explored','obstacles'):
                    np.testing.assert_array_equal(getattr(env,attribute),getattr(reference,attribute))
                if env.estimator:
                    np.testing.assert_allclose(env.signal_belief[env.explored],env.signal[env.explored],rtol=0,atol=1e-12)
            trace.update(reference.signal.tobytes()); trace.update(reference.observed_s.tobytes())
        for env in routes.values():
            require(env.observations==reference.observations and env.events==reference.events,'Raw observations/events changed')
            np.testing.assert_array_equal(env.fields._physical_signal,reference.fields._physical_signal)
        require(digest(row['path'])==row['sha256'],'Audit changed its source package')
        rows.append(dict(parent=row['parent_terrain'],mechanism=row['mechanism'],seed=11,
                         decisions=count,raw_trace_sha256=trace.hexdigest()))
    return dict(passed=True,development_events=len(rows),route_episode_comparisons=len(rows)*len(METHODS),
                only_changed_actor_channels=[3,7],raw_records_and_truth_unchanged=True,rows=rows)


def supplemental(study):
    study=Path(study); contract=frozen_contract(study)
    result=dict(schema=SCHEMA,audit_tool_sha256=digest(__file__),contract=contract,
                permissions_and_selected_effects=permission_and_null_audit(study),
                actual_e2_record_regression=e2_record_regression(study),
                confirmation_performance_read=False,completed_study_claim=False)
    require(frozen_contract(study)==contract,'Study inputs changed during independent audit')
    write_evidence(study/'independent-supplemental-preflight.json',result)
    return result


class CachedPhysicalCase:
    """Read-only in-memory physical frames; no study or solver file is modified."""
    def __init__(self,case):
        self.base=case; self.frames=[]
        with zipfile.ZipFile(case.path) as archive:
            for entry in case.process['frames']:
                array=np.frombuffer(archive.read(entry['url']),dtype='<f4').reshape(4,*case.shape).copy()
                array.setflags(write=False); self.frames.append(array)

    def __getattr__(self,name): return getattr(self.base,name)
    def values(self,time_s): return self.frames[self.base.index(time_s)]
    def labels(self,time_s): return self.values(time_s)[0]>=.05


def independent_saved_action_replays(study,rows,seal):
    """Rebuild every recorded run using the original environment, without the estimator.

    This independently verifies sensor values, task transitions and rewards. It
    does not claim to re-evaluate the neural policy or select any new model.
    """
    from sur_rescue.environment import ACTIONS
    physical={(row['parent_terrain'],row['mechanism']):CachedPhysicalCase(case)
              for row,case in cases(study/'data/dataset.json','holdout')}
    selected={(row['method'],row['seed']):row for row in seal['models']}
    environments={}; count=0; records=0; decisions=0
    duration=read(study/'task-selection.json')['duration_s']
    for row in rows:
        if row['method'] not in METHODS: continue
        path=Path(row['replay']).resolve()
        require(path.parent==(study/'confirmation').resolve(),'Replay points outside this study')
        saved=read(path); chosen=selected[(row['method'],row['training_seed'])]
        case=physical[(row['parent'],row['mechanism'])]
        require(saved['source_sha256']==case.digest and saved['actor_sha256']==chosen['actor_sha256'] and
                saved['parameters_sha256']==digest(study/'parameters.json'),'Replay provenance differs from seal')
        key=(row['parent'],row['mechanism'],row['scenario_seed'])
        if key not in environments:
            cfg=replace(config('E2','B3',row['scenario_seed']),duration_s=float(duration))
            environments[key]=ScaleEnv(case,cfg)
        reference=environments[key]; reference.reset(); total=0.; done=False
        for index,action in enumerate(saved['actions']):
            require(action['time_s']==index*5 and len(action['actions'])==2,'Saved action chronology invalid')
            require(not done,'Actions continued after the task deadline')
            _,reward,done=reference.step([ACTIONS.index(value) for value in action['actions']])
            total+=reward
        require(done and reference.time_s==duration,'Saved trajectory did not reach its registered deadline')
        require(reference.observations==saved['observations'],'Actual raw sensor records differ on independent replay')
        require(reference.events==saved['events'],'Discovery/service events differ on independent replay')
        require(total==saved['return_'],'Task return differs on independent replay')
        expected=reference.metrics()
        expected.update(first_coverage_auc=float(np.mean([np.mean(s['explored']) for s in reference.snapshots[:-1]])),
            first_discovery_s=next((e['time_s'] for e in reference.events if e['type']=='discovered'),None),
            first_rescue_s=next((e['time_s'] for e in reference.events if e['type']=='rescued'),None),
            unrescued=5-sum(s=='rescued' for s in reference.target_status.values()),
            service_interruptions=sum(e['type']=='service_interrupted' for e in reference.events))
        for name,value in expected.items():
            require(saved['metrics'][name]==value,f'Task metric {name} differs on independent replay')
        count+=1; records+=len(reference.observations); decisions+=len(reference.actions)
        if count%96==0: print(json.dumps(dict(audit_stage='independent_raw_saved_action_replays',completed=count,total=1728)),flush=True)
    require(count==1728,'Not every model replay was independently reconstructed')
    return dict(passed=True,model_runs=count,actual_sensor_records=records,decisions=decisions,
                physical_case_layouts=len(environments),raw_signals_water_timestamps_events_and_rewards_equal=True,
                neural_policy_recomputed=False,selection_changed=False)


def independent_rule_replays(study,rows):
    from scale100_development import legal_rule
    from scale100_rules import water_aware
    physical={(row['parent_terrain'],row['mechanism']):CachedPhysicalCase(case)
              for row,case in cases(study/'data/dataset.json','holdout')}
    duration=read(study/'task-selection.json')['duration_s']; count=0
    for row in rows:
        if row['method'] in METHODS: continue
        env=ScaleEnv(physical[(row['parent'],row['mechanism'])],
                     replace(config('E2','B3',row['scenario_seed']),duration_s=float(duration)))
        env.reset(); done=False
        while not done:
            actions=water_aware(env) if row['method']=='water_aware' else legal_rule(env,row['method'])
            _,_,done=env.step(actions)
        metrics=env.metrics(); metrics.update(rescue_fraction=metrics['rescued']/5,
            service_interruptions=sum(e['type']=='service_interrupted' for e in env.events))
        require(metrics==row['metrics'],'Rule result differs on independent recomputation')
        count+=1
    require(count==216,'Not every rule episode was independently recomputed')
    return dict(passed=True,rule_runs=count,actual_task_metrics_equal=True)


def selftest():
    """Explicit in-memory engineering fixtures; never write study completion evidence."""
    base=np.full(8,.5)
    values={method:base.copy() for method in (*METHODS,'frontier','sweep_refresh','water_aware')}
    values['R1']=np.full(8,.6)
    values['R3']=np.array([.6,.4]*4)
    statistics=independent_statistics(values)
    np.testing.assert_allclose(statistics[('R1','Bpost')]['ci95'],[.1,.1],rtol=0,atol=1e-14)
    require(statistics[('R1','Bpost')]['exact_sign_flip_p']==2/256,'Constant-effect exact sign-flip fixture failed')
    require(statistics[('R1','Bpost')]['holm_p']==6/256,'Three-comparison Holm fixture failed')
    require(statistics[('R2','Bpost')]['exact_sign_flip_p']==1 and
            statistics[('R2','Bpost')]['ci95']==[0.,0.],'Null-effect statistics fixture failed')
    rows=[dict(method=method,parent=f'engineering-parent-{i}',metrics={'rescue_fraction':float(value)})
          for method,array in values.items() for i,value in enumerate(array)]
    comparisons=[dict(method=method,baseline=baseline,**entry) for (method,baseline),entry in statistics.items()]
    report=dict(parent_values={key:value.tolist() for key,value in values.items()},
                means={key:float(value.mean()) for key,value in values.items()},comparisons=comparisons,
                primary_holm=sorted([row for row in comparisons if row['baseline']=='Bpost'],key=lambda row:row['exact_sign_flip_p']))
    report_audit(rows,report)
    import copy
    rejected=[]
    for name in ('confidence_interval','holm','matched_control','parent_mean'):
        changed=copy.deepcopy(report)
        if name=='confidence_interval': changed['comparisons'][0]['ci95'][0]+=.01
        elif name=='holm': changed['comparisons'][0]['holm_p']+=.01
        elif name=='matched_control': changed['comparisons']=[r for r in changed['comparisons'] if (r['method'],r['baseline'])!=('R2','C2')]
        else: changed['parent_values']['Bpost'][0]+=.01
        try: report_audit(rows,changed)
        except (AssertionError,KeyError): rejected.append(name)
        else: raise AssertionError(f'Failed to reject altered engineering fixture: {name}')
    return dict(passed=True,scope='in-memory engineering fixtures only',statistics_cases=['constant_effect','zero_effect','balanced_signs'],
                altered_evidence_rejected=rejected,real_study_read=False,completion_evidence_written=False)


def normalized_training_features(study):
    records=load_records(study/'fit-records.npz'); parameters=read(study/'parameters.json')
    require(parameters['fit_only'],'Parameters not declared fit-only')
    require(parameters['fit_records_sha256']==digest(study/'fit-records.npz'),'Fit record digest changed')
    data=[estimator('Bpost',parameters['parameters']['Bpost'],record).raw_features() for record in records]
    f=np.concatenate([r[0] for r in data]); mask=np.concatenate([r[1] for r in data])
    mean=[float(f[mask[:,i],i].mean()) if mask[:,i].any() else 0. for i in range(4)]
    scale=[max(float(f[mask[:,i],i].std()),1e-6) if mask[:,i].any() else 1. for i in range(4)]
    for method in METHODS:
        p=parameters['parameters'][method]
        np.testing.assert_allclose(p['feature_mean'],mean,rtol=0,atol=1e-14)
        np.testing.assert_allclose(p['feature_scale'],scale,rtol=0,atol=1e-14)
    return dict(passed=True,records=len(records),available_counts=mask.sum(axis=0).tolist(),
                means=mean,scales=scale,parameters_sha256=digest(study/'parameters.json'))


def e1_regression(study):
    rows=[]; params=read(study/'parameters.json')['parameters']
    for row,case in cases(study/'data/dataset.json','development'):
        cfg=replace(config('E1','B3',11),duration_s=900.)
        reference=ScaleEnv(case,cfg); view=reference.reset()
        routes={method:RouteEnv(case,cfg,method,Parameters(**params[method])) for method in METHODS}
        for env in routes.values(): np.testing.assert_array_equal(env.reset().images,view.images)
        h=hashlib.sha256(); decisions=0; reward_sum=0.
        done=False
        while not done:
            actions=rule_actions(view); view,reward,done=reference.step(actions)
            h.update(view.images.tobytes()); reward_sum+=reward; decisions+=1
            for method,env in routes.items():
                other,r,d=env.step(actions)
                require(r==reward and d==done,'E1 task/reward changed')
                np.testing.assert_array_equal(other.images,view.images)
                np.testing.assert_array_equal(other.images[:,[3,7]],0.)
        for env in routes.values():
            require(env.observations==reference.observations,'E1 observation records differ')
            require(env.events==reference.events,'E1 events differ')
            require(env._targets==[],'E1 targets are not empty')
        rows.append(dict(parent=row['parent_terrain'],mechanism=row['mechanism'],seed=11,
                         decisions=decisions,matched_methods=list(METHODS),actor_input_trace_sha256=h.hexdigest(),return_=reward_sum))
    return dict(passed=True,full_actual_events=len(rows),route_episode_comparisons=len(rows)*8,rows=rows)


def data_audit(study,final=False):
    manifest=read(study/'data/dataset.json'); rows=manifest['cases']; split={}
    expected={(name,seed,mechanism) for name,seeds in registration()['split'].items() if name in ('fit','development','holdout')
              for seed in seeds for mechanism in (('slow','rapid','double') if name=='holdout' else ('slow','rapid'))}
    actual={(r['split'],r['seed'],r['mechanism']) for r in rows}
    require(len(rows)==42 and actual==expected,'Missing/duplicate physical event membership')
    for row in rows: split.setdefault(row['split'],set()).add(row['parent_terrain'])
    require({k:len(v) for k,v in split.items()}==dict(fit=6,development=3,holdout=8),'Wrong parent counts')
    for a,b in itertools.combinations(split,2): require(not(split[a]&split[b]),'Parent leakage')
    checked=[]; errors=[]; frame_count=0
    torch.set_num_threads(1); operator=SWEOperator(dx=5.)
    for row in rows:
        require(digest(row['path'])==row['sha256'],'Hydraulic package changed')
        with zipfile.ZipFile(row['path']) as archive:
            process=json.loads(archive.read('process.json'))
            require([f['time_s'] for f in process['frames']]==list(range(0,901,5)),'Nonactual/missing time frames')
            require(process['grid']['shape']==[20,20] and process['grid']['dx_m']==5,'Wrong grid geometry')
            if row['split']=='holdout' and not final: continue
            for frame in process['frames']:
                fields=np.frombuffer(archive.read(frame['url']),dtype='<f4').reshape(4,20,20)
                require(np.isfinite(fields).all() and fields[0].min()>=-1e-8,'Invalid hydraulic frame')
                frame_count+=1
        from scale100_env import FloodCase
        case=FloodCase(row['path'])
        for i in (0,19,20,79,80,139,140,180):
            prediction=operator.step(torch.tensor(case.probes['before'][i])[None],torch.tensor(case.bed),
                         float(case.probes['dt'][i]),source=torch.tensor(case.probes['source'][i]))
            errors.append(float((prediction[0]-torch.tensor(case.probes['after'][i])).abs().max()))
        checked.append(dict(parent=row['parent_terrain'],mechanism=row['mechanism'],split=row['split'],sha256=row['sha256']))
    require(max(errors)<1e-10,'New physical events fail accepted-step SWE consistency')
    return dict(passed=True,events=len(rows),split_parents={k:len(v) for k,v in split.items()},
                full_numerical_events=len(checked),actual_frames_checked=frame_count,
                max_internal_step_error=max(errors),holdout_numerical_content_read=final,performance_scored=False,checked=checked)


def preflight(study):
    study=Path(study); register=read(study/'registration.json')
    require(register==registration(),'Registration changed')
    result=dict(schema=SCHEMA,audit_tool_sha256=digest(__file__),training_sources=sources(),
                feature_fit=normalized_training_features(study),
                data=data_audit(study,False),e1=e1_regression(study))
    write_evidence(study/'independent-preflight.json',result)
    return result


def completed(study):
    study=Path(study); registration_data=read(study/'registration.json')
    require(read(study/'status.json')['status']=='completed','Controller is not completed')
    require(registration_data==registration(),'Registration mismatch')
    require(read(study/'checks.json')['passed'],'Numerical/permission checks missing')
    require(read(study/'checks.json')['sources']==sources(),'Training checks have stale source')
    recovery=read(study/'recovery-proof.json')
    require(recovery['sources']==sources() and recovery['actor_bitwise_equal'] and recovery['history_equal'],'Recovery evidence invalid')
    require(read(study/'pilot-acceptance.json')['passed'],'Pilot gate missing')
    require(read(study/'independent-null-audit.json')['donor_parent_always_different'],'Independent null diagnostic missing')
    require(read(study/'independent-null-audit.json')['main_policy_parameters_sha256']==digest(study/'parameters.json'),'Main parameters changed')
    pre=read(study/'independent-preflight.json')
    require(pre['training_sources']==sources() and pre['e1']['passed'],'Actual E1 regression evidence missing/stale')
    require(pre['feature_fit']['parameters_sha256']==digest(study/'parameters.json'),'Training feature-normalization evidence stale')
    contract=frozen_contract(study)
    additional=read(study/'independent-supplemental-preflight.json')
    require(additional['contract']==contract,'Supplemental permission/regression evidence stale')
    require(additional['permissions_and_selected_effects']['passed'] and
            additional['actual_e2_record_regression']['passed'],'Supplemental raw-record evidence failed')
    require(not additional['confirmation_performance_read'],'Supplemental preflight used confirmation performance')
    training=[]; initial={}; selected_counts=[]
    for method in METHODS:
        pilot=read(study/'pilot'/f'{method}-s11'/'progress.json')
        require(pilot['status']=='completed' and pilot['episode']==128 and pilot['updates']==125,'Pilot budget incomplete')
        for seed in (11,29,47):
            root=study/'formal'/f'{method}-s{seed}'; meta=read(root/'actor.json'); progress=read(root/'progress.json')
            require(meta['schema']==SCHEMA and meta['sources']==sources(),'Formal model source/version changed')
            require(meta['method']==method and meta['seed']==seed and meta['phase']=='formal','Mixed model identity')
            require(meta['dataset_sha256']==contract['dataset_sha256'] and
                    meta['task_selection_sha256']==contract['task_selection_sha256'],'Formal data/task contract changed')
            require(meta['auxiliary_losses'] is False and meta['surprise_training_only'] is True,'Unexpected learning mechanism')
            require(meta['trained_episodes']==1024 and meta['updates']==1021,'Formal budget incomplete')
            require(progress['status']=='completed' and progress['episode']==1024,'Formal progress not completed')
            require(meta['actor_sha256']==digest(root/'actor.pt'),'Formal actor digest mismatch')
            require(meta['parameters_sha256']==digest(study/'parameters.json'),'Mixed estimator parameters')
            history=read(root/'history.json'); require(len(history)==1024,'Training history incomplete')
            require([r['episode'] for r in history]==list(range(1,1025)),'Missing/repeated episodes')
            for i,r in enumerate(history):
                if i<3: require(r['update'] is None,'Unexpected early optimizer update')
                else:
                    update=r['update']; require(update['updates']==i-2,'Missing/repeated optimizer update')
                    require(update['data']==0 and update['physics']==0,'Unexpected auxiliary mechanism')
                    require(np.isfinite([update['td'],update['gradient_norm']]).all(),'Nonfinite TD optimization')
            init=torch.load(root/'initial-actor.pt',weights_only=True,map_location='cpu')
            if seed in initial:
                require(all(torch.equal(init[k],initial[seed][k]) for k in init),'Unmatched actor initialization')
            else: initial[seed]=init
            for episode in (256,512,1024):
                path=root/f'actor-{episode:04d}.pt'; candidate=read(path.with_suffix('.json'))
                require(candidate['trained_episodes']==episode and candidate['actor_sha256']==digest(path),'Selection checkpoint missing/changed')
            training.append(dict(method=method,seed=seed,episodes=1024,updates=1021,actor_sha256=meta['actor_sha256']))
    seal=read(study/'sealed-models.json'); opened=read(study/'confirmation/opened.json')
    require(len(seal['models'])==24 and len(seal['candidates'])==72,'Selection opportunities differ')
    require(seal['development_only'] and not seal['confirmation_opened'],'Seal did not precede confirmation')
    require(seal['parameters_sha256']==contract['parameters_sha256'] and
            seal['dataset_sha256']==contract['dataset_sha256'],'Seal inputs differ from training')
    require({(r['method'],r['seed'],r['episode']) for r in seal['candidates']}==
            {(m,s,e) for m in METHODS for s in (11,29,47) for e in (256,512,1024)},'Incomplete development candidate identities')
    require({(r['method'],r['seed']) for r in seal['models']}==
            {(m,s) for m in METHODS for s in (11,29,47)},'Incomplete selected model identities')
    require(opened['seal_sha256']==digest(study/'sealed-models.json'),'Confirmation is not bound to model seal')
    require(seal['sources']==sources() and opened['sources']==sources(),'Mixed confirmation source')
    for model in seal['models']:
        require(digest(model['actor'])==model['actor_sha256'],'Sealed actor changed')
        pool=[r for r in seal['candidates'] if (r['method'],r['seed'])==(model['method'],model['seed'])]
        require(len(pool)==3 and max(pool,key=lambda r:r['score'])==model,'Development selection rule violated')
    for candidate in seal['candidates']:
        require(len(candidate['score'])==4 and np.isfinite(candidate['score']).all() and
                candidate['score'][-1]==-candidate['episode'],'Invalid checkpoint selection score')
        require(digest(candidate['actor'])==candidate['actor_sha256'],'Development candidate actor changed')
    result=read(study/'confirmation/results.json'); rows=result['rows']; known=set(); audits=0
    steps=int(read(study/'task-selection.json')['duration_s']/5)
    require(len(rows)==1944 and result['models']==1728 and result['rules']==216,'Incomplete confirmation matrix')
    for row in rows:
        key=(row['method'],row['training_seed'],row['scenario_seed'],row['parent'],row['mechanism'])
        require(key not in known,'Duplicate confirmation episode'); known.add(key)
        metrics=row['metrics']; require(0<=metrics['rescue_fraction']<=1,'Invalid service metric')
        require(metrics['rescue_fraction']==metrics['rescued']/5,'Wrong service denominator')
        if row['method'] in METHODS:
            replay=read(row['replay']); audit=replay['replay_audit']
            require(all(audit[k] for k in ('raw_records_equal','final_actor_view_equal','return_equal')),'Replay reconstruction failed')
            require(replay['metrics']==metrics,'Replay and aggregate results differ')
            require(len(replay['actions'])==steps,'Incomplete decision trajectory')
            require(len(replay['observations'])==2*(steps+1),'Incomplete actual sensor records')
            for observation in replay['observations']:
                require(observation['observed_s'] in range(0,901,5),'Nonactual observation timestamp')
                n=len(observation['cells'])
                require(all(len(observation[k])==n for k in ('signal','obstacles','labels','depth_m','u_m_s','v_m_s')),'Observation arrays differ in length')
            audits+=1
    expected={(m,s,l,f'signal-parent-{p}',mechanism) for m in METHODS for s in (11,29,47)
              for l in (11,29,47) for p in range(7301,7309) for mechanism in ('slow','rapid','double')}
    expected|={(m,None,l,f'signal-parent-{p}',mechanism) for m in ('frontier','sweep_refresh','water_aware')
               for l in (11,29,47) for p in range(7301,7309) for mechanism in ('slow','rapid','double')}
    require(known==expected,'Confirmation matrix membership mismatch')
    for file in ('report.md','report.json','per-episode.csv','offline-diagnostics.json'):
        require((study/file).is_file() and (study/file).stat().st_size>0,f'Missing deliverable {file}')
    statistics=report_audit(rows,read(study/'report.json'))
    with (study/'per-episode.csv').open(encoding='utf-8-sig',newline='') as stream:
        exported=list(csv.DictReader(stream))
    require(len(exported)==len(rows),'CSV omitted episodes')
    for actual,expected_row in zip(exported,rows):
        require(actual['method']==expected_row['method'] and actual['parent']==expected_row['parent'] and
                actual['mechanism']==expected_row['mechanism'],'CSV episode identity changed')
        require(abs(float(actual['rescue_fraction'])-expected_row['metrics']['rescue_fraction'])<1e-14,'CSV service result changed')
    raw_replay=independent_saved_action_replays(study,rows,seal)
    rule_replay=independent_rule_replays(study,rows)
    data=data_audit(study,True)
    require(frozen_contract(study)==contract,'Frozen inputs changed during final audit')
    audit=dict(schema=SCHEMA,passed=True,scope='full registered v2 study; bounded synthetic evidence',
               audit_tool_sha256=digest(__file__),sources=sources(),training_jobs=training,
               completed_formal_jobs=24,completed_pilot_jobs=8,confirmation_model_replays=audits,
               rule_evaluations=216,independent_parents=8,data=data,
               independent_statistics=statistics,independent_raw_saved_action_replays=raw_replay,
               independent_rule_replays=rule_replay,
               selected_effects=additional['permissions_and_selected_effects']['selected_effects'],
               supplemental_preflight_sha256=digest(study/'independent-supplemental-preflight.json'),
               report_sha256=digest(study/'report.md'),results_sha256=digest(study/'confirmation/results.json'))
    write_evidence(study/'completion-audit.json',audit)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['preflight','supplemental','selftest','final']); p.add_argument('--study',type=Path,default=DEFAULT)
    args=p.parse_args()
    result=selftest() if args.mode=='selftest' else {'preflight':preflight,'supplemental':supplemental,'final':completed}[args.mode](args.study)
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','schema','completed_formal_jobs','confirmation_model_replays')},ensure_ascii=False))
