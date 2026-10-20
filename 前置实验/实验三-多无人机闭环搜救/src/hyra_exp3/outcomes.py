"""Post-execution target accounting; true windows never feed the controller."""
import numpy as np
from .windows import service_start_intervals


def analyze_outcomes(truth,result,predictor,config):
    categories={name:0 for name in ('undiscovered','no_true_window','discovered_after_last_window',
        'window_misclassification','execution_failure','resource_or_time_exhausted')}
    windows={}
    for target in result['targets']:
        r,c=target['cell']
        windows[target['id']]=service_start_intervals(truth['times_s'],
            truth['depths_m'][:,r*config.map_shape[1]+c],config.service_duration_s,config.max_service_depth_m)
    failed_forecast=set();failed_execution=set();false_unsafe=set();eligible=[]
    for event in result['events']:
        if event['event']=='service_complete' and not event['success']:
            target=result['targets'][event['target']]
            estimate=predictor.published_windows[event['prediction_version'],tuple(target['cell'])]
            (failed_forecast if estimate.can_start(event['start_s']) is True else failed_execution).add(target['id'])
        if event['event']!='plan':
            continue
        now=event['time_s'];version=event['version']
        free=[r for r in event.get('planning_state',[]) if r['mode']!='service' and r['payload']>0]
        for target in result['targets']:
            if target['served_at'] is not None or target['discovered_at'] is None or target['discovered_at']>now+1e-8:
                continue
            estimate=predictor.published_windows.get((version,tuple(target['cell'])))
            if estimate is None:
                continue
            r,c=target['cell'];position=np.array([(c+.5)*config.cell_size_m,(r+.5)*config.cell_size_m])
            for robot in free:
                arrival=now+float(np.linalg.norm(position-robot['position_m']))/config.speed_mps
                true_start=next((max(a,arrival) for a,b in windows[target['id']] if max(a,arrival)<=b+1e-8),None)
                if true_start is not None and estimate.can_start(true_start) is False:
                    false_unsafe.add(target['id'])
        if 'periodic' not in event['reason']:
            states={r['drone']:r for r in event.get('planning_state',[])}
            for change in event.get('reassignment_details',[]):
                robot=states[change['drone']];tid=change['previous_target'];target=result['targets'][tid]
                if robot['payload']<=0 or (target['served_at'] is not None and target['served_at']<=now):
                    continue
                r,c=target['cell'];xy=np.array([(c+.5)*config.cell_size_m,(r+.5)*config.cell_size_m])
                arrival=now+float(np.linalg.norm(xy-robot['position_m']))/config.speed_mps
                start=max(arrival,robot['planned_start'] if robot['planned_start'] is not None else arrival)
                estimate=predictor.published_windows.get((version,tuple(target['cell'])))
                if estimate is not None and estimate.can_start(start) is True:
                    eligible.append(dict(time_s=now,drone=robot['drone'],target=tid,version=version))
    rows=[]
    for target in result['targets']:
        tid=target['id']
        if target['served_at'] is not None:
            label='served'
        elif target['discovered_at'] is None:
            label='undiscovered'
        elif not windows[tid]:
            label='no_true_window'
        elif target['discovered_at']>windows[tid][-1][1]+1e-8:
            label='discovered_after_last_window'
        elif tid in failed_execution:
            label='execution_failure'
        elif tid in failed_forecast or tid in false_unsafe:
            label='window_misclassification'
        else:
            label='resource_or_time_exhausted'
        if label!='served':
            categories[label]+=1
        rows.append(dict(target=tid,category=label,true_safe_start_intervals=windows[tid],
                         supported_false_unsafe_evidence=tid in false_unsafe))
    if sum(categories.values())!=config.n_targets-result['summary']['successful_services']:
        raise AssertionError('Failure target accounting does not conserve the denominator')
    return dict(categories=categories,targets=rows,no_true_window_all_hidden=sum(not v for v in windows.values()),
        new_evidence_reassignments=sum(e.get('reassignments',0) for e in result['events'] if e['event']=='plan' and 'periodic' not in e['reason']),
        eligible_defer_interventions=eligible,policy_interventions=sum(len(e.get('policy_interventions',[])) for e in result['events']),
        note='Exclusive descriptive attribution; supported false-unsafe evidence alone is not a causal service-gain claim')
