"""Independent two-agent actual-frame geometry, memory and metric audit.

Does not call TeamEnv, move, footprint, Memory, map_metrics or Task.result.
Views for actor checks are built from independently reconstructed observations.
"""
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_uav.core import Pose,PolicyView
from hyra_learning.team_env import TeamView


def metrics(truth,labels):
    present=labels!=-1;right=(labels==truth)&present
    balanced=[right[truth==klass].sum()/np.count_nonzero(truth==klass) for klass in (0,1) if np.any(truth==klass)]
    return dict(correct_coverage=float(np.mean(balanced)),unknown_fraction=float(1-present.mean()),
        observed_error_rate=float(np.count_nonzero(labels[present]!=truth[present])/present.sum()) if present.any() else None,
        first_coverage=float(present.mean()))


def visible_cells(pose,config,shape,dx):
    z,x=np.indices(shape);vx=x*dx-pose[0];vz=z*dx-pose[1]
    bearings=np.rad2deg(np.arctan2(vz,vx));difference=np.remainder(bearings-pose[2]+180,360)-180
    visible=(np.sqrt(vx*vx+vz*vz)<=config['range_m']+1e-9)&(np.abs(difference)<=config['fov_deg']/2+1e-9)
    col=max(0,min(shape[1]-1,math.floor(pose[0]/dx+.5)));row=max(0,min(shape[0]-1,math.floor(pose[1]/dx+.5)))
    visible[row,col]=True
    return np.flatnonzero(visible)


def motion(before,actions,config,team,shape,dx):
    result=before.copy();dt=config['sample_s']
    for agent,action in enumerate(actions):
        if action=='forward':
            radians=math.radians(before[agent,2])
            result[agent,0]=min((shape[1]-.5)*dx-1e-9,max(-dx/2,before[agent,0]+config['speed_m_s']*dt*math.cos(radians)))
            result[agent,1]=min((shape[0]-.5)*dx-1e-9,max(-dx/2,before[agent,1]+config['speed_m_s']*dt*math.sin(radians)))
        elif action=='left':result[agent,2]-=config['angular_speed_deg_s']*dt
        elif action=='right':result[agent,2]+=config['angular_speed_deg_s']*dt
        elif action!='hover':raise ValueError('Unknown action')
        result[agent,2]%=360
    relative=before[0,:2]-before[1,:2];change=(result[0,:2]-result[1,:2])-relative
    denominator=float(np.dot(change,change));fraction=np.clip(-np.dot(relative,change)/denominator,0,1) if denominator else 0
    collision=int(np.linalg.norm(relative+fraction*change)<team['separation_m']-1e-9)
    if collision:
        for agent,action in enumerate(actions):
            if action=='forward':result[agent]=before[agent]
    distance=sum(math.hypot(*(result[i,:2]-before[i,:2])) for i in range(2))
    turn=sum(action in ('left','right') for action in actions)*dt
    return result,distance,turn,collision


def assert_close(a,b):
    if a is None or b is None:assert a is b,(a,b)
    else:np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-9)


def rebuild(process,config,team,actions,initial_poses):
    """Rebuild all121 frames solely from requested actions and actual ZIP truth."""
    assert len(initial_poses)==2
    shape=process.shape;dx=process.dx
    poses=np.asarray([[p['x_m'],p['z_m'],p['heading_deg']] for p in initial_poses],dtype=float)
    labels=np.full(shape,-1,dtype=np.int8);timestamps=np.full(shape,-1.);quality=np.zeros(shape)
    history=[[] for _ in range(labels.size)]
    snapshots=[];observations=[];views=[];distance=turn=0.;collisions=0
    samples=round(config['action_s']/config['sample_s']);count=round(config['duration_s']/config['sample_s'])
    assert len(actions)==count//samples
    for i in range(count+1):
        now=i*config['sample_s'];action_index=(i-1)//samples
        if i:
            action_row=actions[action_index];assert action_row['time_s']==action_index*config['action_s']
            joint=action_row['actions'];assert len(joint)==2
            poses,dd,dt,dc=motion(poses,joint,config,team,shape,dx)
            distance+=dd;turn+=dt;collisions+=dc
        matches=np.flatnonzero(np.abs(process.times-now)<1e-7);assert len(matches)==1
        frame=int(matches[0]);truth=process.labels[frame]
        for agent in range(2):
            cells=visible_cells(poses[agent],config,shape,dx);values=truth.ravel()[cells].astype(int)
            observations.append(dict(agent_id=agent,frame_index=frame,observed_s=now,arrived_s=now,quality=1.,
                cells=cells.tolist(),labels=values.tolist(),action_index=action_index,
                sample_offset_s=0. if i==0 else ((i-1)%samples+1)*config['sample_s']))
            # Ordered agent IDs, first equal-time arrival wins. No model data.
            for cell,value in zip(cells,values):
                if timestamps.flat[cell]<now:
                    timestamps.flat[cell]=now;labels.flat[cell]=value;quality.flat[cell]=1
                    history[cell].append((now,now,int(value),1.));history[cell]=history[cell][-config['history_length']:]
        snapshot=dict(time_s=now,frame_index=frame,poses=[dict(x_m=float(p[0]),z_m=float(p[1]),heading_deg=float(p[2])) for p in poses],
            labels=labels.ravel().tolist(),observed_s=timestamps.ravel().tolist(),metrics=metrics(truth,labels),
            distance_m=distance,turn_s=turn,collision_events=collisions)
        snapshots.append(snapshot)
        if i%samples==0:
            agents=tuple(PolicyView(Pose(*pose),now,shape,dx,labels.copy(),timestamps.copy(),quality.copy(),
                tuple(tuple(h) for h in history)) for pose in poses)
            views.append(TeamView(agents,process.bed))
    return dict(snapshots=snapshots,observations=observations,views=views)


def audit_replay(replay,process):
    assert replay['schema']=='hyra-team-run-v1' and replay['n_agents']==2 and replay['source_sha256']==process.digest
    rebuilt=rebuild(process,replay['config'],replay['team'],replay['actions'],replay['initial_poses'])
    snapshots=rebuilt['snapshots'];saved=replay['snapshots'];observations=rebuilt['observations']
    assert len(saved)==len(snapshots)==121 and observations==replay['observations']
    for actual,expected,truth in zip(saved,snapshots,replay['truth']):
        assert actual['time_s']==expected['time_s'] and actual['frame_index']==expected['frame_index']
        for field in ('labels','observed_s'):assert actual[field]==expected[field]
        for field in ('distance_m','turn_s','collision_events'):assert_close(actual[field],expected[field])
        for a,b in zip(actual['poses'],expected['poses']):
            for field in a:assert_close(a[field],b[field])
        for field in expected['metrics']:assert_close(actual['metrics'][field],expected['metrics'][field])
        np.testing.assert_array_equal(truth,process.labels[expected['frame_index']].ravel())
    # Track each changed state until its next actual transition or cutoff.
    truths=np.asarray(replay['truth']);records=np.asarray([s['labels'] for s in snapshots]);arrivals=np.asarray([s['observed_s'] for s in snapshots])
    times=np.asarray([s['time_s'] for s in snapshots]);changes=[]
    for cell in range(truths.shape[1]):
        pending=None;segment=0
        for index in range(1,len(times)+1):
            changed=index<len(times) and truths[index,cell]!=truths[index-1,cell]
            if pending is not None and (changed or index==len(times)):
                begin,detected=pending;stop=float(times[min(index,len(times)-1)])
                changes.append(dict(cell=cell,segment=segment,changed_s=float(times[begin]),segment_end_s=stop,
                    detected_s=detected,delay_s=detected-times[begin] if detected is not None else None,
                    restricted_delay_s=(detected if detected is not None else stop)-times[begin]))
                segment+=1;pending=None
            if changed:pending=(index,None)
            if pending is not None and pending[1] is None and arrivals[index,cell]>=times[pending[0]] and records[index,cell]==truths[pending[0],cell]:
                pending=(pending[0],float(times[index]))
    assert changes==replay['changes']
    summary=replay['summary'];quality=sum(snapshots[i]['metrics']['correct_coverage']*(times[i+1]-times[i]) for i in range(len(times)-1))/times[-1]
    assert_close(summary['time_average_correct_coverage'],quality)
    for field in ('distance_m','turn_s','collision_events'):assert_close(summary[field],snapshots[-1][field])
    assert summary['change_count']==len(changes)
    assert_close(summary['undiscovered_change_fraction'],sum(c['detected_s'] is None for c in changes)/len(changes) if changes else None)
    assert_close(summary['restricted_change_delay_s'],np.mean([c['restricted_delay_s'] for c in changes]) if changes else None)
    for field in snapshots[-1]['metrics']:assert_close(summary['final'][field],snapshots[-1]['metrics'][field])
    reached=next((s['time_s'] for s in snapshots if s['metrics']['first_coverage']>=.9),None)
    assert_close(summary['first_90_percent_coverage_s'],reached)
    # Simultaneous observations by both agents are one visit to the union map.
    # Existing replay's continuous-gap statistic retains per-agent duplicates;
    # report an explicit union statistic without changing the frozen metric.
    visits={cell:[] for cell in range(truths.shape[1])}
    for observation in observations:
        for cell in observation['cells']:
            if not visits[cell] or visits[cell][-1]!=observation['observed_s']:visits[cell].append(observation['observed_s'])
    gaps=[b-a for values in visits.values() for a,b in zip(values,values[1:])]
    revisit=[gap for gap in gaps if gap>replay['config']['sample_s']+1e-7]
    assert_close(summary['revisit_median_s'],float(np.median(revisit)) if revisit else None)
    assert_close(summary['revisit_p90_s'],float(np.quantile(revisit,.9)) if revisit else None)
    motion_cost=snapshots[-1]['distance_m']*replay['team']['distance_cost_per_m']+snapshots[-1]['turn_s']*replay['team']['turn_cost_per_s']
    return dict(status='passed',frames=121,observations=242,actions=60,views=rebuilt['views'],
        metrics=dict(time_average_correct_coverage=quality,motion_cost=motion_cost,
            return_=quality-motion_cost-snapshots[-1]['collision_events']*replay['team']['collision_cost'],
            distance_m=snapshots[-1]['distance_m'],turn_s=snapshots[-1]['turn_s'],collision_events=snapshots[-1]['collision_events'],final=snapshots[-1]['metrics']),
        secondary=dict(change_count=len(changes),undiscovered_change_fraction=summary['undiscovered_change_fraction'],
            restricted_change_delay_s=summary['restricted_change_delay_s'],
            time_average_unknown_fraction=float(np.dot([s['metrics']['unknown_fraction'] for s in snapshots[:-1]],np.diff(times))/times[-1]),
            union_continuous_gap_median_s=float(np.median(gaps)) if gaps else None,
            saved_per_agent_continuous_gap_median_s=summary['continuous_sample_gap_median_s']))
