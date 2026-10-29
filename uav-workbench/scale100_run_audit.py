"""Independent replay audit and richer task metrics for the frozen training contract.

Reads evaluator truth only AFTER rollout. No change to actor inputs or rewards.
"""
import argparse,json,math
from pathlib import Path
from scale100_train import Agent,FloodCase,ScaleEnv,config,digest,sources,rollout,torch,write
import numpy as np

def pool(array):
    # Independent block means; the 20x20 contract divides exactly into 5x5 bins.
    return array.reshape(*array.shape[:-2],5,4,5,4).mean(axis=(-3,-1))

def require(condition,message):
    if not bool(condition):raise AssertionError(message)

def audit(env,images=None):
    shape=env.shape;n=env.config.n_agents;dx=env.dx;steps=len(env.snapshots);size=math.prod(shape)
    require(steps==181,'Expected 181 actual snapshots for a 900s task')
    require(len(env.observations)==steps*n,'Missing sensor records')
    labels=np.full(size,-1,dtype=int);observed=np.full(size,-1.);water=np.zeros((3,size));explored=np.zeros(size,dtype=bool)
    scores=[];pixel_scores=[];first=[];stale=[];ages=[];distances=0.;service_starts={};rescued_at={};seen_targets=set();first_seen={};repeat_records=0
    previous=None;stationary_streak=[0]*n;max_streak=[0]*n;all_changed=0;unseen_changes=0;memory_errors=0
    pending_changes={};post_change_delays=[];superseded_changes=0;reason_counts={}
    for k,snapshot in enumerate(env.snapshots):
        time=float(snapshot['time_s']);require(time==k*5.,'Wrong physical clock')
        values=env.source.values(time);truth=(values[0]>=.05).ravel();poses=np.array([[p['z_m']/dx,p['x_m']/dx] for p in snapshot['poses']])
        require(np.array_equal(poses,np.round(poses)),'Off-grid poses');poses=poses.astype(int)
        require(len(set(map(tuple,poses)))==n,'Actual co-location')
        if k:
            for reason in snapshot['motion_status']:reason_counts[reason]=reason_counts.get(reason,0)+1
            delta=poses-previous;distances+=float(np.linalg.norm(delta,axis=1).sum()*dx)
            require((np.max(np.abs(delta),axis=1)<=1).all(),'Motion exceeds one grid cell')
            for i in range(n):
                stationary_streak[i]=stationary_streak[i]+1 if not np.any(delta[i]) else 0;max_streak[i]=max(max_streak[i],stationary_streak[i])
                r,c=poses[i];require(not env._blocked[r,c],'Agent entered obstacle')
                if delta[i,0] and delta[i,1]:require(not env._blocked[previous[i,0],c] and not env._blocked[r,previous[i,1]],'Corner cutting')
                for j in range(i):
                    relative=previous[i]-previous[j];velocity=delta[i]-delta[j];denom=float(velocity@velocity)
                    t=float(np.clip(-(relative@velocity)/denom,0,1)) if denom else 0.
                    require(np.linalg.norm(relative+t*velocity)>1e-8,'Actual segment crossing or swap')
            previous_truth=(env.source.values(time-5)[0]>=.05).ravel();changed=truth!=previous_truth
            all_changed+=int(changed.sum());unseen_changes+=int((changed&~explored).sum())
            for cell in np.flatnonzero(changed):
                if int(cell) in pending_changes:superseded_changes+=1
                pending_changes[int(cell)]=time
        for agent in range(n):
            obs=env.observations[k*n+agent];require(obs['agent_id']==agent and obs['observed_s']==time,'Future/delayed or mismatched record')
            r,c=poses[agent];expected={rr*shape[1]+cc for rr in range(max(0,r-2),min(shape[0],r+3)) for cc in range(max(0,c-2),min(shape[1],c+3))}
            cells=np.asarray(obs['cells'],dtype=int);require(len(cells)==len(expected) and set(cells)==expected,'Sensor footprint differs from legal pose/window')
            require(np.array_equal(obs['labels'],truth[cells]),'Logged labels differ from actual hydraulic frame')
            measured=np.array([obs['depth_m'],obs['u_m_s'],obs['v_m_s']]);require(np.array_equal(measured,values[:3].reshape(3,-1)[:,cells]),'Water log contains unmeasured or wrong values')
            require(np.array_equal(obs['obstacles'],env._blocked.ravel()[cells]),'Wrong measured obstacle record')
            repeat_records+=int(explored[cells].sum());labels[cells]=obs['labels'];observed[cells]=time;water[:,cells]=measured;explored[cells]=True
            for cell in cells:
                if int(cell) in pending_changes:post_change_delays.append(time-pending_changes.pop(int(cell)))
            for target in env._targets:
                if target[0]*shape[1]+target[1] in expected:seen_targets.add(target);first_seen.setdefault(target,time)
        require(np.array_equal(labels,snapshot['labels']),'Map contains records not in sensor log')
        require(np.array_equal(observed,snapshot['observed_s']),'Map timestamp was overwritten without a measurement')
        require(np.array_equal(explored,snapshot['explored']),'Exploration mask differs from sensor log')
        correct=(labels==truth)&explored;balanced=float(np.mean([correct[truth==v].mean() for v in (False,True) if (truth==v).any()]))
        require(abs(balanced-snapshot['metrics']['correct_coverage'])<1e-12,'Wrong balanced coverage metric')
        scores.append(balanced);pixel_scores.append(float(correct.mean()));first.append(float(explored.mean()));stale.append(float(((labels!=truth)&explored).mean()))
        memory_errors+=int(((labels!=truth)&explored).sum());ages.append(float(np.mean(time-observed[explored])))
        # Every actor water pixel must reconstruct from the arrived records.
        if images is not None:
            hydro=np.concatenate((water.reshape(3,*shape),explored.reshape(1,*shape),np.where(explored,(time-observed)/900.,0.).reshape(1,*shape)))
            static=(env.source.bed-env.source.bed.mean())[None];hp=np.pad(hydro,((0,0),(2,2),(2,2)));bp=np.pad(static,((0,0),(2,2),(2,2)))
            expected_images=np.stack([np.concatenate((hp[:,r:r+5,c:c+5],pool(hydro),bp[:,r:r+5,c:c+5],pool(static))) for r,c in poses])
            require(np.allclose(images[k,:,9:],expected_images,rtol=1e-6,atol=1e-7),'Actor water/static features contain hidden/future/unlogged information')
        visible={(t['row'],t['col']):t['status'] for t in snapshot['targets']}
        require(set(visible)<=seen_targets,'Hidden target coordinate exposed before first observation')
        for target,status in visible.items():
            if status=='rescued' and target not in rescued_at:
                # The latest two intervals plus entry time must all satisfy the service condition.
                for tau in (time-10.,time-5.,time):
                    require(tau>=0,'Rescue completed before a full dwell')
                    field=env.source.values(tau);r,c=target
                    require(field[0,r,c]<=env.config.service_depth_limit_m and math.hypot(float(field[1,r,c]),float(field[2,r,c]))<=env.config.service_speed_limit_m_s,'Rescue succeeded outside its service window')
                for earlier in env.snapshots[k-2:k+1]:
                    positions={(round(p['z_m']/dx),round(p['x_m']/dx)) for p in earlier['poses']}
                    require(target in positions,'Rescue dwell did not remain on the target')
                rescued_at[target]=time
        previous=poses.copy()
    require(abs(distances-env.distance_m)<1e-7,'Logged distance differs from actual trajectory')
    if images is not None:require(np.array_equal(water.reshape(3,*shape),env.water),'Final actor memory differs from raw records')
    n_targets=len(env._targets);completed=len(rescued_at)==n_targets if n_targets else None
    return dict(audit_passed=True,snapshots=steps,actual_sensor_records=steps*n,actor_hydraulic_features_checked=images is not None,physical_time_s=900.,balanced_current_correct_coverage_auc=float(np.mean(scores[:-1])),pixel_current_correct_coverage_auc=float(np.mean(pixel_scores[:-1])),first_coverage_auc=float(np.mean(first[:-1])),final_first_coverage=first[-1],first_90_s=next((k*5. for k,v in enumerate(first) if v>=.9),None),stale_error_area_auc=float(np.mean(stale[:-1])),mean_record_age_s=float(np.mean(ages[:-1])),undiscovered_change_fraction=unseen_changes/all_changed if all_changed else None,change_cell_events=all_changed,post_change_observation_events=len(post_change_delays),post_change_observation_mean_delay_s=float(np.mean(post_change_delays)) if post_change_delays else None,changes_superseded_without_observation=superseded_changes,changes_right_censored=len(pending_changes),latency_scope='first actual sample after a change; not proof of explicit change detection; unobserved/superseded events retained separately',repeated_cell_samples=repeat_records,maximum_stationary_streak_s=max(max_streak)*5.,motion_reason_counts=reason_counts,distance_m=distances,collision_attempt_events=env.collision_events,discovered=len(first_seen),rescued=len(rescued_at),targets=n_targets,rescue_fraction=len(rescued_at)/n_targets if n_targets else None,first_discovery_s=min(first_seen.values()) if first_seen else None,first_rescue_s=min(rescued_at.values()) if rescued_at else None,all_rescue_completion_s=max(rescued_at.values()) if n_targets and completed else None,all_rescue_right_censored=not completed if n_targets else None,cutoff_s=900.,scope='independent trajectory/measurement/service audit; not convergence or generalization proof')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    meta=json.loads((a.model/'actor.json').read_text());require(meta['sources']==sources(),'Model sources changed');require(digest(a.model/'actor.pt')==meta['actor_sha256'],'Actor checksum mismatch')
    dataset=json.loads(a.dataset.read_text());row=next(r for r in dataset['cases'] if r['split']=='development');case=FloodCase(row['path']);require(case.digest==row['sha256'],'Case checksum mismatch')
    actor=Agent();actor.load_state_dict(torch.load(a.model/'actor.pt',weights_only=True));actor.eval();rows=[]
    for seed in (11,29,47):
        env=ScaleEnv(case,config(meta['task'],meta['method'],seed));total,data,_=rollout(actor,env,np.random.default_rng(seed),0.,True)
        report=audit(env,data['images'].numpy());rows.append(dict(seed=seed,return_=total,**report));print(json.dumps(dict(seed=seed,audit_passed=True,coverage=report['final_first_coverage'],rescued=report['rescued'])),flush=True)
    write(a.output,dict(schema='hyra-scale100-independent-run-audit-v1',audit_source_sha256=digest(__file__),model=str(a.model.resolve()),actor_sha256=meta['actor_sha256'],source_sha256=sources(),task=meta['task'],method=meta['method'],training_episodes=meta['trained_episodes'],parent=row['parent_terrain'],split='development',holdout_evaluated=False,rows=rows))
if __name__=='__main__':main()
