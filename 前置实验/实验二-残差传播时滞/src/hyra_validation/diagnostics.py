"""Offline-only truth diagnostics; outputs never feed the online controller."""
import numpy as np
from .data import prepare_observations,match_prior
from .representation import fit_structure,distance_structure,build_representation,causal_history
from .envs import pool
from .evaluation import score_event,compare_paired_events
from .artifacts import write_json


def residual_samples(event,lead):
    samples=[]
    for t in event["time_s"]:
        if t>=lead and t<=300 and t%10==0:
            prior=match_prior(event,0,t,lead)
            if prior is not None:
                f=prior["matched_forecast_id"]
                v=int(np.flatnonzero(event["forecast_valid_s"][f]==t)[0])
                index=int(np.flatnonzero(event["time_s"]==t)[0])
                samples.append(event["truth_depth_m"][index]-event["forecast_depth_m"][f,v])
    return np.array(samples)


def correlation(x):
    x=x-x.mean(axis=0)
    covariance=x.T@x
    scale=np.sqrt(np.maximum(np.diag(covariance),1e-12))
    return np.clip(covariance/np.outer(scale,scale),-1,1)


def freeze_graph(events,config,path):
    rc=config["representation"]
    graph=fit_structure(events["train"],rc)
    validation_correlations=[correlation(residual_samples(e,rc["lead_time_s"])) for e in events["validation"]]
    validation_families={}
    for event,c in zip(events["validation"],validation_correlations):
        validation_families.setdefault(event["metadata"]["event_family_id"],[]).append(c)
    required=int(np.ceil(2*len(validation_families)/3))
    w=np.array(graph["weights"])
    original=w.copy()
    # Prespecified validation stability filter. Test is never used.
    for j,i in zip(*np.nonzero(w)):
        if i!=j:
            agreement=sum(all(np.sign(c[j,i])==np.sign(w[j,i]) and abs(c[j,i])>=.5 for c in members)
                for members in validation_families.values())
            if agreement<required:
                w[j,i]=0
    w/=np.abs(w).sum(axis=1,keepdims=True)
    graph.update(weights=w.tolist(),validation_event_ids=[e["metadata"]["event_id"] for e in events["validation"]],
                 validation_rule="same sign and |corr|>=0.5 in every condition of at least2/3 validation families",frozen=True)
    write_json(path/"frozen_structure.json",graph)
    distance=distance_structure(events["train"][0]["xy_m"],rc)
    # Same number of neighbours and same self-link policy after pruning.
    d=np.linalg.norm(events["train"][0]["xy_m"][:,None]-events["train"][0]["xy_m"][None,:],axis=-1)
    dw=np.zeros_like(w)
    for j in range(len(w)):
        links=np.argsort(d[j],kind="stable")[:np.count_nonzero(w[j])]
        dw[j,links]=1/np.maximum(d[j,links],10.)
        dw[j]/=dw[j].sum()
    distance["weights"]=dw.tolist()
    write_json(path/"distance_structure.json",distance)
    write_json(path/"graph_validation.json",dict(candidate_edges=int(np.count_nonzero(original)-len(w)),
        retained_edges=int(np.count_nonzero(w)-len(w)),validation_event_count=len(validation_correlations),
        validation_family_count=len(validation_families),required_agreeing_families=required))
    return graph,distance


def mechanism_report(events,graph,config):
    w=np.array(graph["weights"])
    j,i=np.nonzero(w-np.diag(np.diag(w)))
    report=[]
    for split,sequence in events.items():
        for event in sequence:
            x=residual_samples(event,config["representation"]["lead_time_s"])
            raw=correlation(x)
            centered=correlation(x-x.mean(axis=1,keepdims=True))
            common=x.mean(axis=1)
            fraction=float(np.var(common)/np.maximum(np.mean(np.var(x,axis=0)),1e-12))
            agreement=float((np.sign(raw[j,i])==np.sign(w[j,i])).mean()) if len(i) else None
            report.append(dict(event_id=event["metadata"]["event_id"],split=split,
                selected_link_mean_correlation=float(np.abs(raw[j,i]).mean()) if len(i) else None,
                after_global_mean_removal_correlation=float(np.abs(centered[j,i]).mean()) if len(i) else None,
                common_mode_variance_ratio=fraction,signed_link_agreement=agreement))
    return report


def replay_samples(event,method,structure,config):
    observation_config=dict(config["observations"],route_regions=[[r,r+1,r+10,r+11] for r in range(0,90,11)])
    records=prepare_observations(event,observation_config)
    times=np.arange(20,301,10)
    x,y,coverage,strata=[],[],[],[]
    xyz=event["xy_m"]/100
    for t in times:
        rep=build_representation(records,None,float(t),structure,dict(config["representation"],method=method))
        image=rep["features"].T.reshape(-1,10,10)
        global_features=pool(image).ravel()
        targets=residual_samples_at(event,t,config["representation"]["lead_time_s"])
        history=causal_history(records,float(t),config["representation"])
        for r in range(100):
            x.append(np.r_[rep["features"][r],global_features,xyz[r],t/300])
            y.append(targets[r]); coverage.append(float(rep["coverage"][r]))
            age=t-history[r][0]["valid_time_s"] if history[r] else None
            strata.append("unobserved" if age is None else "fresh" if age<=10 else "stale")
    return dict(x=np.array(x),y=np.array(y),coverage=np.array(coverage),strata=np.array(strata),records=records)


def residual_samples_at(event,t,lead):
    p=match_prior(event,0,t,lead)
    f=p["matched_forecast_id"]
    v=int(np.flatnonzero(event["forecast_valid_s"][f]==t)[0])
    ti=int(np.flatnonzero(event["time_s"]==t)[0])
    return event["truth_depth_m"][ti]-event["forecast_depth_m"][f,v]


def score_fixed_order(times,depths,order,distances,now,step_s,service_s,limit,end_time=300):
    """Offline known-target diagnostic; abort a failed attempt at decision tick.

    The path/order is fixed before water evaluation. No online planner is added.
    """
    from .envs import interval_valid
    event=dict(time_s=times,truth_depth_m=depths)
    clock=float(now);success=0;trace=[]
    for index,region in enumerate(order):
        clock+=max(distances[index],1)*step_s
        start=clock
        completed=False
        for duration in np.arange(step_s,service_s+step_s,step_s):
            check_end=min(start+duration,start+service_s)
            if check_end>end_time or not interval_valid(event,region,start,check_end,limit):
                clock=check_end
                break
            clock=check_end
            if check_end>=start+service_s:
                completed=True;success+=1
                break
        trace.append(dict(region=int(region),start_s=start,end_s=clock,completed=completed))
    return dict(service_yield=success,trace=trace)


def replay_evaluate(events,graphs,config,path):
    readers={}; samples={}; validation_error={}
    for method in config["methods"]:
        structure=graphs[1] if method=="B1_distance" else graphs[0]
        for split in ("train","validation"):
            samples[method,split]=[replay_samples(e,method,structure,config) for e in events[split]]
        train=samples[method,"train"]
        x=np.concatenate([s["x"] for s in train]); y=np.concatenate([s["y"] for s in train])
        mean,std=x.mean(axis=0),x.std(axis=0)
        std[std<1e-8]=1
        z=np.c_[np.ones(len(x)),(x-mean)/std]
        xtx,xty=z.T@z,z.T@y
        readers[method]=dict(mean=mean,std=std,weights={})
        for alpha in config["evaluation"]["ridge_candidates"]:
            penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
            coefficient=np.linalg.solve(xtx+penalty,xty)
            readers[method]["weights"][str(alpha)]=coefficient
            errors=[]
            for sample in samples[method,"validation"]:
                pred=np.c_[np.ones(len(sample["x"])),(sample["x"]-mean)/std]@coefficient
                errors.append(np.abs(pred-sample["y"]).mean())
            validation_error.setdefault(str(alpha),[]).append(float(np.mean(errors)))
    # One alpha shared across all four groups, selected on average validation MAE.
    selected=min(validation_error,key=lambda a:np.mean(validation_error[a]))
    write_json(path/"diagnostic_readers.json",dict(selected_alpha=selected,validation_errors=validation_error,
        readers={m:dict(mean=r["mean"],std=r["std"],coefficient=r["weights"][selected]) for m,r in readers.items()},
        online_use=False,input_dim=len(next(iter(readers.values()))["mean"])))
    scores=[]
    for method,reader in readers.items():
        structure=graphs[1] if method=="B1_distance" else graphs[0]
        samples[method,"test"]=[replay_samples(e,method,structure,config) for e in events["test"]]
        for event,sample in zip(events["test"],samples[method,"test"]):
            pred=np.c_[np.ones(len(sample["x"])),(sample["x"]-reader["mean"])/reader["std"]]@reader["weights"][selected]
            row=dict(event_id=event["metadata"]["event_id"],event_family_id=event["metadata"]["event_family_id"],method=method,
                **score_event(sample["y"],pred,config["evaluation"]),effective_coverage=float((sample["coverage"]>0).mean()))
            row["strata"]={}
            for group in ("fresh","stale","unobserved"):
                mask=sample["strata"]==group
                row["strata"][group]=score_event(sample["y"][mask],pred[mask],config["evaluation"]) if mask.any() else None
            scores.append(row)
            np.savez_compressed(path/f"replay_{method}_{row['event_id']}.npz",truth_residual_m=sample["y"],
                                diagnostic_residual_m=pred,coverage=sample["coverage"],strata=sample["strata"])
            if method=="B0_raw":
                write_json(path/f"records_{row['event_id']}.json",sample["records"])
    write_json(path/"replay_scores.json",scores)
    write_json(path/"replay_paired.json",compare_paired_events(scores,dict(metric="mae_m",seed=config["evaluation"]["bootstrap_seed"])))
    write_json(path/"mechanism_report.json",mechanism_report(events,graphs[0],config))
    global_scores=[]
    for event in events["test"]:
        sample=replay_samples(event,"B0_raw",graphs[0],config)
        prediction=[]
        for t in np.arange(20,301,10):
            histories=causal_history(sample["records"],float(t),config["representation"])
            latest=[h[0]["residual_m"] for h in histories if h]
            prediction.extend([float(np.mean(latest)) if latest else 0.]*100)
        global_scores.append(dict(event_id=event["metadata"]["event_id"],
            reference="arithmetic mean of latest available local residuals; offline only",
            **score_event(sample["y"],prediction,config["evaluation"])))
    write_json(path/"global_mean_reference.json",global_scores)
    return scores
