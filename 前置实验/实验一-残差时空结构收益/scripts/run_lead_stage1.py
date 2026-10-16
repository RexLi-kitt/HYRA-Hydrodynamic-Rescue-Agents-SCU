"""Exploration only: physical calibration, 12 new families, causal information gate."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.adapters.anuga import simulate,export_event
from hyra_validation.lead import hydraulic_domain,XY,SOURCES,TARGETS,ROWS,sensor_records,features,future_label
from hyra_validation.data import load_events

METHODS=["I0_self","I1_common","I2_unaligned","I3_physical_lag","I4_wrong_lag","I5_reverse_direction"]


def response(params,mesh,center=25,amp=.1,plateau=20,yieldstep=2):
    t,b,_,_=simulate(hydraulic_domain(params,mesh),220,XY,yieldstep)
    tt,p,_,_=simulate(hydraulic_domain(params,mesh,pulse_amplitude=amp,pulse_center=center,plateau=plateau),220,XY,yieldstep)
    np.testing.assert_array_equal(t,tt)
    return t,p-b


def arrival(t,x,threshold=.1):
    peak=np.max(np.abs(x))
    if peak<.001:return None
    ids=np.flatnonzero((t>=90)&(np.abs(x)>=threshold*peak))
    return float(t[ids[0]]) if len(ids) else None


def calibrate(run):
    params=dict(slope=.0015,ridge=.07,rate=.003,onset=25,friction=.03)
    rows=[];responses={}
    for mesh in (10,20):
        t,x=response(params,mesh)
        responses[mesh]=(t,x)
        np.savez_compressed(run/f"calibration_mesh{mesh}.npz",time_s=t,response_m=x)
    t,a=responses[10];_,b=responses[20]
    selected=[r for r in TARGETS if np.max(abs(b[:,r]))>=.005]
    assert selected,"Local pulse does not reach monitored downstream cells"
    peak_errors=[abs(np.max(abs(a[:,r]))-np.max(abs(b[:,r])))/np.max(abs(b[:,r])) for r in selected]
    timing=[abs(arrival(t,a[:,r])-arrival(t,b[:,r])) for r in selected]
    coarse_ok=np.median(peak_errors)<=.2 and np.median(timing)<=4
    mesh=10 if coarse_ok else 20
    checks=[dict(mesh_pair=[10,20],cells=len(selected),median_peak_relative_error=float(np.median(peak_errors)),
        median_arrival_difference_s=float(np.median(timing)),passed=bool(coarse_ok))]
    if not coarse_ok:
        tc,c=response(params,40);np.savez_compressed(run/"calibration_mesh40.npz",time_s=tc,response_m=c)
        selected=[r for r in TARGETS if np.max(abs(c[:,r]))>=.005]
        pe=[abs(np.max(abs(b[:,r]))-np.max(abs(c[:,r])))/np.max(abs(c[:,r])) for r in selected]
        timing=[abs(arrival(t,b[:,r])-arrival(tc,c[:,r])) for r in selected]
        fine_ok=np.median(pe)<=.2 and np.median(timing)<=4
        checks.append(dict(mesh_pair=[20,40],cells=len(selected),median_peak_relative_error=float(np.median(pe)),median_arrival_difference_s=float(np.median(timing)),passed=bool(fine_ok)))
        if not fine_ok:
            write_json(run/"calibration.json",dict(checks=checks,passed=False))
            raise RuntimeError("Mesh sensitivity gate failed; do not train information models")
    t,x=response(params,mesh,yieldstep=1)
    baseline=responses[mesh][1]
    output_error=float(np.max(abs(x[::2]-baseline)))
    output_arrival=max(abs(arrival(t,x[:,r])-arrival(responses[mesh][0],baseline[:,r])) for r in selected)
    assert output_error<=.001 and output_arrival<=2,"Output interval materially changes response"
    for amp in (.05,.1,.2):
        for plateau in (20,40):
            t,x=response(params,mesh,amp=amp,plateau=plateau)
            rows.append(dict(amplitude_m=amp,plateau_s=plateau,max_downstream_response_m=float(np.max(abs(x[:,TARGETS]))),
                first_upstream_s=min(v for r in SOURCES if (v:=arrival(t,x[:,r])) is not None),
                first_downstream_s=min(v for r in TARGETS if (v:=arrival(t,x[:,r])) is not None)))
    result=dict(checks=checks,passed=True,nmesh=mesh,selected_amplitude_m=.1,selected_plateau_s=20,
        scan=rows,selection="Middle amplitude and shorter plateau prespecified; no residual/policy outcome selection",
        output_step_check=dict(max_response_difference_m=output_error,max_arrival_difference_s=output_arrival,
            tolerance_m=.001,tolerance_s=2),limitations="Representative calibration, not full parameter-range convergence")
    write_json(run/"calibration.json",result)
    print("calibration",result,flush=True)
    return mesh


def forecast(params,mesh,switched=False):
    nt,nd,states,mapping=simulate(hydraulic_domain(params,mesh,switched=switched),300,XY)
    releases=np.arange(0,301,10,dtype=float);fds=[];fts=[]
    for release in releases:
        ti=int(np.flatnonzero(nt==release)[0])
        d=hydraulic_domain(params,mesh,initial=states[ti],start=release,switched=switched and release>=90)
        ft,fd,_,_=simulate(d,release+60,XY)
        np.testing.assert_allclose(fd[0],nd[ti],atol=1e-8)
        fds.append(fd);fts.append(ft)
    return dict(release_s=releases,forecast_valid_s=np.array(fts),forecast_depth_m=np.array(fds),nominal_time_s=nt,nominal_depth_m=nd),states,mapping


def generate(run,mesh):
    import anuga
    rng=np.random.default_rng(2026101501);manifest={s:[] for s in ("train","validation","test")};params_all=[]
    for index in range(12):
        split="train" if index<6 else "validation" if index<9 else "test"
        params=dict(slope=float(rng.uniform(.001,.002)),ridge=float(rng.uniform(.04,.1)),rate=float(rng.uniform(.0025,.0035)),
            onset=float(rng.uniform(15,30)),friction=float(rng.uniform(.025,.035)))
        params_all.append(dict(family=index,split=split,params=params))
        start=float((60,90)[index%2]);center=float((25,75)[(index//2)%2])
        t,base,base_states,mapping=simulate(hydraulic_domain(params,mesh,factor=1),360,XY)
        tt,perturbed,pulse_states,_=simulate(hydraulic_domain(params,mesh,factor=1,pulse_amplitude=.1,pulse_start=start,pulse_center=center),360,XY)
        np.testing.assert_array_equal(t,tt);np.testing.assert_allclose(perturbed[t<start],base[t<start],atol=1e-8)
        prior,nominal_states,_=forecast(params,mesh)
        common,common_states,_=forecast(params,mesh,True)
        np.testing.assert_allclose(prior["forecast_depth_m"][prior["release_s"]<90],common["forecast_depth_m"][common["release_s"]<90],atol=1e-7)
        for condition,truth,refstates,f,nstates in [("C0_stable",base,base_states,prior,nominal_states),
                ("C1_local",perturbed,pulse_states,prior,nominal_states),("C2_common",base,base_states,common,common_states)]:
            eid=f"lead_{index:03}_{condition}";path=run/f"{eid}.npz"
            metadata=dict(event_id=eid,event_family_id=f"lead_{index:03}",family_index=index,split=split,condition=condition,
                source="ANUGA",solver_version=anuga.__version__,units=dict(time="s",xy="m",depth="m"),params=params,mapping=mapping,
                pulse_start_s=start,pulse_center_y_m=center,local_pulse_amplitude_m=.1 if condition=="C1_local" else 0,
                nmesh=mesh,paired_layout_index=index%3,raw_solver_file=path.with_suffix(".solver.npz").name,
                forecasting="Independent nominal, factor0.9; common shift0.65 at90 known only then; no observation/reference reset")
            np.savez_compressed(path.with_suffix(".solver.npz"),time_s=t,nominal_time_s=f["nominal_time_s"],
                **{f"reference_{k}":np.array([s[k] for s in refstates]) for k in refstates[0]},
                **{f"nominal_{k}":np.array([s[k] for s in nstates]) for k in nstates[0]})
            export_event(dict(time_s=t,truth_depth_m=truth,xy_m=XY),f,path,metadata)
            manifest[split].append(path.name)
        print("family",index,split,"generated",flush=True)
    write_json(run/"manifest.json",manifest);write_json(run/"family_parameters.json",params_all)
    return load_events(run/"manifest.json")


def physical_graph(events,mesh,run):
    measurements={str(r):[] for r in TARGETS}
    for e in events["train"]:
        for center,ys in [(25,ROWS[:3]),(75,ROWS[3:])]:
            t,x=response(e["metadata"]["params"],mesh,center)
            for target in TARGETS:
                if target//10 not in ys:continue
                td=arrival(t,x[:,target])
                for y in ys:
                    source=10*y;ts=arrival(t,x[:,source])
                    if td is not None and ts is not None:
                        measurements[str(target)].append(dict(source=source,travel_s=max(td-ts,0),gain=float(np.max(abs(x[:,target])))))
    graph={};calibration=[]
    for target,rows in measurements.items():
        links=[]
        for source in sorted({r["source"] for r in rows}):
            rr=[r for r in rows if r["source"]==source]
            lag=float(np.median([r["travel_s"] for r in rr]))
            links.append(dict(source=source,travel_s=lag,weight=1/3))
            calibration.append(dict(target=int(target),source=source,travel_median_s=lag,travel_range_s=[min(r["travel_s"] for r in rr),max(r["travel_s"] for r in rr)]))
        assert len(links)==3
        graph[target]=links
    write_json(run/"physical_graph.json",dict(graph=graph,fitted_families=[e["metadata"]["event_family_id"] for e in events["train"]],
        basis="Independent nominal solver pulses,10% response arrival; no reference/test response for edges",measurements=calibration))
    return graph


def ridge_fit(x,y,alpha):
    mean=x.mean(axis=0);std=x.std(axis=0);std[std<1e-8]=1
    z=np.c_[np.ones(len(x)),(x-mean)/std];penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
    weights=np.linalg.solve(z.T@z+penalty,z.T@y)
    return mean,std,weights


def predict(model,x):
    mean,std,w=model;return np.c_[np.ones(len(x)),(x-mean)/std]@w


def information(events,graph,run):
    scores=[];all_primary=[];readers=[]
    for moving in (False,True):
        for noise,delay in ((0,0),(.02,0),(0,10),(.02,10)):
            for tau in (10,20,30):
                samples={};cases={}
                for split,sequence in events.items():
                    cases[split]=[];xs={m:[] for m in METHODS};ys=[]
                    for event in sequence:
                        records=sensor_records(event,noise,delay,moving)
                        for now in range(80,241,10):
                            for region in TARGETS:
                                ys.append(future_label(event,region,now,tau))
                                cases[split].append(dict(family=event["metadata"]["event_family_id"],condition=event["metadata"]["condition"],time_s=now,region=region))
                                for method in METHODS:xs[method].append(features(event,records,region,now,tau,graph,method))
                    samples[split]=dict(y=np.array(ys),x={m:np.array(x) for m,x in xs.items()})
                for method in METHODS:
                    candidates={a:ridge_fit(samples["train"]["x"][method],samples["train"]["y"],a) for a in (.1,1,10)}
                    errors={a:float(np.mean(abs(predict(model,samples["validation"]["x"][method])-samples["validation"]["y"]))) for a,model in candidates.items()}
                    alpha=min(errors,key=errors.get);model=candidates[alpha]
                    readers.append(dict(moving=moving,noise=noise,delay=delay,tau=tau,method=method,alpha=alpha,validation_errors=errors,
                        mean=model[0],std=model[1],weights=model[2],test_used=False))
                    p=predict(model,samples["test"]["x"][method]);y=samples["test"]["y"]
                    keys=sorted({(r["family"],r["condition"]) for r in cases["test"]})
                    for family,condition in keys:
                        mask=np.array([r["family"]==family and r["condition"]==condition for r in cases["test"]])
                        scores.append(dict(moving=moving,noise=noise,delay=delay,tau=tau,method=method,family=family,condition=condition,
                            mae_m=float(np.mean(abs(p[mask]-y[mask]))),rmse_m=float(np.sqrt(np.mean((p[mask]-y[mask])**2))),samples=int(mask.sum())))
                    if tau==20 and noise==0 and delay==0:
                        all_primary.extend(dict(case,method=method,moving=moving,prediction=float(pp),label=float(yy)) for case,pp,yy in zip(cases["test"],p,y))
                print("information",moving,noise,delay,tau,"complete",flush=True)
                write_json(run/"information_scores.json",scores)
    write_json(run/"information_readers.json",readers);write_json(run/"primary_cases.json",all_primary)
    return scores


def report(run,scores):
    rows=[];index={(r["moving"],r["noise"],r["delay"],r["tau"],r["method"],r["family"],r["condition"]):r for r in scores}
    families=sorted({r["family"] for r in scores});rng=np.random.default_rng(91)
    for moving in (False,True):
        for condition in ("C0_stable","C1_local","C2_common"):
            for baseline in [m for m in METHODS if m!="I3_physical_lag"]:
                base=np.array([index[moving,0,0,20,baseline,f,condition]["mae_m"] for f in families])
                proposed=np.array([index[moving,0,0,20,"I3_physical_lag",f,condition]["mae_m"] for f in families])
                diff=proposed-base;boot=rng.choice(diff,(2000,len(diff)),replace=True).mean(axis=1)
                rows.append(dict(moving=moving,condition=condition,baseline=baseline,mae_delta_m=float(diff.mean()),
                    relative_reduction=float(1-proposed.mean()/base.mean()),interval_95=np.quantile(boot,[.025,.975]).tolist(),
                    favorable_families=int(sum(diff<0)),independent_families=len(families)))
    primary=[r for r in rows if r["condition"]=="C1_local"]
    gate=all(r["relative_reduction"]>=.05 and r["favorable_families"]>=2 for r in primary)
    # Internal diagnostic is explicitly iterative, not formal holdout.
    write_json(run/"comparisons.json",rows)
    write_json(run/"stage_gate.json",dict(information_gate_passed=gate,formal_blind_test=False,
        criterion="C1 tau20, all five baselines, stationary/moving, >=5% MAE reduction and >=2/3 diagnostic families favorable",
        next="Complete risk/opportunity diagnostics and freeze formal protocol" if gate else "Do not start large strategy training; locate failed information/lag hypothesis",
        task_efficacy_tested=False))
    lines=["# 时空先行信息探索阶段报告\n","本轮是12个新家族上的探索实验（6训练/3验证/3内部诊断），不是正式36家族盲测。局部边界扰动经ANUGA传播；预测器不读取参考/观测重置。\n",
        "六组读出共享下游自身6条历史；除自身基线外均含可见共同趋势。I2对相同源历史均匀汇总，I3按名义水动力时滞对齐，I4错配过去时滞，I5使用下游截面源作反向对照。所有读出特征维度相同，正则只由验证选择。\n",
        "未来误差标签锁定当前已发布预测。主指标20s提前量，统计以家族为单位；3个内部诊断家族区间仅作探索，不能将时间/格点当独立样本。\n",
        "| 观测路线 | 条件 | 对照 | I3相对MAE下降 | I3−对照MAE(mm) | 家族95%区间(mm) | 有利家族 |\n|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lo,hi=r["interval_95"]
        lines.append(f"| {'移动' if r['moving'] else '固定截面'} | {r['condition']} | {r['baseline']} | {r['relative_reduction']*100:+.1f}% | {r['mae_delta_m']*1000:+.3f} | [{lo*1000:+.3f},{hi*1000:+.3f}] | {r['favorable_families']}/3 |")
    lines.append(f"\n信息门槛：{'通过' if gate else '未通过'}。这不是救援收益结果。")
    lines.append("\n若不通过，不自动训练策略或解封正式测试；继续查看采样/时滞/自身趋势与共同偏差影响，必要时收窄或放弃假设。风险阈值和任务机会诊断尚须完成，不能将本报告说成全方案已验证。")
    lines.append(f"\n协议：[实验方案]({(ROOT/'docs/RESIDUAL_LEAD_EXPERIMENT_PLAN.md').as_posix()})。校准见calibration.json；读出、数据、图与比较均保存在本目录。")
    path=run/"REPORT.md";path.write_text("\n".join(lines),encoding="utf-8")
    write_json(run/"completion.json",dict(completed=True,scope="exploratory physical and information stage",information_gate_passed=gate,full_plan_completed=False))
    write_json(ROOT/"results/latest_lead_report.json",dict(report=str(path),stage_gate=str(run/"stage_gate.json"),full_plan_completed=False))
    print("REPORT",path,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--resume");args=parser.parse_args()
    if args.resume:
        run=Path(args.resume);mesh=json.loads((run/"calibration.json").read_text(encoding="utf-8"))["nmesh"]
        events=load_events(run/"manifest.json")
    else:
        protocol=dict(seed=2026101501,families=[6,3,3],stage="exploration",forecast_lead_s=20,
            label="future truth minus current published forecast",history_length=6,alpha_candidates=[.1,1,10],
            noise=[0,.02],delay=[0,10],tau=[10,20,30],methods=METHODS,formal_test_opened=False)
        run=new_run("lead_stage1",protocol)
        mesh=calibrate(run);events=generate(run,mesh)
    graph=physical_graph(events,mesh,run);scores=information(events,graph,run);report(run,scores)
