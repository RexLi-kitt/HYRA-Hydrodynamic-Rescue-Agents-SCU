"""Timing availability and matched recency controls on existing exploration data."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hyra_validation.artifacts import ROOT, new_run, write_json
from hyra_validation.data import load_events
from hyra_validation.lead import TARGETS, future_label
from hyra_validation.lead_diagnostic import METHODS, matched_features, phased_records, timing_rows
from run_lead_stage1 import ridge_fit, predict


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scenarios():
    rows=[]
    for moving in (False, True):
        for noise, delay in ((0,0),(.02,0),(0,10),(.02,10)):
            rows.append(dict(moving=moving,noise=noise,delay=delay,phase=0,
                name=f"{'rotating' if moving else 'fixed'}_n{noise:g}_d{delay}_p0"))
    rows.extend(dict(moving=True,noise=0,delay=0,phase=p,name=f"rotating_n0_d0_p{p}") for p in (1,2,3))
    return rows


def coverage(events, graph, run):
    # Sampling schedule and the frozen graph are common to all families.
    # Count unique schedules, not 12 repetitions as independent observations.
    event=events['train'][0]; details=[]; summaries=[]; targets=[]
    for s in scenarios():
        if s['noise'] != 0:continue
        rec=phased_records(event,s['noise'],s['delay'],s['moving'],s['phase'])
        for tau in (10,20,30):
            for wait in (0,6):
                rr=[]; tt=[]
                for now in range(80,241,10):
                    for region in TARGETS:
                        rows=timing_rows(rec,graph,now,region,tau,wait)
                        rr.extend(dict(s,**r) for r in rows)
                        tt.append(dict(s,time_s=now,region=region,tau=tau,wait_s=wait,
                            any_observed=any(r['observed'] for r in rows),
                            any_positive=any(r['positive_remaining'] for r in rows),
                            any_horizon=any(r['horizon_covered'] for r in rows),
                            all_horizon=all(r['horizon_covered'] for r in rows)))
                ages=[r['age_s'] for r in rr if r['observed']]
                leads=[r['nominal_remaining_s'] for r in rr if r['observed']]
                summaries.append(dict(s,tau=tau,wait_s=wait,edge_cases=len(rr),target_cases=len(tt),
                    observed_rate=float(np.mean([r['observed'] for r in rr])),
                    positive_edge_rate=float(np.mean([r['positive_remaining'] for r in rr])),
                    horizon_edge_rate=float(np.mean([r['horizon_covered'] for r in rr])),
                    positive_target_rate=float(np.mean([r['any_positive'] for r in tt])),
                    horizon_target_rate=float(np.mean([r['any_horizon'] for r in tt])),
                    clipped_kernel_rate=float(np.mean([r['desired_clipped'] for r in rr])),
                    age_quantiles_s=np.quantile(ages,[0,.25,.5,.75,1]) if ages else None,
                    remaining_quantiles_s=np.quantile(leads,[0,.25,.5,.75,1]) if leads else None))
                details.extend(rr); targets.extend(tt)
    write_json(run/'timing_edges.json',details)
    write_json(run/'timing_targets.json',targets)
    write_json(run/'timing_summary.json',summaries)
    unique=[dict(target=int(target),**link) for target,links in graph.items() for link in links]
    write_json(run/'graph_timing_audit.json',dict(edges=unique,
        horizons=[dict(tau=tau,clipped=sum(r['travel_s']<tau for r in unique),
            centers_s=sorted({max(r['travel_s']-tau,0) for r in unique})) for tau in (10,20,30)],
        interpretation='Nominal response timing potential only; not measured alarm lead or an information-theoretic upper bound'))


def information(events, graph, run):
    scores=[]; readers=[]; legacy=read(run/'input_source.json')['run']
    old=read(Path(legacy)/'information_scores.json')
    old_index={(r['moving'],r['noise'],r['delay'],r['tau'],r['method'],r['family'],r['condition']):r for r in old}
    replication=[]
    for s in scenarios():
        records={e['metadata']['event_id']:phased_records(e,s['noise'],s['delay'],s['moving'],s['phase'])
                 for seq in events.values() for e in seq}
        for tau in (10,20,30):
            datasets={}; cases={}
            for split, seq in events.items():
                xs={m:[] for m in METHODS}; y=[]; cases[split]=[]
                for event in seq:
                    rec=records[event['metadata']['event_id']]
                    for now in range(80,241,10):
                        for region in TARGETS:
                            ff=matched_features(event,rec,region,now,tau,graph)
                            for m in METHODS:xs[m].append(ff[m])
                            y.append(future_label(event,region,now,tau))
                            tr=timing_rows(rec,graph,now,region,tau,6)
                            cases[split].append(dict(family=event['metadata']['event_family_id'],
                                condition=event['metadata']['condition'],time_s=now,region=region,
                                pulse_start_s=event['metadata']['pulse_start_s'],
                                pulse_center_y_m=event['metadata']['pulse_center_y_m'],
                                horizon_available=any(r['horizon_covered'] for r in tr),
                                positive_remaining=any(r['positive_remaining'] for r in tr)))
                datasets[split]=dict(y=np.array(y),x={m:np.array(v) for m,v in xs.items()})
            predictions=[]
            for m in METHODS:
                candidates={a:ridge_fit(datasets['train']['x'][m],datasets['train']['y'],a) for a in (.1,1,10)}
                errors={a:float(np.mean(abs(predict(model,datasets['validation']['x'][m])-datasets['validation']['y'])))
                        for a,model in candidates.items()}
                alpha=min(errors,key=errors.get); model=candidates[alpha]
                readers.append(dict(s,tau=tau,method=m,alpha=alpha,validation_errors=errors,
                    mean=model[0],std=model[1],weights=model[2],test_used_for_fit=False))
                p=predict(model,datasets['test']['x'][m]); y=datasets['test']['y']; predictions.append(p)
                cc=cases['test']
                for family, condition in sorted({(r['family'],r['condition']) for r in cc}):
                    family_mask=np.array([r['family']==family and r['condition']==condition for r in cc])
                    for stratum in ('all','horizon_available','horizon_unavailable','positive_remaining','no_positive_remaining'):
                        if stratum=='all':mask=family_mask
                        else:
                            field='horizon_available' if stratum.startswith('horizon') else 'positive_remaining'
                            value=stratum in ('horizon_available','positive_remaining')
                            mask=family_mask & np.array([r[field]==value for r in cc])
                        if not mask.any():continue
                        row=dict(s,tau=tau,method=m,family=family,condition=condition,stratum=stratum,
                            mae_m=float(np.mean(abs(p[mask]-y[mask]))),rmse_m=float(np.sqrt(np.mean((p[mask]-y[mask])**2))),samples=int(mask.sum()))
                        scores.append(row)
                        if s['phase']==0 and m in ('I1_common','I2_unaligned','I3_physical_lag') and stratum=='all':
                            original=old_index[s['moving'],s['noise'],s['delay'],tau,m,family,condition]
                            diff=abs(original['mae_m']-row['mae_m'])
                            replication.append(dict(s,tau=tau,method=m,family=family,condition=condition,mae_difference_m=diff))
                            assert diff<1e-10, f'Legacy reproduction mismatch: {replication[-1]}'
            pp=np.array(predictions)
            if tau==30:
                # Every T is below 30: the original clipping erases all lag centers.
                np.testing.assert_allclose(pp[METHODS.index('I3_physical_lag')],pp[METHODS.index('N0_recent')],atol=1e-12)
                np.testing.assert_allclose(pp[METHODS.index('N1_physical_norm')],pp[METHODS.index('N2_recent_norm')],atol=1e-12)
            file=f"predictions_{s['name']}_t{tau}.npz"
            np.savez_compressed(run/file,methods=np.array(METHODS),prediction=pp,label=y,
                **{key:np.array([r[key] for r in cases['test']]) for key in cases['test'][0]})
            write_json(run/'information_scores.json',scores)
            write_json(run/'information_readers.json',readers)
            print('completed',s['name'],'horizon',tau,flush=True)
    write_json(run/'legacy_reproduction.json',dict(passed=True,cases=replication,
        max_mae_difference_m=max(r['mae_difference_m'] for r in replication)))
    return scores


def comparisons(scores, run):
    rng=np.random.default_rng(2026101502); rows=[]
    def compare(group, proposed, baseline):
        result=[]
        for family in sorted({r['family'] for r in group}):
            a=[r['mae_m'] for r in group if r['family']==family and r['method']==proposed]
            b=[r['mae_m'] for r in group if r['family']==family and r['method']==baseline]
            if a and b:result.append((family,float(np.mean(a)),float(np.mean(b))))
        if not result:return None
        a=np.array([r[1] for r in result]); b=np.array([r[2] for r in result]); delta=a-b
        boots=rng.choice(delta,(4000,len(delta)),replace=True).mean(axis=1)
        return dict(proposed=proposed,baseline=baseline,proposed_mae_m=a.mean(),baseline_mae_m=b.mean(),
            relative_reduction=1-a.mean()/b.mean(),mae_delta_m=delta.mean(),
            interval_95=np.quantile(boots,[.025,.975]),favorable_families=int(sum(delta<0)),
            independent_families=len(delta),family_values=result)
    pairs=[('I3_physical_lag',b) for b in ('I1_common','I2_unaligned','N0_recent')]
    pairs += [('N1_physical_norm',b) for b in ('I1_common','N3_uniform_norm','N2_recent_norm','N4_wrong_norm')]
    for s in scenarios():
        for tau in (10,20,30):
            for condition in ('C0_stable','C1_local','C2_common'):
                for stratum in ('all','horizon_available','horizon_unavailable','positive_remaining','no_positive_remaining'):
                    group=[r for r in scores if r['name']==s['name'] and r['tau']==tau and r['condition']==condition and r['stratum']==stratum]
                    for p,b in pairs:
                        value=compare(group,p,b)
                        if value:rows.append(dict(s,tau=tau,condition=condition,stratum=stratum,**value))
    for tau in (10,20,30):
        for condition in ('C0_stable','C1_local','C2_common'):
            group=[r for r in scores if r['moving'] and r['noise']==0 and r['delay']==0
                   and r['tau']==tau and r['condition']==condition and r['stratum']=='all']
            for p,b in pairs:
                value=compare(group,p,b)
                rows.append(dict(name='rotating_phase_average',moving=True,noise=0,delay=0,phase='average',
                    tau=tau,condition=condition,stratum='all',**value))
    write_json(run/'comparisons.json',rows)
    selected=[r for r in rows if r['name'] in ('fixed_n0_d0_p0','rotating_phase_average') and r['tau']==20
        and r['condition']=='C1_local' and r['stratum']=='all' and r['proposed']=='N1_physical_norm']
    assert len(selected)==8
    passed=all(r['relative_reduction']>=.05 and r['favorable_families']>=2 and r['interval_95'][1]<0 for r in selected)
    write_json(run/'diagnostic_gate.json',dict(passed=bool(passed),checks=selected,
        original_information_gate_changed=False,formal_test_opened=False,policy_trained=False,
        decision='Proceed to finite two-target diagnosis only' if passed else 'Stop before rescue-policy/critical-order testing; lag-specific information gate failed'))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',required=True);parser.add_argument('--resume')
    args=parser.parse_args(); source=Path(args.source_run).resolve()
    config=dict(stage='exploratory timing and matched-recency diagnosis',source_run=str(source),
        family_split=[6,3,3],tau=[10,20,30],primary_tau=20,methods=METHODS,
        scenarios=scenarios(),wait_s=[0,6],formal_test_opened=False,
        protocol='docs/LEAD_TIMING_DIAGNOSTIC.md')
    run=Path(args.resume).resolve() if args.resume else new_run('lead_timing_diagnostic',config)
    if not args.resume:
        write_json(run/'input_source.json',dict(run=str(source),
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
                   if p.suffix in ('.npz','.json') and not p.name.endswith('.solver.npz')}))
        shutil.copyfile(ROOT/'docs/LEAD_TIMING_DIAGNOSTIC.md',run/'PROTOCOL.md')
        for path in [ROOT/'src/hyra_validation/lead_diagnostic.py',Path(__file__).resolve()]:
            dst=run/'source_snapshot'/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
        write_json(ROOT/'results/latest_timing_diagnostic.json',dict(run=str(run),status='running',formal_test_opened=False))
    print('RUN',run,flush=True)
    events=load_events(source/'manifest.json');graph=read(source/'physical_graph.json')['graph']
    coverage(events,graph,run)
    scores=information(events,graph,run)
    comparisons(scores,run)
    write_json(run/'information_completion.json',dict(completed=True,formal_test_opened=False,policy_trained=False,
        model_count=len(read(run/'information_readers.json'))))
    print('INFORMATION_COMPLETE',run,flush=True)
