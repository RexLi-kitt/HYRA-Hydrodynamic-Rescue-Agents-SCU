"""Independently aggregate raw confirmation shards and recompute all gates."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hashlib import sha256


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return sha256(path.read_bytes()).hexdigest()
def avg(values):
    valid=[v for v in values if v is not None]
    return float(sum(valid)/len(valid)) if valid else None
def equal(actual,expected):
    if expected is None:assert actual is None
    else:np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output.resolve();summary=read(root/'confirmation-summary.json')
    plan=read(root/'formal-plan.json');registration=read(root/'evaluation-plan.json')
    held={e['id']:e for e in read(root/'index.json') if e['split']=='holdout'};assert len(held)==9
    files=list((root/'evaluation/policies').glob('*.json'));raw=[read(p) for p in files];assert len(raw)==108
    rules=[read(p) for p in (root/'evaluation/rules').glob('*.json')];assert len(rules)==27
    methods=plan['methods'];seeds=plan['seeds']
    assert {(r['case'],r['method'],r['seed']) for r in raw}=={(c,m,s) for c in held for m in methods for s in seeds}
    assert {(r['case'],r['method']) for r in rules}=={(c,m) for c in held for m in ('R0','R1','R2')}
    events={}
    for case in held:
        for method in methods:
            rows=[r for r in raw if r['case']==case and r['method']==method]
            assert len(rows)==3
            for row in rows:assert row['family']==held[case]['family'] and row['forcing_family']==held[case]['forcing_family']
            events[case,method]=dict(quality=avg([r['metrics']['time_average_correct_coverage'] for r in rows]),
                collisions=avg([r['metrics']['collision_events'] for r in rows]),motion_cost=avg([r['metrics']['motion_cost'] for r in rows]),
                return_=avg([r['metrics']['return_'] for r in rows]),
                training_seed_quality_std=float(np.std([r['metrics']['time_average_correct_coverage'] for r in rows],ddof=1)))
    assert len(summary['events'])==36
    assert {(r['case'],r['method']) for r in summary['events']}==set(events)
    for row in summary['events']:
        for field,value in events[row['case'],row['method']].items():equal(row[field],value)
    for method in methods:equal(summary['main_quality'][method],avg([events[c,method]['quality'] for c in held]))
    for method in ('R0','R1','R2'):equal(summary['rule_quality'][method],avg([r['metrics']['time_average_correct_coverage'] for r in rules if r['method']==method]))
    paired=[]
    for case in held:
        physics=events[case,'Sur-QMIX+Physics'];data=events[case,'Sur-QMIX+Data'];sur=events[case,'Sur-QMIX']
        paired.append(dict(case=case,family=held[case]['family'],forcing_family=held[case]['forcing_family'],
            delta_Data=physics['quality']-data['quality'],delta_Sur=physics['quality']-sur['quality'],delta_QMIX=physics['quality']-events[case,'QMIX']['quality'],
            delta_collisions=physics['collisions']-data['collisions'],delta_motion_cost=physics['motion_cost']-data['motion_cost']))
    gate=summary['gate'];assert len(gate['paired'])==9
    for saved in gate['paired']:
        rebuilt=next(r for r in paired if r['case']==saved['case'])
        for k,v in rebuilt.items():
            if k.startswith('delta_'):equal(saved[k],v)
            else:assert saved[k]==v
    means={k:avg([r['delta_'+k] for r in paired]) for k in ('Data','Sur','QMIX','collisions','motion_cost')}
    for k,v in means.items():equal(gate['mean_deltas'][k],v)
    forcing={f:{k:avg([r['delta_'+k] for r in paired if r['forcing_family']==f]) for k in ('Data','Sur')}
             for f in {r['forcing_family'] for r in paired}}
    for family,values in forcing.items():
        for k,v in values.items():equal(gate['forcing_deltas'][family][k],v)
    positive=sum(r['delta_Data']>0 and r['delta_Sur']>0 for r in paired);assert positive==gate['positive_events']
    criteria=registration['gate']
    checks=dict(mean_Physics_Data=means['Data']*100>=criteria['physics_vs_data_mean_min_pp'],
        mean_Physics_Sur=means['Sur']*100>=criteria['physics_vs_sur_mean_min_pp'],
        simultaneous_positive_events=positive>=criteria['positive_events_min'],
        each_forcing_positive=all(v>0 for row in forcing.values() for v in row.values()),
        collision_increase=means['collisions']<=criteria['mean_collision_increase_max_per_episode'],
        motion_cost_increase=means['motion_cost']<=criteria['mean_motion_cost_increase_max'])
    assert gate['criteria']==checks and gate['passed']==all(checks.values())
    assert set(gate['failed_criteria'])=={k for k,v in checks.items() if not v}
    forecasts=[]
    for path in (root/'evaluation/forecasts').glob('*.jsonl'):forecasts.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    assert len(forecasts)==9720
    expected={(c,m,s,r,t) for c in held for m in ('Sur-QMIX+Data','Sur-QMIX+Physics') for s in seeds for r in ('R0','R1','R2') for t in range(0,600,10)}
    assert {(r['case'],r['method'],r['seed'],r['route'],r['time_s']) for r in forecasts}==expected
    fields=['future_h_mae_m','future_h_unknown_mae_m','future_h_stale_mae_m','future_wet_classbalanced_error',
        'future_momentum_mae_m2_s','current_h_mae_m','supervised_loss','physical_rate_loss','absolute_volume_rate_residual_m3_s']
    for row in forecasts:
        assert (row['future_h_unknown_mae_m'] is None)==(row['unknown_cells']==0)
        assert (row['future_h_stale_mae_m'] is None)==(row['stale_cells']==0)
        assert all(row[k] is None or np.isfinite(row[k]) for k in fields)
    grouped=defaultdict(list)
    for row in forecasts:grouped[row['case'],row['method'],row['seed'],row['route']].append(row)
    route_means={key:{f:avg([r[f] for r in rows]) for f in fields} for key,rows in grouped.items()}
    assert all(len(rows)==60 for rows in grouped.values()) and len(route_means)==162
    for row in summary['predictions']:
        case,method=row['case'],row['method']
        for field in fields:
            seed_values=[avg([route_means[case,method,seed,route][field] for route in ('R0','R1','R2')]) for seed in seeds]
            equal(row[field],avg(seed_values))
        for count,field in (('empty_unknown_frames','unknown_cells'),('empty_stale_frames','stale_cells')):
            assert row[count]==sum(r[field]==0 for r in forecasts if r['case']==case and r['method']==method)
    assert len(summary['predictions'])==18 and {(r['case'],r['method']) for r in summary['predictions']}=={(c,m) for c in held for m in methods[2:]}
    drift={r['job']:r for r in summary['late_training_drift']}
    assert set(drift)=={m.replace('+','-')+f'-s{s}' for m in methods for s in seeds}
    for job,row in drift.items():
        values={r['episode']:r['mean_return'] for r in read(root/'formal'/job/'validation.json')}
        assert set(values)==set(range(200,2001,200))
        delta=(values[1800]+values[2000]-values[1400]-values[1600])/2
        equal(row['late_validation_return_change'],delta);assert row['late_drift_flag']==(abs(delta)>.02)
    result=dict(status='passed',event_method_rows=36,policy_episodes=108,rule_episodes=27,forecast_rows=9720,
        criteria=checks,gate_passed=all(checks.values()),na_unknown_rows=sum(r['unknown_cells']==0 for r in forecasts),
        na_stale_rows=sum(r['stale_cells']==0 for r in forecasts),
        summary_sha256=digest(root/'confirmation-summary.json'),auditor_sha256=digest(Path(__file__)),
        aggregation='time -> common route -> training seed -> event; event/parent/forcing dependence retained')
    out=ROOT/'docs/evidence/20261025-sciml-development/statistics-audit-final.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('PASSED independent domains,all aggregation,NA accounting and all gate decisions')


if __name__=='__main__':main()
