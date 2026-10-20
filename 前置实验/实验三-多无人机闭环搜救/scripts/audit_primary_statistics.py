"""Recompute paired primary statistics from integer service counts independently."""
from pathlib import Path
from fractions import Fraction
import argparse
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
import numpy as np
from hyra_exp3.artifacts import sha256,write_json
from hyra_exp3.protocol import load_protocol
from hyra_exp3.freeze import verify_frozen


def recompute_comparison(rows,families,baseline,targets,stored):
    counts={(r['family_id'],r['training_seed'],r['layout'],r['method']):r['summary']['successful_services'] for r in rows}
    seeds=(101,202,303)
    matrix=np.asarray([[float(Fraction(100*sum(counts[f,s,l,'P']-counts[f,s,l,baseline] for l in (0,1)),2*targets))
        for s in seeds] for f in families])
    assert stored['family_ids']==families and stored['training_seeds']==list(seeds) and stored['layouts_averaged_first']
    assert stored['method']=='P' and stored['baseline']==baseline
    assert np.allclose(matrix,np.asarray(stored['family_seed_differences_pp']),atol=1e-12,rtol=0)
    # A multiplicity-weighted crossed mean gives the same resampling rule without
    # using the production paired_effect implementation or resampling layouts.
    rng=np.random.default_rng(2026101803);replicates=10000;boot=np.empty(replicates)
    for i in range(replicates):
        family_counts=np.bincount(rng.integers(0,len(families),len(families)),minlength=len(families))
        seed_counts=np.bincount(rng.integers(0,3,3),minlength=3)
        boot[i]=float(family_counts@matrix@seed_counts)/(len(families)*3)
    point=math.fsum(matrix.ravel())/matrix.size
    interval=np.quantile(boot,[.025,.975]).tolist()
    assert math.isclose(stored['delta_percentage_points'],point,abs_tol=1e-12,rel_tol=0)
    assert np.allclose(interval,stored['ci95_percentage_points'],atol=1e-12,rtol=0)
    assert stored['bootstrap_replicates']==10000 and stored['bootstrap_seed']==2026101803
    assert stored['primary_positive']==(interval[0]>0) and stored['engineering_target_3pp_met']==(point>=3)
    return dict(baseline=baseline,delta_percentage_points=point,ci95_percentage_points=interval,
        family_seed_differences_pp=matrix.tolist(),all_observed_paired_service_differences_zero=bool(np.all(matrix==0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();directory=args.run.resolve()
    frozen_path=ROOT/'configs/frozen_experiment3.v1.json';frozen=verify_frozen(ROOT,frozen_path)
    summary_path=directory/'formal_summary.json';summary=load_protocol(summary_path)
    raw_audit_path=directory/'formal_integrity_audit.json';raw_audit=load_protocol(raw_audit_path)
    assert raw_audit['status']=='passed' and raw_audit['episodes']==3024
    assert raw_audit['formal_summary_sha256']==sha256(summary_path) and raw_audit['frozen_sha256']==sha256(frozen_path)
    rows=summary['rows'];planned={r['id']:r for r in frozen['formal_matrix']}
    identities=[r['id'] for r in rows]
    assert summary['status']=='completed' and summary['episodes']==3024
    assert summary['main_episodes']==2376 and summary['delayed_episodes']==648 and summary['B_star']==frozen['B_star']
    assert len(identities)==len(set(identities))==3024 and set(identities)==set(planned)
    assert all(all(row[k]==v for k,v in planned[row['id']].items()) for row in rows)
    expected_groups={(32,0.):3,(50,0.):5,(50,10.):3,(70,0.):3}
    assert {(g['side'],g['delay_s']) for g in summary['groups']}==set(expected_groups) and len(summary['groups'])==4
    results=[]
    for stored in summary['groups']:
        side,delay=stored['side'],stored['delay_s'];group=[r for r in rows if (r['side'],r['delay_s'])==(side,delay)]
        families=sorted({r['family_id'] for r in group});targets={32:12,50:30,70:60}[side]
        assert len(families)==36 and len(group)==216*expected_groups[side,delay]
        assert all(r['summary']['hidden_targets']==targets and 0<=r['summary']['successful_services']<=targets for r in group)
        names={r['method'] for r in group};assert names==set(stored['rates'])
        rates={}
        for method in sorted(names):
            values=[r for r in group if r['method']==method];assert len(values)==216
            successes=sum(r['summary']['successful_services'] for r in values)
            rate=float(Fraction(successes,216*targets))
            assert math.isclose(stored['rates'][method]['success_rate'],rate,abs_tol=1e-12,rel_tol=0)
            rates[method]=dict(success_rate=rate,successful_service_instances=successes,hidden_target_instances=216*targets)
        comparisons=[recompute_comparison(group,families,frozen['B_star'],targets,stored['primary_effect'])]
        for baseline,effect in stored['mechanism_contrasts'].items():
            comparisons.append(recompute_comparison(group,families,baseline,targets,effect))
        assert set(stored['mechanism_contrasts'])==({'P_wrong_links','P_uniform_time'} if (side,delay)==(50,0.) else set())
        results.append(dict(side=side,delay_s=delay,rates=rates,comparisons=comparisons))
    result=dict(status='passed',episodes=3024,groups=results,comparisons_recomputed=sum(len(g['comparisons']) for g in results),
        complete_registered_matrix_verified=True,integer_service_counts_used=True,layouts_averaged_before_crossed_resampling=True,
        bootstrap_replicates=10000,bootstrap_seed=2026101803,
        target_counts_are_execution_instances_not_unique_people=True,
        formal_summary=dict(path=str(summary_path),sha256=sha256(summary_path)),
        full_raw_episode_audit=dict(path=str(raw_audit_path),sha256=sha256(raw_audit_path)),
        independent_audit_source=dict(path=str(Path(__file__).resolve()),sha256=sha256(Path(__file__))))
    write_json(directory/'summary_statistics_integrity_audit.json',result)
    print({k:v for k,v in result.items() if k!='groups'})
