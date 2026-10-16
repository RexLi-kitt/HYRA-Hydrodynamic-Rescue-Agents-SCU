"""Paired robust contrasts: take minimum within family before family averaging."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hyra_validation.artifacts import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();run=args.run.resolve()
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'));specs=json.loads((run/'candidates.json').read_text(encoding='utf-8'))
    names=[s['name'] for s in specs];winners=selection['winners'];chosen={g:w['name'] for g,w in winners.items()}
    p4=chosen['P4'];matched=p4.replace('P4_','P5_',1);assert matched in names
    chosen['P5_exact_P4']=matched
    families=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('metric_v2_*.json')) if not p.name.endswith('_information_action.json')]
    contrasts=[('H2_MAE_6_12_improvement_vs_P5','mae_6_12','P5','P4'),
        ('H2_MAE_6_12_improvement_vs_matched_P5','mae_6_12','P5_exact_P4','P4'),
        ('H4_common_window_error_reduction_vs_Bstar','window_error','B_star','P4'),
        ('H5_completed_service_gain_vs_Bstar','services','P4','B_star'),
        ('service_gain_vs_P5','services','P4','P5')]
    results=[]
    for family in families:
        if not all(m['active'] for m in family['meshes'].values()):continue
        row=dict(family=family['family'],values={})
        for label,metric,left,right in contrasts:
            differences={mesh:m[metric][names.index(chosen[left])]-m[metric][names.index(chosen[right])] for mesh,m in family['meshes'].items()}
            row['values'][label]=dict(by_mesh=differences,robust=min(differences.values()))
        row['oracle_headroom_vs_Bstar']=min(m['oracle_services']-m['services'][names.index(chosen['B_star'])] for m in family['meshes'].values())
        results.append(row)
    mean={key:float(np.mean([r['values'][key]['robust'] for r in results])) for key,*_ in contrasts} if results else {}
    result=dict(chosen=chosen,independent_families=len(results),training_diagnostic=selection['training_diagnostic'],
        mean_family_robust_contrasts=mean,mean_robust_oracle_headroom_vs_Bstar=float(np.mean([r['oracle_headroom_vs_Bstar'] for r in results])) if results else None,
        families=results,confirmation_claim=False,CI_policy='No confirmatory confidence claim on reused training diagnostics')
    write_json(run/'paired_comparisons.json',result);print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
