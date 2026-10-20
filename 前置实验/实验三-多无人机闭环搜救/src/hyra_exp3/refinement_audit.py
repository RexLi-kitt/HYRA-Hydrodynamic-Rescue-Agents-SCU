"""Recompute all completed numerical comparisons from their saved fields."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from .artifacts import sha256,write_json
from .diagnostics import load_field
from .execution_plan import register
from .phase_probe import mission_view
from .protocol import load_protocol
from .refinement import metrics,passes


def _check_field(item,criteria):
    records=[item['coarse'],item['reference_5m2s'],item['reference_5m1s']]
    fields=[]
    for record in records:
        manifest=record['manifest']
        if sha256(manifest['field_path'])!=manifest['field_sha256']:
            raise AssertionError('Numerical comparison field changed')
        fields.append(load_field(manifest['field_path']))
    okay=True
    for row in item['comparisons']:
        a,b,c=[mission_view(f,row['phase_s']) for f in fields]
        calculated={'spatial_10_vs_5':metrics(a,b),'temporal_2_vs_1':metrics(b,c),
                    'joint_10m2s_vs_5m1s':metrics(a,c)}
        for name,result in calculated.items():
            for key in ('depth_mae_m','depth_p99_abs_m','complete_window_disagreement_fraction'):
                if not np.isclose(result[key],row[name][key],rtol=0,atol=1e-12):
                    raise AssertionError('Numerical comparison statistics changed')
            okay=okay and passes(result,criteria)
    if okay!=item['tolerances_passed']:
        raise AssertionError('Numerical tolerance verdict differs')
    return dict(fields=[r['manifest']['field_path'] for r in records],passed=okay)


def audit_refinement(root,run,workers=3):
    root,run=Path(root),Path(run);plan=register(root)
    summary=load_protocol(run/'refinement_summary.json')
    if summary['status']!='completed' or summary['cases']!=216:
        raise ValueError('Full numerical matrix required')
    families={r['family_id'] for r in load_protocol(root/'manifests/family_split.json')['families'] if r['split']=='design'}
    cases=[load_protocol(p) for p in sorted((run/'jobs').glob('*/condition_*/refinement_case.json'))]
    identities={(r['family']['family_id'],r['side'],r['condition']['rise_m'],r['condition']['rise_duration_s'],r['condition']['perturbation_m']) for r in cases}
    expected={(f,s,r,d,p) for f in families for s in (32,50,70) for r in (.6,.9) for d in (600,900) for p in (0.,.1,.2)}
    if identities!=expected or len(cases)!=216:
        raise AssertionError('Numerical family/scale/hydrology matrix incomplete')
    unique={}
    for case in cases:
        for item in case['fields'].values():
            key=item['reference_5m1s']['manifest']['field_path']
            if key in unique and unique[key]!=item:
                raise AssertionError('Shared nominal comparison differs across perturbations')
            unique[key]=item
    if len(unique)!=288:raise AssertionError('Expected 216 true and 72 nominal independent comparison triples')
    paths=set();passed=True
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_check_field,item,plan['numerical_criteria']) for item in unique.values()]
        for i,future in enumerate(as_completed(futures),1):
            row=future.result();paths.update(row['fields']);passed=passed and row['passed']
            if i%24==0:print(f'Numerical independent recheck {i}/288 triples',flush=True)
    result=dict(status='passed',comparisons_recomputed=288,cases=216,unique_fields=len(paths),
        numerical_tolerances_passed=passed,criteria=plan['numerical_criteria'],
        source_summary_sha256=sha256(run/'refinement_summary.json'),asymptotic_convergence_claimed=False)
    write_json(run/'refinement_integrity_audit.json',result)
    return result
