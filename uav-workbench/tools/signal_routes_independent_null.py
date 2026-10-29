"""Strengthen synthetic null to independent-parent water, without changing policies.

Run before confirmation opens. Only diagnostic artifacts are revised; policy
parameters.json, sources, checkpoints and task selection are never modified.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import signal_routes
import numpy as np
from signal_routes.common import DEFAULT,read,write,digest
from signal_routes.offline import load_records,synthetic_record,fit,estimator
from signal_routes import METHODS


def independent(records):
    parents=sorted({r['parent'] for r in records}); result=[]
    for r in records:
        donor=parents[(parents.index(r['parent'])+1)%len(parents)]
        other=next(p for p in records if p['parent']==donor and p['seed']==r['seed']
                   and p['step']==r['step'] and p['mechanism']==r['mechanism'])
        result.append({**r,'water_truth':other['water_truth'].copy(),'null_donor_parent':donor})
    return result


def main(study=DEFAULT):
    study=Path(study)
    marker=study/'independent-null-audit.json'
    if marker.exists():
        audit=read(marker)
        if not audit['passed'] or not audit['donor_parent_always_different']:
            raise ValueError('Independent-parent null audit did not pass')
        if audit['main_policy_parameters_sha256']!=digest(study/'parameters.json'):
            raise ValueError('Main policy parameters changed after null diagnostic')
        if audit['fit_source_sha256']!=digest(study/'fit-independent-null-records.npz'):
            raise ValueError('Independent-parent null records changed')
        return
    if (study/'confirmation/opened.json').exists(): raise ValueError('Cannot revise diagnostics after confirmation')
    history=study/'diagnostic-history'; history.mkdir(exist_ok=True)
    write(history/'initial-null-parameters.json',read(study/'parameters-null.json'))
    old=read(study/'offline-diagnostics.json')
    write(history/'initial-local-shuffle-null.json',dict(rows=[r for r in old['rows'] if r['condition']=='null'],
        scope='initial local spatial shuffle; superseded by independent-parent null'))
    records=independent(load_records(study/'fit-records.npz'))
    path=study/'fit-independent-null-records.npz'
    np.savez_compressed(path,records=np.array(records,dtype=object))
    params=fit(path,study/'parameters-null.json','null')
    rows=[]
    for original in independent(load_records(study/'development-records.npz')):
        if original['step']!=12: continue
        r=synthetic_record(original,'null'); truth=r['truth']; missing=~r['valid']
        for method in METHODS:
            if method=='Braw':
                score=dict(rmse=float(np.sqrt(np.mean(truth[missing]**2))),nll=None,coverage95=None,n=int(missing.sum())); residual=0.
            else:
                model=estimator(method,params[method],r); score=model.score(truth)
                residual=model.posterior()['diagnostics']['residual_inf']
            rows.append(dict(method=method,condition='null',parent=r['parent'],forcing=r['mechanism'],
                seed=r['seed'],null_donor_parent=r['null_donor_parent'],**score,residual=residual))
    old['rows']=[r for r in old['rows'] if r['condition']!='null']+rows
    old['null_recipe']='different independent parent water, matching forcing/time/layout, then joint spatial permutation'
    old['null_recipe_sha256']=digest(__file__); write(study/'offline-diagnostics.json',old)
    write(marker,dict(passed=True,rows=len(rows),fit_source_sha256=digest(path),tool_sha256=digest(__file__),
        main_policy_parameters_sha256=digest(study/'parameters.json'),policy_inputs_or_checkpoints_changed=False,
        donor_parent_always_different=all(r['null_donor_parent']!=r['parent'] for r in records)))


if __name__=='__main__': main()
