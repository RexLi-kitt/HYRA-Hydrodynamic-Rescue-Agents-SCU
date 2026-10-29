"""Development-only flow reversal check for the frozen scalar mean route."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import signal_routes
import numpy as np
from signal_routes.common import DEFAULT,read,write,digest,sources
from signal_routes.offline import load_records,estimator


def main(study=DEFAULT):
    study=Path(study); output=study/'mean-direction-diagnostic.json'
    contract=dict(sources=sources(),parameters_sha256=digest(study/'parameters.json'),
                  development_sha256=digest(study/'development-records.npz'))
    if output.exists():
        previous=read(output)
        if not previous['passed'] or previous['contract']!=contract: raise ValueError('Stale mean direction diagnostic')
        return previous
    if (study/'confirmation/opened.json').exists(): raise ValueError('Cannot add development diagnostics after confirmation opens')
    parameters=read(study/'parameters.json')['parameters']; rows=[]
    for original in load_records(study/'development-records.npz'):
        if original['step']!=12: continue
        # Only legal estimator inputs are retained, without any offline truth fields.
        record={k:original[k] for k in ('signal','water','valid','stamps','now','obstacles','bed','dx_m')}
        reverse={**record,'water':record['water'].copy()}; reverse['water'][1:]*=-1
        for method in ('R2','C2'):
            forward=estimator(method,parameters[method],record)
            backward=estimator(method,parameters[method],reverse)
            fmask=forward.raw_features(); bmask=backward.raw_features()
            for a,b in zip(fmask,bmask): np.testing.assert_array_equal(a,b)
            left=forward.posterior(True); right=backward.posterior(True)
            np.testing.assert_array_equal(left['mean'],right['mean'])
            np.testing.assert_array_equal(left['variance'],right['variance'])
            rows.append(dict(method=method,parent=original['parent'],mechanism=original['mechanism'],seed=original['seed'],
                nonzero_fitted_mean=bool(np.any(np.asarray(parameters[method]['beta'])!=0)),
                nonzero_arrived_velocity=bool(np.any(record['water'][1:,record['valid']]!=0)),
                mean_max_difference=float(np.max(np.abs(left['mean']-right['mean']))),
                variance_max_difference=float(np.max(np.abs(left['variance']-right['variance'])))))
    assert len(rows)==36 and len({r['parent'] for r in rows})==3
    assert all(r['nonzero_fitted_mean'] for r in rows)
    assert any(r['nonzero_arrived_velocity'] for r in rows)
    assert sources()==contract['sources'] and digest(study/'parameters.json')==contract['parameters_sha256']
    result=dict(passed=True,scope='development records only; estimator-input perturbation, not another physical event',
        tool_sha256=digest(__file__),contract=contract,rows=rows,confirmation_performance_read=False,
        interpretation='The registered R2/C2 scalar mean features are invariant under simultaneous u/v reversal. '
                       'This route can test scalar water covariate association but cannot establish upstream/downstream directionality.')
    write(output,result); return result


if __name__=='__main__':
    result=main(); print(dict(passed=result['passed'],rows=len(result['rows']),maximum_mean_difference=max(r['mean_max_difference'] for r in result['rows'])))
