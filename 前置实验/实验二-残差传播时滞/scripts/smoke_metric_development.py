"""Exercise fresh, actual causal histories against the pinned native FilterPy."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from hyra_validation.metric_dataset import MetricFamilyData
from hyra_validation.nominal_response import NominalResponseLibrary,response_predictions
from hyra_validation.full_upstream import ordinary_full
from hyra_validation.local_service import load_design
from hyra_validation.artifacts import write_json


def main():
    pointer=json.loads((ROOT/'results/latest_metric_development_data.json').read_text(encoding='utf-8'))
    run=Path(pointer['run']);index=json.loads((run/'family_index.json').read_text(encoding='utf-8'))
    design=json.loads((ROOT/'configs/metric_development.design.v1.json').read_text(encoding='utf-8'))
    data=MetricFamilyData(index[0]['manifest_path'],design)
    inputs=data.inputs(80,'E1',240,20,0,.02,0,0)
    labels=data.labels(80,'E1',240,20)
    assert inputs['base'].shape==(6,12,26) and labels['truth'].shape==(6,12,26)
    assert inputs['history']['values'].shape==(6,6,12,6)
    refs=inputs['observation_references'];assert all(r['prior_release_s']<=r['valid_s']-20 for r in refs)
    allowed={'now_s','base','common','nominal_mean','nominal_drift','current_release_s','source_current',
             'history','observation_sha256','observation_references'}
    assert set(inputs)==allowed
    library=NominalResponseLibrary(data.nominal)
    predictions=[]
    for t in range(6):
        problem=library.source_problem(inputs['history']['values'][t],inputs['now_s'][t],inputs['nominal_mean'][t],inputs['nominal_drift'][t])
        pred=response_predictions(problem)
        assert np.all(np.isfinite(pred['physical']))
        predictions.append(dict(now_s=int(inputs['now_s'][t]),event_probability=pred['event_probability'],
            physical_MAE_m=float(abs(pred['physical']-labels['residual'][t]).mean()),
            recency_MAE_m=float(abs(pred['recency']-labels['residual'][t]).mean()),
            physical_recency_max_difference_m=float(abs(pred['physical']-pred['recency']).max())))
    source=ROOT/'third_party/filterpy';sys.path.insert(0,str(source))
    from filterpy.kalman import KalmanFilter
    import filterpy.kalman.kalman_filter as reference
    assert Path(reference.__file__).resolve().is_relative_to(source.resolve())
    config,_=load_design(ROOT);models=config['models'];maximum=0.;count=0
    for delay in (0,10):
        history=data.inputs(80,'E1',240,20,1,.02,delay,1)['history']
        ours=ordinary_full(history,models)
        for kind,qs in [('rw',models['kf_rw_process_variance_rate_m2_per_s']),('cv',models['kf_cv_process_spectral_density_m2_per_s3'])]:
            for q in qs:
                for r in models['kf_measurement_std_m']:
                    name=f'kf_{kind}_q{q:g}_r{r:g}'
                    for t in range(6):
                        for s in range(6):
                            f=KalmanFilter(dim_x=2,dim_z=1);f.x=np.zeros(2);f.H=np.array([[1.,0.]])
                            f.P=np.diag([models['kf_initial_bias_std_m']**2,
                                models['kf_initial_bias_rate_std_m_per_s']**2 if kind=='cv' else 0.]);f.R=np.array([[r*r]])
                            last=None
                            for record in history['values'][t,s,::-1]:
                                if not record[5]:continue
                                time=-record[3];dt=0 if last is None else time-last
                                f.F=np.array([[1.,dt if kind=='cv' else 0.],[0.,1.]])
                                f.Q=q*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]]) if kind=='cv' else np.diag([q*dt,0.])
                                f.predict();f.update(np.array([record[2]]));last=time
                            expected=f.x[0]+(np.arange(0,51,2)-(last or 0))*f.x[1]
                            error=float(abs(expected-ours[name][t,s]).max());maximum=max(maximum,error);count+=1
                            np.testing.assert_allclose(expected,ours[name][t,s],atol=1e-12,rtol=1e-12)
    result=dict(passed=True,family=data.manifest['family'],shape=list(inputs['common'].shape),
        reference_curves=count,maximum_FilterPy_difference_m=maximum,FilterPy_commit=pinned['commit'],
        reference_module_sha256=hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
        physical_predictions=predictions,efficacy_claim=False)
    output=run/'actual_full_history_smoke.json';write_json(output,result);print(json.dumps(result))


if __name__=='__main__':main()
