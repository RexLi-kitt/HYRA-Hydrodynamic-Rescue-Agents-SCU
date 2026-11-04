"""Audit pinned Okushiri laboratory data; no solver calibration or time shifting."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def audit(root):
    root=Path(root)
    provenance=json.loads((root/'provenance.json').read_text(encoding='utf8'))
    for item in provenance['files']:
        assert hashlib.sha256((root/item['file']).read_bytes()).hexdigest()==item['sha256']
    wave=np.loadtxt(root/'Benchmark_2_input.txt',skiprows=1)
    obs=np.loadtxt(root/'output_ch5-7-9.txt',skiprows=1)
    obs[:,1:]/=100.0  # Header explicitly states cm; boundary file already m.
    header={}
    with (root/'Benchmark_2_Bathymetry.asc').open() as f:
        for _ in range(6):
            key,value=next(f).split();header[key.lower()]=float(value)
    bed=np.loadtxt(root/'Benchmark_2_Bathymetry.asc',skiprows=6)
    assert bed.shape==(int(header['nrows']),int(header['ncols']))
    assert np.isfinite(bed).all() and not (bed==header['nodata_value']).any()
    for values in (wave,obs):
        assert np.isfinite(values).all() and (np.diff(values[:,0])>0).all()
        assert np.allclose(np.diff(values[:,0]),.05,rtol=0,atol=1e-10)
    start=max(wave[0,0],obs[0,0]);end=min(wave[-1,0],obs[-1,0])
    select=(obs[:,0]>=start)&(obs[:,0]<=end)
    assert np.allclose(obs[select,0],wave[:,0],rtol=0,atol=1e-10)
    gauges=[]
    for j,(name,y) in enumerate([('ch5',1.196),('ch7',1.696),('ch9',2.196)],1):
        v=obs[select,j];t=obs[select,0]
        gauges.append(dict(name=name,x_m=4.521,y_m=y,minimum_m=float(v.min()),maximum_m=float(v.max()),peak_time_s=float(t[v.argmax()])))
    result=dict(evidence='laboratory scaled tsunami experiment, not field flood observations',source_commit=provenance['commit'],boundary_samples=len(wave),observation_samples=len(obs),boundary_time_s=wave[[0,-1],0].tolist(),observation_time_s=obs[[0,-1],0].tolist(),common_window_s=[float(start),float(end)],common_samples=int(select.sum()),interval_s=.05,bed_header=header,bed_range_m=[float(bed.min()),float(bed.max())],gauges=gauges,solver_run=False)
    (root/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (root/'observations_si.json').write_text(json.dumps(dict(time_s=obs[:,0].tolist(),stage_m=obs[:,1:].tolist(),boundary_time_s=wave[:,0].tolist(),boundary_stage_m=wave[:,1].tolist()),allow_nan=False),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True)
    axs[0].plot(wave[:,0],wave[:,1]);axs[0].set_ylabel('Boundary stage (m)')
    for j,g in enumerate(gauges,1):axs[1].plot(obs[select,0],obs[select,j],label=g['name'])
    axs[1].set_ylabel('Observed stage (m)');axs[1].set_xlabel('Original time (s)');axs[1].legend()
    for ax in axs:ax.grid(alpha=.3)
    fig.suptitle('Okushiri laboratory data: common input/observation window')
    fig.tight_layout();fig.savefig(root/'continuous_observations.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('data');audit(parser.parse_args().data)
