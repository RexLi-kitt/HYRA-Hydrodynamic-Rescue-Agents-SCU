"""Summarize measured EA4 model agreement and synthetic Eden capacity probes."""
from pathlib import Path
import csv
import json
import math
import sys

LAB=Path(__file__).resolve().parents[1];WORK=LAB/'.runtime/lisflood-comparison'
sys.path.insert(0,str(WORK/'python-packages'))
import numpy as np

def load_samples(path):
    rows=[]
    for line in path.read_text().splitlines():
        parts=line.split()
        if len(parts)!=7:continue
        try:rows.append([float(x) for x in parts])
        except ValueError:pass
    return np.array(rows)

def main():
    folder=WORK/'ea4-10m'
    results={mode:json.loads((folder/f'result-{mode}.json').read_text()) for mode in ['stepwise','linear']}
    assert results['stepwise']['input_sha256']==results['linear']['input_sha256']
    assert results['stepwise']['solver_sha256']==results['linear']['solver_sha256']
    assert all(r['complete'] and r['duration_s']==18000 and r['hyra_finite'] and r['hyra_nonnegative'] for r in results.values())
    assert abs(results['linear']['hyra_balance']['external_inflow_m3']-285000)<1
    stations=[]
    ref_h=load_samples(folder/'dg2/ea4.stage');ref_v=load_samples(folder/'dg2/ea4.velocity')
    coordinates=np.loadtxt(folder/'ea4.stage',skiprows=1)
    for mode in ['stepwise','linear']:
        with np.load(folder/f'hyra-{mode}.npz') as saved:rows=saved['records']
        rh=np.stack([np.interp(rows[:,0],ref_h[:,0],ref_h[:,j]) for j in range(1,7)],axis=1)
        rv=np.stack([np.interp(rows[:,0],ref_v[:,0],ref_v[:,j]) for j in range(1,7)],axis=1)
        h=rows[:,1:7];v=rows[:,7:13]
        for j in range(6):
            stations.append(dict(mode=mode,station=j+1,x_m=coordinates[j,0],y_m=coordinates[j,1],
                depth_rmse_m=float(np.sqrt(np.mean((h[:,j]-rh[:,j])**2))),
                speed_rmse_m_s=float(np.sqrt(np.mean((v[:,j]-rv[:,j])**2))),
                hyra_sampled_peak_depth_m=float(h[:,j].max()),dg2_sampled_peak_depth_m=float(rh[:,j].max())))
        results[mode]['station_depth_rmse_m']=float(np.sqrt(np.mean((h-rh)**2)))
        results[mode]['station_speed_rmse_m_s']=float(np.sqrt(np.mean((v-rv)**2)))
        # Raster outputs use the first adaptive step crossing each save time.
        results[mode]['dg2_snapshot_actual_times_s']=[float(ref_h[np.searchsorted(ref_h[:,0],t),0]) for t in range(3600,18001,3600)]
    # Native result was preserved before the script gained mode metadata.
    results['stepwise']['forcing_interpretation']='native piecewise constant'
    results['stepwise']['original_settings'].update(numeric_precision='float32',solver='CPU DG2 8.0.3')
    capacity=[json.loads(p.read_text()) for p in sorted((WORK/'capacity').glob('*.json'),key=lambda p:int(p.stem))]
    estimates=[]
    for dx,n in [(40,1250),(20,2500)]:
        r=next(x for x in capacity if x['n']==n)
        median_dt=float(np.median(r['dt_s']));seconds=r['median_wall_per_step_s']*475200/min(1.,median_dt)
        estimates.append(dict(dx_m=dx,cells=n*n,synthetic_wall_per_step_s=r['median_wall_per_step_s'],
            synthetic_median_dt_s=median_dt,synthetic_peak_private_GiB=r['peak_private_commit_bytes']/2**30,
            full_event_hours_if_same_step_cost_and_dt_with_default_cap=seconds/3600,
            full_event_hours_if_30s_cap_and_same_synthetic_dt=r['median_wall_per_step_s']*475200/median_dt/3600,
            label='conditional extrapolation from 3 synthetic steps, not an Eden runtime prediction'))
    out=LAB.parent/'docs/experiments/2026-10-24/lisflood-comparison';out.mkdir(parents=True,exist_ok=True)
    report=dict(date='2026-10-24',results=results,capacity=capacity,conditional_estimates=estimates,
        hardware=dict(cpu_logical_processors=24,physical_RAM_GiB=31.718063354492188,GPU='RTX 5070 Ti Laptop, 12227 MiB; not used'),
        verified_checks=dict(identical_input_hashes=True,unchanged_production_solver=True,complete_both_HYRA_runs=True,
            linear_inflow_within_1m3_of_analytic_integral=True),
        limitations=['one EA4 benchmark only, no independent measured truth',
            'linear input adapter changes forcing interpretation only; production solver unchanged',
            'native precision, CFL, limiter and boundary differences are retained',
            'DG2 raster snapshots cross nominal save times by an adaptive-step offset; exact times retained',
            'station outputs interpolated to HYRA 100-second output times; speed extraction semantics can differ',
            'capacity tests use synthetic uniform rainfall and a thin layer plus channel, not Eden data',
            'Eden full DEM/rainfall/masks/observations not obtained; no full Eden event executed'])
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'station-comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(stations[0]));w.writeheader();w.writerows(stations)
    print(json.dumps(dict(results=results,conditional_estimates=estimates),indent=2))

if __name__=='__main__':main()
