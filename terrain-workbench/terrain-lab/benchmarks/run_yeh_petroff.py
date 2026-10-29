"""HYRA comparison with ANUGA's archived Yeh-Petroff laboratory force data.

Reads published data only; does not execute ANUGA. No tuning or data rejection.
"""
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--solver-dir',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--wall-flux-mode',choices=['hydrostatic','reflective'],default='hydrostatic')
    p.add_argument('--grid-widths',nargs='+',type=float,choices=[.01,.005],default=[.01,.005])
    p.add_argument('--solid-only',action='store_true')
    p.add_argument('--coarse-only',action='store_true');args=p.parse_args()
    sys.path.insert(0,str(args.solver_dir));from flood_solver import FloodSolver
    solver_hash=hashlib.sha256((args.solver_dir/'flood_solver.py').read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    obs=[]
    for line in (args.data/'for_exp.txt').read_text().splitlines():
        parts=line.split();obs.append((float(parts[0]),float(parts[-1])))
    obs=np.array(obs);order=np.argsort(obs[:,0],kind='stable');obs=obs[order]
    # Keep repeated time values and negative instrument readings; no trimming.
    output=[]
    cases=[(m,d,'solid') for d in ((.01,) if args.coarse_only else args.grid_widths) for m in ('rusanov','knp')]
    if not args.coarse_only and not args.solid_only:cases += [(m,.01,'raised_bed') for m in ('rusanov','knp')]
    for method,dx,representation in cases:
        nx,nz=round(1.6/dx),round(.61/dx)
        x=(np.arange(nx)+.5)*dx;y=(np.arange(nz)+.5)*dx
        obstacle=(x[None,:]>=.9)&(x[None,:]<1.02)&(y[:,None]>=.25)&(y[:,None]<.37)
        bed=np.where(obstacle,.75,0.) if representation=='raised_bed' else np.zeros((nz,nx))
        stage=np.broadcast_to(np.where(x<=.4,.3,.01),(nz,nx))
        h=np.maximum(stage-bed,0.)
        if representation=='solid':h[obstacle]=0.
        s=FloodSolver(bed,h,dx,boundary='closed',manning=.03,flux_method=method,max_timestep=.05,wall_flux_mode=args.wall_flux_mode)
        if representation=='solid':s.set_environment(solid=obstacle)
        rows=np.flatnonzero((y>=.25)&(y<.37));left=round(.9/dx)-1;right=round(1.02/dx)
        series=[];start=time.perf_counter()
        for t in np.linspace(0,3,301):
            s.advance(float(t))
            # Hydrostatic net force on upstream/downstream vertical column faces.
            # rho=1024 and g=9.8 follow the source comparison's postprocessing.
            force=.5*1024*9.8*dx*np.sum(s.h[rows,left]**2-s.h[rows,right]**2)
            series.append((float(t),float(force)))
        series=np.asarray(series);metrics={}
        # +0.1 s is a published plotting convention, not a fitted alignment.
        for shift in (0.,.1):
            valid=(obs[:,0]>=shift)&(obs[:,0]<=3+shift)
            predicted=np.interp(obs[valid,0],series[:,0]+shift,series[:,1]);error=predicted-obs[valid,1]
            metrics[str(shift)]=dict(n=int(valid.sum()),rmse_N=float(np.sqrt(np.mean(error**2))),mae_N=float(np.mean(np.abs(error))),bias_N=float(np.mean(error)),window_s=[shift,3+shift])
        name=f'{method}-{dx:g}-{representation}'
        np.savetxt(args.output/(name+'.csv'),series,delimiter=',',header='time_s,hydrostatic_force_N',comments='')
        row=dict(case=name,method=method,dx_m=dx,shape=[nz,nx],representation=representation,elapsed_s=time.perf_counter()-start,
                 metrics=metrics,peak_force_N=float(series[:,1].max()),peak_time_s=float(series[series[:,1].argmax(),0]),summary=s.summary())
        output.append(row)
        (args.output/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
        print(json.dumps({k:row[k] for k in ('case','elapsed_s','metrics','peak_force_N','peak_time_s')}),flush=True)
    manifest=dict(source='https://github.com/anuga-community/anuga_core/tree/main/validation_tests/experimental_data/dam_break_yeh_petroff',wall_flux_mode=args.wall_flux_mode,
        data_sha256=hashlib.sha256((args.data/'for_exp.txt').read_bytes()).hexdigest(),solver_sha256=solver_hash,
        python=platform.python_version(),numpy=np.__version__,n_observations=len(obs),unique_times=len(np.unique(obs[:,0])),time_range=obs[[0,-1],0].tolist(),
        observed_max_force_N=float(obs[:,1].max()),rho_postprocessing=1024,g_postprocessing=9.8,
        note='Independent HYRA reproduction of archived setup, not an ANUGA run. Cell-centre adjacent-face hydrostatic force; vertical dynamic impact unresolved. Fixed source plot shift +0.1s reported alongside unshifted metrics.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
