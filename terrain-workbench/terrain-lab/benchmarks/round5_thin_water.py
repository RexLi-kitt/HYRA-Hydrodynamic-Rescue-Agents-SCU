"""Single-factor wet/dry threshold audit; production solver is not modified."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--solver-dir',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.solver_dir));from flood_solver import FloodSolver
a.output.mkdir(parents=True,exist_ok=True)
plan=dict(hypothesis='The 1e-6 m momentum cutoff may explain the observed return-flow error.',
    factors=dict(velocity_depth=[1e-8,1e-6,1e-4],wall_flux_mode=['hydrostatic','reflective']),
    fixed=dict(dx_m=.01,flux_method='rusanov',manning=.03,end_s=3.,output_interval_s=.01,max_timestep_s=.05),
    prespecified_checks=dict(mass_residual_m3=1e-9,baseline_max_force_difference_N=1e-10,
        note='These are numerical verification tolerances, not observational accuracy targets. No threshold selected by RMSE fitting.'),
    solver_sha256=hashlib.sha256((a.solver_dir/'flood_solver.py').read_bytes()).hexdigest(),
    observation_sha256=hashlib.sha256((a.data/'for_exp.txt').read_bytes()).hexdigest())
(a.output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
class Instrumented(FloodSolver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cut=dict(steps_with_nonzero_discard=0,cell_step_count=0,sum_discarded_momentum_norm_m4_s=0.,max_discarded_cell_momentum_m2_s=0.,min_positive_fluid_depth_m=None)
    def _complete_dry_sinks(self,*args,**kwargs):
        super()._complete_dry_sinks(*args,**kwargs)
        fluid=~self.solid;positive=self.h[fluid&(self.h>0)]
        if positive.size:self.cut['min_positive_fluid_depth_m']=min(self.cut['min_positive_fluid_depth_m'] or float('inf'),float(positive.min()))
        thin=fluid&(self.h<=self.velocity_depth);q=np.hypot(self.mx[thin],self.mz[thin]);n=int(np.count_nonzero(q))
        if n:
            self.cut['steps_with_nonzero_discard']+=1;self.cut['cell_step_count']+=n
            self.cut['sum_discarded_momentum_norm_m4_s']+=float(q.sum()*self.dx**2)
            self.cut['max_discarded_cell_momentum_m2_s']=max(self.cut['max_discarded_cell_momentum_m2_s'],float(q.max()))
obs=np.array([[float(s.split()[0]),float(s.split()[-1])] for s in (a.data/'for_exp.txt').read_text().splitlines()])
x=(np.arange(160)+.5)*.01;y=(np.arange(61)+.5)*.01
solid=(x[None,:]>=.9)&(x[None,:]<1.02)&(y[:,None]>=.25)&(y[:,None]<.37)
depth=np.broadcast_to(np.where(x<=.4,.3,.01),(61,160)).copy();depth[solid]=0
rows=np.flatnonzero((y>=.25)&(y<.37));all_results=[]
for wall in ('hydrostatic','reflective'):
    baseline=None
    for cutoff in (1e-6,1e-8,1e-4):
        s=Instrumented(np.zeros_like(depth),depth,.01,boundary='closed',manning=.03,
            velocity_depth=cutoff,wall_flux_mode=wall,max_timestep=.05)
        s.set_environment(solid=solid);start=time.perf_counter();curve=[]
        for t in np.linspace(0,3,301):
            s.advance(float(t));force=.5*1024*9.8*.01*np.sum(s.h[rows,89]**2-s.h[rows,102]**2);curve.append([t,force])
        curve=np.asarray(curve)
        if baseline is None:baseline=curve.copy()
        metrics={}
        for shift in (0.,.1):
            for label,lo,hi in [('whole',shift,3+shift),('return_flow',1.5,1.8)]:
                valid=(obs[:,0]>=lo)&(obs[:,0]<=hi);e=np.interp(obs[valid,0],curve[:,0]+shift,curve[:,1])-obs[valid,1]
                metrics[f'{label}_shift_{shift}']=dict(n=int(valid.sum()),rmse_N=float(np.sqrt(np.mean(e**2))),mae_N=float(np.mean(abs(e))))
        name=f'{wall}-{cutoff:g}';np.savetxt(a.output/(name+'.csv'),curve,delimiter=',',header='time_s,force_N',comments='')
        result=dict(name=name,wall=wall,velocity_depth=cutoff,diagnostics=s.cut,metrics=metrics,
            max_difference_from_default_N=float(np.max(abs(curve[:,1]-baseline[:,1]))),summary=s.summary(),elapsed_s=time.perf_counter()-start)
        assert abs(result['summary']['balance']['residual_m3'])<1e-9
        all_results.append(result);(a.output/'results.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
