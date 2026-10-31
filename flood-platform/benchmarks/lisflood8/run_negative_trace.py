"""Execute once; verify an observer before interpreting its trace."""
from pathlib import Path
import hashlib, importlib.util, json, os, shutil, subprocess, time
import numpy as np

out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
r=out/'round75'; case=r/'trace_case'; case.mkdir(exist_ok=False)
base=out/'round74/smoke60_stride'
names=['case.par','carlisle.bci','carlisle.bdy','carlisle.stage',
 'CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw',
 'CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y']
for n in names: shutil.copy2(base/n,case/n)
exe=r/'build-trace-cpu64/Release/lisflood.exe'
start=time.perf_counter()
with (case/'run.log').open('w') as f:
 p=subprocess.run([str(exe),'-v','case.par'],cwd=case,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,timeout=180)
elapsed=time.perf_counter()-start
traces=[json.loads(l) for l in (case/'negative-depth.jsonl').read_text().splitlines()]
steps=np.atleast_2d(np.loadtxt(case/'accepted_timesteps.txt'))
baseline=np.atleast_2d(np.loadtxt(base/'accepted_timesteps.txt'))
checks=[]
for t in traces:
 fw,fe,fn,fs=t['mass_flux_w_e_n_s'];rhs=-((fe-fw)/t['dx_m']+(fn-fs)/t['dy_m'])
 h=(t['H_n_postsource']+t['dt_s']*rhs) if t['stage']==1 else .5*(t['H_n_postsource']+t['H_stage']+t['dt_s']*rhs)
 tol=1e-15
 checks.append(dict(stage=t['stage'],t_s=t['t_s'],rhs_reconstructed=rhs,h_reconstructed=h,
  rhs_matches=abs(rhs-t['rhs_H'])<tol,h_matches=abs(h-t['H_after_predicted'])<tol,
  net_outflow_depth=t['dt_s']*(-rhs),available_depth=t['H_n_postsource'] if t['stage']==1 else t['H_n_postsource']+t['H_stage']))
spec=importlib.util.spec_from_file_location('probe',out/'round73/probe_boundary_series.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
control,field=mod.run_case(exe,r/'square_control',multi=True)
oldfield=np.loadtxt(out/'round74/square_multiple_regression/results/check-0000.wd',skiprows=6)
result=dict(diagnostic_exit=p.returncode,expected_exit_42=p.returncode==42,wall_s=elapsed,
 steps_logged=len(steps),note='Last logged step was selected but interrupted before completion; do not count it as a completed physical step.',
 timestep_prefix_exact=bool(np.array_equal(steps,baseline[:len(steps)])),input_bytes_unchanged=all((case/n).read_bytes()==(base/n).read_bytes() for n in names),
 trace_count=len(traces),independent_reconstruction=checks,control=control,
 square_field_exact=bool(field is not None and np.array_equal(field,oldfield)),
 observer_no_arithmetic_change_verified=False,trace_sha256=hashlib.sha256((case/'negative-depth.jsonl').read_bytes()).hexdigest())
result['observer_no_arithmetic_change_verified']=all([result['expected_exit_42'],result['timestep_prefix_exact'],result['input_bytes_unchanged'],result['square_field_exact'],control.get('finite_nonnegative',False),control.get('volume_pass',False),all(c['rhs_matches'] and c['h_matches'] for c in checks)])
(r/'trace-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
print(json.dumps(traces),flush=True)
