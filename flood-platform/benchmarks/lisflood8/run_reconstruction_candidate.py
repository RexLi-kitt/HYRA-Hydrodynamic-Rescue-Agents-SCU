"""Stationary-water gates before one original-input startup candidate."""
from pathlib import Path
import importlib.util, json, os, shutil, subprocess, time
import numpy as np

out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');r=out/'round75'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
probe=module('probe',out/'round73/probe_boundary_series.py')
audit=module('audit',out/'round73/audit_carlisle_rong2023.py')
baseline=r/'build-trace-cpu64/Release/lisflood.exe';candidate=r/'build-reconstruction-cpu64/Release/lisflood.exe'

def execute(exe,case):
 start=time.perf_counter()
 with (case/'run.log').open('w') as f:
  try:p=subprocess.run([str(exe),'-v','case.par'],cwd=case,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,timeout=180);code=p.returncode
  except subprocess.TimeoutExpired:code='timeout_180s'
 return code,time.perf_counter()-start

def lake(exe,name,partial):
 case=r/name;case.mkdir(exist_ok=False)
 ny,nx=8,12;X=np.broadcast_to((np.arange(nx)+.5)/nx,(ny,nx))
 z=np.where(X<.5,.25,1.5) if partial else .5+.15*np.cos(2*np.pi*X)
 zx=np.zeros_like(z) if partial else -.15*2*np.pi*np.sin(2*np.pi*X)/nx/(2*np.sqrt(3))
 zx[:,[0,-1]]=0;zy=np.zeros_like(z);h=np.maximum(1-z,0)
 header='ncols 12\nnrows 8\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999'
 for n,a in [('bed.asc',z),('bed.asc1x',zx),('bed.asc1y',zy),('water.asc',h),('water.asc1x',-zx),('water.asc1y',zy)]:np.savetxt(case/n,a,header=header,comments='',fmt='%.17g')
 (case/'case.par').write_text('DEMfile bed.asc\nstartfile water.asc\ndirroot results\nresroot check\ndg2\nsim_time 3\ninitial_tstep .02\nmassint 1\nsaveint 3\nfpfric 0\noutput_precision 15\n')
 code,elapsed=execute(exe,case);a=None
 result=dict(exit_code=code,wall_s=elapsed,partial_dry=partial)
 if code==0:
  a=np.loadtxt(case/'results/check-0000.wd',skiprows=6)
  result.update(max_depth_drift_m=float(np.max(np.abs(a-h))),volume_error_m3=float(a.sum()-h.sum()),finite_nonnegative=bool(np.isfinite(a).all() and (a>=0).all()))
 result['passed']=code==0 and result['finite_nonnegative'] and result['max_depth_drift_m']<1e-10 and abs(result['volume_error_m3'])<1e-8
 (case/'result.json').write_text(json.dumps(result,indent=2));return result,a

result={'stationary':{}}
for partial in [False,True]:
 label='partial' if partial else 'wet_sloped'
 old,oa=lake(baseline,'lake_old_'+label,partial)
 new,na=lake(candidate,'lake_candidate_'+label,partial)
 result['stationary'][label]=dict(baseline=old,candidate=new,fields_identical=bool(oa is not None and na is not None and np.array_equal(oa,na)))
control,z=probe.run_case(candidate,r/'candidate_square',multi=True)
result['flat_source_control']=control
small_pass=all(v['baseline']['passed'] and v['candidate']['passed'] for v in result['stationary'].values()) and control.get('finite_nonnegative',False) and control.get('volume_pass',False)
result['small_gates_passed']=small_pass
(r/'candidate-controls.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if small_pass:
 case=r/'candidate_case';case.mkdir(exist_ok=False)
 base=out/'round74/smoke60_stride'
 names=['case.par','carlisle.bci','carlisle.bdy','carlisle.stage','CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw','CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y']
 for n in names:shutil.copy2(base/n,case/n)
 code,elapsed=execute(candidate,case)
 actual=dict(exit_code=code,wall_s=elapsed,inputs_unchanged=all((case/n).read_bytes()==(base/n).read_bytes() for n in names))
 steps=np.loadtxt(case/'accepted_timesteps.txt',ndmin=2)
 actual['logged_steps']=len(steps);actual['last_logged_t_and_dt']=steps[-1].tolist()
 traces=[json.loads(l) for l in (case/'negative-depth.jsonl').read_text().splitlines()] if (case/'negative-depth.jsonl').exists() else []
 actual['negative_update_traces']=traces
 if code==0:
  head,h=audit.read_asc(case/'results/check-0000.wd');dx=head['cellsize'];curves=audit.read_bdy(case/'carlisle.bdy')
  q=sum(np.interp(steps[:,0],curves[n]['time_s'],curves[n]['q_as_stored'])*dx for n in ['upstream1','upstream2','upstream3'])
  vol=float(np.dot(q,steps[:,1]));raster=float(h.sum()*dx*dx)
  mass=np.loadtxt(case/'results/check.mass',skiprows=1,ndmin=2)
  actual.update(end_s=float(steps[-1].sum()),min_depth_m=float(h.min()),max_depth_m=float(h.max()),negative_cells=int((h<0).sum()),finite_nonnegative=bool(np.isfinite(h).all() and (h>=0).all()),accepted_source_volume_m3=vol,raster_volume_m3=raster,ledger_error_m3=raster-vol,no_outflow=bool(np.all(np.abs(mass[:,8])<1e-9)))
 actual['passed']=code==0 and actual.get('finite_nonnegative',False) and len(traces)==0 and abs(actual.get('ledger_error_m3',1))<1e-6 and actual.get('no_outflow',False) and abs(actual.get('end_s',0)-60)<1e-9 and actual['inputs_unchanged']
 (case/'result.json').write_text(json.dumps(actual,indent=2));result['actual_startup']=actual
else:result['actual_startup']={'skipped':'Stationary/source gates failed; reject before another actual startup.'}
result['production_adopted']=False
(r/'candidate-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
