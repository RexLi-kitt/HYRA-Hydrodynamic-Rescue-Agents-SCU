from pathlib import Path
import importlib.util,json,os,shutil,subprocess,time
import numpy as np
out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');r=out/'round76'
exe=r/'build-integrated-cpu64/Release/lisflood.exe'
spec=importlib.util.spec_from_file_location('probe',out/'round73/probe_boundary_series.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

def run(case):
 st=time.perf_counter()
 with (case/'run.log').open('w') as f:
  try:p=subprocess.run([str(exe),'-v','case.par'],cwd=case,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,timeout=180);code=p.returncode
  except subprocess.TimeoutExpired:code='timeout_180s'
 return code,time.perf_counter()-st
result={}
control,h=probe.run_case(exe,r/'constant_source',multi=True)
old=np.loadtxt(out/'round75/candidate_square/results/check-0000.wd',skiprows=6)
control['field_identical_round75']=bool(h is not None and np.array_equal(h,old));result['constant_source']=control
result['stationary']={}
for name in ['lake_candidate_wet_sloped','lake_candidate_partial','plane_shore_candidate']:
 base=out/'round75'/name;case=r/name;case.mkdir(exist_ok=False)
 for n in ['case.par','bed.asc','bed.asc1x','bed.asc1y','water.asc','water.asc1x','water.asc1y']:shutil.copy2(base/n,case/n)
 code,wall=run(case);rec=dict(exit_code=code,wall_s=wall,passed=False)
 if code==0:
  initial=np.loadtxt(case/'water.asc',skiprows=6);h=np.loadtxt(case/'results/check-0000.wd',skiprows=6);old=np.loadtxt(base/'results/check-0000.wd',skiprows=6)
  rec.update(field_identical_round75=bool(np.array_equal(h,old)),max_drift_m=float(np.max(np.abs(h-initial))),volume_error_m3=float(h.sum()-initial.sum()),finite_nonnegative=bool(np.isfinite(h).all() and (h>=0).all()))
  rec['passed']=rec['field_identical_round75'] and rec['finite_nonnegative'] and rec['max_drift_m']<1e-10 and abs(rec['volume_error_m3'])<1e-8
 (case/'result.json').write_text(json.dumps(rec,indent=2));result['stationary'][name]=rec
result['small_gates_passed']=control.get('finite_nonnegative',False) and control.get('volume_pass',False) and control['field_identical_round75'] and all(x['passed'] for x in result['stationary'].values())
(r/'controls.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
result['actual']={}
if result['small_gates_passed']:
 for name,base_name in [('dtmax1','candidate_case'),('dtmax_half','candidate_half_step')]:
  case=r/name;case.mkdir(exist_ok=False);base=out/'round75'/base_name
  names=['case.par','carlisle.bci','carlisle.bdy','carlisle.stage','CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw','CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y']
  for n in names:shutil.copy2(base/n,case/n)
  code,wall=run(case);rec=dict(exit_code=code,wall_s=wall,passed=False,inputs_unchanged=all((base/n).read_bytes()==(case/n).read_bytes() for n in names))
  if code==0:
   h=np.loadtxt(case/'results/check-0000.wd',skiprows=6);steps=np.loadtxt(case/'accepted_timesteps.txt',ndmin=2);v=np.loadtxt(case/'accepted_source_volumes.txt',ndmin=2);mass=np.loadtxt(case/'results/check.mass',skiprows=1,ndmin=2)
   old=np.loadtxt(base/'results/check-0000.wd',skiprows=6);total=float(v[:,2].sum());raster=float(h.sum()*100)
   # Original all-three hydrographs are linear on 0..60, derived in round73.
   expected=111.0936
   rec.update(steps=len(steps),source_records=len(v),source_t_dt_exact=bool(np.array_equal(steps,v[:,:2])),end_s=float(steps[-1].sum()),min_depth_m=float(h.min()),max_depth_m=float(h.max()),negative_cells=int((h<0).sum()),finite_nonnegative=bool(np.isfinite(h).all() and (h>=0).all()),negative_trace_present=(case/'negative-depth.jsonl').exists(),prescribed_integral_m3=expected,accepted_volume_m3=total,raster_volume_m3=raster,source_integral_error_m3=total-expected,ledger_error_m3=raster-total,no_outflow=bool(np.all(np.abs(mass[:,8])<1e-9)),mass_max_interval_error=float(np.max(np.abs(mass[:,10]))),max_retry_attempts=int(v[:,3].max()),retry_steps=int((v[:,3]>1).sum()),max_depth_change_from_old_m=float(np.max(np.abs(h-old))))
   rec['passed']=rec['inputs_unchanged'] and rec['source_t_dt_exact'] and abs(rec['end_s']-60)<1e-9 and rec['finite_nonnegative'] and not rec['negative_trace_present'] and abs(rec['source_integral_error_m3'])<1e-6 and abs(rec['ledger_error_m3'])<1e-6 and rec['no_outflow']
  (case/'result.json').write_text(json.dumps(rec,indent=2));result['actual'][name]=rec;print(json.dumps({name:rec}),flush=True)
  if not rec['passed']:break
 if len(result['actual'])==2 and all(x['passed'] for x in result['actual'].values()):
  h1=np.loadtxt(r/'dtmax1/results/check-0000.wd',skiprows=6);h2=np.loadtxt(r/'dtmax_half/results/check-0000.wd',skiprows=6)
  result['time_sensitivity']=dict(max_depth_change_m=float(np.max(np.abs(h1-h2))),volume_difference_m3=result['actual']['dtmax1']['accepted_volume_m3']-result['actual']['dtmax_half']['accepted_volume_m3'],note='Prescribed input volume invariant, hydrodynamic fields need not be time-step invariant.')
else:result['skip']='Small controls failed, no actual startup launched.'
result['field_qualitative_passes_added']=0;result['production_adopted']=False
(r/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
