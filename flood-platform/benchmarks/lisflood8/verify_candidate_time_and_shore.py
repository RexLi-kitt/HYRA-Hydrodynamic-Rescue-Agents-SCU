"""Check unresolved timestep coupling and a sloped wet/dry equilibrium."""
from pathlib import Path
import importlib.util,json,os,shutil,subprocess,time
import numpy as np
out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');r=out/'round75'
protocol=dict(reason='Slope limiting alone is not a proof under the unchanged CFL; source integration uses actual dt. The earlier dry-step control had zero depth slopes, so also check a sloped wet/dry equilibrium.',
 cases=['Same archived 60s case with only initial_tstep 1 -> .5','Analytic stationary plane shore at a cell face: h=max(1-z,0), hu=hv=0, exact planar wet depth slopes, three seconds, baseline and candidate'],
 criteria=['Strict nonnegative final depth and no negative RK trace','Source ledger error <1e-6 m3 in actual case','Shore depth drift <1e-10 m and mass error <1e-8 m3'],
 interpretation='Different wet footprints without observation cannot establish realism. No time alignment or mean-depth clipping. No wider time-step sweep.')
(r/'additional-verification-protocol.json').write_text(json.dumps(protocol,indent=2))
spec=importlib.util.spec_from_file_location('audit',out/'round73/audit_carlisle_rong2023.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
exe=r/'build-reconstruction-cpu64/Release/lisflood.exe'
def run(exe,case):
 st=time.perf_counter()
 with (case/'run.log').open('w') as f:p=subprocess.run([str(exe),'-v','case.par'],cwd=case,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,timeout=180)
 return p.returncode,time.perf_counter()-st
case=r/'candidate_half_step';case.mkdir(exist_ok=False)
for p in (r/'candidate_case').iterdir():
 if p.is_file() and p.name in ['case.par','carlisle.bci','carlisle.bdy','carlisle.stage','CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw','CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y']:shutil.copy2(p,case/p.name)
p=case/'case.par';txt=p.read_text();assert txt.count('initial_tstep 1\n')==1;p.write_text(txt.replace('initial_tstep 1\n','initial_tstep .5\n'))
code,wall=run(exe,case);result={'half_step':dict(exit_code=code,wall_s=wall)}
if code==0:
 head,h=audit.read_asc(case/'results/check-0000.wd');steps=np.loadtxt(case/'accepted_timesteps.txt',ndmin=2);curves=audit.read_bdy(case/'carlisle.bdy');dx=head['cellsize']
 q=sum(np.interp(steps[:,0],curves[n]['time_s'],curves[n]['q_as_stored'])*dx for n in ['upstream1','upstream2','upstream3']);v=float(np.dot(q,steps[:,1]));err=float(h.sum()*dx*dx-v)
 old=np.loadtxt(r/'candidate_case/results/check-0000.wd',skiprows=6)
 result['half_step'].update(steps=len(steps),end_s=float(steps[-1].sum()),min_depth_m=float(h.min()),max_depth_m=float(h.max()),finite_nonnegative=bool(np.isfinite(h).all() and (h>=0).all()),accepted_source_volume_m3=v,source_quadrature_error_m3=v-111.0936,ledger_error_m3=err,negative_trace_present=(case/'negative-depth.jsonl').exists(),max_change_depth_m=float(np.max(np.abs(h-old))))
 result['half_step']['passed']=result['half_step']['finite_nonnegative'] and abs(err)<1e-6 and not result['half_step']['negative_trace_present'] and abs(result['half_step']['end_s']-60)<1e-9
(case/'result.json').write_text(json.dumps(result['half_step'],indent=2))
result['shore']={}
for label,ex in [('baseline',r/'build-trace-cpu64/Release/lisflood.exe'),('candidate',exe)]:
 c=r/('plane_shore_'+label);c.mkdir(exist_ok=False);ny,nx=8,12
 z=np.broadcast_to(1+(np.arange(nx)-5.5)*.1,(ny,nx)).copy();zx=np.full_like(z,.1/(2*np.sqrt(3)));zx[:,[0,-1]]=0;zero=np.zeros_like(z);h=np.maximum(1-z,0);hx=np.where(h>0,-zx,0)
 header='ncols 12\nnrows 8\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999'
 for n,a in [('bed.asc',z),('bed.asc1x',zx),('bed.asc1y',zero),('water.asc',h),('water.asc1x',hx),('water.asc1y',zero)]:np.savetxt(c/n,a,header=header,comments='',fmt='%.17g')
 (c/'case.par').write_text('DEMfile bed.asc\nstartfile water.asc\ndirroot results\nresroot check\ndg2\nsim_time 3\ninitial_tstep .02\nmassint 1\nsaveint 3\nfpfric 0\noutput_precision 15\n')
 code,wall=run(ex,c);rec=dict(exit_code=code,wall_s=wall)
 if code==0:
  a=np.loadtxt(c/'results/check-0000.wd',skiprows=6);rec.update(max_depth_drift_m=float(np.max(np.abs(a-h))),volume_error_m3=float(a.sum()-h.sum()),finite_nonnegative=bool(np.isfinite(a).all() and (a>=0).all()))
 rec['passed']=code==0 and rec['finite_nonnegative'] and rec['max_depth_drift_m']<1e-10 and abs(rec['volume_error_m3'])<1e-8
 result['shore'][label]=rec;(c/'result.json').write_text(json.dumps(rec,indent=2))
result['source_quadrature_caution']='Unchanged left-endpoint source integration: baseline stride60 error -0.43966149m3; candidate dtmax1 error -1.34801704m3. Larger accepted steps increase integration bias; this is not missing water in the ledger. Half-step is sensitivity, not calibration.'
(r/'additional-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
