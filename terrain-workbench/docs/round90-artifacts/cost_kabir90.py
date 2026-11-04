from pathlib import Path
import json,hashlib,shutil,os,subprocess,time,struct
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round90';C=R/'cost7200';C.mkdir(exist_ok=False)
assert json.loads((R/'startup-analysis.json').read_text())['gate']
E=O/'round86/build-isotropic-cpu64/Release/lisflood.exe'
for n in ['Carlisle_5m.asc','carlisle.bci','carlisle_run7.bdy']:shutil.copy2(R/'cap2'/n,C/n)
p=(R/'cap2/case.par').read_text().replace('sim_time 60','sim_time 7200').replace('saveint 30','saveint 1800')+'binary_out\n'
(C/'case.par').write_text(p)
protocol={'sim_time_s':7200,'wall_limit_s':240,'purpose':'single early-event full-grid cost and mass probe after startup gate; no repetition or automatic extension','same_physics_as':'cap2','changes':['duration','1800s snapshots','binary output to avoid hardcoded ASCII rounding'],'full_event_not_started':True,'stopping':'timeout/failure keep all outputs; no restart loop','expected_volume':'CSV independent quadrature plus left-point-rule bounds; no field scoring','input_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in C.iterdir() if x.is_file()}}
(R/'cost-protocol.json').write_text(json.dumps(protocol,indent=2));(R/'cost-status.json').write_text('{"state":"running"}')
env=os.environ.copy();env['OMP_NUM_THREADS']='1';st=time.perf_counter()
with (C/'run.log').open('w') as f:
 try:p=subprocess.run([str(E),'-v','case.par'],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240);code=p.returncode
 except subprocess.TimeoutExpired:code='timeout'
r={'state':'completed' if code==0 else 'failed','exit_code':code,'wall_s':time.perf_counter()-st,'frames':[],'field_passes_added':0}
for p in sorted((C/'results').glob('*.wdb')):
 with p.open('rb') as f:
  nc,nr=struct.unpack('<ii',f.read(8));head=struct.unpack('<dddd',f.read(32));h=np.fromfile(f,dtype='<f8')
 assert (nc,nr)==(951,611) and h.size==nc*nr and head[:3]==(338500.,554700.,5.)
 r['frames'].append({'name':p.name,'finite_nonnegative':bool(np.isfinite(h).all() and h.min()>=0),'volume_m3':float(h.sum()*25),'wet_cells_over_1mm':int((h>.001).sum()),'max_h_m':float(h.max())})
if (C/'results/check.mass').exists():
 m=np.loadtxt(C/'results/check.mass',skiprows=1);r.update(end_s=float(m[-1,0]),steps=int(m[-1,3]),median_step_s=float(np.median(m[:,1])),last_volume_reported_m3=float(m[-1,5]),sum_abs_mass_residual_m3=float(abs(m[:,10]).sum()),max_outflow_m3_s=float(m[:,8].max()))
(R/'cost-status.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
