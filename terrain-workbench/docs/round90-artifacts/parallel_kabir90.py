from pathlib import Path
import os,json,shutil,time,subprocess,struct,hashlib
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round90';B=R/'cost7200';C=R/'cost7200_omp4';C.mkdir(exist_ok=False)
assert json.loads((R/'cost-analysis.json').read_text())['left_rule_match_within_time_precision']
for n in ['Carlisle_5m.asc','carlisle.bci','carlisle_run7.bdy','case.par']:shutil.copy2(B/n,C/n)
protocol={'single_changed_variable':'OMP_NUM_THREADS 1 to4','case':'same7200s source/geometry/solver/outputs','wall_limit_s':240,'predeclared_equivalence':'all saved h maxabs <=1e-9 m; all time/step/iteration text columns equal; finite nonnegative; cumulative mass discrepancy <=1e-4m3','adoption':'retain 4 threads for this isolated reference path only if equivalence passes and measured wall runtime is lower; no production changes','GPU_not_used':True}
(R/'parallel-protocol.json').write_text(json.dumps(protocol,indent=2));(R/'parallel-status.json').write_text('{"state":"running"}')
env=os.environ.copy();env['OMP_NUM_THREADS']='4';E=O/'round86/build-isotropic-cpu64/Release/lisflood.exe';st=time.perf_counter()
with (C/'run.log').open('w') as f:
 try:p=subprocess.run([str(E),'-v','case.par'],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240);code=p.returncode
 except subprocess.TimeoutExpired:code='timeout'
r={'state':'completed' if code==0 else 'failed','exit_code':code,'wall_s':time.perf_counter()-st,'frames':[]}
def read(p):
 with p.open('rb') as f:
  nc,nr=struct.unpack('<ii',f.read(8));hdr=struct.unpack('<dddd',f.read(32));a=np.fromfile(f,dtype='<f8')
 assert a.size==nc*nr
 return a
if code==0:
 for p in sorted((B/'results').glob('*.wdb')):
  a=read(p);b=read(C/'results'/p.name);r['frames'].append({'name':p.name,'max_abs_h_difference_m':float(abs(a-b).max()),'volume_difference_m3':float((b-a).sum()*25),'finite_nonnegative':bool(np.isfinite(b).all() and b.min()>=0)})
 m=np.loadtxt(C/'results/check.mass',skiprows=1);base=np.loadtxt(B/'results/check.mass',skiprows=1)
 r['step_columns_equal']=bool(m.shape==base.shape and np.array_equal(m[:,:4],base[:,:4]));r['sum_abs_mass_residual_m3']=float(abs(m[:,10]).sum())
 r['equivalence']=r['step_columns_equal'] and r['sum_abs_mass_residual_m3']<=1e-4 and len(r['frames'])==5 and all(x['finite_nonnegative'] and x['max_abs_h_difference_m']<=1e-9 for x in r['frames'])
basewall=json.loads((R/'cost-status.json').read_text())['wall_s'];r['speedup_vs_same_case']=basewall/r['wall_s'];r['use_four_for_reference']=bool(r.get('equivalence') and r['wall_s']<basewall)
(R/'parallel-status.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
