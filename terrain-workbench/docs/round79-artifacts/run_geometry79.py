from pathlib import Path
import sys,os,json,hashlib,shutil,subprocess,time
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs'); R=O/'round79'
sys.path.insert(0,str(O/'round78'))
from prepare_meanpreserving_bed78 import slopes
EX=O/'round77/build-rk-ledger-cpu64/Release/lisflood.exe'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def arr(p):return np.loadtxt(p,skiprows=6)
def write(p,a,header):np.savetxt(p,a,fmt='%.17g',header=header,comments='')
def run(c):
 t=time.perf_counter()
 with (c/'run.log').open('w') as f:
  try:rc=subprocess.run([str(EX),'-v','case.par'],cwd=c,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
  except subprocess.TimeoutExpired:rc='timeout180'
 return dict(exit_code=rc,wall_s=time.perf_counter()-t)
if __name__=='__main__':
 R.mkdir(exist_ok=False)
 dump(R/'protocol.json',dict(single_factor='whole-domain raw-value-preserving bed representation from round78; same round77 solver and physical inputs',exe_sha256=sha(EX),static_cases=['wet sloped','partial step','plane shore'],static_tolerances=dict(depth_drift_m=1e-10,volume_m3=1e-8),dynamic_tolerances=dict(source_volume_m3=1e-6,ledger_m3=1e-6,negative_cells=0,end_s=60),tolerance_basis='Existing round75/76 double-precision internal checks, not physical accuracy or qualitative thresholds. No post-result relaxation.',decision='Fail any static gate: no actual run. Fail dynamic: retain and stop; no rerun. Static and short hydrodynamic success cannot establish real flood match.',cost='3 tiny equilibrium cases then paired60s; each process timeout180s, serial OMP1',field_acceptance='not scored, epoch/outlet/initial provenance unresolved'))
 result={'static':{},'actual':{},'field_passes_added':0,'production_adopted':False}
 for name in ['lake_candidate_wet_sloped','lake_candidate_partial','plane_shore_candidate']:
  base=O/'round75'/name;c=R/name;c.mkdir();shutil.copy2(base/'case.par',c/'case.par')
  z=arr(base/'bed.asc');initial=arr(base/'water.asc');sx,sy=slopes(z)
  # Preserve the analytic free surface from the existing equilibrium; rederive h slopes from NEW bed.
  hx=np.where(initial>0,-sx,0.);hy=np.where(initial>0,-sy,0.)
  head='\n'.join((base/'bed.asc').read_text().splitlines()[:6])
  for n,a in [('bed.asc',z),('bed.asc1x',sx),('bed.asc1y',sy),('water.asc',initial),('water.asc1x',hx),('water.asc1y',hy)]:write(c/n,a,head)
  rec=run(c);rec['passed']=False
  if rec['exit_code']==0:
   h=arr(c/'results/check-0000.wd');rec.update(finite_nonnegative=bool(np.isfinite(h).all() and (h>=0).all()),max_drift_m=float(abs(h-initial).max()),volume_error_m3=float(h.sum()-initial.sum()),negative_trace=(c/'negative-depth.jsonl').exists())
   rec['passed']=rec['finite_nonnegative'] and rec['max_drift_m']<1e-10 and abs(rec['volume_error_m3'])<1e-8 and not rec['negative_trace']
  result['static'][name]=rec;dump(c/'verification.json',rec);print(name,json.dumps(rec),flush=True)
 result['static_pass']=all(v['passed'] for v in result['static'].values());dump(R/'verification.json',result)
 if result['static_pass']:
  base=O/'round76/dtmax1';candidate=O/'round78/mean-preserving-geometry'
  names=['case.par','carlisle.bci','carlisle.bdy','carlisle.stage','CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw','CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y']
  for label in ['archive','mean_preserving']:
   c=R/label;c.mkdir()
   for n in names:shutil.copy2(base/n,c/n)
   if label=='mean_preserving':
    for a,b in [('bed.asc','CarlislefixDEMwithbedDG2.asc'),('bed.asc1x','CarlislefixDEMwithbedDG2.asc1x'),('bed.asc1y','CarlislefixDEMwithbedDG2.asc1y')]:shutil.copy2(candidate/a,c/b)
   dump(c/'inputs.json',{n:sha(c/n) for n in names});rec=run(c);rec['passed']=False
   if rec['exit_code']==0:
    h=arr(c/'results/check-0000.wd');t=np.loadtxt(c/'accepted_timesteps.txt',ndmin=2);s=np.loadtxt(c/'accepted_source_volumes.txt',ndmin=2);b=np.loadtxt(c/'accepted_boundary_exchange.txt',ndmin=2)
    assert np.array_equal(t,b[:,:2]) and np.array_equal(t,s[:,:2])
    # accepted boundary columns verified against header; point terms already in source integral.
    net=float(np.sum(b[:,1]*.5*(b[:,2]-b[:,3]+b[:,4]-b[:,5])))
    volume=float(h.sum()*100);source=float(s[:,2].sum());rec.update(steps=len(t),end_s=float(t[-1].sum()),finite_nonnegative=bool(np.isfinite(h).all() and (h>=0).all()),min_depth_m=float(h.min()),max_depth_m=float(h.max()),negative_trace=(c/'negative-depth.jsonl').exists(),source_m3=source,boundary_net_m3=net,volume_m3=volume,ledger_error_m3=volume-source-net,source_error_m3=source-111.0936,wet_cells_numerical=int((h>1e-6).sum()),boundary_header=(c/'accepted_boundary_exchange.txt').read_text().splitlines()[0])
    rec['passed']=rec['finite_nonnegative'] and not rec['negative_trace'] and abs(rec['end_s']-60)<1e-9 and abs(rec['ledger_error_m3'])<1e-6 and abs(rec['source_error_m3'])<1e-6
   result['actual'][label]=rec;dump(c/'verification.json',rec);dump(R/'verification.json',result);print(label,json.dumps(rec),flush=True)
   if not rec['passed']:break
  if len(result['actual'])==2 and all(v['passed'] for v in result['actual'].values()):
   a=arr(R/'archive/results/check-0000.wd');b=arr(R/'mean_preserving/results/check-0000.wd')
   result['comparison']=dict(max_abs_depth_difference_m=float(abs(a-b).max()),archive_only_numerically_wet=int(((a>1e-6)&(b<=1e-6)).sum()),candidate_only_numerically_wet=int(((b>1e-6)&(a<=1e-6)).sum()),threshold_purpose='1e-6 numerical diagnostic only, not field wet/dry or safety threshold',baseline_equal_round76=bool(np.array_equal(a,arr(base/'results/check-0000.wd'))))
 dump(R/'verification.json',result);print(json.dumps(result),flush=True)
