from pathlib import Path
import json,hashlib,os,subprocess,time,shutil
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round90';P=O/'round89/original';R.mkdir(exist_ok=False)
E=O/'round86/build-isotropic-cpu64/Release/lisflood.exe'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol={'case':'Kabir5m corrected identity Run7.bdy=2005 Run1.csv','purpose':'original acceleration+SGC-enable startup and input mass audit; not field validation','duration_s':60,'caps_s':[2,1],'wall_timeout_each_s':180,'OMP_NUM_THREADS':1,'unchanged':['DEM','23 point locations','original three hydrographs','n .055','FREE .0006','drycheckoff','acceleration','sgc_enable','dry initial state'],'numerical_gate':'finite nonnegative saved H; mass residual and prescribed inlet quadrature discrepancy explained; failures retained, no automatic retry','expected_source':'independent CSV piecewise linear total; left-endpoint quadrature error bounded by max slope*T*cap/2; roundoff tolerance 1e-5 m3','scope':'No DG2 friction/RK retry applied in this backend; no restart claim, fast checkpoint is commented out in source','exe_sha256':sha(E)}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2))
status={'state':'running','runs':[]};(R/'status.json').write_text(json.dumps(status))
for cap in [2,1]:
 C=R/f'cap{cap}';C.mkdir()
 for name in ['Carlisle_5m.asc','carlisle.bci','carlisle_run7.bdy']:shutil.copy2(P/name,C/name)
 lines=(P/'carlisle_run1.par').read_text().splitlines();out=[]
 replace={'resroot':'check','dirroot':'results','sim_time':'60','initial_tstep':str(cap),'massint':'0.000000001','saveint':'30','bdyfile':'carlisle_run7.bdy'}
 for line in lines:
  tok=line.split()
  out.append((tok[0]+' '+replace[tok[0]]) if tok and tok[0] in replace else line)
 out.extend(['output_precision 12'])
 (C/'case.par').write_text('\n'.join(out)+'\n')
 (C/'hashes.json').write_text(json.dumps({p.name:sha(p) for p in C.iterdir() if p.is_file()},indent=2))
 env=os.environ.copy();env['OMP_NUM_THREADS']='1';start=time.perf_counter()
 with (C/'run.log').open('w') as f:
  try:res=subprocess.run([str(E),'-v','case.par'],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=res.returncode
  except subprocess.TimeoutExpired:code='timeout'
 row={'cap_s':cap,'exit_code':code,'wall_s':time.perf_counter()-start,'frames':[]}
 for p in sorted((C/'results').glob('*.wd')):
  a=np.loadtxt(p,skiprows=6);row['frames'].append({'name':p.name,'finite':bool(np.isfinite(a).all()),'minimum_h':float(a.min()),'volume_m3':float(a.sum()*25)})
 row['finite_nonnegative_frames']=bool(row['frames']) and all(x['finite'] and x['minimum_h']>=0 for x in row['frames'])
 status['runs'].append(row);(R/'status.json').write_text(json.dumps(status,indent=2))
 if code!=0 or not row['finite_nonnegative_frames']:break
status['state']='completed' if len(status['runs'])==2 and all(x['exit_code']==0 and x['finite_nonnegative_frames'] for x in status['runs']) else 'failed'
(R/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
