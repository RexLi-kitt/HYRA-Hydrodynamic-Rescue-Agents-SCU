from pathlib import Path
import os,sys,json,time,shutil,subprocess,hashlib
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round81';C=R/'archive_conditional'
EX=O/'round77/build-rk-ledger-cpu64/Release/lisflood.exe'
sys.path.insert(0,str(O/'round73'))
from audit_carlisle_public_inputs import read_bdy
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def integrate(c,end):
 t=np.array(c['time_s']);q=np.array(c['q_as_stored']);knots=np.r_[0,t[(t>0)&(t<end)],end];v=np.interp(knots,t,q)
 return float(np.sum((v[:-1]+v[1:])*.5*np.diff(knots)))
def latest_step():
 p=C/'accepted_timesteps.txt'
 if not p.exists():return None
 with p.open('rb') as f:
  f.seek(0,2);n=f.tell();f.seek(max(0,n-4096));lines=f.read().decode('ascii',errors='ignore').splitlines()
 for line in reversed(lines[:-1]):
  try:a,b=map(float,line.split());return a+b
  except ValueError:pass
 return None
if __name__=='__main__':
 R.mkdir(exist_ok=False);C.mkdir()
 base=O/'round79/archive'
 for p in base.iterdir():
  if p.is_file() and p.name in json.loads((base/'inputs.json').read_text()):shutil.copy2(p,C/p.name)
 text=(C/'case.par').read_text();text=text.replace('sim_time 60\n','sim_time 164000\n').replace('massint 10\n','massint 300\n').replace('saveint 60\n','saveint 4100\n')+'qoutput\n'
 assert 'sim_time 164000' in text and 'initial_tstep 1' in text
 (C/'case.par').write_text(text)
 curves=read_bdy(C/'carlisle.bdy')
 volumes={k:10*integrate(curves[k],164000) for k in ['upstream1','upstream2','upstream3']}
 protocol={'scope':'Conditional author-archive baseline, complete registered0..164000s; not complete recession to245700s or a validated field reproduction','evidence_class':'same-event archived input simulation, uncertain metadata retained','physical_choices':'Unchanged archived DG2 bed, HVAR west, three original unit-width sources, n=.06, dry start implied by archive. No selection by score.','numerical_choices':'round77 patched CPU64 OMP1; dt cap1 as successful round79 (archive par10), output mass300s/snapshots4100s, qoutput enabled. Exact output events affect step partition, so record accepted steps.','exe_sha256':sha(EX),'inputs':{p.name:sha(p) for p in C.iterdir() if p.is_file()},'expected_source_volume_m3':volumes,'internal_gate':'Finite/nonnegative means in all saved snapshots; no negative RK trace; end164000; exact integrated source; independently reconstruct volume from accepted RK boundary fluxes. Volume tolerance max(1e-4m3,1e-9 times gross accepted volume); not field accuracy.','unresolved':['epoch','HVAR vertical datum vs paper free outlet','initial-state realism','164000 truncates boundary series before245700','actual channel width support'],'observations':{'all_outline_ids':['1494','20431','25550','26602','4078483'],'all_marks':263,'outside_or_partial':'retained unknown/uncovered, not excluded or dry','maximum_extent':'compare only model registered-window maximum; no claim event maximum equivalence without timing coverage','qualitative_acceptance':'not eligible while input correspondence unresolved; no retrospective threshold fitting','reconstruction':'cell means only; raw unbounded depth polynomials not used'},'cost':'one heavy process; disk132GB free at launch; anticipated hours uncertain from wet-area-dependent CFL, record progress.8h resource cap, no automatic retry.','candidate_queue':'none; do not auto-run mean-preserving candidate until baseline assessed','production_adopted':False,'field_passes_added':0}
 dump(R/'protocol.json',protocol)
 dump(R/'curve-window-diagnostics.json',{k:{'initial':float(c['q_as_stored'][0]),'at164000':float(np.interp(164000,c['time_s'],c['q_as_stored'])),'final':float(c['q_as_stored'][-1]),'peak_time_s':float(c['time_s'][int(np.argmax(c['q_as_stored']))])} for k,c in curves.items()})
 start=time.time()
 with (C/'run.log').open('w') as f:
  p=subprocess.Popen([str(EX),'-v','case.par'],cwd=C,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
  print(json.dumps({'started_pid':p.pid,'wrapper_pid':os.getpid(),'case':str(C)}),flush=True)
  while True:
   code=p.poll();status={'state':'running' if code is None else 'exited','pid':p.pid,'wrapper_pid':os.getpid(),'exit_code':code,'elapsed_wall_s':time.time()-start,'last_buffered_accepted_end_s':latest_step(),'simulation_target_s':164000,'no_auto_restart':True}
   dump(R/'status.json',status)
   if code is not None:break
   if time.time()-start>8*3600:
    p.terminate();p.wait();status.update(state='resource_cap_stopped',exit_code=p.returncode);dump(R/'status.json',status);break
   time.sleep(15)
 if code==0:
  st=np.loadtxt(C/'accepted_timesteps.txt',ndmin=2);s=np.loadtxt(C/'accepted_source_volumes.txt',ndmin=2);b=np.loadtxt(C/'accepted_boundary_exchange.txt',ndmin=2)
  assert np.array_equal(st,s[:,:2]) and np.array_equal(st,b[:,:2])
  net=float(np.sum(b[:,1]*.5*(b[:,2]-b[:,3]+b[:,4]-b[:,5])));src=float(s[:,2].sum());gross=src+float(np.sum(b[:,1]*.5*np.sum(abs(b[:,2:6]),axis=1)));tol=max(1e-4,1e-9*gross)
  fields={};last=None
  for fn in sorted((C/'results').glob('check-*.wd')):
   h=np.loadtxt(fn,skiprows=6);last=h;fields[fn.name]={'finite':bool(np.isfinite(h).all()),'negative_cells':int((h<0).sum()),'volume_m3':float(h.sum()*100)}
  badq=[]
  for fn in (C/'results').glob('check-*.Q*'):
   if not np.isfinite(np.loadtxt(fn,skiprows=6)).all():badq.append(fn.name)
  end=float(st[-1].sum());err=float(last.sum()*100-src-net)
  rec={'end_s':end,'steps':len(st),'source_m3':src,'expected_source_m3':sum(volumes.values()),'boundary_net_m3':net,'ledger_error_m3':err,'volume_tolerance_m3':tol,'snapshots':fields,'nonfinite_momentum_files':badq,'negative_trace':(C/'negative-depth.jsonl').exists(),'field_passes_added':0}
  rec['internal_pass']=abs(end-164000)<1e-7 and len(fields)==40 and all(x['finite'] and x['negative_cells']==0 for x in fields.values()) and not badq and not rec['negative_trace'] and abs(err)<=tol and abs(src-sum(volumes.values()))<=tol
  dump(R/'completion.json',rec);print(json.dumps(rec),flush=True)
 else:print(json.dumps(status),flush=True)
