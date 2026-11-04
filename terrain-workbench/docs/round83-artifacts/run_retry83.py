from pathlib import Path
import json,shutil,subprocess,time,os,hashlib
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round83'
assert json.loads((R/'controls.json').read_text())['all_gates_passed']
C=R/'carlisle_2400';C.mkdir(exist_ok=False)
base=O/'round81/archive_conditional'
names=['case.par','carlisle.bci','carlisle.bdy','carlisle.stage','CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw','CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y']
for n in names:shutil.copy2(base/n,C/n)
par=(C/'case.par').read_text().replace('sim_time 164000','sim_time 2400').replace('saveint 4100','saveint 300')
(C/'case.par').write_text(par)
def save(obj): (R/'status.json').write_text(json.dumps(obj,indent=2))
exe=R/'build-rk-retry-cpu64/Release/lisflood.exe';st=time.perf_counter()
with (C/'run.log').open('w') as log:
    p=subprocess.Popen([str(exe),'-v','case.par'],cwd=C,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
    save({'state':'running','pid':p.pid,'wrapper_pid':os.getpid(),'target_s':2400,'wall_limit_s':900,'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()})
    try:code=p.wait(timeout=900)
    except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=15);code='resource_limit_900s'
rec={'state':'completed' if code==0 else 'failed','exit_code':code,'wall_s':time.perf_counter()-st,'target_s':2400,'field_passes_added':0,'production_adopted':False}
steps=np.loadtxt(C/'accepted_timesteps.txt',ndmin=2) if (C/'accepted_timesteps.txt').stat().st_size else np.empty((0,2))
if len(steps):rec.update(accepted_steps=len(steps),end_s=float(steps[-1].sum()),minimum_dt=float(steps[:,1].min()))
rej=C/'rejected_rk_attempts.txt'
if rej.exists() and rej.stat().st_size:
    rr=np.loadtxt(rej,ndmin=2);rec.update(rejections=len(rr),first_rejection_s=float(rr[0,0]),max_attempt=int(rr[:,2].max()),first_rejection_stage=int(rr[0,3]))
else:rec['rejections']=0
rec['snapshots']=[]
for f in sorted((C/'results').glob('check-*.wd')):
    h=np.loadtxt(f,skiprows=6);row={'file':f.name,'finite':bool(np.isfinite(h).all()),'nonnegative':bool((h>=0).all()),'min':float(h.min()),'max':float(h.max()),'volume_m3':float(h.sum()*100)}
    row['momentum_finite']=all(np.isfinite(np.loadtxt(f.with_suffix(ext),skiprows=6)).all() for ext in ['.Qx','.Qy'])
    rec['snapshots'].append(row)
if code==0:
    mass=np.loadtxt(C/'results/check.mass',skiprows=1,ndmin=2)
    vol=np.loadtxt(C/'accepted_source_volumes.txt',ndmin=2)
    b=np.loadtxt(C/'accepted_boundary_exchange.txt',ndmin=2)
    rec.update(source_step_times_equal=bool(np.array_equal(vol[:,:2],steps)),boundary_step_times_equal=bool(np.array_equal(b[:,:2],steps)),mass_max_interval_error=float(np.abs(mass[:,10]).max()),source_total_m3=float(vol[:,2].sum()))
    boundary_net=float((.5*(b[:,2]-b[:,3]+b[:,4]-b[:,5])*b[:,1]).sum())
    rec['boundary_net_m3']=boundary_net
    rec['independent_balance_error_m3']=rec['snapshots'][-1]['volume_m3']-rec['source_total_m3']-boundary_net
    rec['internal_gate_pass']=abs(rec['end_s']-2400)<1e-9 and len(rec['snapshots'])==8 and all(x['finite'] and x['nonnegative'] and x['momentum_finite'] for x in rec['snapshots']) and rec['source_step_times_equal'] and rec['boundary_step_times_equal'] and abs(rec['independent_balance_error_m3'])<1e-5
save(rec);print(json.dumps(rec),flush=True)
