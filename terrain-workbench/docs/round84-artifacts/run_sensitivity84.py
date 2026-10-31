from pathlib import Path
import json,shutil,subprocess,os,time,hashlib
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round84';B=O/'round83/carlisle_2400'
R.mkdir(exist_ok=False);C=R/'carlisle_2400_dt_half';C.mkdir()
exe=O/'round83/build-rk-retry-cpu64/Release/lisflood.exe'
assert json.loads((O/'round83/independent-verification.json').read_text())['independent_gate_pass']
protocol={'date':'2026-10-30','baseline':'round83 same2400s/300soutput DG2 RK retry',
 'single_change':'initial_tstep (dt maximum) 1 to0.5 seconds; unchanged executable, inputs, all observation coordinates and geometry',
 'predeclared_times_s':list(range(300,2401,300)),
 'internal_checks':['exit0 end2400 finite nonnegative all8 H fields and finite momentum','source integral and accepted interval equality','independent total volume balance <1e-5m3'],
 'internal_sensitivity_only':{'depth_thresholds_m':[1e-6,1e-3,1e-2],'metrics':['all-domain L1/max differences and volume','bothwet/baselineonly/candidateonly/neither per frame and threshold','actual accepted dt distribution and rejected attempts','first saved wet-frame differences; no subframe arrival claims']},
 'not_accuracy_acceptance':'No observation-supported tolerance for these short-time fields; do not choose a timestep by agreement with score, claim convergence from two steps, or call software agreement field accuracy.',
 'cost':'oneCPU task wallcap900s; no restart; unchanged limited positive-Q/no-rain/no-restart scope',
 'next':'If numerical gate fails preserve and diagnose; if both pass quantify sensitivity before choosing full-window run, no invented real-world pass.'}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2))
names=['case.par','carlisle.bci','carlisle.bdy','carlisle.stage','CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw','CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y']
for name in names:shutil.copy2(B/name,C/name)
p=C/'case.par';s=p.read_text();assert s.count('initial_tstep 1\n')==1;p.write_text(s.replace('initial_tstep 1\n','initial_tstep .5\n'))
config={name:{'baseline_sha256':hashlib.sha256((B/name).read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((C/name).read_bytes()).hexdigest()} for name in names}
assert all(v['baseline_sha256']==v['candidate_sha256'] for k,v in config.items() if k!='case.par')
(R/'configuration-hashes.json').write_text(json.dumps(config,indent=2))
def save(o):(R/'status.json').write_text(json.dumps(o,indent=2))
start=time.perf_counter()
with (C/'run.log').open('w') as log:
    p=subprocess.Popen([str(exe),'-v','case.par'],cwd=C,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
    save({'state':'running','pid':p.pid,'wrapper_pid':os.getpid(),'target_s':2400,'wall_limit_s':900,'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()})
    try:code=p.wait(timeout=900)
    except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=15);code='resource_limit_900s'
rec={'state':'completed' if code==0 else 'failed','exit_code':code,'wall_s':time.perf_counter()-start,'field_passes_added':0,'production_adopted':False}
steps=np.loadtxt(C/'accepted_timesteps.txt',ndmin=2)
rec.update(accepted_steps=len(steps),end_s=float(steps[-1].sum()),minimum_dt=float(steps[:,1].min()),median_dt=float(np.median(steps[:,1])),max_dt=float(steps[:,1].max()))
rej=C/'rejected_rk_attempts.txt'
if rej.exists():
    x=np.loadtxt(rej,ndmin=2);rec.update(rejections=len(x),stage_counts={str(i):int((x[:,3]==i).sum()) for i in [1,2]},first_rejection_s=float(x[0,0]),max_attempt=int(x[:,2].max()))
else:rec['rejections']=0
rec['snapshots']=[]
for f in sorted((C/'results').glob('check-*.wd')):
    h=np.loadtxt(f,skiprows=6)
    rec['snapshots'].append({'file':f.name,'finite_nonnegative':bool(np.isfinite(h).all() and (h>=0).all()),'volume_m3':float(h.sum()*100),'momentum_finite':all(np.isfinite(np.loadtxt(f.with_suffix(ext),skiprows=6)).all() for ext in ['.Qx','.Qy'])})
if code==0:
    v=np.loadtxt(C/'accepted_source_volumes.txt',ndmin=2);b=np.loadtxt(C/'accepted_boundary_exchange.txt',ndmin=2);m=np.loadtxt(C/'results/check.mass',skiprows=1,ndmin=2)
    boundary=float((.5*(b[:,2]-b[:,3]+b[:,4]-b[:,5])*b[:,1]).sum())
    rec.update(source_m3=float(v[:,2].sum()),boundary_net_m3=boundary,source_times_equal=bool(np.array_equal(v[:,:2],steps)),boundary_times_equal=bool(np.array_equal(b[:,:2],steps)),mass_max_interval_error=float(np.abs(m[:,10]).max()))
    rec['independent_balance_error_m3']=rec['snapshots'][-1]['volume_m3']-rec['source_m3']-boundary
    rec['internal_gate_pass']=abs(rec['end_s']-2400)<1e-9 and len(rec['snapshots'])==8 and all(v['finite_nonnegative'] and v['momentum_finite'] for v in rec['snapshots']) and rec['source_times_equal'] and rec['boundary_times_equal'] and abs(rec['independent_balance_error_m3'])<1e-5
save(rec);print(json.dumps(rec),flush=True)
