from pathlib import Path
import json,hashlib,shutil,subprocess,os,time,numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round88';R.mkdir(exist_ok=False);B=O/'round86/cfl0165';C=R/'carlisle_2400_cfl0165'
assert json.loads((O/'round87/regressions.json').read_text())['all_passed']; C.mkdir(exist_ok=False)
exe=O/'round86/build-isotropic-cpu64/Release/lisflood.exe';sha=hashlib.sha256(exe.read_bytes()).hexdigest();assert sha=='884a074e36474ae86949579525af295a0e9550c24d2ca138cd2612dcd4e7a771'
protocol={'baseline':'round87 candidate2400CFL.33 plus round86/cfl0165 early600 prefix; old round83/85 pair retained','candidate_exe_sha256':sha,'single_change_from_round83':'same isolated friction code as round87; single paired parameter CFL.33->.165','extension':'600->2400s same output300; no restart supported, first600 rerun is necessary prefix and must match stored fields/accepted steps, not counted as new independent evidence','gate':'finite nonnegative all8 H, finite q, accepted time/source/boundary ledger and total water error<1e-5; early600 fields and accepted steps must match bitwise','comparison':'all8 frames of candidate .33/.165 pair versus old83/85 pair; fixed thresholds1e-6/.001/.01, all differences retained; no field accuracy or convergence theorem claim','cost':'one serial run,900s wall limit,no automatic restart','next':'Complete this paired sensitivity branch; no further CFL sweep. Use full-frame adverse outcomes to decide next mechanism/input evidence work; no automatic full event.'}
(R/'extension-protocol.json').write_text(json.dumps(protocol,indent=2))
names=['case.par','carlisle.bci','carlisle.bdy','carlisle.stage','CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw','CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y']
for n in names:shutil.copy2(B/n,C/n)
p=C/'case.par';s=p.read_text();assert 'sim_time 600\n' in s;p.write_text(s.replace('sim_time 600\n','sim_time 2400\n'))
(R/'extension-hashes.json').write_text(json.dumps({n:{'baseline600_sha256':hashlib.sha256((B/n).read_bytes()).hexdigest(),'candidate2400_sha256':hashlib.sha256((C/n).read_bytes()).hexdigest()} for n in names},indent=2))
start=time.perf_counter()
with (C/'run.log').open('w') as log:
    p=subprocess.Popen([str(exe),'-v','case.par'],cwd=C,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
    (R/'status.json').write_text(json.dumps({'state':'running','pid':p.pid,'wrapper_pid':os.getpid(),'target_s':2400,'wall_limit_s':900}))
    try:rc=p.wait(timeout=900)
    except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=15);rc='timeout900'
r={'state':'completed' if rc==0 else 'failed','exit_code':rc,'wall_s':time.perf_counter()-start,'exe_sha256':sha,'production_adopted':False,'field_passes_added':0}
steps=np.loadtxt(C/'accepted_timesteps.txt',ndmin=2);r.update(accepted_steps=len(steps),end_s=float(steps[-1].sum()),median_dt=float(np.median(steps[:,1])),minimum_dt=float(steps[:,1].min()))
rej=C/'rejected_rk_attempts.txt';r['rejections']=len(rej.read_text().splitlines()) if rej.exists() else 0
frames=[]
for f in sorted((C/'results').glob('check-*.wd')):
    h=np.loadtxt(f,skiprows=6);frames.append({'file':f.name,'finite_nonnegative':bool(np.isfinite(h).all() and (h>=0).all()),'volume_m3':float(h.sum()*100),'momentum_finite':all(np.isfinite(np.loadtxt(f.with_suffix(e),skiprows=6)).all() for e in ['.Qx','.Qy'])})
r['frames']=frames
if rc==0:
    v=np.loadtxt(C/'accepted_source_volumes.txt',ndmin=2);b=np.loadtxt(C/'accepted_boundary_exchange.txt',ndmin=2)
    net=float((.5*(b[:,2]-b[:,3]+b[:,4]-b[:,5])*b[:,1]).sum());sv=float(v[:,2].sum());r.update(source_m3=sv,boundary_net_m3=net,balance_error_m3=frames[-1]['volume_m3']-sv-net,source_times_equal=bool(np.array_equal(v[:,:2],steps)),boundary_times_equal=bool(np.array_equal(b[:,:2],steps)))
    early=np.loadtxt(B/'accepted_timesteps.txt',ndmin=2);r['early_steps_bitwise_equal']=bool(np.array_equal(early,steps[:len(early)]))
    r['early_all_fields_byte_equal']=all((C/'results'/f.name).read_bytes()==f.read_bytes() for f in (B/'results').glob('check-000[01].*') if f.suffix not in ['.mass','.stage'])
    r['internal_gate_pass']=abs(r['end_s']-2400)<1e-9 and len(frames)==8 and all(x['finite_nonnegative'] and x['momentum_finite'] for x in frames) and abs(r['balance_error_m3'])<1e-5 and abs(sv-108345.16)<1e-6 and r['source_times_equal'] and r['boundary_times_equal'] and r['early_steps_bitwise_equal'] and r['early_all_fields_byte_equal']
(R/'status.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

