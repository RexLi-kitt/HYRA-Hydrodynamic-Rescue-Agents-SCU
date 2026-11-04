from pathlib import Path
import importlib.util,json,shutil,subprocess,os,time
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round83'
spec=importlib.util.spec_from_file_location('old77',O/'round77/prepare_round77.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
exe=R/'build-rk-retry-cpu64/Release/lisflood.exe'
def run(exe,c,iv):
    rec=m.run_case(exe,c,iv)
    rej=c/'rejected_rk_attempts.txt'
    rec['rejections']=len(rej.read_text().splitlines()) if rej.exists() else 0
    return rec
res={}
for name in ['closed','drain_E','point_and_outlet']:
    c=R/'controls'/name;iv=m.make_case(c,name);rec=run(exe,c,iv)
    if rec['exit_code']==0:
        b=O/'round77/candidate'/name
        rec['mean_bitwise_identical']=np.array_equal(np.loadtxt(c/'results/check-0000.wd',skiprows=6),np.loadtxt(b/'results/check-0000.wd',skiprows=6))
        rec['steps_bitwise_identical']=(c/'accepted_timesteps.txt').read_bytes()==(b/'accepted_timesteps.txt').read_bytes()
        rec['passed']=rec['finite_nonnegative'] and abs(rec['total_ledger_residual_m3'])<1e-8 and rec['mean_bitwise_identical'] and rec['steps_bitwise_identical'] and rec['rejections']==0
    else:rec['passed']=False
    res[name]=rec
    print(name,json.dumps(rec),flush=True)
def synthetic(c):
    c.mkdir(parents=True,exist_ok=False)
    header='ncols 5\nnrows 5\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n'
    for name in ['bed.asc','bed.asc1x','bed.asc1y','water.asc','water.asc1x','water.asc1y','water.asc.Qx','water.asc.Qy','water.asc.Qx1x','water.asc.Qx1y','water.asc.Qy1x','water.asc.Qy1y']:
        a=np.zeros((5,5))
        if name=='water.asc': a[:]=.01
        if name=='water.asc.Qx1x':a[2,2]=1
        (c/name).write_text(header+'\n'.join(' '.join(map(str,row)) for row in a)+'\n')
    (c/'input.bci').write_text('P 1.5 1.5 QVAR source\n')
    (c/'input.bdy').write_text('# break-point source integration during RK retries\nsource\n4 seconds\n.01 0\n.03 .05\n.01 .1\n.01 .2\n')
    (c/'case.par').write_text('DEMfile bed.asc\nstartfile water.asc\nstartq2d\nbcifile input.bci\nbdyfile input.bdy\ndirroot results\nresroot check\ndg2\nsim_time .2\ninitial_tstep .2\nmassint .2\nsaveint .2\nfpfric 0\noutput_precision 15\nqoutput\n')
for label,x in [('baseline',O/'round77/build-rk-ledger-cpu64/Release/lisflood.exe'),('candidate',exe)]:
    c=R/('retry_source_'+label);synthetic(c);rec=run(x,c,.25)
    if rec['exit_code']==0:
        source=np.loadtxt(c/'accepted_source_volumes.txt',ndmin=2);steps=np.loadtxt(c/'accepted_timesteps.txt',ndmin=2)
        rec['source_integral_m3']=float(source[:,2].sum())
        rec['source_matches_steps']=bool(np.array_equal(source[:,:2],steps))
        rec['source_error_m3']=rec['source_integral_m3']-.003
        rec['volume_error_m3']=rec['final_volume_m3']-.253
        rec['passed']=rec['rejections']>0 and rec['finite_nonnegative'] and rec['source_matches_steps'] and abs(rec['source_error_m3'])<1e-10 and abs(rec['volume_error_m3'])<1e-9
    res['retry_source_'+label]=rec
    print(label,json.dumps(rec),flush=True)
res['all_gates_passed']=all(res[n]['passed'] for n in ['closed','drain_E','point_and_outlet','retry_source_candidate']) and res['retry_source_baseline']['exit_code']==42
(R/'controls.json').write_text(json.dumps(res,indent=2))
if not res['all_gates_passed']:raise SystemExit(1)
