from pathlib import Path
import json,hashlib,shutil,difflib,subprocess,os,time,importlib.util
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round86';S=Path(__file__).parent
source=R/'lisflood8-isotropic-friction'; base=O/'round83/lisflood8-rk-retry'
protocol={'single_change':'dg2::friction local frozen-depth ODE, replace componentwise diagonal denominators by shared exact vector damping; retain caller, thresholds, n=.06, source, flux, boundaries, RK retry and production default.',
 'formula':'dq/dt=-g*n^2*|q|*q/h^(7/3); fixed h,n gives q1=q0/(1+dt*g*n^2*|q0|/h^(7/3)). Scope is local friction substep only, not exact coupled SWE.',
 'operator_gate':'compiled function agrees independent analytic vector formula; preserves direction <=1e-10 degree; semigroup relative1e-12 on fixed H active cells, zero n identity, dissipative finite.',
 'coupled_gate':'3 unchanged n=0 controls agree available fields and accepted steps; two predeclared 600s Carlisle runs cfl .33/.165 dtmax1, all inputs and output times300/600 frozen; finite nonnegative fields/source and water balance.',
 'paired_sensitivity':'compare both CFLs at300 and600, all3 thresholds as before, whole-domain depth L1/max, full masks. If sensitivity grows or failures occur do not promote candidate or hide the adverse result. Decreased sensitivity alone is not field validation.',
 'cost':'build<=240s per command, each600s run<=300s wall; serial, no restart; no whole2400 repeat','field_passes_added':0,'production_adopted':False}
(R/'candidate-protocol.json').write_text(json.dumps(protocol,indent=2))
shutil.copytree(base,source,ignore=shutil.ignore_patterns('.git','build','__pycache__'))
p=source/'swe/dg2/friction.cpp';old=p.read_text()
start=old.index('\tNUMERIC_TYPE Sfx =');new=old[:start]+'''    // Exact isotropic drag for this fixed-depth, fixed-roughness substep only.
    const NUMERIC_TYPE damping = C(1.0) /
        (C(1.0) + Solverptr->Tstep * Cf * speed / H);
    HU *= damping;
    HV *= damping;
}
'''
p.write_text(new)
(R/'isotropic-friction.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/swe/dg2/friction.cpp',tofile='b/swe/dg2/friction.cpp')))
changed=[]
for f in base.rglob('*'):
    if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts:
        target=source/f.relative_to(base)
        if target.exists() and hashlib.sha256(f.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():changed.append(str(f.relative_to(base)))
assert changed==[str(Path('swe/dg2/friction.cpp'))],changed
cmake=r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
build=R/'build-isotropic-cpu64'
t0=time.perf_counter()
with (R/'candidate-build.log').open('w') as log:
    for cmd in [[cmake,'-S',str(source),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64','-D_CONFIG='+str(O/'round65/cpu64.cmake'),'-DCMAKE_CUDA_COMPILER=NOTFOUND'],[cmake,'--build',str(build),'--config','Release','--target','lisflood','--parallel','1']]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
exe=build/'Release/lisflood.exe'
(R/'candidate-build.json').write_text(json.dumps({'wall_s':time.perf_counter()-t0,'only_changed_source':changed,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()},indent=2))
# Recompile small harness with candidate function, retaining original harness for audit.
probe=R/'candidate-probe';probe.mkdir()
raw=(R/'friction_probe.cpp').read_text();prefix=raw[:raw.index('void dg2::friction\n')];main=raw[raw.index('int main()'):]
(probe/'friction_probe.cpp').write_text(prefix+new[new.index('void dg2::friction\n'):]+main)
shutil.copy2(R/'CMakeLists.txt',probe/'CMakeLists.txt')
with (R/'candidate-probe-build.log').open('w') as log:
    for cmd in [[cmake,'-S',str(probe),'-B',str(probe/'build'),'-G','Visual Studio 17 2022','-A','x64'],[cmake,'--build',str(probe/'build'),'--config','Release','--parallel','1']]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
pex=probe/'build/Release/friction_probe.exe'
def friction(h,q,dt,n=.06):
    data=''.join(f'{a:.17g} {b:.17g} {c:.17g} {n:.17g} {dt:.17g}\n' for a,(b,c) in zip(h,q))
    result=subprocess.run([str(pex)],input=data,capture_output=True,text=True,check=True,timeout=30)
    return np.array([[float(v) for v in line.split()] for line in result.stdout.splitlines()])
d=np.load(R/'full-diagnostics.npz');diag=json.loads((R/'diagnosis.json').read_text());op=[]
for rec in diag['fixed_depth_friction_probes']:
    a=d[f"friction_t{rec['time_s']}"];h=a[:,2];q=a[:,3:5];dt=rec['fixed_depth_dt_s'];ans=friction(h,q,dt);exact=a[:,9:11]
    half=friction(h,q,dt/2);two=friction(h,half,dt/2)
    angle=np.abs(np.degrees(np.arctan2(q[:,0]*ans[:,1]-q[:,1]*ans[:,0],np.sum(q*ans,axis=1))))
    assert np.allclose(ans,exact,rtol=1e-12,atol=1e-14)
    assert np.allclose(ans,two,rtol=1e-12,atol=1e-14)
    assert np.max(angle)<1e-10 and np.array_equal(friction(h,q,dt,0),q)
    assert np.isfinite(ans).all() and np.all(np.linalg.norm(ans,axis=1)<=np.linalg.norm(q,axis=1)*(1+1e-14))
    op.append({'time_s':rec['time_s'],'cells':len(h),'max_rotation_deg':float(angle.max()),'max_exact_difference':float(abs(ans-exact).max()),'max_one_vs_two_difference':float(abs(ans-two).max()),'passed':True})
(R/'operator-verification.json').write_text(json.dumps(op,indent=2))
spec=importlib.util.spec_from_file_location('control77',O/'round77/prepare_round77.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
controls={}
for name in ['closed','drain_E','point_and_outlet']:
    c=R/'controls'/name;iv=m.make_case(c,name);rec=m.run_case(exe,c,iv);b=O/'round83/controls'/name
    rec['mean_bitwise_identical']=np.array_equal(np.loadtxt(c/'results/check-0000.wd',skiprows=6),np.loadtxt(b/'results/check-0000.wd',skiprows=6))
    rec['steps_bitwise_identical']=(c/'accepted_timesteps.txt').read_bytes()==(b/'accepted_timesteps.txt').read_bytes()
    rec['passed']=rec['exit_code']==0 and rec['finite_nonnegative'] and abs(rec['total_ledger_residual_m3'])<1e-8 and rec['mean_bitwise_identical'] and rec['steps_bitwise_identical'];controls[name]=rec
(R/'controls.json').write_text(json.dumps(controls,indent=2));assert all(x['passed'] for x in controls.values())
names=['case.par','carlisle.bci','carlisle.bdy','carlisle.stage','CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw','CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y']
runs={}
for tag,origin in [('cfl033',O/'round83/carlisle_2400'),('cfl0165',O/'round85/carlisle_2400_cfl_half')]:
    c=R/tag;c.mkdir()
    for name in names:shutil.copy2(origin/name,c/name)
    par=c/'case.par';text=par.read_text();assert 'sim_time 2400\n' in text;par.write_text(text.replace('sim_time 2400\n','sim_time 600\n'))
    (c/'input-hashes.json').write_text(json.dumps({n:hashlib.sha256((c/n).read_bytes()).hexdigest() for n in names},indent=2))
    t0=time.perf_counter()
    with (c/'run.log').open('w') as log:
        p=subprocess.Popen([str(exe),'-v','case.par'],cwd=c,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
        (R/'status.json').write_text(json.dumps({'state':'running','case':tag,'pid':p.pid,'wrapper_pid':os.getpid()}))
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=15);code='timeout300'
    assert code==0,code
    steps=np.loadtxt(c/'accepted_timesteps.txt',ndmin=2);sourcevol=np.loadtxt(c/'accepted_source_volumes.txt',ndmin=2);boundary=np.loadtxt(c/'accepted_boundary_exchange.txt',ndmin=2)
    frames=[]
    for idx in range(2):
        h=np.loadtxt(c/f'results/check-{idx:04d}.wd',skiprows=6)
        frames.append({'time_s':300*(idx+1),'finite_nonnegative':bool(np.isfinite(h).all() and (h>=0).all()),'volume_m3':float(h.sum()*100),'momentum_finite':all(np.isfinite(np.loadtxt(c/f'results/check-{idx:04d}.{q}',skiprows=6)).all() for q in ['Qx','Qy'])})
    net=float((.5*(boundary[:,2]-boundary[:,3]+boundary[:,4]-boundary[:,5])*boundary[:,1]).sum());sv=float(sourcevol[:,2].sum());balance=frames[-1]['volume_m3']-sv-net
    rec={'exit':code,'wall_s':time.perf_counter()-t0,'steps':len(steps),'median_dt':float(np.median(steps[:,1])),'end_s':float(steps[-1].sum()),'source_m3':sv,'boundary_m3':net,'balance_error_m3':balance,'source_equals_baseline600':abs(sv-11109.36)<1e-6,'source_times_equal':bool(np.array_equal(steps,sourcevol[:,:2])),'boundary_times_equal':bool(np.array_equal(steps,boundary[:,:2])),'frames':frames,'rejections':len((c/'rejected_rk_attempts.txt').read_text().splitlines()) if (c/'rejected_rk_attempts.txt').exists() else 0}
    rec['passed']=abs(rec['end_s']-600)<1e-8 and abs(balance)<1e-5 and rec['source_equals_baseline600'] and rec['source_times_equal'] and rec['boundary_times_equal'] and all(x['finite_nonnegative'] and x['momentum_finite'] for x in frames)
    runs[tag]=rec;(R/'runs.json').write_text(json.dumps(runs,indent=2));assert rec['passed']
    print(tag,json.dumps(rec),flush=True)
(R/'status.json').write_text(json.dumps({'state':'completed','all_internal_passed':True,'field_passes_added':0}))
