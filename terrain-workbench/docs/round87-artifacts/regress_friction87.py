from pathlib import Path
import json,hashlib,shutil,subprocess,os,time
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round87';R.mkdir(exist_ok=False)
exes={'baseline':O/'round83/build-rk-retry-cpu64/Release/lisflood.exe','candidate':O/'round86/build-isotropic-cpu64/Release/lisflood.exe'}
protocol={'purpose':'Nonzero roughness coupled regressions before extending isolated exact-friction candidate. No production edits or observation tuning.',
 'lake':'Reuse round79 analytic wet slope/partial step/plane shore fixtures, only n0->.06, same3s/dtmax.02; baseline and candidate. h drift<1e-10m, volume<1e-8m3, finite nonnegative and finite q.',
 'uniform_slope':'80x8 cells dx1m, z=-.001*x, H=1m, qx=H^(5/3)*sqrt(.001)/.06, qy0, DG bed slope=-.001/(2sqrt3), H and q slopes0. Closed walls;2s, interior rows2:6 cols30:50 before fastest characteristic reaches from x walls (~7.4m vs30m). All boundary cells retained but not mistaken for uniform infinite-domain solution.',
 'slope_gate':'Internal analytic regression: finite nonnegative and volume error<1e-8m3; core velocity drift <=2*g*S*dt, halfdt error<=.8 full error unless <1e-12; candidate no worse than baseline by >5%+1e-12. This is a splitting/truncation diagnostic, not field tolerance.',
 'timesteps':[.02,.01],'gravity':9.80655,'n':.06,'cost':'10 tiny serial runs<=30s each; if any gate fails do not start2400 extension; preserve failures','field_passes_added':0}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2))
def run(exe,c,initial):
    start=time.perf_counter()
    with (c/'run.log').open('w') as f:p=subprocess.run([str(exe),'-v','case.par'],cwd=c,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,timeout=30)
    r={'exit':p.returncode,'wall_s':time.perf_counter()-start,'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
    if p.returncode==0:
        h=np.loadtxt(c/'results/check-0000.wd',skiprows=6);qx=np.loadtxt(c/'results/check-0000.Qx',skiprows=6);qy=np.loadtxt(c/'results/check-0000.Qy',skiprows=6)
        steps=np.loadtxt(c/'accepted_timesteps.txt',ndmin=2)
        r.update(finite_nonnegative=bool(np.isfinite(h).all() and (h>=0).all()),momentum_finite=bool(np.isfinite(qx).all() and np.isfinite(qy).all()),volume_error_m3=float(h.sum()-initial.sum()),max_depth_drift_m=float(abs(h-initial).max()),max_momentum_m2_s=float(np.hypot(qx,qy).max()),end_s=float(steps[-1].sum()),steps=len(steps))
    return r
result={'lakes':{},'slope':{},'field_passes_added':0}
for label,exe in exes.items():
    result['lakes'][label]={}
    for name in ['lake_candidate_wet_sloped','lake_candidate_partial','plane_shore_candidate']:
        base=O/'round79'/name;c=R/label/name;c.mkdir(parents=True)
        for f in base.glob('*.asc*'):shutil.copy2(f,c/f.name)
        text=(base/'case.par').read_text();assert 'fpfric 0\n' in text
        (c/'case.par').write_text(text.replace('fpfric 0\n','fpfric .06\n')+'qoutput\n')
        initial=np.loadtxt(c/'water.asc',skiprows=6);r=run(exe,c,initial)
        r['passed']=r['exit']==0 and r['finite_nonnegative'] and r['momentum_finite'] and r['max_depth_drift_m']<1e-10 and abs(r['volume_error_m3'])<1e-8
        result['lakes'][label][name]=r
    result['slope'][label]={}
    for dt in [.02,.01]:
        c=R/label/f'slope_dt{dt}';c.mkdir()
        header='ncols 80\nnrows 8\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n'
        z=np.tile(-.001*(np.arange(80)+.5),(8,1));h=np.ones_like(z);u=np.sqrt(.001)/.06
        fields={'bed.asc':z,'bed.asc1x':np.full_like(z,-.001/(2*np.sqrt(3))),'bed.asc1y':np.zeros_like(z),'water.asc':h,'water.asc.Qx':np.full_like(z,u),'water.asc.Qy':np.zeros_like(z)}
        for suffix in ['1x','1y','.Qx1x','.Qx1y','.Qy1x','.Qy1y']:fields['water.asc'+suffix]=np.zeros_like(z)
        for n,a in fields.items():np.savetxt(c/n,a,header=header,comments='',fmt='%.17g')
        (c/'case.par').write_text(f'DEMfile bed.asc\nstartfile water.asc\nstartq2d\ndirroot results\nresroot check\ndg2\nsim_time 2\ninitial_tstep {dt}\nmassint 2\nsaveint 2\nfpfric .06\noutput_precision 15\nqoutput\n')
        r=run(exe,c,h)
        if r['exit']==0:
            hh=np.loadtxt(c/'results/check-0000.wd',skiprows=6);qq=np.loadtxt(c/'results/check-0000.Qx',skiprows=6);vv=np.loadtxt(c/'results/check-0000.Qy',skiprows=6)
            core=(slice(2,6),slice(30,50));r['core_max_u_error_m_s']=float(abs(qq[core]/hh[core]-u).max());r['core_max_v_m_s']=float(abs(vv[core]/hh[core]).max());r['core_h_drift_m']=float(abs(hh[core]-1).max());r['expected_u_m_s']=float(u);r['declared_splitting_bound_m_s']=2*9.80655*.001*dt
            r['passed']=r['finite_nonnegative'] and r['momentum_finite'] and abs(r['volume_error_m3'])<1e-8 and r['core_max_u_error_m_s']<=r['declared_splitting_bound_m_s'] and r['core_max_v_m_s']<1e-10 and r['core_h_drift_m']<1e-10
        else:r['passed']=False
        result['slope'][label][str(dt)]=r
    (R/'regressions.json').write_text(json.dumps(result,indent=2));print(label,json.dumps(result['slope'][label]),flush=True)
checks={}
for label in exes:
    a=result['slope'][label]['0.02']['core_max_u_error_m_s'];b=result['slope'][label]['0.01']['core_max_u_error_m_s'];checks[label+'_dt_reduction']=b<=.8*a or max(a,b)<1e-12
for dt in ['0.02','0.01']:checks['candidate_no_worse_'+dt]=result['slope']['candidate'][dt]['core_max_u_error_m_s']<=1.05*result['slope']['baseline'][dt]['core_max_u_error_m_s']+1e-12
result['cross_checks']=checks
result['all_passed']=all(checks.values()) and all(v['passed'] for cat in ['lakes','slope'] for sub in result[cat].values() for v in sub.values())
(R/'regressions.json').write_text(json.dumps(result,indent=2));print('ALL_PASSED',result['all_passed'])
