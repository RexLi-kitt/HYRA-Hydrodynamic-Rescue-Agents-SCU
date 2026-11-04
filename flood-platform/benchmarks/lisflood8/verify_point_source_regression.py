from pathlib import Path
import hashlib,json,os,subprocess
import numpy as np

import argparse
p=argparse.ArgumentParser();p.add_argument('evidence_root');args=p.parse_args()
r=Path(args.evidence_root).resolve()
old=r.parent/'round65/reference-inlet-event-precision12'
exe=r/'build-point-cpu64/Release/lisflood.exe'
dest=r/'regression'; dest.mkdir(exist_ok=False)

def call(case, failure=None):
    with (case/'run.log').open('w') as log:
        p=subprocess.run([str(exe),'-v','case.par'],cwd=case,
                         env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,
                         stderr=subprocess.STDOUT,timeout=60)
    if failure:
        assert p.returncode and failure in (case/'run.log').read_text(),case
    else:
        assert p.returncode==0,case

boundary=dest/'unchanged_boundary'; boundary.mkdir()
for f in old.iterdir():
    if f.is_file() and f.name not in ('run.log','accepted_timesteps.txt'):
        (boundary/f.name).write_bytes(f.read_bytes())
call(boundary)
matches={}
for name in ('accepted_timesteps.txt','results/check.mass','results/check.velocity',
             'results/check.stage','results/check-0000.wd'):
    matches[name]=(boundary/name).read_bytes()==(old/name).read_bytes()
assert all(matches.values()),matches

def fixture(name,dx,q,maxdt,variable=False,reject=None):
    case=dest/name; case.mkdir()
    header=f'ncols 20\nnrows 20\nxllcorner 0\nyllcorner 0\ncellsize {dx}\nNODATA_value -9999\n'
    for f in ('flat.dem','flat.dem1x','flat.dem1y'):
        (case/f).write_text(header+(' '.join(['0']*20)+'\n')*20)
    boundary='QVAR pulse' if variable else f'QFIX {q}'
    if reject=='height': boundary='HFIX .1'
    (case/'input.bci').write_text(f'P {dx*10.5} {dx*10.5} {boundary}\n')
    (case/'input.bdy').write_text('# linear synthetic source\npulse\n2 seconds\n.01 0\n.03 .2\n')
    (case/'case.par').write_text('DEMfile flat.dem\nbcifile input.bci\n'+('bdyfile input.bdy\n' if variable else '')+
         f'dirroot results\nresroot check\ndg2\nsim_time .2\ninitial_tstep {maxdt}\nCFL .3\nmassint .05\nsaveint .2\nfpfric 0\nlimitslopes\noutput_precision 12\n')
    if reject:
        call(case,'Only nonnegative' if reject=='height' else 'Nonfinite/negative')
        return {'name':name,'explicitly_rejected':True}
    call(case)
    s=np.loadtxt(case/'accepted_timesteps.txt',ndmin=2)
    h=np.loadtxt(case/'results/check-0000.wd',skiprows=6)
    mass=np.loadtxt(case/'results/check.mass',skiprows=1,ndmin=2)
    assert np.isfinite(h).all() and h.min()>=0
    expected=float(np.sum((.01+.1*s[:,0] if variable else q)*dx*s[:,1]))
    volume=float(h.sum()*dx*dx)
    assert abs(volume-expected)<1e-7,(name,volume,expected)
    assert abs(mass[:,10]).max()<1e-7
    assert abs(s[-1].sum()-.2)<1e-12
    result={'name':name,'steps':len(s),'first_dt':float(s[0,1]),'volume_m3':volume,
            'accepted_left_quadrature_m3':expected,'volume_mismatch_m3':volume-expected,
            'max_abs_reported_mass_error_m3':float(abs(mass[:,10]).max()),'finite_nonnegative':True}
    if variable:
        result['analytic_integral_m3']=.02*dx*.2
        result['time_quadrature_error_m3']=volume-.02*dx*.2
    else:
        post_source_h=q*s[0,1]/dx
        cfl_bound=.3*dx/np.sqrt(9.80665*post_source_h)
        # Approximate g here only for a conservative independent sanity bound;
        # the executable's source/CFL check is authoritative for its exact g.
        assert s[0,1] <= cfl_bound and s[0,1]<maxdt
        result['first_step_source_added_cfl_bound_s']=float(cfl_bound)
    return result

results=[fixture('strong_dry_source',.1,1.,1.),
         fixture('linear_source_dt005',.1,0,.005,True),
         fixture('linear_source_dt0025',.1,0,.0025,True),
         fixture('negative_source_rejected',5,-.02,.005,reject='negative'),
         fixture('height_source_rejected',5,0,.005,reject='height')]
ratio=abs(results[2]['time_quadrature_error_m3']/results[1]['time_quadrature_error_m3'])
assert .45<ratio<.55,ratio
report={'no_source_bitwise_unchanged':matches,'additional_cases':results,
        'linear_source_half_dt_error_ratio':ratio,
        'scope':'internal source timing tests; first-order source quadrature retained; no field validation'}
(r/'regression.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report,indent=2))
