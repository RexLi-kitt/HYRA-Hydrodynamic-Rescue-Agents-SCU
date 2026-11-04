"""Integration regression for TimeSeries pointer lifetime, using physical volumes."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
import numpy as np


def run_case(exe,directory,multi=False,reverse=False,unused=False):
    case=Path(directory);case.mkdir(parents=True,exist_ok=False)
    header='ncols 20\nnrows 20\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value -9999\n'
    for name in ['flat.dem','flat.dem1x','flat.dem1y']:
        (case/name).write_text(header+(' '.join(['0']*20)+'\n')*20)
    names=['a','b','c'] if multi else ['a'];qs=dict(a=.01,b=.02,c=.03)
    points=''.join(f'P {52.5+5*i} 52.5 QVAR {n}\n' for i,n in enumerate(names))
    if multi: points+='W 0 100 HVAR level\n'
    (case/'input.bci').write_text(points)
    curves=[f'{n}\n2 seconds\n{qs[n]} 0\n{qs[n]} 1\n' for n in names]
    if multi: curves.append('level\n2 seconds\n0 0\n0 1\n')
    if reverse:curves=curves[::-1]
    if unused:curves.insert(1,'unused\n2 seconds\n99 0\n99 1\n')
    (case/'input.bdy').write_text('# synthetic lifetime regression\n'+''.join(curves))
    (case/'case.par').write_text('DEMfile flat.dem\nbcifile input.bci\nbdyfile input.bdy\n'
        'dirroot results\nresroot check\ndg2\nsim_time .2\ninitial_tstep .005\nCFL .3\n'
        'massint .05\nsaveint .2\nfpfric 0\nlimitslopes\noutput_precision 12\n')
    start=time.perf_counter()
    with (case/'run.log').open('w') as log:
        proc=subprocess.run([str(exe),'-v','case.par'],cwd=case,env=dict(os.environ,OMP_NUM_THREADS='1'),
                            stdout=log,stderr=subprocess.STDOUT,timeout=30)
    result=dict(name=case.name,exit_code=proc.returncode,elapsed_s=time.perf_counter()-start,
        exe_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),expected_m3=sum(qs[n] for n in names)*5*.2)
    z=None
    if proc.returncode==0:
        z=np.loadtxt(case/'results/check-0000.wd',skiprows=6)
        result.update(finite_nonnegative=bool(np.isfinite(z).all() and (z>=0).all()),volume_m3=float(z.sum()*25))
        result['volume_pass']=abs(result['volume_m3']-result['expected_m3'])<1e-8
    (case/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result,z


def run(old_exe,new_exe,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    specs=[('old_single',old_exe,False,False,False),('old_multiple',old_exe,True,False,False),
           ('new_single',new_exe,False,False,False),('new_multiple',new_exe,True,False,False),
           ('new_reordered_unused',new_exe,True,True,True)]
    records={};arrays={}
    for name,exe,multi,reverse,unused in specs:
        records[name],arrays[name]=run_case(exe,out/name,multi,reverse,unused)
    checks=dict(old_multiple_failure_reproduced=records['old_multiple']['exit_code']!=0,
        new_cases_finite_conserved=all(records[n].get('finite_nonnegative',False) and records[n].get('volume_pass',False)
                                     for n in ['new_single','new_multiple','new_reordered_unused']),
        single_series_field_unchanged=arrays['old_single'] is not None and np.array_equal(arrays['old_single'],arrays['new_single']),
        multi_series_order_and_unused_invariant=arrays['new_multiple'] is not None and np.array_equal(arrays['new_multiple'],arrays['new_reordered_unused']))
    result=dict(cases=records,checks=checks,passed=all(checks.values()),purpose='software correctness, no field-accuracy claim')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('old_exe');p.add_argument('new_exe');p.add_argument('out')
    a=p.parse_args();run(a.old_exe,a.new_exe,a.out)
