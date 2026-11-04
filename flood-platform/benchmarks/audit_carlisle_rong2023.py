"""Inspect author-supplied Carlisle inputs and run a bounded internal smoke case.

The smoke case is not a field validation or a reproduction of the paper.
Never change time origins, boundary types, source locations or observations here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
from audit_carlisle_public_inputs import read_bdy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf8")


def read_asc(path):
    with Path(path).open() as f:
        header = dict(f.readline().split() for _ in range(6))
    h = {k.lower(): float(v) for k, v in header.items()}
    z = np.loadtxt(path, skiprows=6)
    if z.shape != (int(h['nrows']), int(h['ncols'])):
        raise ValueError('raster shape does not match header')
    if not np.isfinite(z).all() or np.any(z == h['nodata_value']):
        raise ValueError('nonfinite/nodata cells require an explicit policy')
    return h, z


def audit(source, old_contract, out):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    curves = read_bdy(source/'carlisle.bdy')
    files = ['CarlislefixDEMwithbed.asc', 'CarlislefixDEMwithbedDG2.asc',
             'bed.asc', 'width.asc']
    rasters = {f: read_asc(source/f) for f in files}
    h = rasters[files[0]][0]
    checks = {'shared_grid_headers': all(v[0] == h for v in rasters.values()),
              'expected_10m_476x306': h['cellsize'] == 10 and h['ncols'] == 476 and h['nrows'] == 306,
              'raw_file_equals_unprocessed_dem': sha(source/'CarlislefixDEMwithbedDG2.asc.raw') == sha(source/files[0]),
              'all_four_curves_274x900s': all(len(c['time_s']) == 274 and np.array_equal(c['time_s'], np.arange(274)*900.) for c in curves.values()),
              'three_nonnegative_inflow_curves': all(np.all(curves[n]['q_as_stored'] >= 0) for n in ['upstream1','upstream2','upstream3'])}
    if set(curves) != {'upstream1','upstream2','upstream3','downstream'}:
        raise ValueError('unexpected boundary names')
    raw = rasters[files[0]][1]; dg = rasters[files[1]][1]
    bed = rasters['bed.asc'][1]; width = rasters['width.asc'][1]
    channel = width > 0
    checks['bed_width_positive_masks_match'] = bool(np.array_equal(bed > 0, channel))
    old = json.loads(Path(old_contract).read_text(encoding='utf8'))
    comparisons = []
    for name in ['upstream1','upstream2','upstream3']:
        o = next(c for c in old['hydrographs'] if c['name'] == name)
        oq = np.asarray(o['unit_width_values_m2_s'])*o['point_count']*o['original_grid_dx_m']
        c = curves[name]; newq = c['q_as_stored']*h['cellsize']
        if not np.array_equal(c['time_s'], o['time_since_model_start_s']):
            raise ValueError('cannot compare different stored time axes')
        bad = np.flatnonzero(np.abs(newq-oq) > 1e-5)
        comparisons.append(dict(name=name, semantics='DG2 one point: Q=q*10m; not SGC semantics',
            identical_time_axis=True, same_values=int(len(oq)-len(bad)), different_values=int(len(bad)),
            peak_m3_s=float(newq.max()), peak_relative_s=float(c['time_s'][newq.argmax()]),
            differences=[dict(row=int(i), time_s=float(c['time_s'][i]), old_m3_s=float(oq[i]),
                              archive_m3_s=float(newq[i])) for i in bad]))
    sources=[]; other=[]
    for line in (source/'carlisle.bci').read_text().splitlines():
        f=line.split()
        if not f: continue
        if f[0]!='P': other.append(line.strip()); continue
        if len(f)!=5 or f[3]!='QVAR': raise ValueError('unsupported point record')
        x,y=map(float,f[1:3]); dx=h['cellsize']
        col=int(np.floor((x-h['xllcorner'])/dx))
        row=int(np.floor((h['yllcorner']+h['nrows']*dx-y)/dx))
        if not (0<=row<dg.shape[0] and 0<=col<dg.shape[1]): raise ValueError('point outside raster')
        sources.append(dict(name=f[4], x=x,y=y,row=row,col=col, raw_bed_m=float(raw[row,col]),
                            dg2_bed_m=float(dg[row,col]), channel_bed_m=float(bed[row,col]),
                            channel_width_m=float(width[row,col])))
    checks['three_distinct_source_cells']=len(sources)==3 and len({(s['row'],s['col']) for s in sources})==3
    checks['three_sources_on_channel_mask']=all(s['channel_width_m']>0 for s in sources)
    checks['finite_slope_arrays']=all(np.isfinite(read_asc(source/('CarlislefixDEMwithbedDG2.asc'+s))[1]).all() for s in ['1x','1y'])
    result=dict(evidence_class='author-archived model inputs, not new observations',
        source_doi='10.5281/zenodo.7064320', paper_doi='10.5194/gmd-16-3291-2023',
        grid=h, checks=checks, all_input_checks_pass=all(checks.values()), sources=sources,
        boundary_records=other, channel_cells=int(channel.sum()),
        dg2_minus_raw_m=dict(min=float((dg-raw).min()),max=float((dg-raw).max()),
                            channel_median=float(np.median((dg-raw)[channel]))),
        inflow_comparison_with_T07=comparisons,
        unresolved=[
            'Paper section 3.5 states free outlet; archive uses W HVAR downstream. Neither is silently substituted.',
            'Absolute start UTC and downstream datum/derivation are not established by the archive.',
            'Initial inflow samples differ from T07; no copy/shift/rescale to improve agreement.',
            'Archived sim_time is 164000 s whereas input series end at 245700 s; it is not a complete recession run.',
            'Archived fpfric=.06 has no separate river friction map; not the optimized two-roughness configuration in the paper.',
            '18 stage locations have no accompanying observational time series in this archive.'
        ], field_validation_ready=False, qualitative_field_passes=0,
        original_hashes={f.name:sha(f) for f in source.iterdir() if f.is_file()})
    write_json(out/'archive-input-audit.json',result)
    np.savez_compressed(out/'archive-terrain-audit.npz',raw=raw,dg2=dg,bed=bed,width=width)
    return result


def smoke(source, exe, destination):
    source, exe, case=Path(source),Path(exe),Path(destination)
    # Exclusive directory: reruns must use a separately registered case.
    case.mkdir(parents=True, exist_ok=False)
    for name in ['CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw',
                 'CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y',
                 'carlisle.bci','carlisle.bdy','carlisle.stage']:
        shutil.copyfile(source/name,case/name)
    protocol=dict(purpose='internal startup/CFL/source-volume smoke test only; not field validation',
        predeclared_duration_s=60, max_dt_s=1, output_interval_s=10, cpu_threads=1,
        original_data_unchanged=True, production_default_changed=False,
        initial_depth='dry default, archive provides no initial state',
        criteria=['return code zero','end time exactly 60 s','finite nonnegative final depth',
                  'no outlet exchange at startup when prescribed stage is below all west boundary beds',
                  'raster volume matches accepted-step left-end source quadrature within max(1e-5 m3, 1e-6*source_volume)',
                  'all copied source hashes unchanged after execution'],
        physical_caution='Source quadrature is first order in time; report trapezoidal input volume separately, without treating it as ledger loss.',
        exe_sha256=sha(exe), source_hashes={f.name:sha(f) for f in case.iterdir()})
    write_json(case/'protocol.json',protocol)
    (case/'case.par').write_text('DEMfile CarlislefixDEMwithbedDG2.asc\nbdyfile carlisle.bdy\n'
        'bcifile carlisle.bci\nstagefile carlisle.stage\ndirroot results\nresroot check\n'
        'dg2\nmint_hk\nmax_Froude 1\nfpfric .06\nsim_time 60\ninitial_tstep 1\n'
        'massint 10\nsaveint 60\noutput_precision 12\n',encoding='utf8')
    start=time.perf_counter()
    with (case/'run.log').open('w') as log:
        completed=subprocess.run([str(exe),'-v','case.par'], cwd=case,
            env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,timeout=300)
    elapsed=time.perf_counter()-start
    if completed.returncode:
        write_json(case/'result.json',dict(return_code=completed.returncode,elapsed_s=elapsed,passed=False))
        raise RuntimeError('smoke solver failed; see run.log; no automatic restart')
    steps=np.loadtxt(case/'accepted_timesteps.txt',ndmin=2)
    mass=np.loadtxt(case/'results/check.mass',skiprows=1,ndmin=2)
    h,z=read_asc(case/'results/check-0000.wd')
    curves=read_bdy(case/'carlisle.bdy');dx=h['cellsize']
    qleft=sum(np.interp(steps[:,0],curves[n]['time_s'],curves[n]['q_as_stored'])*dx
              for n in ['upstream1','upstream2','upstream3'])
    accepted_volume=float(np.dot(qleft,steps[:,1]))
    exact_linear_volume=sum(float(np.interp(60,curves[n]['time_s'],curves[n]['q_as_stored']))*dx*60/2
                            for n in ['upstream1','upstream2','upstream3'])
    raster_volume=float(z.sum()*dx*dx)
    end=float(steps[-1].sum()); tolerance=max(1e-5,1e-6*accepted_volume)
    checks=dict(end_at_60=abs(end-60)<1e-9, finite_nonnegative=bool(np.isfinite(z).all() and (z>=0).all()),
        accepted_volume_conserved=abs(raster_volume-accepted_volume)<=tolerance,
        no_startup_outflow=bool(np.all(np.abs(mass[:,8])<1e-9)),
        input_hashes_unchanged=all(sha(case/n)==v for n,v in protocol['source_hashes'].items()))
    result=dict(purpose=protocol['purpose'],return_code=0, elapsed_s=elapsed,steps=len(steps),
        end_s=end,dt_min_s=float(steps[:,1].min()),dt_max_s=float(steps[:,1].max()),
        accepted_source_volume_m3=accepted_volume,continuous_linear_source_volume_m3=exact_linear_volume,
        temporal_source_quadrature_error_m3=accepted_volume-exact_linear_volume,
        raster_volume_m3=raster_volume,ledger_residual_m3=raster_volume-accepted_volume,
        mass_file_max_abs_interval_error_m3=float(np.abs(mass[:,10]).max()),
        max_depth_m=float(z.max()),wet_cells=int((z>0).sum()),criteria=checks,passed=all(checks.values()))
    write_json(case/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('old_contract');p.add_argument('out')
    p.add_argument('--smoke-exe');a=p.parse_args()
    r=audit(a.source,a.old_contract,a.out)
    print(json.dumps({'checks':r['checks'],'unresolved':r['unresolved']},ensure_ascii=False),flush=True)
    if not r['all_input_checks_pass']: raise SystemExit('input checks failed; solver not started')
    if a.smoke_exe: print(json.dumps(smoke(a.source,a.smoke_exe,Path(a.out)/'smoke60'),ensure_ascii=False),flush=True)
