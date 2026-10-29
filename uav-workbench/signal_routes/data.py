"""New parent-disjoint SWE events; existing generator and packages stay frozen."""
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
from .common import ROOT, read, write, digest, registration

TERRAIN = ROOT.parent / 'terrain-workbench/terrain-lab'


def generate_one(root, seed, split, mechanism):
    sys.path.insert(0,str(TERRAIN))
    from flood_solver import FloodSolver
    path = Path(root)/f'{split}-{seed}-{mechanism}.zip'
    # A recipe sidecar is the completion marker. Interrupted ZIPs are not accepted.
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        row = read(sidecar)
        if not path.exists() or digest(path) != row['sha256']: raise ValueError('Changed event package')
        return row
    if path.exists(): raise FileExistsError(f'Incomplete package requires inspection: {path}')
    rng = np.random.default_rng(seed); z,x = np.indices((20,20)); bed = np.zeros((20,20))
    bed += rng.uniform(-.001,.001)*x*5+rng.uniform(-.001,.001)*z*5
    for _ in range(5):
        r,c = rng.uniform(0,20,2)
        bed += rng.uniform(-.12,.18)*np.exp(-((z-r)**2+(x-c)**2)/rng.uniform(10,60))
    bed -= bed.min(); h = np.maximum(0,.06-bed)
    r,c = map(int,rng.integers(3,17,2)); h += .12*np.exp(-((z-r)**2+(x-c)**2)/5)
    q = float(rng.uniform(1.2,3.))
    graph = [[t,ratio*q] for t,ratio in registration()['hydrographs'][mechanism]]
    solver = FloodSolver(bed,h,5.,boundary='closed',max_timestep=.5)
    solver.set_source_schedule([dict(cells=[[r,c]],hydrograph=graph)])
    frames=[]; before=[]; after=[]; dts=[]; rates=[]
    tmp = path.with_suffix('.zip.tmp')
    def npbytes(a):
        f=io.BytesIO(); np.save(f,np.asarray(a),allow_pickle=False); return f.getvalue()
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for t in range(0,901,5):
            solver.advance(float(t)); u,v=solver.velocity(); name=f'frame-{t:04d}.bin'
            fields=np.stack((solver.h,u,v,solver.maximum))
            if not np.isfinite(fields).all() or np.min(fields[0]) < -1e-9: raise ArithmeticError('Invalid SWE state')
            archive.writestr(name,fields.astype('<f4').tobytes()); frames.append(dict(time_s=float(t),url=name))
            before.append(np.stack((solver.h,solver.mx,solver.mz)).copy())
            dt=solver.timestep(.5); rate=solver._scheduled_rate(solver.time)
            solver.step(dt); after.append(np.stack((solver.h,solver.mx,solver.mz)).copy())
            dts.append(dt); rates.append(rate)
        process=dict(grid=dict(shape=[20,20],dx_m=5.,axis_order=['z','x']),frames=frames,
                     frame_fields=['depth','u','v','maximum_depth'],frame_dtype='<f4')
        archive.writestr('process.json',json.dumps(process))
        for name,array in [('bed',bed),('probe_before',before),('probe_after',after),('probe_dt',dts),('probe_source',rates)]:
            archive.writestr(name+'.npy',npbytes(array))
        manifest=dict(schema='hyra-signal-routes-event-v1',seed=seed,split=split,
            parent_terrain=f'signal-parent-{seed}',mechanism=mechanism,source_schedule=graph,
            source_cells=[[r,c]],duration_s=900.,grid=process['grid'],frame_count=len(frames),
            numerics=solver.numerical_config(),solver_sha256=digest(TERRAIN/'flood_solver.py'),
            generator_sha256=digest(__file__),scope='synthetic SWE, not calibrated real flood')
        archive.writestr('case.json',json.dumps(manifest))
    tmp.replace(path)
    row=dict(path=str(path.resolve()),sha256=digest(path),**manifest); write(sidecar,row)
    return row


def prepare(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    frozen=registration(); rows=[]
    existing=output/'dataset.json'
    if existing.exists():
        manifest=read(existing)
        expected={(split,seed,mechanism) for split in ('fit','development','holdout')
                  for seed in frozen['split'][split]
                  for mechanism in (('slow','rapid','double') if split=='holdout' else ('slow','rapid'))}
        actual={(r['split'],r['seed'],r['mechanism']) for r in manifest['cases']}
        if actual!=expected or len(manifest['cases'])!=42: raise ValueError('Changed parent/event split')
        for row in manifest['cases']:
            if digest(row['path'])!=row['sha256']: raise ValueError('Changed reused hydraulic event')
        write(output/'reuse-audit.json',dict(passed=True,events=42,registration_schema=frozen['schema'],
              reused_existing_physical_inputs=True,confirmation_scores_opened=False))
        return existing
    for split in ('fit','development','holdout'):
        mechanisms=['slow','rapid','double'] if split=='holdout' else ['slow','rapid']
        for seed in frozen['split'][split]:
            for mechanism in mechanisms:
                row=generate_one(output,seed,split,mechanism); rows.append(row)
                write(output/'generation-progress.json',dict(status='running',completed=len(rows),total=42,
                     last=f'{split}-{seed}-{mechanism}',holdout_scores_accessed=False))
                print(f'generated {split}-{seed}-{mechanism} ({len(rows)}/42)',flush=True)
    manifest=dict(schema='hyra-signal-routes-dataset-v1',cases=rows,split_unit='parent_terrain',
                  registration=frozen,holdout_scores_accessed=False)
    write(output/'dataset.json',manifest)
    write(output/'generation-progress.json',dict(status='completed',completed=42,total=42,holdout_scores_accessed=False))
    return output/'dataset.json'


def cases(dataset, split):
    from scale100_env import FloodCase
    for row in read(dataset)['cases']:
        if row['split']!=split: continue
        if digest(row['path'])!=row['sha256']: raise ValueError('Dataset digest mismatch')
        yield row,FloodCase(row['path'])


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output)
