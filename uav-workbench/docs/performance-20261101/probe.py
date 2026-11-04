"""Bounded performance audit on existing TRAIN data, never production training."""
import cProfile
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pstats
import statistics
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
SINGLE = ROOT / 'uav-singles1024-20261101'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def profile_rows(prof):
    rows = []
    for (file, line, name), (primitive, calls, own, cumulative, callers) in pstats.Stats(prof).stats.items():
        rows.append(dict(file=file, line=line, function=name, calls=calls, own_s=own, cumulative_s=cumulative))
    return dict(by_own=sorted(rows, key=lambda r:r['own_s'], reverse=True)[:20],
                by_cumulative=sorted(rows, key=lambda r:r['cumulative_s'], reverse=True)[:20])

def main():
    started = time.perf_counter()
    protocol = read(SINGLE / 'run01/protocol.json')
    sources = dict(protocol['source_hashes'])
    for path, expected in sources.items():
        assert digest(path) == expected, path
    dataset = read(SINGLE / 'run01/dataset.json')['cases']
    row = next(r for r in dataset if r['split'] == 'train')
    assert digest(row['path']) == row['sha256']
    prereg = dict(scope='Performance only; existing TRAIN field, unchanged frozen source and weights',
        date='2026-11-01', source_hashes=sources, source_field=row,
        methods=['happo', 'hasac'], layout=778801, action_seed=778802,
        variants=['mmap_2threads', 'ram_2threads', 'ram_1thread', 'ram_4threads'],
        repetitions=2, order='forward then reverse, identical seeds and fresh controllers',
        checks='Metrics and every trace array exactly equal; no weight mutation during rollout',
        updates='One local cloned HASAC update block, never saved as a production checkpoint',
        limit_s=180, caveat='One short probe while live training continues; warmed OS cache, not isolated benchmark; no new test data',
        accept='RAM preload only if meaningful end-to-end speedup and exact equality; thread change needs future registration, not live edits')
    write(BASE/'protocol.json', prereg)
    spec = importlib.util.spec_from_file_location('perf_single', SINGLE/'singles.py')
    s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s); s.bootstrap()
    e, np, torch = s.e, s.np, s.torch
    t = time.perf_counter(); mmap = np.load(row['path'], mmap_mode='r', allow_pickle=False)
    ram = np.array(mmap, copy=True); ram.flags.writeable = False
    read_s = time.perf_counter()-t
    results=[]; equality=[]; profiles={}; (BASE/'traces').mkdir(exist_ok=True)
    variants = [('mmap_2threads',mmap,2),('ram_2threads',ram,2),('ram_1thread',ram,1),('ram_4threads',ram,4)]
    for algorithm in ('happo','hasac'):
        torch.set_num_threads(2); e.seed_all(778800)
        models=[e.adapted(algorithm)[0]]; initial=e.model_hash(models)
        reference=None; reference_metric=None
        for repeat in range(2):
            for label, water, threads in variants[::1 if repeat==0 else -1]:
                if time.perf_counter()-started > 180: raise TimeoutError('Bounded probe limit')
                torch.set_num_threads(threads); e.seed_all(778802)
                path=BASE/'traces'/f'{algorithm}-{label}-{repeat}.npz'
                t=time.perf_counter()
                metric,_=e.rollout(models,water,778801,s.ROLE,trace_path=path)
                elapsed=time.perf_counter()-t
                assert e.model_hash(models)==initial
                with np.load(path, allow_pickle=False) as archive:
                    arrays={k:archive[k].copy() for k in archive.files}
                if reference is None:reference=arrays; reference_metric=metric
                equal=metric==reference_metric and arrays.keys()==reference.keys() and all(np.array_equal(v,reference[k]) for k,v in arrays.items())
                equality.append(dict(algorithm=algorithm, variant=label, repeat=repeat, exactly_equal=equal))
                results.append(dict(algorithm=algorithm,variant=label,repeat=repeat,seconds=elapsed,steps=metric['ST']))
        torch.set_num_threads(2); e.seed_all(778802); prof=cProfile.Profile();prof.enable()
        _, rows=e.rollout(models,ram,778801,s.ROLE,collect=algorithm=='hasac',trace_path=BASE/'traces'/f'{algorithm}-profile.npz')
        prof.disable();profiles[algorithm+'_rollout']=profile_rows(prof)
        if algorithm=='hasac':
            e.seed_all(778803); prof=cProfile.Profile(); t=time.perf_counter();prof.enable()
            s.e.update_block(models, rows, [])
            prof.disable(); profiles['hasac_update']=dict(seconds=time.perf_counter()-t,**profile_rows(prof))
        del models;gc.collect()
    summary=[]
    for algorithm in ('happo','hasac'):
        for label,_,_ in variants:
            times=[r['seconds'] for r in results if r['algorithm']==algorithm and r['variant']==label]
            summary.append(dict(algorithm=algorithm,variant=label,median_s=statistics.median(times),min_s=min(times),max_s=max(times)))
    for path, expected in sources.items():assert digest(path)==expected,path
    output=dict(status='completed',runtime_s=time.perf_counter()-started,torch=torch.__version__,cuda_available=torch.cuda.is_available(),
        field_bytes=ram.nbytes,all22_fields_bytes=sum(Path(r['path']).stat().st_size for r in dataset),copy_seconds=read_s,
        measurements=results,summary=summary,exact_trace_checks=equality,all_equal=all(r['exactly_equal'] for r in equality),
        profiles=profiles,production_sources_unchanged=True,probe_sha256=digest(__file__),protocol_sha256=digest(BASE/'protocol.json'))
    write(BASE/'results.json',output)
    print(json.dumps({k:output[k] for k in ('status','runtime_s','field_bytes','all22_fields_bytes','summary','all_equal','production_sources_unchanged')}),flush=True)

if __name__=='__main__':
    main()
