"""Existing training fields only; exact-trajectory acceptance before adoption."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from geometry_cache import install

BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
def write(name,value): (BASE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    write('cache-protocol.json',dict(date='2026-11-01',scope='Existing TRAIN data only, no production file edits',
      candidate='65536 exact separation entries +512 immutable view masks; never caches observations or policy decisions',
      inputs='Exact dtype and bytes; finite public geometry. No coordinate rounding, no future or hidden state',
      success='Every metric and trace array identical, source/weight hashes unchanged, median paired total speedup >=1.10',
      rejection='Any difference or insufficient gain means do not deploy; no changed scores or thresholds',
      limits='At most180s; one small process; no CUDA install or live model mutation'))
    spec=importlib.util.spec_from_file_location('perf_strict',ROOT/'uav-strict-eval-20261101/strict_eval.py')
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);v.bootstrap();e,np,torch=v.e,v.np,v.torch
    torch.set_num_threads(2)
    cases=[r for r in read(v.SINGLE/'run01/dataset.json')['cases'] if r['split']=='train'][:2]
    sources=read(v.SINGLE/'run01/protocol.json')['source_hashes']
    for p,h in sources.items():assert sha(p)==h,p
    control=install(); records=[]; (BASE/'cache-traces').mkdir(exist_ok=True);start=time.perf_counter()
    candidates=[(a,) for a in v.ALGOS]+[('hatrpo','hasac'),('hasac','mat')]
    for j,names in enumerate(candidates):
        models=[e.adapted(a)[0] for a in names];initial=e.model_hash(models)
        water=np.load(cases[j%2]['path'],mmap_mode='r',allow_pickle=False)
        role=(0,0,0,0) if len(names)==1 else ((0,1,1,0) if j%2 else (1,0,0,1))
        for cover in (True,False):
            baseline=None;current={}
            for cached in ([False,True] if j%2==0 else [True,False]):
                if time.perf_counter()-start>180:raise TimeoutError('Probe limit')
                control['clear']();control['enable' if cached else 'disable']();e.seed_all(778804)
                p=BASE/'cache-traces'/f'{"_".join(names)}-{cover}-{cached}.npz';t=time.perf_counter()
                metric,_=e.rollout(models,water,778805,role,deterministic=j%2==0,coverage=cover,trace_path=p)
                elapsed=time.perf_counter()-t
                with np.load(p,allow_pickle=False) as f: arrays={k:f[k].copy() for k in f.files}
                if baseline is None:baseline=(metric,arrays)
                equal=metric==baseline[0] and all(np.array_equal(x,baseline[1][k]) for k,x in arrays.items())
                assert e.model_hash(models)==initial
                current[str(cached)]=elapsed
                records.append(dict(candidate='_'.join(names),coverage=cover,cache=cached,seconds=elapsed,steps=metric['ST'],exactly_equal=equal,stats=control['stats']()))
        del models
    control['disable']();control['clear']()
    for p,h in sources.items():assert sha(p)==h,p
    pairs=[]
    for names in candidates:
        for cover in (True,False):
            rows=[r for r in records if r['candidate']=='_'.join(names) and r['coverage']==cover]
            base=next(r['seconds'] for r in rows if not r['cache']);fast=next(r['seconds'] for r in rows if r['cache'])
            pairs.append(dict(candidate='_'.join(names),coverage=cover,baseline_s=base,cached_s=fast,speedup=base/fast))
    median=statistics.median(r['speedup'] for r in pairs);equal=all(r['exactly_equal'] for r in records)
    result=dict(status='completed',seconds=time.perf_counter()-start,records=records,pairs=pairs,median_speedup=median,all_exactly_equal=equal,
      candidate_eligible=equal and median>=1.10,production_sources_unchanged=True,adapter_sha256=sha(BASE/'geometry_cache.py'),protocol_sha256=sha(BASE/'cache-protocol.json'))
    write('cache-results.json',result)
    print(json.dumps({k:result[k] for k in ('status','seconds','median_speedup','all_exactly_equal','candidate_eligible','pairs')}),flush=True)

if __name__=='__main__':main()
