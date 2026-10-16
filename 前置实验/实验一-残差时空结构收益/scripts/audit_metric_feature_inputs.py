"""Independently reconstruct every cached source input from saved source fields."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from hyra_validation.artifacts import write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(path,design):
    receipt=read(path.with_suffix('.json'));assert digest(path)==receipt['cache_sha256'];manifest=read(receipt['manifest_path'])
    assert digest(receipt['manifest_path'])==receipt['manifest_sha256'];fields={}
    def field(key):
        if key not in fields:
            info=manifest['fields'][key];assert digest(info['projected_path'])==info['projected_sha256']
            with np.load(info['projected_path'],allow_pickle=False) as z:fields[key]=(z['time_s'],z['linear'])
        return fields[key]
    source=np.array(design['source_ids']);versions=manifest['forecast_versions'];releases=np.array(sorted(map(int,versions)))
    observation_cache={};scalars=0;records=0;maximum=0.
    with np.load(path,allow_pickle=False) as z:
        meta=z['metadata'];histories=z['history']
        assert histories.shape==(576,6,6,12,6);assert meta.shape==(576,9)
        for b,m in enumerate(meta):
            mesh,anchor,condition,center,phase,noise,delay,replicate,weight=m;anchor=int(anchor);phase=int(phase)
            key=tuple(m[:8])
            if key not in observation_cache:
                ref=manifest['references'][str(int(mesh))];times,depth=field(ref['base'])
                if condition:
                    pt,pd=field(ref['branches'][f'a{anchor}_y{int(center)}']);take=times<anchor
                    times=np.r_[times[take],pt];depth=np.vstack([depth[take],pd])
                valid=np.arange(anchor-60+5*design['sample_phase_fractions'][phase],anchor+30+1e-9,5.)
                seed=[*manifest['metadata']['seed_identity'],937,anchor,phase,int(replicate)]
                errors=noise*np.random.default_rng(np.random.SeedSequence(seed)).normal(size=(len(valid),6))
                observed=np.column_stack([np.interp(valid,times,depth[:,s]) for s in source])+errors;prior=[]
                for validtime in valid:
                    issue=releases[np.flatnonzero(releases<=validtime-20)[-1]]
                    ft,fd=field(versions[str(issue)]['field_key']);assert ft[0]==issue and ft[30]==issue+60
                    prior.append([np.interp(validtime,ft,fd[:,s]) for s in source])
                prior=np.array(prior);observation_cache[key]=(valid,observed,prior)
            valid,observed,prior=observation_cache[key];expected=np.zeros((6,6,12,6))
            for t,offset in enumerate(design['decision_offsets_s']):
                now=anchor+offset
                for j,v in enumerate(valid):
                    age=now-v
                    if age<0 or age>=60 or v+delay>now:continue
                    slot=int(age//5);issue=releases[np.flatnonzero(releases<=v-20)[-1]]
                    expected[t,:,slot]=np.stack([observed[j],prior[j],observed[j]-prior[j],np.full(6,age),np.full(6,v-issue),np.ones(6)],axis=-1)
            # Age, actual lead and masks are exactly representable at this schedule.
            np.testing.assert_array_equal(expected[:,:,:,3:],histories[b,:,:,:,3:])
            difference=float(abs(expected[:,:,:,:3]-histories[b,:,:,:,:3]).max());maximum=max(maximum,difference)
            np.testing.assert_allclose(expected[:,:,:,:3],histories[b,:,:,:,:3],atol=1e-7,rtol=0)
            np.testing.assert_array_equal(expected[:,:,:,5].sum(axis=(1,2)),72 if delay==0 else 60)
            assert abs(weight-((2. if condition==0 else 1.)/(1 if noise==0 else 3)))<1e-12
            records+=int(expected[:,:,:,5].sum());scalars+=expected.size
    return dict(passed=True,family=path.stem,cache_sha256=receipt['cache_sha256'],contexts=576,history_scalars=scalars,
        arrived_records=records,maximum_float32_depth_rounding_m=maximum,audit_source_sha256=digest(__file__),
        scope='Every cached source value, old forecast, residual, age, lead, mask, arrival, full record count and E0 multiplicity; no target observations accessed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--limit',type=int,default=36);args=parser.parse_args()
    folder=Path(read(ROOT/'results/latest_metric_feature_cache.json')['folder']);design=read(ROOT/'configs/metric_development.design.v1.json');audits=folder/'input_audit';audits.mkdir(exist_ok=True);done={}
    while True:
        paths=sorted(folder.glob('metric_v2_*.npz'))[:args.limit]
        for path in paths:
            if not path.with_suffix('.json').exists() or path.stem in done:continue
            output=audits/(path.stem+'.json')
            if output.exists():
                result=read(output);assert result['cache_sha256']==read(path.with_suffix('.json'))['cache_sha256']
            else:result=audit(path,design);write_json(output,result)
            done[path.stem]=result;write_json(audits/'progress.json',dict(passed=True,audited_families=len(done),families=sorted(done),arrived_records=sum(r['arrived_records'] for r in done.values())))
            print('INPUT_AUDITED '+path.stem,flush=True)
        if len(done)>=args.limit or not args.watch:break
        time.sleep(20)
    print('DONE_INPUT_AUDIT '+str(len(done)),flush=True)


if __name__=='__main__':main()
