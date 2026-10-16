"""Incremental, content-addressed feature caches of completed fresh families."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from hyra_validation.metric_dataset import MetricFamilyData
from hyra_validation.metric_features import build_family_features,physical_grid,METADATA
from hyra_validation.artifacts import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=36);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();pointer=json.loads((ROOT/'results/latest_metric_development_data.json').read_text(encoding='utf-8'))
    run=Path(pointer['run']);design=json.loads((ROOT/'configs/metric_development.design.v1.json').read_text(encoding='utf-8'))
    files=[ROOT/'src/hyra_validation'/f for f in ('metric_features.py','metric_dataset.py','nominal_response.py','full_upstream.py','upstream_data.py','hidden_target_inputs.py')]
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest();folder=run/('feature_cache_'+fingerprint[:12]);folder.mkdir(exist_ok=True)
    config=dict(fingerprint=fingerprint,sources=sources,physical_grid=physical_grid(design),metadata_columns=METADATA,
        label_split_policy='No pilot or confirmation; metadata columns are evaluation-only',E0_centers='one identical copy with multiplicity two')
    write_json(folder/'config.json',config);write_json(ROOT/'results/latest_metric_feature_cache.json',dict(folder=str(folder),fingerprint=fingerprint))
    processed=set();started=time.perf_counter();print('CACHE '+str(folder),flush=True)
    while True:
        for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==sources[str(p.relative_to(ROOT))],str(p)
        index=json.loads((run/'family_index.json').read_text(encoding='utf-8'))[:args.limit]
        for entry in index:
            family=entry['family'];output=folder/(family+'.npz');receipt=folder/(family+'.json')
            if family in processed:continue
            if receipt.exists():
                info=json.loads(receipt.read_text(encoding='utf-8'));assert hashlib.sha256(output.read_bytes()).hexdigest()==info['cache_sha256']
            else:
                print('BUILD '+family,flush=True);data=MetricFamilyData(entry['manifest_path'],design)
                arrays=build_family_features(data,lambda p:print('BLOCK '+family+' '+json.dumps(p),flush=True))
                temporary=folder/(family+'.partial.npz');np.savez_compressed(temporary,**arrays);temporary.replace(output)
                write_json(receipt,dict(family=family,split=entry['split'],fingerprint=fingerprint,manifest_path=entry['manifest_path'],
                    manifest_sha256=hashlib.sha256(Path(entry['manifest_path']).read_bytes()).hexdigest(),
                    cache_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),blocks=len(arrays['metadata']),bytes=output.stat().st_size))
            processed.add(family);write_json(folder/'progress.json',dict(completed_families=sorted(processed),elapsed_s=time.perf_counter()-started))
            print('CACHED '+family,flush=True)
        if len(processed)>=args.limit:break
        if not args.watch:break
        # This process waits on generator completions; it never launches replacements.
        time.sleep(20)
    write_json(folder/'completion.json',dict(completed=True,requested=args.limit,cached=len(processed),watch=args.watch))
    print('DONE '+str(folder),flush=True)


if __name__=='__main__':main()
