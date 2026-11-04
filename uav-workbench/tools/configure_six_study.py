"""Bind an existing local research archive; does not download, train or select."""
import argparse
import json
from pathlib import Path
import sys

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP))
from hyra_workbench.six_study import digest, read


def configure(archive):
    archive=Path(archive).resolve()
    run=archive/'uav-strict-eval-20261101/run01'
    dataset=archive/'uav-singles1024-20261101/run01/dataset.json'
    for path,expected in read(run/'registration.json')['source_hashes'].items():
        if digest(path)!=expected: raise ValueError('Frozen source differs: '+path)
    for candidate in read(run/'frozen-candidates.json')['candidates']:
        if digest(candidate['path'])!=candidate['sha256']: raise ValueError('Checkpoint differs: '+candidate['id'])
    files=[run/n for n in ('registration.json','frozen-candidates.json','scenario-manifest.json','results.jsonl','strict-summary.json')]
    files += [dataset,archive/'uav-singles1024-20261101/run01/protocol.json']
    rows=[json.loads(line) for line in (run/'results.jsonl').read_text(encoding='utf-8').splitlines() if line]
    hashes={}
    for row in rows:
        trace=run/'traces'/Path(row['trace']).name
        hashes[trace.name]=digest(trace)
    cfg=dict(study=str(run),adapter=str(archive/'uav-strict-eval-20261101/strict_eval.py'),
             dataset=str(dataset),seals={str(p):digest(p) for p in files},trace_hashes=hashes)
    target=APP/'generated/six-study/local.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise FileExistsError('Configuration already exists; preserve and review it before replacing: '+str(target))
    target.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',type=Path,required=True)
    print(configure(parser.parse_args().archive_root))
