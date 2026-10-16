"""Check optimized replay and frozen-source provenance against the first replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hyra_validation.artifacts import write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--old',type=Path,required=True);parser.add_argument('--new',type=Path,required=True);args=parser.parse_args()
    old=args.old.resolve();new=args.new.resolve();namesold=[s['name'] for s in read(old/'candidates.json')];namesnew=[s['name'] for s in read(new/'candidates.json')]
    shared=sorted(set(namesold)&set(namesnew));differences={k:0. for k in ('services','window_error','mae_6_12')};comparisons=0
    for run in (old,new):
        for rel,sha in read(run/'provenance.json')['files'].items():assert digest(run/'frozen_source'/rel)==sha,rel
        modelrun=Path(read(run/'config.json')['model_run']);modelrun=modelrun if modelrun.is_absolute() else ROOT/modelrun
        for spec in read(modelrun/'registry.json'):assert digest(spec['path'])==spec['sha256']
    selection=read(old/'selection.json');config=read(old/'config.json')
    for i in range(config['family_start'],config['family_start']+config['family_count']):
        name=f'metric_v2_{config["split"]}_{i:03}.json';left=read(old/name);right=read(new/name)
        for mesh in ('80','40'):
            a,b=left['meshes'][mesh],right['meshes'][mesh];assert a['total_weighted_cases']==b['total_weighted_cases']
            for metric in differences:
                x=np.array([a[metric][namesold.index(s)] for s in shared]);y=np.array([b[metric][namesnew.index(s)] for s in shared])
                difference=float(abs(x-y).max());differences[metric]=max(differences[metric],difference);comparisons+=len(shared)
                if metric=='services':np.testing.assert_array_equal(x,y)
                else:np.testing.assert_allclose(x,y,atol=1e-14,rtol=0)
    result=dict(passed=True,old_run=str(old),new_run=str(new),independent_families=selection['independent_families'],
        common_candidates=len(shared),compared_family_mesh_candidate_metrics=comparisons,maximum_differences=differences,
        frozen_source_and_model_checksums_verified=True,aggregate_tolerance={'services':0.,'window_error':1e-14,'mae_6_12':1e-14},
        scope='Service aggregates exact; error means allow floating summation rounding. All-case unit test compares individual window values and choices bitwise. No efficacy claim')
    write_json(new/'replay_equivalence_audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
