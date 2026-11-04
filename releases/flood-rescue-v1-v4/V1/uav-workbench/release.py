"""Standalone generation, evaluation, archived replay and integrity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from flood_dqn import maps
from flood_dqn.env import Environment, layout
from release_io import read
from v1_rescue95.study import OUT, evaluate, generate, summarize

ROOT = Path(__file__).resolve().parent.parent


def verify():
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest:
        file = ROOT/row['bundled']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == row['bundled_sha256'], file
    assert shutil.which('node'), 'Install Node.js and add it to PATH.'
    print(f'Verified {len(manifest)} packaged source and result files.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'smoke', 'run', 'replay-archive'])
    parser.add_argument('--parents', type=int, nargs='+', default=[189001])
    parser.add_argument('--layouts', type=int, nargs='+', default=[0])
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    verify()
    if args.command == 'verify':
        return
    if args.command == 'smoke':
        from v1_rescue95.study import public
        from v1_rescue95.rescue import Rescue
        from v1_rescue95.gaps import Gaps
        env = Environment(np.zeros((301,3,100,100),np.float32), np.ones((300,100,100),bool), layout(42))
        planner, rescue = Gaps(), Rescue()
        try:
            for _ in range(5):
                view = public(env)
                env.step(planner.actions(view,rescue.actions(view)))
            print('Synthetic dry-field smoke check passed:', env.metrics())
        finally:
            planner.close()
        return
    if args.workers < 1 or any(p not in range(189001,189051) for p in args.parents) or any(n not in range(4) for n in args.layouts):
        parser.error('Use workers >= 1, parents 189001..189050 and layouts 0..3.')
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        manifest = list(pool.map(generate,args.parents))
    maps.save_json(OUT/'maps-selected.json',manifest)
    jobs = [('test','gaps',p,n) for p in args.parents for n in args.layouts]
    if args.command == 'run':
        with ProcessPoolExecutor(args.workers) as pool:
            result = list(pool.map(evaluate,jobs))
        summarize(result,'test')
    else:
        for _,_,p,n in jobs:
            archived = read(ROOT/'results/actions'/f'{p}-{n}.gz')
            with np.load(OUT/'water/maps'/f'{p}.npz') as data:
                env = Environment(data['fields'],data['safe'],layout(archived['row']['layout_seed']))
                for actions in archived['actions']:
                    _, done = env.step(actions)
                assert done
                for key,value in env.metrics().items():
                    assert value == archived['row'][key], (p,n,key,value,archived['row'][key])
            print(f'Archived replay matched: {p}-{n}')


if __name__ == '__main__':
    main()
