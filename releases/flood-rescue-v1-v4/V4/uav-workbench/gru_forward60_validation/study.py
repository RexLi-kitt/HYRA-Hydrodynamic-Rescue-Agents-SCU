from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from functools import lru_cache
from flood_dqn import maps
from release_io import save,read

OUT=maps.ROOT/'generated/gru-forward60-validation-v1'
SEEDS=(11,29,47)


@lru_cache(maxsize=3)
def policy(seed):
    from .learning import predictor
    return predictor(seed)


def case(job):
    from .env import run
    from .scenario import make
    stage,p,n,orientation=job
    path=OUT/stage/f'{p}-{n}-{orientation}.gz'
    if path.exists():return
    data=make(p,n,orientation)
    records=[]
    for arm in ('P0','P1','M0','M1'):
        for s in SEEDS if arm.startswith('M') else (0,):
            row,people,*_=run(data,arm,predict=policy(s) if s else None)
            assert len(people)==40 and 0<=row['saved']<=row['found']<=40
            records.append(dict(row=dict(**row)|dict(parent=p,layout=n,orientation=orientation,arm=arm,seed=s),persons=people))
    save(path,dict(records=records,meta=data[-1]))


def generate(p):
    maps.DATA=OUT/'water'
    return maps.make_map(p)


def execute(jobs):
    with ProcessPoolExecutor(8) as pool:list(pool.map(case,jobs))


def main():
    from .learning import prepare_train
    from .report import report
    from . import env,scenario
    OUT.mkdir(parents=True,exist_ok=True)
    code={str(p):maps.digest(p) for p in [*sorted(Path(__file__).parent.glob('*.py')),Path(maps.__file__),maps.TERRAIN/'flood_solver.py']}
    protocol=OUT/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==code
    else:maps.save_json(protocol,code)
    prepare_train()
    from .checks import verify
    verify()
    execute([('development',p,0,'main') for p in range(31001,31007)])
    seal=dict(code=code,models={p.name:maps.digest(p) for p in sorted(OUT.glob('M-*.pt'))},test=[188001,188024],seeds=list(SEEDS),stats=20261103)
    assert len(seal['models'])==3
    lock=OUT/'seal.json'
    if lock.exists():assert json.loads(lock.read_text())==seal
    else:maps.save_json(lock,seal)
    with ProcessPoolExecutor(4) as pool:manifest=list(pool.map(generate,range(188001,188025)))
    maps.save_json(OUT/'maps.json',manifest)
    execute([('test',p,n,'main') for p in range(188001,188025) for n in range(2)]+
            [('test',p,0,o) for p in range(188001,188007) for o in ('rot','mirror')])
    report()

