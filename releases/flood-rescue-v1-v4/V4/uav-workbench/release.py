"""Portable entry point for a single published HyRA experiment branch."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from flood_dqn import maps
from release_io import read,save

ROOT=Path(__file__).resolve().parent.parent
VERSION=json.loads((ROOT/'VERSION.json').read_text(encoding='utf-8'))['version']


def verify():
    records=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for record in records:
        path=ROOT/record['bundled']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['bundled_sha256'], path
    print(f'{VERSION}: verified {len(records)} source, model and archive files.',flush=True)


def model_smoke():
    if VERSION in ('V2','V3'):
        from dynamic_drop_rebuild.learning import predictor
        import torch
        from release_utils import model_file
        from dynamic_drop_rebuild.scenarios import OUT
        for arm in ('T0','TG'):
            for seed in (11,29,47):
                ck=torch.load(model_file(OUT,f'{arm}-{seed}.pt'),weights_only=True)
                result=predictor(arm,seed)(np.zeros((100,12,len(ck['mu'])),dtype=np.float32))
                assert result.shape==(100,) and np.isfinite(result).all() and (result>=0).all()
    else:
        from gru_forward60_validation.learning import predictor
        for seed in (11,29,47):
            result=predictor(seed)(np.zeros((100,4,14),dtype=np.float32))
            assert result.shape==(100,) and np.isfinite(result).all() and (result>=0).all()
    print('All deployed GRU checkpoints loaded and inferred on CPU.',flush=True)


def run_case(args):
    maps.make_map(args.parent)
    outputs=[]
    if VERSION=='V2':
        from dynamic_drop_rebuild.sim import run
        from dynamic_drop_rebuild.learning import predictor
        choice=json.loads((ROOT/'results/selection.json').read_text(encoding='utf-8'))['alpha']
        for arm in args.arms or ['B0','T0','TG']:
            if arm not in ('B0','L0','LG','T0','TG'):raise ValueError(arm)
            seed=args.seed if arm.startswith('T') else 11
            alpha=choice.get(arm,0)
            result=run(args.parent,args.layout,args.radius,arm=arm,
                predictor=None if arm=='B0' else predictor(arm,seed),alpha=alpha)[0]
            outputs.append(dict(**result,seed=0 if arm=='B0' else seed,alpha=alpha))
        archived=ROOT/'results/cases'/f'{args.parent}-{args.layout}-{args.radius}.json'
    elif VERSION=='V3':
        from guizhou_mixed_rescue.scenario import make
        from guizhou_mixed_rescue.study import run
        data=make(args.parent,args.layout)
        for arm in args.arms or ['H','T','G']:
            if arm not in ('H','T','G'):raise ValueError(arm)
            outputs.append(run(data,arm,0 if arm=='H' else args.seed))
        archived=ROOT/'results/cases'/f'{args.parent}-{args.layout}.gz'
    else:
        from gru_forward60_validation.scenario import make
        from gru_forward60_validation.env import run
        from gru_forward60_validation.learning import predictor
        data=make(args.parent,args.layout,args.orientation)
        for arm in args.arms or ['P0','P1','M0','M1']:
            if arm not in ('P0','P1','M0','M1'):raise ValueError(arm)
            seed=args.seed if arm.startswith('M') else 0
            row,people,*_=run(data,arm,predict=predictor(seed) if seed else None)
            outputs.append(dict(row=dict(**row,parent=args.parent,layout=args.layout,
                                        orientation=args.orientation,arm=arm,seed=seed),persons=people))
        archived=ROOT/'results/cases'/f'{args.parent}-{args.layout}-{args.orientation}.gz'
    if args.compare_archive:
        if not archived.exists():raise FileNotFoundError(archived)
        old=json.loads(archived.read_text(encoding='utf-8')) if archived.suffix=='.json' else read(archived)['records']
        for new in outputs:
            def match(r):
                return r['row']['arm']==new['row']['arm'] and r.get('seed',r['row'].get('seed',0))==new.get('seed',new['row'].get('seed',0)) and (VERSION!='V2' or r['alpha']==new['alpha'])
            previous=next(r for r in old if match(r))
            assert new['persons']==previous['persons'], (VERSION,new['row']['arm'],'person events differ')
            for key,value in new['row'].items():
                if key not in ('ms',):assert value==previous['row'][key],(key,value,previous['row'][key])
            for key in ('choices','trace'):
                if key in new:assert new[key]==previous[key],key
        print('Archived outcomes, person events and available decisions matched.',flush=True)
    output=maps.ROOT/'generated/release-runs'/f'{VERSION}-{args.parent}-{args.layout}-{args.radius}-{args.orientation}-{args.seed}.gz'
    save(output,dict(records=outputs))
    print(json.dumps([r['row'] for r in outputs],ensure_ascii=False),flush=True)
    print(output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','smoke','run','prepare-training','train'])
    parser.add_argument('--parent',type=int,default={'V2':181001,'V3':184001,'V4':188001}[VERSION])
    parser.add_argument('--layout',type=int,default=0)
    parser.add_argument('--radius',type=int,default=40 if VERSION=='V3' else 60,choices=[40,60,100])
    parser.add_argument('--seed',type=int,default=11,choices=[11,29,47])
    parser.add_argument('--arms',nargs='+')
    parser.add_argument('--orientation',default='main',choices=['main','rot','mirror'])
    parser.add_argument('--compare-archive',action='store_true')
    args=parser.parse_args()
    verify()
    if args.command=='smoke':model_smoke()
    elif args.command=='run':run_case(args)
    elif args.command=='prepare-training':
        for parent in range(21001,21017):maps.make_map(parent)
        print('16 training and calibration water maps generated.')
    elif args.command=='train':
        if VERSION=='V4':
            from gru_forward60_validation.learning import prepare_train
            prepare_train()
        elif VERSION=='V2':
            from dynamic_drop_rebuild.learning import collect,train
            from dynamic_drop_rebuild.scenarios import OUT
            import shutil
            OUT.mkdir(parents=True,exist_ok=True)
            if not (OUT/'A-gate.json').exists():shutil.copy2(ROOT/'results/A-gate.json',OUT/'A-gate.json')
            collect();train()
        else:parser.error('V3 reuses V2 models. Train them in the V2 directory.')


if __name__=='__main__':main()
