from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.posttest import run_posttest
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['reassignment_counterfactual','numerical_direction'])
    p.add_argument('--frozen',type=Path,default=ROOT/'configs/frozen_experiment3.v1.json')
    p.add_argument('--resume',type=Path)
    p.add_argument('--workers',type=int,default=4,choices=range(1,7))
    a=p.parse_args()
    print(run_posttest(ROOT,a.frozen,a.kind,a.workers,a.resume))
