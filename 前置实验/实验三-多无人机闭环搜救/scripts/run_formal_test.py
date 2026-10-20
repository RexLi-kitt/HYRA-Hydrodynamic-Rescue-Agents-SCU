from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.formal import run_formal
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--frozen',type=Path,default=ROOT/'configs/frozen_experiment3.v1.json')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--workers',type=int,default=6,choices=range(1,7))
    args=parser.parse_args()
    print(run_formal(ROOT,args.frozen,args.workers,args.resume))
