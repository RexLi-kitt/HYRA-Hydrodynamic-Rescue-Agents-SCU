from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.training import train_all
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-run',type=Path,required=True)
    parser.add_argument('--wait-for-data',action='store_true')
    args=parser.parse_args()
    print(train_all(ROOT,args.data_run,wait_for_data=args.wait_for_data))
