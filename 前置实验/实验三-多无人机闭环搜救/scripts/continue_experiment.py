from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.workflow import run_workflow
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('data','training','refinement','sensitivity'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    print(run_workflow(ROOT,{k:getattr(a,k) for k in ('data','training','refinement','sensitivity')},a.resume))
