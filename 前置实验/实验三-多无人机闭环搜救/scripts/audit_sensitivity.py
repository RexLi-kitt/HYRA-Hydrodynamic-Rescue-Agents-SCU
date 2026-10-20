"""Independently read back the complete prespecified planner sensitivity matrix."""
from pathlib import Path
import argparse
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.artifacts import new_run,finish_run,write_json
from hyra_exp3.sensitivity_audit import audit_sensitivity

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3,choices=range(1,5))
    args=parser.parse_args()
    target=new_run(ROOT,'planner_sensitivity_independent_audit',dict(run=str(args.run.resolve()),workers=args.workers))
    print(f'Planner sensitivity independent audit directory: {target}',flush=True)
    try:
        result=audit_sensitivity(ROOT,args.run.resolve(),workers=args.workers)
        write_json(target/'audit_result.json',result)
        finish_run(ROOT,target)
        print(result,flush=True)
    except BaseException as error:
        (target/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(ROOT,target,status='failed',error=repr(error))
        raise
