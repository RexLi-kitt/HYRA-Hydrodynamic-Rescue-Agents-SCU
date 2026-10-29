"""Fresh-process runtime/source guard for each formal training segment."""
import argparse,json
from pathlib import Path
from scale100_formal import validate_plan,runtime,ROOT
from scale100_train import write
from scale100_backend import train_backend

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--job',required=True);p.add_argument('--stop-after',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text());validate_plan(plan);job=next(j for j in plan['jobs'] if j['id']==a.job)
    if a.stop_after not in plan['development_checkpoints']:raise ValueError('Unregistered formal checkpoint')
    output=a.plan.parent/job['id'];train_backend(plan.get('training_backend','absolute_head_v1'),Path(plan['dataset_path']),output,job['task'],job['method'],job['seed'],plan['episodes_per_job'],a.stop_after,a.resume,'formal')
    write(output/f'runtime-ep{a.stop_after:04d}.json',dict(runtime=runtime(),job=job['id'],episode=a.stop_after,source_sha256=plan['training_sources']))
if __name__=='__main__':main()
