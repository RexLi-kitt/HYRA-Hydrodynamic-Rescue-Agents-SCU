"""Wait for one verified training process, then run registered evaluation.

Never starts/restarts training. A missing training process without complete
model freeze is a failure requiring inspection, not permission to refit.
"""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return sha256(path.read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(path)


def process(pid):
    result=subprocess.run(['powershell','-NoProfile','-Command',
        f'Get-CimInstance Win32_Process -Filter "ProcessId={int(pid)}" | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress'],
        check=True,capture_output=True,text=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--training-pid',type=int,required=True)
    args=parser.parse_args();root=args.output.resolve();token=uuid.uuid4().hex
    lock=root/'finish.lock'
    with lock.open('x',encoding='utf-8') as file:json.dump(dict(pid=os.getpid(),token=token),file)
    started=time.monotonic();plan_hash=digest(root/'formal-plan.json')
    def state(stage,**extra):
        write(root/'finish-state.json',dict(stage=stage,pid=os.getpid(),training_pid=args.training_pid,
            utc=datetime.now(timezone.utc).isoformat(),formal_plan_sha256=plan_hash,**extra))
    try:
        while True:
            assert digest(root/'formal-plan.json')==plan_hash
            live=process(args.training_pid)
            if live is not None:
                if 'hyra_learning.sciml_study train generated/sciml-round1' not in live['CommandLine']:
                    raise RuntimeError('Training PID has been reused by another command')
                state('verified_training_wait',command=live['CommandLine'])
            else:
                if not (root/'model-freeze.json').exists():raise RuntimeError('Training process missing before all models froze; inspect logs and recovery, do not duplicate training')
                summary=read(root/'training-summary.json');freeze=read(root/'model-freeze.json')
                assert summary['status']=='completed' and len(summary['jobs'])==len(freeze['selected_models'])==12
                assert freeze['formal_plan_sha256']==plan_hash
                break
            if time.monotonic()-started>24*3600:raise TimeoutError('24-hour supervision limit; training was not interrupted')
            time.sleep(60)
        state('registered_confirmation_evaluation')
        subprocess.run([sys.executable,'-u','tools/evaluate_sciml.py','evaluate',str(root)],cwd=ROOT,check=True)
        state('full_hydraulic_label_audit')
        subprocess.run([sys.executable,'-u','tools/audit_sciml_data.py','--include-holdout'],cwd=ROOT,check=True)
        state('evaluation_and_label_audit_completed',remaining='Independent training/trajectory/forecast audit, analysis, runnable delivery and full goal acceptance remain required')
    except BaseException as error:
        state('failed',error=repr(error));raise
    finally:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()


if __name__=='__main__':main()
