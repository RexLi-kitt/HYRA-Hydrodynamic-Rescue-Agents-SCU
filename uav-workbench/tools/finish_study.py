"""Continue an already-authorized frozen study after its existing fit finishes.

Never starts/restarts fitting, changes budgets, selects another checkpoint or
launches a further mechanism round. A local supervisor can finish scoring even
while the conversational assistant is unavailable.
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


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');temporary.replace(path)


def main(root,wait_seconds):
    root=root.resolve();token=uuid.uuid4().hex;lock=root/'evaluation-supervisor.lock'
    with lock.open('x',encoding='utf-8') as file:json.dump(dict(pid=os.getpid(),token=token),file)
    state_file=ROOT/'generated/strong-study-supervisor.json'
    def state(stage,**extra):
        write(state_file,dict(pid=os.getpid(),token=token,study=str(root),stage=stage,
             updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
        print(stage,flush=True)
    try:
        state('waiting_for_model_freeze');deadline=time.monotonic()+wait_seconds
        while True:
            if time.monotonic()>deadline:raise TimeoutError('No complete freeze before supervisor timeout; fitting is not restarted')
            try:
                freeze=json.loads((root/'model-freeze.json').read_text(encoding='utf-8'))
            except (FileNotFoundError,json.JSONDecodeError):
                time.sleep(2);continue
            if len(freeze['models'])!=15:raise ValueError('Frozen model count is not 15')
            if sha256((root/'plan.json').read_bytes()).hexdigest()!=freeze['plan_sha256']:raise ValueError('Plan changed')
            for row in freeze['models']:
                if sha256((root/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Frozen model changed')
            break
        def run(command):
            subprocess.run([sys.executable,'-u',*command],cwd=ROOT,check=True)
        state('scoring_frozen_holdout')
        run(['-m','hyra_uav.strong_study','evaluate',str(root)])
        state('auditing_all_actual_replays')
        run(['tools/engineering_audit.py','--source',str(root),'--output',str(root/'input-audit-final'),'--expected','54'])
        state('publishing_complete_result')
        os.environ['MPLCONFIGDIR']=str(ROOT/'.runtime/plot-cache')
        plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
        output=ROOT/'docs/evidence'/('20261025-p3-nonlinear-round'+str(plan['round']))
        run(['tools/strong_report.py',str(root),str(output)])
        summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        if summary['metrics_rows']!=summary['expected_metrics_rows']:raise ValueError('Incomplete score table')
        state('completed',gate_passed=summary['gate_passed'],report=str(output),
              limits='Study job completed only; full goal acceptance and strict-ablation review remain separate')
    except BaseException as error:
        state('stopped',reason=type(error).__name__+': '+str(error));raise
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8')).get('token')==token:lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('study',type=Path)
    parser.add_argument('--wait-seconds',type=float,default=7200);args=parser.parse_args()
    main(args.study,args.wait_seconds)
