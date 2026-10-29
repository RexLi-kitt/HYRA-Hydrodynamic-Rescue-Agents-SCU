"""Follow the verified confirmation supervisor with full independent audits."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from finish_sciml import ROOT,read,write,digest,process


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--evaluation-pid',type=int,required=True)
    args=parser.parse_args();root=args.output.resolve();token=uuid.uuid4().hex;lock=root/'audit-finish.lock'
    with lock.open('x',encoding='utf-8') as file:
        import json
        json.dump(dict(pid=os.getpid(),token=token),file)
    programs=['audit_sciml_training.py','audit_sciml_transitions.py','audit_sciml_trajectories.py','audit_sciml_forecasts.py']
    sources={name:digest(ROOT/'tools'/name) for name in programs+['audit_team_replays.py','finish_sciml.py','finish_sciml_audits.py']}
    started=time.monotonic()
    def state(stage,**extra):
        write(root/'audit-finish-state.json',dict(stage=stage,pid=os.getpid(),evaluation_pid=args.evaluation_pid,
            utc=datetime.now(timezone.utc).isoformat(),audit_sources=sources,**extra))
    try:
        while True:
            live=process(args.evaluation_pid)
            if live is not None:
                if 'tools/finish_sciml.py generated/sciml-round1 --training-pid 37420' not in live['CommandLine']:
                    raise RuntimeError('Evaluation supervisor PID reused')
                state('verified_evaluation_supervisor_wait',command=live['CommandLine'])
            else:
                previous=read(root/'finish-state.json')
                if previous['stage']!='evaluation_and_label_audit_completed':raise RuntimeError('Confirmation supervisor stopped before evaluation and label audit passed')
                if read(root/'confirmation-summary.json')['status']!='completed':raise RuntimeError('Confirmation summary incomplete')
                break
            if time.monotonic()-started>24*3600:raise TimeoutError('24-hour supervision limit, no experiment interrupted')
            time.sleep(60)
        for program in programs:
            for name,h in sources.items():
                if digest(ROOT/'tools'/name)!=h:raise RuntimeError('Audit source changed while scheduled: '+name)
            state('running_independent_audit',program=program)
            subprocess.run([sys.executable,'-u',str(ROOT/'tools'/program),str(root)],cwd=ROOT,check=True)
        state('all_independent_audits_completed',remaining='Result interpretation/plots, runnable delivery, full artifact index and goal-by-goal final acceptance remain required')
    except BaseException as error:
        state('failed',error=repr(error));raise
    finally:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()


if __name__=='__main__':main()
