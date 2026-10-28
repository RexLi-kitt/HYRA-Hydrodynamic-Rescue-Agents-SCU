"""After actual full-audit completion, verify statistics and prepare report.

Does not claim final delivery: figures need visual QA and final artifacts need
requirement-by-requirement acceptance by the active goal worker.
"""
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
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--audit-pid',type=int,required=True)
    parser.add_argument('--plot-python',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();token=uuid.uuid4().hex;lock=root/'report-finish.lock'
    with lock.open('x',encoding='utf-8') as file:
        import json
        json.dump(dict(pid=os.getpid(),token=token),file)
    programs=['audit_sciml_targets.py','audit_sciml_statistics.py','report_sciml.py','experiment_index.py']
    sources={name:digest(ROOT/'tools'/name) for name in programs+['finish_sciml_report.py']}
    def state(stage,**extra):
        write(root/'report-finish-state.json',dict(stage=stage,pid=os.getpid(),audit_pid=args.audit_pid,
            utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,**extra))
    started=time.monotonic()
    try:
        while True:
            live=process(args.audit_pid)
            if live is not None:
                if 'tools/finish_sciml_audits.py generated/sciml-round1 --evaluation-pid 37500' not in live['CommandLine']:
                    raise RuntimeError('Audit supervisor PID reused')
                state('verified_audit_supervisor_wait',command=live['CommandLine'])
            else:
                if read(root/'audit-finish-state.json')['stage']!='all_independent_audits_completed':raise RuntimeError('Independent audits stopped before completion')
                break
            if time.monotonic()-started>24*3600:raise TimeoutError('24-hour supervision limit, no experiment interrupted')
            time.sleep(60)
        for program in programs:
            for name,h in sources.items():
                if digest(ROOT/'tools'/name)!=h:raise RuntimeError('Scheduled report tool changed: '+name)
            state('running_final_analysis',program=program)
            python=sys.executable if program=='audit_sciml_targets.py' else str(args.plot_python)
            command=[python,'-u',str(ROOT/'tools'/program)]
            if program!='experiment_index.py':command.append(str(root))
            subprocess.run(command,cwd=ROOT,check=True)
        state('report_generated_final_acceptance_pending',remaining='Visually inspect exported plots; deliver complete source/config/log/checkpoint artifact manifest and reproducible commands; audit all goal requirements. Do not mark overall goal complete from this state alone.')
    except BaseException as error:
        state('failed',error=repr(error));raise
    finally:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()


if __name__=='__main__':main()
