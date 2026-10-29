"""Run/resume the registered study, then verify its complete deliverables.

This is an external orchestration tool. It never changes frozen training
sources, estimators, checkpoint selection, or the confirmation matrix.
Only launch --run-study after verifying no prior controller is alive.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
import shutil
import subprocess
import time
import traceback
import signal_routes
from signal_routes.common import DEFAULT,ROOT,read,write,digest,sources


def finalize(study):
    study=Path(study)
    if read(study/'status.json')['status']!='completed':
        raise ValueError('Training, selection and confirmation must finish before finalization')
    from report_signal_routes_study import supplemental_report
    from audit_signal_routes_study import completed
    report=supplemental_report(study)
    audit=completed(study)
    if not audit['passed'] or audit['report_sha256']!=digest(study/'report.md'):
        raise AssertionError('Independent completion evidence does not bind the final report')
    return dict(report=report,audit_sha256=digest(study/'completion-audit.json'),
                report_sha256=digest(study/'report.md'),sources=sources())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5))
    parser.add_argument('--run-study',action='store_true')
    args=parser.parse_args(); study=args.study.resolve()
    state=dict(status='running',pid=os.getpid(),started_at=time.time(),tool_sha256=digest(__file__))
    path=study/'finalization-status.json'
    write(path,state)
    try:
        if args.run_study and (not (study/'status.json').exists() or read(study/'status.json')['status']!='completed'):
            history=study/'resumptions'/str(time.time_ns()); history.mkdir(parents=True)
            for name in ('status.json','workers.json','launcher.json'):
                if (study/name).exists(): shutil.copyfile(study/name,history/name)
            state['stage']='registered_study'; write(path,state)
            subprocess.run([sys.executable,'-u','-B',str(ROOT/'tools/run_signal_routes_study.py'),
                            '--study',str(study),'--workers',str(args.workers)],cwd=ROOT,check=True)
        state['stage']='supplemental_report_and_independent_completion_audit'; write(path,state)
        evidence=finalize(study)
        state.update(status='completed',stage='completed',finished_at=time.time(),evidence=evidence)
        write(path,state)
        print('FULL STUDY AND INDEPENDENT COMPLETION AUDIT PASSED',flush=True)
    except Exception as error:
        state.update(status='failed',finished_at=time.time(),error=repr(error),traceback=traceback.format_exc())
        write(path,state)
        raise


if __name__=='__main__': main()
