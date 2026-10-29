"""Safe continuation for the frozen single-round study,including completed runs.

The archived native trainer is left unchanged. Completed model freeze is never
rewritten after the holdout index has been appended.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return sha256(p.read_bytes()).hexdigest()
def alive(pid):
    if os.name=='nt':
        r=subprocess.run(['powershell','-NoProfile','-Command',f'Get-CimInstance Win32_Process -Filter "ProcessId={int(pid)}" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'],check=True,capture_output=True,text=True)
        return json.loads(r.stdout) if r.stdout.strip() else None
    try:os.kill(int(pid),0);return dict(ProcessId=pid)
    except ProcessLookupError:return None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output.resolve();plan=read(root/'formal-plan.json')
    for name,h in plan['sources'].items():
        if digest(Path(name))!=h:raise ValueError('Frozen scientific source changed: '+name)
    freeze=root/'model-freeze.json'
    if freeze.exists():
        frozen=read(freeze);summary=read(root/'training-summary.json')
        assert frozen['formal_plan_sha256']==digest(root/'formal-plan.json')
        assert len(frozen['selected_models'])==len(summary['jobs'])==12 and summary['status']=='completed'
        for row in frozen['selected_models']:assert digest(root/row['path'])==row['sha256']
        for row in summary['jobs']:assert row['episodes']==2000 and row['training_joint_steps']==120000 and row['updates']==1993
        original=root/'training-index-frozen.json'
        if original.exists():assert digest(original)==frozen['training_input_sha256']
        print('Already completed:all12frozen models verified;no training,no evaluation,no metadata rewrite.');return
    assert not any(e['split']=='holdout' for e in read(root/'index.json'))
    lock=root/'execution.lock'
    if lock.exists():
        record=read(lock);live=alive(record['pid'])
        if live:
            print(json.dumps(dict(status='existing_process_alive_no_duplicate_start',process=live)));return
        history=root/'retired-locks';history.mkdir(exist_ok=True)
        target=history/f'execution-{int(time.time_ns())}.json'
        assert target.resolve().is_relative_to(root)
        lock.rename(target)
    subprocess.run([sys.executable,'-u','-m','hyra_learning.sciml_study','train',str(root),'--device',plan['runtime']['device']],cwd=ROOT,check=True)

if __name__=='__main__':main()
