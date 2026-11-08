"""Freeze inputs before execution and fail if they change during a run."""
import shutil
import core as c

s=c.search
LOCAL=('core.py','train.py','demo.py','micro.py','teacher_multistep.py','evaluate.py',
       'policy_adapter.py','provenance.py')


def begin(output, seed, resume=None):
    output.mkdir(parents=True,exist_ok=False)
    lock=s.source_lock()
    for name in LOCAL: lock[str(c.HERE/name)]=s.sha(c.HERE/name)
    lock[str(c.source_checkpoint(seed))]=s.sha(c.source_checkpoint(seed))
    if resume: lock[str(resume.resolve())]=s.sha(resume)
    snapshot=output/'source_snapshot';snapshot.mkdir()
    files={}
    for index,path in enumerate(lock):
        from pathlib import Path
        p=Path(path); destination=snapshot/f'{index:03d}_{p.name}'
        shutil.copyfile(p,destination);files[str(destination.relative_to(output))]=lock[path]
    s.json_write(output/'source_manifest.json',dict(inputs=lock,snapshots=files,check='must match at completion'))
    return lock


def finish(lock):
    s.check_lock(lock)
