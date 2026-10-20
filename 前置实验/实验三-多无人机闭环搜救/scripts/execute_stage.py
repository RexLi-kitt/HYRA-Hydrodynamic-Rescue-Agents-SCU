from pathlib import Path
from contextlib import contextmanager
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.workflow import execute_stage
from hyra_exp3.protocol import load_protocol
from hyra_exp3.artifacts import write_json


@contextmanager
def stage_lock(path):
    """Serialize callers for one workflow marker; the OS releases crashed owners."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0,os.SEEK_END)
        if stream.tell()==0:stream.write(b'\0');stream.flush()
        while True:
            stream.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:
                time.sleep(.25)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def request_identity(spec):
    return dict(root=spec['root'],action=spec['action'],
        arguments={k:v for k,v in spec['arguments'].items() if k!='resume'})


if __name__=='__main__':
    spec=load_protocol(sys.argv[1])
    marker=Path(spec['result_path'])
    with stage_lock(marker.with_suffix('.lock')):
        identity=request_identity(spec)
        if marker.exists() and load_protocol(marker)['status']=='completed':
            completed=load_protocol(marker)
            if completed.get('request_identity',identity)!=identity:
                raise ValueError('Concurrent workflow stage has different inputs')
            print(f"Workflow stage already completed: {spec['action']}",flush=True)
        else:
            result=execute_stage(spec['root'],spec['action'],spec['arguments'])
            write_json(marker,dict(status='completed',result=str(result),request_identity=identity))
