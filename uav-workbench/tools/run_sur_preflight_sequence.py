"""Run the registered six-episode checkpoint and resume to twelve, once."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    output = ROOT/'generated/sur-preflight-20261025'
    state = ROOT/'generated/sur-preflight-supervisor.json'
    lock = ROOT/'generated/sur-preflight-supervisor.lock'
    token = uuid.uuid4().hex
    with lock.open('x', encoding='utf-8') as file:
        json.dump(dict(pid=os.getpid(), token=token), file)
    try:
        if output.exists():
            raise ValueError('Preserve existing preflight; this sequence must not launch twice')
        write(state, dict(stage='first_six_episodes', pid=os.getpid()))
        subprocess.run([sys.executable, '-u', 'learning.py', 'preflight', str(output), '--stop-after', '6'], cwd=ROOT, check=True)
        checkpointed = read(output/'summary.json')
        assert checkpointed['status'] == 'checkpointed_for_registered_resume'
        pointers = {}
        for method in ('QMIX', 'Sur-QMIX-adapted'):
            pointer = read(output/method/'latest.json')
            assert pointer['episode'] == 6
            assert sha256((output/method/pointer['path']).read_bytes()).hexdigest() == pointer['sha256']
            pointers[method] = pointer
        write(output/'planned-resume.json', dict(state='six_completed_each_before_resume', checkpoints=pointers,
                                                supervisor_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
        write(state, dict(stage='resume_through_twelve', pid=os.getpid()))
        subprocess.run([sys.executable, '-u', 'learning.py', 'preflight', str(output), '--resume'], cwd=ROOT, check=True)
        summary = read(output/'summary.json')
        assert summary['status'] == 'completed'
        for result in summary['methods']:
            assert result['episodes'] == 12 and result['training_joint_steps'] == 720
            assert result['evaluation_joint_steps'] == 180 and result['updates'] == 9
            assert result['segments'][0]['first_episode'] == 0 and result['segments'][0]['last_episode'] == 6
            assert result['segments'][1]['first_episode'] == 6 and result['segments'][1]['last_episode'] == 12
        capacities = [r['capacity'] for r in summary['methods']]
        assert capacities[0] == capacities[1]
        write(state, dict(stage='completed', pid=os.getpid(), training_joint_steps=1440, evaluation_joint_steps=360,
                          limitation='Preflight execution and resources only; formal physical-model study remains incomplete.'))
    except BaseException as error:
        write(state, dict(stage='stopped', pid=os.getpid(), reason=type(error).__name__+': '+str(error)))
        raise
    finally:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()


if __name__ == '__main__':
    main()
