"""Verify bundled bytes, all published actors, and optional real CPU replays."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    bundle = ROOT / 'runtime-assets'
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    rows = manifest['files'] + manifest['frozen_sources']
    for row in rows:
        data = (bundle / row['path']).read_bytes()
        assert len(data) == row['bytes'], row['path']
        assert hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
    import console_scale100_contract as scale
    import console_signal_routes_contract as routes
    for task in scale.TASKS:
        for group in scale.LABELS:
            for seed in scale.TRAINING_SEEDS:
                scale.model_record(task, group, seed)
    for method in routes.LABELS:
        for seed in routes.TRAINING_SEEDS:
            routes.model_record(method, seed)
    for contract in (scale, routes):
        for row in contract.scene_records():
            assert scale.digest(row['path']) == row['source_sha256'], row['id']
    if args.replay:
        for module, contract, group in [('console_scale100_runner', scale, 'B3'),
                                         ('console_signal_routes_runner', routes, 'R1')]:
            job = dict(scene=contract.scene_records()[0]['id'], task='exploration_rescue',
                       group=group, mode='model', training_seed=11)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'job.json'
                path.write_text(json.dumps(job), encoding='utf-8')
                code = ('import json; from ' + module + ' import execute_job; '
                        'r=execute_job(json.load(open(' + repr(str(path)) + '))); '
                        'assert r["audit"]["audit_passed"]; print("CPU replay passed: ' + group + '")')
                subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)
    print(f'Verified {len(rows)} files, 60 published actors, and all bundled interactive scenes')


if __name__ == '__main__':
    main()
