import importlib.util,json,shutil,hashlib
from pathlib import Path
from datetime import datetime
base=Path(__file__).resolve().parent;root=base.parent
old=root/'uav-parallel4-test-20261101';run=root/'uav-strict-eval-20261101/run01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return tuple(r[k] for k in ('candidate','case','assignment','variant','mode','action_seed'))
assert read(base/'preflight.json')['status']=='passed'
assert read(run/'status.json')['status']=='manual_stopped'
assert read(base/'handoff.json')['old_exited']
spec=importlib.util.spec_from_file_location('more',base/'parallel_eval.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
expected={m.task_key(t) for t in m.make_tasks(read(run/'frozen-candidates.json'),read(run/'scenario-manifest.json'))}
raw=(run/'results.jsonl').read_bytes();assert raw.endswith(b'\n')
(base/'four-worker-results-prefix.jsonl').write_bytes(raw)
rows=[json.loads(x) for x in raw.splitlines()];known={key(r):r for r in rows};assert len(known)==len(rows)
recovered=[]
for path in sorted((old/'workers').glob('*/results.jsonl')):
    for line in path.read_text(encoding='utf8').splitlines():
        r=json.loads(line);k=key(r);assert k in expected
        if k in known:
            assert {a:b for a,b in r.items() if a!='trace'}=={a:b for a,b in known[k].items() if a!='trace'}
            continue
        source=Path(r['trace']);target=run/'traces'/source.name
        if target.exists():assert sha(source)==sha(target)
        else:shutil.copy2(source,target)
        r['trace']=str(target)
        with (run/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
        known[k]=r;recovered.append(k)
audit=dict(timestamp=datetime.now().isoformat(),prior_merged=len(rows),completed_worker_records_recovered=recovered,total_preserved=len(known),
    original_prefix_unchanged=(run/'results.jsonl').read_bytes().startswith(raw),source_prefix_sha256=sha(base/'four-worker-results-prefix.jsonl'),
    note='Old executor completed queued trials while honoring stop; reuse their checked records rather than repeat them.')
(base/'recovery-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(audit))
