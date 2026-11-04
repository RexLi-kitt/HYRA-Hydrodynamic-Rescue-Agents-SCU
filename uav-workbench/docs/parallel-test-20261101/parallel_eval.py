"""Two-process continuation of the SAME frozen test, preserving completed rows."""
import argparse
import concurrent.futures as cf
import gc
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
OLD=ROOT/'uav-strict-eval-20261101';RUN=OLD/'run01'
def engine():
    spec=importlib.util.spec_from_file_location('serial_frozen',OLD/'strict_eval.py')
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def key(r):return (r['candidate'],r['case'],r['assignment'],r['variant'],r['mode'],r['action_seed'])
def task_key(t):
    c,case,role,assignment,variant,mode,seed=t
    return (c['id'],case['id'],assignment,variant,mode,seed)

def initialize(root):
    global V,MODELS,CURRENT
    V=engine();V.bootstrap();V.torch.set_num_threads(2)
    V.RUN=Path(root)/str(os.getpid());(V.RUN/'traces').mkdir(parents=True,exist_ok=True)
    V.emit=lambda *a,**kw:None;MODELS=None;CURRENT=None

def evaluate_task(task):
    global MODELS,CURRENT
    c,case,role,assignment,variant,mode,seed=task
    if CURRENT!=c['id']:
        MODELS=None;gc.collect()
        if c['family']!='rule':
            assert V.sha(c['path'])==c['sha256']
            ck=V.torch.load(c['path'],map_location='cpu',weights_only=False)
            MODELS,_=V.e.restore(ck);del ck
        elif c['id']=='uniform_service':
            m=V.e.Learner('happo',V.e.DIM)
            with V.torch.no_grad():
                for p in m.parameters():p.zero_()
            m.eval();MODELS=[m]
        CURRENT=c['id']
    return V.trial(c,MODELS,case,tuple(role),assignment,variant,mode,seed,0)

def make_tasks(frozen,cases):
    tasks=[]
    for c in frozen['candidates']:
        roles=[(0,0,0,0)] if c['family']=='single' else [(0,1,1,0),(1,0,0,1)]
        for case in cases:
            for i,role in enumerate(roles):
                for variant in ('full','no-cover'):tasks.append((c,case,role,i,variant,'fixed',510001))
        if c['id'] in frozen['validation_chosen_leaders'].values():
            for case in cases[:50]:
                for i,role in enumerate(roles):
                    for seed in (510101,510102,510103):tasks.append((c,case,role,i,'full','sample',seed))
    for cid,variant in [('pure_planner','planner'),('uniform_service','no-cover')]:
        c=dict(id=cid,family='rule')
        for case in cases:tasks.append((c,case,(0,0,0,0),0,variant,'fixed' if cid=='pure_planner' else 'sample',510001))
    return tasks

def preflight():
    v=engine();v.bootstrap()
    frozen=read(RUN/'frozen-candidates.json')
    train=next(r for r in read(v.SINGLE/'run01/dataset.json')['cases'] if r['split']=='train')
    water=v.np.load(train['path'],mmap_mode='r');people=v.e.Env(water,777551).people
    case=dict(id='training-only-performance',stratum='iid50',path=train['path'],layout_seed=777551,people=people.tolist())
    # Four single/pair architectures plus two rule controls, both action modes.
    names=['happo','hasac','mat','hatrpo_hasac','hasac_mat','coma']
    tasks=[]
    for i,name in enumerate(names):
        c=next(c for c in frozen['candidates'] if c['id']==name)
        role=(0,0,0,0) if c['family']=='single' else (1,0,0,1)
        tasks.append((c,case,role,0,'full' if i%2==0 else 'no-cover','fixed' if i%2==0 else 'sample',510001))
    tasks += [(dict(id='pure_planner',family='rule'),case,(0,0,0,0),0,'planner','fixed',510001),
              (dict(id='uniform_service',family='rule'),case,(0,0,0,0),0,'no-cover','sample',510001)]
    v.write(BASE/'preflight-protocol.json',dict(tasks=[task_key(t) for t in tasks],data='existing TRAIN only',workers=2,torch_threads_per_worker=2,
      acceptance='Exact metrics and every trace array; no weight changes; bounded memory; same seeds and sample order within rollout',
      source_sha256=v.sha(OLD/'strict_eval.py'),wrapper_sha256=v.sha(__file__)))
    initialize(BASE/'preflight-serial');start=time.perf_counter();serial=[evaluate_task(t) for t in tasks];serial_s=time.perf_counter()-start
    start=time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(BASE/'preflight-parallel'),)) as pool:
        parallel=list(pool.map(evaluate_task,tasks))
    parallel_s=time.perf_counter()-start
    checks=[]
    for a,b in zip(serial,parallel):
        assert {k:x for k,x in a.items() if k!='trace'}=={k:x for k,x in b.items() if k!='trace'},key(a)
        with v.np.load(a['trace']) as x,v.np.load(b['trace']) as y:
            assert x.files==y.files and all(v.np.array_equal(x[k],y[k]) for k in x.files),key(a)
        checks.append(dict(key=key(a),metrics_and_all_arrays_exact=True))
    result=dict(status='passed',checks=checks,serial_s=serial_s,parallel_s=parallel_s,startup_included=True,
      speedup=serial_s/parallel_s,test_fields_used=False,wrapper_sha256=v.sha(__file__))
    v.write(BASE/'preflight.json',result);print(json.dumps(result),flush=True)

def run():
    v=engine()
    assert read(BASE/'preflight.json')['status']=='passed'
    assert read(BASE/'preflight.json')['wrapper_sha256']==v.sha(__file__)
    assert read(RUN/'status.json')['status']=='manual_stopped','Old evaluator must exit at record boundary first'
    assert not v.stop(),'Handoff stop marker must be archived before continuation'
    # Provenance: original source, all fields and candidates are unchanged.
    original=read(RUN/'registration.json')
    for p,h in original['source_hashes'].items():assert v.sha(p)==h,p
    frozen=read(RUN/'frozen-candidates.json');cases=read(RUN/'scenario-manifest.json')
    for c in frozen['candidates']:assert v.sha(c['path'])==c['sha256']
    for c in cases:assert v.sha(c['path'])==c['sha256']
    tasks=make_tasks(frozen,cases);assert len(tasks)==5630
    raw=(RUN/'results.jsonl').read_bytes();assert raw.endswith(b'\n')
    (BASE/'serial-results-prefix.jsonl').write_bytes(raw)
    rows=[json.loads(x) for x in raw.splitlines() if x.strip()];by_key={key(r):r for r in rows}
    assert len(by_key)==len(rows);expected={task_key(t) for t in tasks};assert len(expected)==5630 and set(by_key)<=expected
    pending=[t for t in tasks if task_key(t) not in by_key]
    start=time.monotonic()
    v.write(BASE/'registration.json',dict(user_request='Accelerate remaining evaluation',workers=2,threads_each=2,old_registration_sha256=v.sha(RUN/'registration.json'),
      wrapper_sha256=v.sha(__file__),preflight_sha256=v.sha(BASE/'preflight.json'),completed_prefix_records=len(rows),remaining=len(pending),
      prefix_sha256=v.sha(BASE/'serial-results-prefix.jsonl'),frozen_sha256=v.sha(RUN/'frozen-candidates.json'),scenario_manifest_sha256=v.sha(RUN/'scenario-manifest.json'),
      preserves='All5630 predefined records; exact seeds, checkpoints, rules, full/no-cover controls, all role assignments; no test-based selection',
      original_time_budget='Shared original8h test budget, not reset',original_elapsed_s=read(BASE/'handoff.json')['elapsed_test_s']))
    elapsed_before=read(BASE/'handoff.json')['elapsed_test_s']
    def guard():
        v.guard(start-elapsed_before)
    def emit(**kw):v.emit('evaluating_parallel',completed_records=len(rows),total_records=5630,workers=2,**kw)
    emit();v.mirror()
    with cf.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(BASE/'workers'),)) as pool:
        todo=iter(pending);running={}
        for _ in range(4):
            task=next(todo,None)
            if task is not None:running[pool.submit(evaluate_task,task)]=task
        while running:
            guard();finished,_=cf.wait(running,timeout=5,return_when=cf.FIRST_COMPLETED)
            for future in finished:
                task=running.pop(future);r=future.result();assert key(r)==task_key(task) and key(r) not in by_key
                source=Path(r['trace']);target=RUN/'traces'/source.name;assert not target.exists()
                shutil.copy2(source,target);r['trace']=str(target)
                with (RUN/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
                rows.append(r);by_key[key(r)]=r;emit(current=r['candidate'],case=r['case'],variant=r['variant'])
                if len(rows)%100==0:v.mirror()
                task=next(todo,None)
                if task is not None:running[pool.submit(evaluate_task,task)]=task
    assert set(by_key)==expected and len(rows)==5630
    # Original order maintains identical grouping and bootstrap RNG interpretation.
    ordered=[by_key[task_key(t)] for t in tasks]
    v.bootstrap();v.aggregate(ordered,frozen)
    for p,h in original['source_hashes'].items():assert v.sha(p)==h
    for c in frozen['candidates']:assert v.sha(c['path'])==c['sha256']
    v.write(RUN/'integrity-audit.json',dict(status='passed',all_trace_metrics_reconciled=True,all_weights_unchanged=True,records=5630,training_used_test=False,
      parallel_addendum_sha256=v.sha(BASE/'registration.json'),unique_complete_task_set=True,aggregation_original_order=True))
    v.emit('completed',records=len(rows),elapsed_s=elapsed_before+time.monotonic()-start,workers=2);v.mirror()
    subprocess.run([sys.executable,str(v.SINGLE/'refresh.py'),'--workbook'],check=True)

if __name__=='__main__':
    import sys
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.preflight:preflight()
    elif args.execute:
        try:run()
        except BaseException as ex:
            v=engine();v.write(BASE/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));v.emit('manual_stopped' if isinstance(ex,InterruptedError) else 'failed',error=repr(ex));v.mirror();raise
    else:raise RuntimeError('Explicit mode required')
