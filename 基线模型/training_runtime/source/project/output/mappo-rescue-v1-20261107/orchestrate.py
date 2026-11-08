"""Serial registered comparison runner. It does nothing unless --execute.

--dry-run prints the planned stages and never launches subprocesses or reads
cross-domain scenarios. Existing incomplete artifacts are never overwritten.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
SDK=Path(r'C:\Users\李\Desktop\hyra\HYRA-referee-20261106.1')
SEEDS=(77126,77127,77128)
DEV=(41004,41005,42012,42015,51001,51002,41006,41007,42017,42018,51003,51004)
LOCAL=('core.py','train.py','demo.py','micro.py','teacher_multistep.py','evaluate.py',
       'policy_adapter.py','provenance.py','audit_results.py','summarize.py','eval_extra.py',
       'protocol_candidate.json','orchestrate.py')
EXTRA={(p,f,r) for p,f in ((72110001,'ordinary'),(95851001,'ordinary'),(88520001,'continuous'),
    (88520001,'rewet'),(95861001,'continuous'),(95861001,'rewet')) for r in (0,1)}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def source_checkpoint(seed):
    return HERE.parent/'mappo-search-v1-20261107'/f'local_mask_{seed}'/'checkpoint-new-025600.pt'


def build_plan(python,panel):
    jobs=[];evaluation_paths={}
    def add(name,kind,output,script,args,seed=None,**kwargs):
        jobs.append(dict(name=name,kind=kind,output=str(output),seed=seed,
            cmd=[str(python),'-B',str(HERE/script),*map(str,args)],**kwargs))
    def eval_and_audit(seed,arm,checkpoint,zero=False,micro=64):
        tag=f'{seed}_{arm}';result=HERE/f'formal_{tag}_dev.json';audit=HERE/f'formal_{tag}_replay.json'
        args=['--seed',seed,'--maps','development','--micro-cases',micro,'--output',result]
        if checkpoint is not None:args+=['--checkpoint',checkpoint]
        if zero:args+=['--zero-rescue']
        add('eval_'+tag,'evaluation',result,'evaluate.py',args,seed=seed,
            checkpoint=str(checkpoint) if checkpoint else None,zero=zero,micro=micro)
        aa=['--result',result,'--output',audit]
        verify=seed==77126 and checkpoint is not None and not zero
        if verify:aa+=['--verify-policy','--device','cuda:0']
        add('audit_'+tag,'audit',audit,'audit_results.py',aa,seed=seed,result=str(result),policy_verify=verify)
        evaluation_paths[tag]=str(result)
    for seed in SEEDS:
        demo=HERE/f'candidate_demo_{seed}';demo_cp=demo/'checkpoint.pt'
        demo_ppo=HERE/f'candidate_ppo_demo_{seed}';ppo_cp=demo_ppo/'checkpoint-008192.pt'
        nodemo=HERE/f'candidate_ppo_no_demo_{seed}';nodemo_cp=nodemo/'checkpoint-008192.pt'
        add(f'demo_{seed}','demo',demo,'demo.py',
            ['--seed',seed,'--cases',512,'--updates',1000,'--batch',32,'--kl-positive',.02,
             '--kl-negative',1,'--search-steps-per-map',480,'--positive-fraction',.5,'--output',demo],seed=seed)
        eval_and_audit(seed,'demo_before_ppo',demo_cp)
        add(f'ppo_no_demo_{seed}','ppo',nodemo,'train.py',
            ['--seed',seed,'--steps',8192,'--rollout',128,'--checkpoint-every',8,'--output',nodemo],
            seed=seed,resume=None)
        eval_and_audit(seed,'no_demo_after_ppo',nodemo_cp)
        add(f'ppo_demo_{seed}','ppo',demo_ppo,'train.py',
            ['--seed',seed,'--steps',8192,'--rollout',128,'--checkpoint-every',8,'--output',demo_ppo,'--resume',demo_cp],
            seed=seed,resume=str(demo_cp))
        eval_and_audit(seed,'demo_after_ppo',ppo_cp)
        eval_and_audit(seed,'search_source',None,micro=0)
        if seed==77126:eval_and_audit(seed,'demo_after_ppo_zero_rescue',ppo_cp,zero=True,micro=0)
    extra=HERE/'formal_extra_77126.json';extra_audit=HERE/'formal_extra_77126_replay.json'
    fixed=HERE/'candidate_ppo_demo_77126'/'checkpoint-008192.pt'
    add('extra_77126','extra',extra,'eval_extra.py',
        ['--panel',panel,'--checkpoint',fixed,'--seed',77126,'--output',extra],seed=77126,checkpoint=str(fixed),panel=str(panel))
    add('audit_extra_77126','extra_audit',extra_audit,'orchestrate.py',
        ['--audit-extra',extra,'--panel',panel,'--audit-output',extra_audit],seed=77126,result=str(extra),panel=str(panel))
    for arm in ('demo_before_ppo','no_demo_after_ppo','demo_after_ppo','search_source'):
        paths=[evaluation_paths[f'{seed}_{arm}'] for seed in SEEDS]
        out=HERE/f'formal_summary_{arm}.json'
        add('summary_'+arm,'summary',out,'summarize.py',['--inputs',*paths,'--output',out],inputs=paths)
    add('summary_zero','summary',HERE/'formal_summary_zero.json','summarize.py',
        ['--inputs',evaluation_paths['77126_demo_after_ppo_zero_rescue'],'--output',HERE/'formal_summary_zero.json'],
        inputs=[evaluation_paths['77126_demo_after_ppo_zero_rescue']])
    add('summary_extra','summary',HERE/'formal_summary_extra.json','summarize.py',
        ['--inputs',extra,'--output',HERE/'formal_summary_extra.json'],inputs=[str(extra)])
    return jobs


def manifest_ok(folder,seed,resume=None):
    manifest=read(folder/'source_manifest.json');inputs=manifest['inputs']
    for path,digest in inputs.items():
        if not Path(path).is_file() or sha(path)!=digest:raise ValueError('Changed/missing source '+path)
    for path,digest in manifest['snapshots'].items():
        if not (folder/path).is_file() or sha(folder/path)!=digest:raise ValueError('Changed/missing snapshot '+path)
    expected=source_checkpoint(seed).resolve()
    if inputs.get(str(expected))!=sha(expected):raise ValueError('Wrong pretrained search source')
    for name in LOCAL[:8]:
        p=(HERE/name).resolve()
        if inputs.get(str(p))!=sha(p):raise ValueError('Missing/changed locked code '+name)
    if resume is not None:
        path=Path(resume).resolve()
        if inputs.get(str(path))!=sha(path):raise ValueError('PPO resume identity mismatch')
    elif any('candidate_demo_' in p and p.endswith('checkpoint.pt') for p in inputs):
        raise ValueError('Unexpected demonstration checkpoint in no-demo source')
    return sha(folder/'source_manifest.json')


def train_ok(job):
    import torch
    folder=Path(job['output']);seed=job['seed']
    if job['kind']=='demo':
        log=read(folder/'demo_log.json');cp=folder/'checkpoint.pt'
        if log.get('source_integrity')!='PASS' or log['checkpoint_sha256']!=sha(cp):raise ValueError('Demo not complete')
        if (log['cases'],log['updates'],log['batch'],log['example_draws'])!=(512,1000,32,32000):raise ValueError('Demo budget mismatch')
        collection=log['train_map_collection']
        if len(log['losses'])!=1000 or [(r['parent'],r['roster']) for r in collection]!=[(41001,0),(42002,0),(41002,0),(42004,0)]:
            raise ValueError('Demo updates or training-map identity mismatch')
        if any(not 0<r['joint_steps']<=480 for r in collection):raise ValueError('Demo collection budget mismatch')
        state=torch.load(cp,map_location='cpu',weights_only=False)
        config=dict(cases=512,batch=32,kl_positive=.02,kl_negative=1.,search_steps_per_map=480,positive_fraction=.5)
        if state['seed']!=seed or state['joint']!=0 or state['demo_updates']!=1000 or state['demo_config']!=config:
            raise ValueError('Demo checkpoint identity mismatch')
        for key,name in (('demo_teacher_sha256','micro.py'),('demo_planner_sha256','teacher_multistep.py'),('demo_code_sha256','demo.py')):
            if state[key]!=sha(HERE/name):raise ValueError('Demo teacher/code identity mismatch '+name)
        source=manifest_ok(folder,seed)
    else:
        status=read(folder/'status.json');cp=folder/'checkpoint-008192.pt'
        if status.get('phase')!='COMPLETED' or status.get('source_integrity')!='PASS':raise ValueError('PPO not complete')
        if (status['joint'],status['updates'],status['checkpoint_sha256'])!=(8192,64,sha(cp)):
            raise ValueError('PPO budget or final checkpoint mismatch')
        updates=read(folder/'updates.json')
        if len(updates)!=64 or any(r['joint']!=(i+1)*128 or len(r['ppo'])!=5 for i,r in enumerate(updates)):
            raise ValueError('PPO rollout or epoch budget mismatch')
        with gzip.open(folder/'trace.jsonl.gz','rt',encoding='utf8') as f:
            if sum(1 for _ in f)!=8192:raise ValueError('PPO trace length mismatch')
        state=torch.load(cp,map_location='cpu',weights_only=False)
        if state['seed']!=seed or state['joint']!=8192 or state['updates']!=64:raise ValueError('PPO checkpoint identity mismatch')
        source=manifest_ok(folder,seed,job['resume'])
    if state['schema']!='MAPPO_RESCUE_V1' or state['source_sha256']!=sha(source_checkpoint(seed)):
        raise ValueError('Checkpoint source mismatch')
    for name,digest in state['code_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Checkpoint code mismatch')
    return dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),source_manifest_sha256=source)


def evaluation_ok(job):
    output=Path(job['output']);data=read(output)
    if data['seed']!=job['seed'] or bool(data['zero_rescue'])!=job['zero']:raise ValueError('Evaluation identity mismatch')
    cp=job['checkpoint']
    if cp is None:
        if data['checkpoint'] is not None:raise ValueError('Expected source search policy')
    elif data['checkpoint_sha256']!=sha(cp):raise ValueError('Evaluation checkpoint mismatch')
    pairs=[(r['parent'],r['roster'],r['mode']) for r in data['full']]
    expected={(p,0,m) for p in DEV for m in ('sampled','greedy')}
    if len(pairs)!=24 or set(pairs)!=expected:raise ValueError('Not complete 12-map dual-mode development evaluation')
    if any(r['evaluation_seed']!=581000+r['parent'] for r in data['full']):raise ValueError('Evaluation RNG mismatch')
    for mode in ('sampled','greedy'):
        rows=data['micro'][mode]
        if len(rows)!=3*job['micro']:raise ValueError('Micro evaluation incomplete')
        for kind in ('near','aligned','unaligned'):
            ids=[r['seed'] for r in rows if r['kind']==kind]
            if len(ids)!=job['micro'] or set(ids)!=set(range(200000,200000+job['micro'])):
                raise ValueError('Micro kind or holdout seeds incomplete')
    return dict(output_sha256=sha(output),checkpoint_sha256=data['checkpoint_sha256'],episodes=len(pairs))


def audit_ok(job):
    result=read(job['output']);source=Path(job['result'])
    if result['status']!='PASS' or result['source_sha256']!=sha(source):raise ValueError('Audit source mismatch')
    expected=read(source)['full']
    if len(result['replays'])!=len(expected) or not all(r['all_metrics_exact'] for r in result['replays']):
        raise ValueError('Incomplete replay')
    if job['policy_verify'] and (len(result['policy_reproductions'])!=2 or not all(r['policy_actions_exact'] for r in result['policy_reproductions'])):
        raise ValueError('Adapter action reproduction incomplete')
    return dict(output_sha256=sha(job['output']),audited_result_sha256=sha(source))


def extra_ok(job):
    result=read(job['output']);rows=result['records']
    keys={(r['parent'],r['family'],r['roster'],r['mode'],r['arm']) for r in rows}
    expected={(p,f,r,m,a) for p,f,r in EXTRA for m in ('sampled','greedy') for a in ('candidate','search_source')}
    if len(rows)!=48 or keys!=expected or result['checkpoint_sha256']!=sha(job['checkpoint']):raise ValueError('Incomplete extra evaluation')
    panel=Path(job['panel'])
    if result['panel_sha256']!=sha(panel/'panel.json') or result['panel_manifest_sha256']!=sha(panel/'SHA256.json'):
        raise ValueError('Extra panel identity mismatch')
    if result['source_policy_checkpoint_sha256']!=sha(source_checkpoint(job['seed'])):raise ValueError('Extra search source mismatch')
    if not all(r['official_replay_metrics_exact'] for r in rows):raise ValueError('Missing built-in replay')
    return dict(output_sha256=sha(job['output']),checkpoint_sha256=sha(job['checkpoint']),episodes=48)


def check_output(job):
    if job['kind'] in ('demo','ppo'):return train_ok(job)
    if job['kind']=='evaluation':return evaluation_ok(job)
    if job['kind']=='audit':return audit_ok(job)
    if job['kind']=='extra':return extra_ok(job)
    if job['kind']=='extra_audit':
        result=read(job['output'])
        if result['status']!='PASS' or result['result_sha256']!=sha(job['result']) or len(result['cases'])!=48:
            raise ValueError('Incomplete independent extra replay')
        return dict(output_sha256=sha(job['output']),audited_result_sha256=sha(job['result']))
    if job['kind']=='summary':
        data=read(job['output'])
        if [x['sha256'] for x in data['inputs']]!=[sha(x) for x in job['inputs']]:raise ValueError('Summary input mismatch')
        return dict(output_sha256=sha(job['output']))
    raise ValueError('Unknown stage')


def run_extra_audit(result_path,panel,output):
    if output.exists() or not output.resolve().is_relative_to(HERE):raise ValueError('New experiment output required')
    result=read(result_path);panel=panel.resolve()
    if sha(panel/'panel.json')!=result['panel_sha256'] or sha(panel/'SHA256.json')!=result['panel_manifest_sha256']:
        raise ValueError('Extra panel identity changed')
    spec=importlib.util.spec_from_file_location('orchestration_official_replay',SDK/'run.py')
    sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk)
    cases={(r['parent'],r['family'],r['roster']):r for r in read(panel/'panel.json')}
    manifest=read(panel/'SHA256.json');audits=[]
    for row in result['records']:
        key=(row['parent'],row['family'],row['roster'])
        if key not in EXTRA:raise ValueError('Unregistered case')
        case=cases[key]
        for field in ('water','population'):
            if sha(panel/case[field])!=manifest[case[field]]:raise ValueError('Extra input hash mismatch')
        import numpy as np
        tape=np.asarray([t['moves']+t['targets'] for t in row['trace']],dtype=np.int64)
        metrics,_=sdk.evaluate(panel,case,tape=tape)
        if metrics!=row['metrics']:raise ValueError('Extra official replay mismatch '+case['id'])
        audits.append(dict(id=case['id'],arm=row['arm'],mode=row['mode'],steps=len(tape),all_metrics_exact=True))
        print('EXTRA REPLAY PASS',case['id'],row['arm'],row['mode'],flush=True)
    if len(audits)!=48:raise ValueError('Extra replay incomplete')
    write(output,dict(status='PASS',result_sha256=sha(result_path),panel_sha256=sha(panel/'panel.json'),
                      official_sdk_sha256=sha(SDK/'run.py'),cases=audits))


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true');mode.add_argument('--dry-run',action='store_true');mode.add_argument('--audit-extra',type=Path)
    parser.add_argument('--panel',type=Path);parser.add_argument('--audit-output',type=Path)
    parser.add_argument('--state-dir',type=Path,default=HERE/'formal_orchestration_v1')
    a=parser.parse_args()
    if a.audit_extra:
        if not a.panel or not a.audit_output:parser.error('--audit-extra requires --panel and --audit-output')
        return run_extra_audit(a.audit_extra,a.panel,a.audit_output)
    if not (a.execute or a.dry_run):parser.error('choose --dry-run or --execute')
    if a.execute and not a.panel:parser.error('--execute requires explicitly supplied dev80 --panel')
    panel=a.panel.resolve() if a.panel else Path('<dev80-panel-required-at-execution>')
    if a.execute and not ((panel/'panel.json').is_file() and (panel/'SHA256.json').is_file()):
        parser.error('dev80 panel directory must contain panel.json and SHA256.json')
    jobs=build_plan(Path(sys.executable).resolve(),panel)
    if a.dry_run:
        print(json.dumps(dict(mode='dry-run',registered_seeds=list(SEEDS),stages=len(jobs),
            note='No subprocesses, no data or checkpoint contents read; an existing artifact is not assumed complete.',
            jobs=[dict(j,output_exists=Path(j['output']).exists()) for j in jobs]),indent=2,ensure_ascii=False))
        return
    state=a.state_dir.resolve()
    if not state.is_relative_to(HERE):parser.error('state directory must be inside experiment')
    state.mkdir(parents=True,exist_ok=True)
    lock=state/'runner.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Runner lock exists; inspect prior runner and stale lock before manually removing it')
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    try:
        frozen={name:sha(HERE/name) for name in LOCAL}
        source={str(seed):sha(source_checkpoint(seed)) for seed in SEEDS}
        plan=state/'plan.json';plan_data=dict(jobs=jobs,local_sha256=frozen,source_checkpoint_sha256=source,
                                               protocol='protocol_candidate.json',python=sys.executable)
        if plan.exists():
            if read(plan)!=plan_data:raise ValueError('Existing orchestration plan or source hashes differ')
        else:write(plan,plan_data)
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
        for job in jobs:
            for name,digest in frozen.items():
                if sha(HERE/name)!=digest:raise ValueError('Orchestration source changed: '+name)
            for seed,digest in source.items():
                if sha(source_checkpoint(int(seed)))!=digest:raise ValueError('Search checkpoint changed: '+seed)
            meta=state/(job['name']+'.json');output=Path(job['output'])
            previous=read(meta) if meta.exists() else None
            if output.exists():
                try:details=check_output(job)
                except Exception as exc:raise RuntimeError('Existing stage incomplete or identity mismatch, left untouched: '+job['name']+'; '+str(exc)) from exc
                if job['kind'] not in ('demo','ppo'):
                    if not previous or previous.get('phase')!='COMPLETED' or previous.get('details')!=details:
                        raise RuntimeError('Existing result has no matching completed runner provenance: '+job['name'])
                write(meta,dict(phase='COMPLETED',job=job,details=details,reused=True,runner_pid=os.getpid()))
                print('REUSE',job['name'],flush=True);continue
            if previous and previous.get('phase')=='COMPLETED':raise RuntimeError('Completed stage output missing: '+job['name'])
            stdout=state/(job['name']+'.stdout.log');stderr=state/(job['name']+'.stderr.log')
            if stdout.exists() or stderr.exists():raise RuntimeError('Previous stage logs exist; inspect before retry: '+job['name'])
            with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
                flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
                proc=subprocess.Popen(job['cmd'],cwd=str(HERE),env=env,stdout=so,stderr=se,creationflags=flags)
                write(meta,dict(phase='RUNNING',job=job,pid=proc.pid,runner_pid=os.getpid(),started_unix=time.time(),
                                stdout=str(stdout),stderr=str(stderr)))
                print('START',job['name'],'PID',proc.pid,flush=True)
                code=proc.wait()
            if code:
                write(meta,dict(phase='FAILED',job=job,pid=proc.pid,exit_code=code,stdout=str(stdout),stderr=str(stderr)))
                raise RuntimeError('Stage failed: '+job['name'])
            details=check_output(job)
            for name,digest in frozen.items():
                if sha(HERE/name)!=digest:raise ValueError('Orchestration source changed: '+name)
            write(meta,dict(phase='COMPLETED',job=job,pid=proc.pid,exit_code=0,details=details,completed_unix=time.time()))
            print('PASS',job['name'],flush=True)
        write(state/'status.json',dict(phase='COMPLETED',runner_pid=os.getpid(),stages=len(jobs),plan_sha256=sha(plan)))
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
