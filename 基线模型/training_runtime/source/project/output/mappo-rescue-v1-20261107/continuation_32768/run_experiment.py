"""Execute the registered paired continuation with immutable stage outputs."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
V1=HERE.parent
SEEDS=(77126,77127,77128)
ARMS=('frozen','joint')
ENDPOINTS=(16384,24576,32768)
EXTRA=Path(r'C:\Users\李\Desktop\hyra\HYRA-dev80-20261106.1')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def checkpoint(arm,seed,endpoint):return HERE/arm/f'seed_{seed}'/f'checkpoint-{endpoint:06d}.pt'


def plan():
    jobs=[]
    def add(name,kind,script,args,output,**kw):
        jobs.append(dict(name=name,kind=kind,cmd=[sys.executable,'-B',str(script),*map(str,args)],output=str(output),**kw))
    for endpoint in ENDPOINTS:
      for seed in SEEDS:
       for arm in ARMS:
        prefix=f'{arm}_{seed}_{endpoint:06d}';folder=HERE/arm/f'seed_{seed}';cp=checkpoint(arm,seed,endpoint)
        args=['--arm',arm,'--seed',seed,'--until',endpoint]
        if endpoint>ENDPOINTS[0]:args+=['--resume',checkpoint(arm,seed,endpoint-8192)]
        add(prefix+'_train','train',HERE/'continue_train.py',args,folder/f'stage_{endpoint:06d}.json',
            arm=arm,seed=seed,endpoint=endpoint,checkpoint=str(cp))
        result=folder/f'eval_{endpoint:06d}.json';final=endpoint==32768
        add(prefix+'_eval','eval',V1/'evaluate.py',
            ['--seed',seed,'--checkpoint',cp,'--maps','development' if final else 'tuning',
             '--micro-cases',64 if final else 0,'--output',result],result,
            arm=arm,seed=seed,endpoint=endpoint,checkpoint=str(cp),episodes=24 if final else 6,micro=64 if final else 0)
        audit=folder/f'audit_{endpoint:06d}.json'
        add(prefix+'_audit','audit',V1/'audit_results.py',
            ['--result',result,'--output',audit,'--verify-policy','--device','cuda:0'],audit,
            result=str(result),episodes=24 if final else 6)
    for arm in ARMS:
        cp=checkpoint(arm,77126,32768);result=HERE/arm/'extra_77126.json';audit=HERE/arm/'extra_77126_audit.json'
        add(arm+'_extra','extra',V1/'eval_extra.py',
            ['--panel',EXTRA,'--checkpoint',cp,'--seed',77126,'--output',result],result,checkpoint=str(cp))
        add(arm+'_extra_audit','extra_audit',V1/'orchestrate.py',
            ['--audit-extra',result,'--panel',EXTRA,'--audit-output',audit],audit,result=str(result))
    return jobs


def check(job):
    path=Path(job['output']);r=read(path)
    if job['kind']=='train':
        import torch
        cp=Path(job['checkpoint']);state=torch.load(cp,map_location='cpu',weights_only=False)
        meta=state['paired_continuation'];endpoint=job['endpoint']
        assert r['phase']=='COMPLETED' and r['source_integrity']=='PASS'
        assert r['joint']==endpoint and r['updates']==64 and r['checkpoint_sha256']==sha(cp)
        assert state['joint']==endpoint and state['updates']==endpoint//128
        assert meta['arm']==job['arm'] and state['seed']==job['seed']
        assert meta['source_lock']==read(HERE/'source_lock.json')
        assert meta['parent_sha256']==sha(meta['parent_checkpoint'])
        changes={k:meta['current_groups'][k]!=meta['initial_groups'][k] for k in meta['initial_groups']}
        if job['arm']=='frozen':assert not any(changes[k] for k in changes if k!='rescue')
        else:assert all(changes.values()) and all(n>0 for n in meta['updates_with_gradient'].values())
        assert changes['rescue'] and r['critic_changed']
        trace=Path(r['trace']);assert sha(trace)==r['trace_sha256']
        count=0
        with gzip.open(trace,'rt',encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);count+=1
                assert row['joint']==endpoint-8192+count
        assert count==8192
        return dict(sha256=sha(path),checkpoint_sha256=sha(cp),new_steps=count,parameter_groups_changed=changes)
    if job['kind']=='eval':
        assert r['seed']==job['seed'] and r['checkpoint_sha256']==sha(job['checkpoint'])
        assert r['zero_rescue'] is False and len(r['full'])==job['episodes']
        expected=[41004,42012,51001] if job['episodes']==6 else [41004,41005,42012,42015,51001,51002,41006,41007,42017,42018,51003,51004]
        assert {(x['parent'],x['roster'],x['mode']) for x in r['full']}=={(p,0,m) for p in expected for m in ('sampled','greedy')}
        assert all(x['metrics']['total']==64 and x['metrics']['boundary']==0 for x in r['full'])
        assert all(len(r['micro'][mode])==3*job['micro'] for mode in ('sampled','greedy'))
        return dict(sha256=sha(path),episodes=len(r['full']),micro_cases=sum(len(x) for x in r['micro'].values()))
    if job['kind']=='audit':
        assert r['status']=='PASS' and r['source_sha256']==sha(job['result'])
        assert len(r['replays'])==job['episodes'] and len(r['policy_reproductions'])==2
        assert all(x['all_metrics_exact'] for x in r['replays'])
        assert all(x['policy_actions_exact'] for x in r['policy_reproductions'])
        return dict(sha256=sha(path),replayed=job['episodes'],policy_episodes=2)
    if job['kind']=='extra':
        assert r['status']=='COMPLETED' and r['checkpoint_sha256']==sha(job['checkpoint'])
        assert len(r['records'])==48 and all(x['official_replay_metrics_exact'] for x in r['records'])
        assert r['panel_sha256']==sha(EXTRA/'panel.json') and r['panel_manifest_sha256']==sha(EXTRA/'SHA256.json')
        return dict(sha256=sha(path),episodes=48)
    if job['kind']=='extra_audit':
        assert r['status']=='PASS' and r['result_sha256']==sha(job['result']) and len(r['cases'])==48
        assert all(x['all_metrics_exact'] for x in r['cases'])
        return dict(sha256=sha(path),replayed=48)
    raise ValueError(job['kind'])


def execute():
    jobs=plan();lock=read(HERE/'source_lock.json')
    for path,digest in lock.items():assert sha(path)==digest,path
    logs=HERE/'logs';logs.mkdir(exist_ok=True)
    state_path=HERE/'execution.json'
    if state_path.exists():
        state=read(state_path);assert state['plan']==jobs
    else:state=dict(status='RUNNING',plan=jobs,completed=[],started=time.time())
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    for index,job in enumerate(jobs):
        if index<len(state['completed']):
            assert state['completed'][index]['evidence']==check(job)
            continue
        if Path(job['output']).exists():raise FileExistsError('Unregistered existing stage '+job['output'])
        state.update(status='RUNNING',current=job['name'],current_index=index,stage_started=time.time());write(state_path,state)
        print(f"START {index+1}/{len(jobs)} {job['name']}",flush=True)
        began=time.monotonic()
        with (logs/(job['name']+'.stdout.log')).open('wb') as stdout,(logs/(job['name']+'.stderr.log')).open('wb') as stderr:
            proc=subprocess.run(job['cmd'],cwd=HERE,env=env,stdout=stdout,stderr=stderr,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if proc.returncode:
            state.update(status='FAILED',returncode=proc.returncode);write(state_path,state)
            raise RuntimeError('Stage failed; inspect logs: '+job['name'])
        evidence=check(job)
        state['completed'].append(dict(name=job['name'],seconds=time.monotonic()-began,evidence=evidence))
        write(state_path,state);print('DONE '+job['name'],flush=True)
    for path,digest in lock.items():assert sha(path)==digest,path
    state.update(status='COMPLETED',current=None,finished=time.time());write(state_path,state)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    if a.execute:execute()
    else:print(json.dumps(plan(),indent=2))
