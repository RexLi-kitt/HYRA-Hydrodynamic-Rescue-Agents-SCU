"""Resumable complete study controller. Confirmation is inaccessible until seal."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from . import METHODS
from .common import DEFAULT,ROOT,read,write,digest,sources,registration


def register(study):
    study=Path(study); study.mkdir(parents=True,exist_ok=True); path=study/'registration.json'
    if path.exists():
        if read(path)!=registration(): raise ValueError('Preregistered design changed')
    else: write(path,registration())


def children(study,jobs,workers=4):
    study=Path(study); pending=list(jobs); active=[]
    try:
        while pending or active:
            while pending and len(active)<workers:
                job=pending.pop(0); phase,method,seed,episodes,stop=job
                output=study/phase/f'{method}-s{seed}'; progress=output/'progress.json'
                if progress.exists() and read(progress)['status']=='completed':
                    if read(output/'plan.json')['sources']!=sources(): raise ValueError('Completed job source changed')
                    continue
                arguments=[sys.executable,'-u','-B','-m','signal_routes.train','--study',str(study),
                           '--method',method,'--seed',str(seed),'--episodes',str(episodes),'--phase',phase]
                if stop is not None: arguments+=['--stop-after',str(stop)]
                if (output/'latest.json').exists(): arguments+=['--resume']
                logpath=study/'logs'/f'{phase}-{method}-s{seed}.log'; logpath.parent.mkdir(exist_ok=True)
                log=logpath.open('a',encoding='utf-8')
                process=subprocess.Popen(arguments,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append((process,log,job,str(logpath)))
            write(study/'workers.json',dict(controller_pid=os.getpid(),pending=len(pending),
                active=[dict(pid=p.pid,job=list(j),log=path) for p,_,j,path in active],updated_at=time.time()))
            for item in active[:]:
                process,log,job,path=item; code=process.poll()
                if code is None: continue
                log.close(); active.remove(item)
                if code: raise RuntimeError(f'Job {job} exited {code}; inspect {path}')
            if active: time.sleep(2)
    finally:
        # Do not leave untracked children after controller failure.
        for process,log,_,_ in active:
            if process.poll() is None: process.terminate(); process.wait(timeout=30)
            log.close()


def recovery_proof(study):
    import torch
    study=Path(study); path=study/'recovery-proof.json'
    if path.exists():
        if read(path)['sources']!=sources(): raise ValueError('Recovery proof has stale sources')
        return
    children(study,[('recovery-split','R3',11,8,4)],1)
    children(study,[('recovery-split','R3',11,8,None),('recovery-full','R3',11,8,None)],1)
    a=study/'recovery-split/R3-s11'; b=study/'recovery-full/R3-s11'
    left=torch.load(a/'actor.pt',weights_only=True); right=torch.load(b/'actor.pt',weights_only=True)
    equal=all(torch.equal(left[k],right[k]) for k in left)
    histories=read(a/'history.json')==read(b/'history.json')
    if not equal or not histories: raise AssertionError('Cross-process recovery differs')
    write(path,dict(actor_bitwise_equal=equal,history_equal=histories,sources=sources(),episodes=[4,8]))


def offline(study):
    from .offline import collect,fit,diagnostics
    study=Path(study); data=study/'data/dataset.json'
    for split in ('fit','development'):
        path=study/f'{split}-records.npz'
        if not path.exists(): collect(data,split,path)
    if not (study/'parameters.json').exists(): fit(study/'fit-records.npz',study/'parameters.json')
    for condition in ('null','attenuation','directional'):
        path=study/f'parameters-{condition}.json'
        if not path.exists(): fit(study/'fit-records.npz',path,condition)
    if not (study/'offline-diagnostics.json').exists():
        diagnostics(study/'development-records.npz',study/'parameters.json',study/'offline-diagnostics.json')


def full(study,workers):
    from .data import prepare
    from .evaluate import select_task,select_models,confirm,report
    study=Path(study); register(study)
    state=dict(status='running',controller_pid=os.getpid(),stage='checks',started_at=time.time())
    def stage(name):
        state['stage']=name; state['updated_at']=time.time(); write(study/'status.json',state)
        print(f'STAGE {name}',flush=True)
    try:
        stage('checks')
        result=subprocess.run([sys.executable,'-B','-m','signal_routes.checks'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        (study/'checks.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError('Numerical/permission acceptance failed')
        write(study/'checks.json',dict(passed=True,sources=sources()))
        stage('data'); prepare(study/'data')
        stage('offline_fit_and_diagnostics'); offline(study)
        stage('development_task_selection'); select_task(study)
        stage('cross_process_recovery'); recovery_proof(study)
        stage('pilot_8_jobs'); children(study,[('pilot',m,11,128,None) for m in METHODS],workers)
        pilot=[read(study/'pilot'/f'{m}-s11'/'progress.json') for m in METHODS]
        for m,row in zip(METHODS,pilot):
            metrics=row['last']['metrics']
            if row['updates']!=125 or metrics['inference_max_residual']>1e-8: raise ArithmeticError(f'Invalid pilot {m}')
        write(study/'pilot-acceptance.json',dict(passed=True,sources=sources(),jobs=8,score_based_elimination=False))
        stage('formal_24_jobs')
        children(study,[('formal',m,s,1024,None) for m in METHODS for s in (11,29,47)],workers)
        stage('development_selection_and_seal'); select_models(study)
        stage('confirmation_and_replay_audit'); confirm(study)
        stage('report'); report(study)
        state.update(status='completed',stage='completed',finished_at=time.time()); write(study/'status.json',state)
    except Exception as error:
        state.update(status='failed',error=repr(error),traceback=traceback.format_exc(),finished_at=time.time())
        write(study/'status.json',state); raise


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['register','data','offline','task','recovery','pilot','formal','select','confirm','report','all','status'])
    p.add_argument('--study',type=Path,default=DEFAULT); p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); study=a.study.resolve()
    if a.workers<1 or a.workers>4: raise ValueError('Use 1–4 bounded local workers')
    if a.stage=='status': print(read(study/'status.json')); return
    register(study)
    if a.stage=='register': return
    if a.stage=='all': full(study,a.workers); return
    if a.stage=='data':
        from .data import prepare
        prepare(study/'data')
    elif a.stage=='offline': offline(study)
    elif a.stage=='recovery': recovery_proof(study)
    elif a.stage in ('pilot','formal'):
        seeds=(11,) if a.stage=='pilot' else (11,29,47); episodes=128 if a.stage=='pilot' else 1024
        if a.stage=='formal' and not read(study/'pilot-acceptance.json')['passed']: raise ValueError('Pilot gate not passed')
        children(study,[(a.stage,m,s,episodes,None) for m in METHODS for s in seeds],a.workers)
    else:
        from .evaluate import select_task,select_models,confirm,report
        {'task':select_task,'select':select_models,'confirm':confirm,'report':report}[a.stage](study)


if __name__=='__main__': main()
