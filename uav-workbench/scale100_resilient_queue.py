"""Bounded, paired auxiliary optimization diagnostic, preserving the v1 queue."""
import os,sys,time,subprocess,json
from pathlib import Path
from scale100_aux_resilient import *
from scale100_budget_pilot import evaluate_segment
from scale100_resilient_forecast import run as forecast

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'
OUT=ROOT/'generated/scale100-aux-resilient-20261026'

def child(args,log,status):
    with log.open('a',encoding='utf-8') as stream:
        proc=subprocess.Popen([sys.executable,'-B','scale100_aux_resilient.py','--dataset',str(DATA),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        status['child_pid']=proc.pid;write(OUT/'queue.json',status);code=proc.wait()
    if code:raise RuntimeError(f'Auxiliary diagnostic failed with exit{code}; see {log}')

def restart_check(status):
    args=['--task','E1','--method','P','--episodes','8'];split=OUT/'restart-split';whole=OUT/'restart-whole'
    child(['--output',str(split),*args,'--stop-after','4'],OUT/'restart.log',status)
    child(['--output',str(split),*args,'--resume'],OUT/'restart.log',status)
    child(['--output',str(whole),*args],OUT/'restart.log',status)
    a=torch.load(split/'actor.pt',weights_only=True);b=torch.load(whole/'actor.pt',weights_only=True)
    exact=all(torch.equal(a[k],b[k]) for k in a);same_history=json.loads((split/'history.json').read_text())==json.loads((whole/'history.json').read_text())
    if not exact or not same_history:raise AssertionError('New optimizer groups are not restart-equivalent')
    write(OUT/'recovery-verification.json',dict(actor_tensors_bitwise_equal=exact,history_equal=same_history,split=[4,4],whole=8,sources=resilient_sources()))

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    write(OUT/'plan.json',dict(schema='hyra-scale100-aux-resilient-development-v4',sources=resilient_sources(),controller_source_sha256=digest(__file__),dataset_sha256=digest(DATA),reason='v3 P decoder has zero active ReLU units and zero latent gradients at all12 fit-only audited states; replace only auxiliary activation by LeakyReLU(.05), preserving v3 settings',tasks=['E1','E2'],methods=['D','P'],seed=11,episodes_per_job=256,evaluation_checkpoints=[64,128,256],main_lr=1e-4,head_lr=1e-3,head_eps=1e-8,auxiliary_activation='leaky_relu_0.05',data_weight=1.,physics_weight=1.,initialization='same fit-only mean U prior for every method; no holdout statistics',formal=False))
    status=dict(status='running',pid=os.getpid(),started_at=time.time(),stage='restart_check',current=None,completed=[],formal_training_started=False);write(OUT/'queue.json',status)
    try:
        restart_check(status)
        for task in ('E1','E2'):
            for method in ('D','P'):
                job=f'{task}-{method}-s11';folder=OUT/job
                for stop in (64,128,256):
                    status.update(stage='training',current=dict(job=job,segment_end=stop));write(OUT/'queue.json',status)
                    args=['--output',str(folder),'--task',task,'--method',method,'--episodes','256','--stop-after',str(stop)]
                    if stop>64:args.append('--resume')
                    child(args,OUT/(job+'.log'),status);summary=evaluate_segment(folder,stop,task,method)
                    spec=json.loads((folder/'plan.json').read_text());progress=json.loads((folder/'progress.json').read_text());actor_path=folder/f'actor-ep{stop:04d}.pt'
                    write(folder/f'actor-ep{stop:04d}.json',dict(**spec,trained_episodes=stop,updates=progress['updates'],actor_sha256=digest(actor_path)))
                    write(OUT/'last-evaluation.json',dict(job=job,**summary))
                status['completed'].append(job);write(OUT/'queue.json',status)
            status.update(stage='matched_forecast',current=dict(task=task));write(OUT/'queue.json',status);forecast(OUT,task,OUT/f'{task}-matched-forecast.json')
        status.update(status='completed',stage='completed',finished_at=time.time());write(OUT/'queue.json',status)
    except Exception as error:
        status.update(status='failed',error=repr(error),finished_at=time.time());write(OUT/'queue.json',status);raise
if __name__=='__main__':main()

