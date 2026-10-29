"""Longer paired DEVELOPMENT budget pilot, no automatic formal/holdout promotion."""
import os,sys,time,json,subprocess
from pathlib import Path
from scale100_train import *
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'
OUT=ROOT/'generated/scale100-budget-pilot-20261026'

def evaluate_segment(folder,episode,task,method):
    pointer=json.loads((folder/'latest.json').read_text());checkpoint=folder/pointer['path']
    if digest(checkpoint)!=pointer['sha256']:raise ValueError('Checkpoint checksum mismatch')
    state=torch.load(checkpoint,weights_only=False,map_location='cpu');actor=Agent();actor.load_state_dict(state['actor']);actor.eval()
    manifest=json.loads(DATA.read_text());entry=next(r for r in manifest['cases'] if r['split']=='development');case=FloodCase(entry['path']);rows=[]
    for seed in (11,29,47):
        env=ScaleEnv(case,config(task,method,seed));total,_,metrics=rollout(actor,env,np.random.default_rng(seed),0.,False)
        rows.append(dict(seed=seed,return_=total,metrics=metrics,parent=entry['parent_terrain'],split='development'))
    path=folder/f'development-ep{episode:04d}.json';write(path,dict(episode=episode,rows=rows,holdout_evaluated=False,checkpoint_sha256=pointer['sha256'],checkpoint_episode=pointer['episode']))
    # Keep a compact actor at each predeclared selection opportunity; optimizer/replay remain in recovery files.
    torch.save(actor.state_dict(),folder/f'actor-ep{episode:04d}.pt')
    return dict(episode=episode,coverage=float(np.mean([r['metrics']['first_coverage'] for r in rows])),rescued=float(np.mean([r['metrics']['rescued'] for r in rows])))

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    plan=dict(schema='hyra-scale100-budget-pilot-v1',phase='development_only',dataset_sha256=digest(DATA),tasks=['E1','E2'],methods=list(METHODS),training_seed=11,episodes_per_job=256,evaluation_episodes=[64,128,256],evaluation_seeds=[11,29,47],source_sha256=sources(),selection_rule='All predeclared checkpoints reported; common formal budget decided after learning/stability/resource review, no holdout tuning',formal_training_started=False)
    write(OUT/'plan.json',plan);status=dict(status='running',pid=os.getpid(),started_at=time.time(),phase='development_only',completed=[],current=None,formal_training_started=False);write(OUT/'queue.json',status)
    try:
        for task in ('E1','E2'):
            for method in METHODS:
                job=f'{task}-{method}-s11';folder=OUT/job
                for stop in (64,128,256):
                    args=[sys.executable,'-B','scale100_train.py','--dataset',str(DATA),'--output',str(folder),'--task',task,'--method',method,'--episodes','256','--stop-after',str(stop)]
                    if stop>64:args.append('--resume')
                    with (OUT/(job+'.log')).open('a',encoding='utf-8') as log:
                        child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        status['current']=dict(job=job,segment_end=stop,child_pid=child.pid);write(OUT/'queue.json',status)
                        code=child.wait()
                    if code:raise RuntimeError(f'{job} segment{stop} exited {code}')
                    evaluation=evaluate_segment(folder,stop,task,method);write(OUT/'last-evaluation.json',dict(job=job,**evaluation))
                status['completed'].append(job);write(OUT/'queue.json',status)
        status.update(status='completed',finished_at=time.time());write(OUT/'queue.json',status)
    except Exception as error:
        status.update(status='failed',error=repr(error),finished_at=time.time());write(OUT/'queue.json',status);raise
if __name__=='__main__':main()
