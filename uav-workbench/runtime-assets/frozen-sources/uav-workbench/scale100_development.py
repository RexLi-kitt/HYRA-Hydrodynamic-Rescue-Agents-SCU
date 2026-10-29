"""Bounded development queue, process liveness, restart proof and legal baselines."""
import os,sys,json,time,subprocess
from collections import deque
from pathlib import Path
from scale100_train import *
from sur_rescue.environment import rule_actions,MOVES

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-900s-20261026/dataset.json'
OUT=ROOT/'generated/scale100-dev-20261026'

def run_child(arguments,log):
    with log.open('a',encoding='utf-8') as f:
        child=subprocess.Popen([sys.executable,'-B','scale100_train.py','--dataset',str(DATA),*arguments],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'live-child.json',dict(pid=child.pid,started_at=time.time(),arguments=arguments,log=str(log)))
        code=child.wait()
    if code:raise RuntimeError(f'Training exited {code}: {log}')

def restart_proof():
    first=OUT/'recovery-check';full=OUT/'uninterrupted-check'
    args=['--task','E1','--method','P','--episodes','8']
    run_child(['--output',str(first),*args,'--resume'],OUT/'recovery-check.log')
    run_child(['--output',str(full),*args],OUT/'uninterrupted-check.log')
    a=torch.load(first/'actor.pt',weights_only=True);b=torch.load(full/'actor.pt',weights_only=True)
    equal=all(torch.equal(a[k],b[k]) for k in a)
    ha=json.loads((first/'history.json').read_text());hb=json.loads((full/'history.json').read_text())
    if not equal or ha!=hb:raise AssertionError('Split-process recovery differs from uninterrupted training')
    write(OUT/'recovery-verification.json',dict(split_process_episodes=[4,4],uninterrupted_episodes=8,all_actor_tensors_bitwise_equal=equal,history_equal=ha==hb,actor_sha256=digest(first/'actor.pt'),scope='E1-P seed11; engineering proof, not convergence'))

def legal_rule(env,name):
    view=env.view();n=env.config.n_agents
    if name=='frontier':return rule_actions(view)
    # Snake sweep with known-obstacle routing; revisit the oldest arrived record after coverage.
    actions=[];reserved=set();nz,nx=env.shape
    for agent,start in enumerate(view.positions):
        victims={(r,c) for r,c,status in view.targets if status=='discovered'}
        lo=nz*agent//n;hi=nz*(agent+1)//n
        ordered=[(r,c) for r in range(lo,hi) for c in (range(nx) if r%2==0 else range(nx-1,-1,-1)) if not view.obstacles[r,c]]
        unknown=[p for p in ordered if not view.explored[p]]
        goal=unknown[0] if unknown else min(ordered,key=lambda p:env.observed_s[p])
        goals=victims or {goal};todo=deque([(start,None)]);visited={start};chosen=None
        while todo:
            (r,c),first=todo.popleft()
            if (r,c) in goals and first is not None:chosen=first;break
            for a,(dz,dx) in enumerate(MOVES):
                p=(r+dz,c+dx)
                if p in visited or p in reserved or not 0<=p[0]<nz or not 0<=p[1]<nx or view.obstacles[p]:continue
                if dz and dx and(view.obstacles[r,p[1]] or view.obstacles[p[0],c]):continue
                visited.add(p);todo.append((p,a if first is None else first))
        if chosen is None:chosen=rule_actions(view)[agent]
        actions.append(chosen);dz,dx=MOVES[chosen];reserved.add((start[0]+dz,start[1]+dx))
    return actions

def baselines():
    manifest=json.loads(DATA.read_text());row=next(r for r in manifest['cases'] if r['split']=='development');case=FloodCase(row['path']);rows=[]
    for task in ('E1','E2'):
        for name in ('frontier','sweep_refresh'):
            for seed in (11,29,47):
                env=ScaleEnv(case,config(task,'B3',seed));env.reset();done=False;total=0
                while not done:_,r,done=env.step(legal_rule(env,name));total+=r
                rows.append(dict(task=task,rule=name,seed=seed,return_=total,metrics=env.metrics(),parent=row['parent_terrain'],split='development'))
    write(OUT/'rule-baselines.json',dict(rows=rows,independent_parent_count=1,seeds_are_not_independent_events=True))

def main():
    OUT.mkdir(parents=True,exist_ok=True);status=dict(schema='hyra-scale100-development-queue-v1',pid=os.getpid(),status='running',stage='recovery',episodes_per_job=24,formal_training_started=False,completed=[],started_at=time.time())
    write(OUT/'queue.json',status)
    try:
        restart_proof();status['stage']='rule-baselines';write(OUT/'queue.json',status);baselines()
        for task in ('E1','E2'):
            for method in METHODS:
                job=f'{task}-{method}-s11';status.update(stage='training',current_job=job);write(OUT/'queue.json',status)
                run_child(['--output',str(OUT/job),'--task',task,'--method',method,'--episodes','24'],OUT/(job+'.log'))
                meta=json.loads((OUT/job/'actor.json').read_text());status['completed'].append(dict(job=job,episodes=meta['trained_episodes'],updates=meta['updates'],actor_sha256=meta['actor_sha256']));write(OUT/'queue.json',status)
        status.update(status='completed',stage='development_complete',finished_at=time.time());write(OUT/'queue.json',status)
    except Exception as error:
        status.update(status='failed',error=repr(error),finished_at=time.time());write(OUT/'queue.json',status);raise
if __name__=='__main__':main()
