"""Bounded parallel execution of independent frozen training jobs.

Each learner remains a separate deterministic, one-thread CPU process. Only
the supervisor writes the global queue; model artifacts are job-local.
"""
import os,time,json,queue,subprocess,shutil,ctypes,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scale100_formal import *

def apply_event(status,identity,event):
    if identity in status['active_jobs'] and identity not in status['completed']:
        status['active_jobs'][identity]=event
        if event['stage']=='training':status['formal_training_started']=True

def worker_limit(available,maximum,requested=None):
    cap=maximum if math.isinf(available) else max(0,int((available-1.5)/.8))
    return min(requested or maximum,maximum,cap)

def available_memory_gib():
    if os.name!='nt':return float('inf')
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),*[(name,ctypes.c_ulonglong) for name in ('total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','extended')]]
    state=Memory();state.length=ctypes.sizeof(Memory);kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(Memory)];kernel.GlobalMemoryStatusEx.restype=ctypes.c_int
    require(kernel.GlobalMemoryStatusEx(ctypes.byref(state)),'Cannot inspect memory before starting workers');return state.available_physical/1024**3

def execute_job(output,plan,job,emit):
    model=output/job['id'];plan_hash=digest(output/'plan.json')
    if (model/'selection.json').exists():return check_selected(model,job,plan,plan_hash)
    for stop in plan['development_checkpoints']:
        evaluation=model/f'development-ep{stop:04d}.json'
        if evaluation.exists():
            old=json.loads(evaluation.read_text());require(old['episode']==stop and digest(model/old['actor_path'])==old['actor_sha256'],'Cached development checkpoint changed');continue
        current=json.loads((model/'latest.json').read_text())['episode'] if (model/'latest.json').exists() else 0
        require(current<=stop,'Checkpoint ahead of missing evaluation; inspect before resuming')
        if current<stop:
            validate_plan(plan);args=[sys.executable,'-B','scale100_formal_worker.py','--plan',str(output/'plan.json'),'--job',job['id'],'--stop-after',str(stop)]
            if current:args.append('--resume')
            with (output/(job['id']+'.log')).open('a',encoding='utf-8') as log:
                child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));emit(job['id'],dict(stage='training',segment_end=stop,child_pid=child.pid));code=child.wait()
            require(code==0,f'{job["id"]} failed with exit{code}')
        emit(job['id'],dict(stage='development_evaluation',episode=stop,child_pid=None));evaluate_checkpoint(model,stop,job,plan)
    select_model(model,job,plan);return check_selected(model,job,plan,plan_hash)

def run_parallel(output,requested=None):
    output=Path(output);plan=json.loads((output/'plan.json').read_text());validate_plan(plan);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    previous=json.loads((output/'queue.json').read_text())
    if previous.get('status')=='completed':
        for job in plan['jobs']:check_selected(output/job['id'],job,plan,digest(output/'plan.json'))
        return previous
    ids=[previous.get('pid'),previous.get('current',{}).get('child_pid'),*[r.get('child_pid') for r in previous.get('active_jobs',{}).values()]]
    for pid in ids:require(not process_alive(pid),'An existing controller/child is live; refuse duplicate execution')
    require(shutil.disk_usage(output).free>4*1024**3,'Insufficient disk space for paired study')
    available=available_memory_gib();limit=worker_limit(available,plan['max_concurrent_workers'],requested)
    require(limit>=1,'Less than2.3GiB available memory; inspect before starting training')
    status=dict(schema='hyra-scale100-parallel-formal-queue-v1',status='running',pid=os.getpid(),started_at=previous.get('started_at',time.time()),formal_training_started=previous.get('formal_training_started',False),workers=limit,available_memory_at_start_gib=available,per_worker_torch_threads=1,completed=[],active_jobs={},failed_jobs=[])
    pending=[]
    for job in plan['jobs']:
        if (output/job['id']/'selection.json').exists():check_selected(output/job['id'],job,plan,digest(output/'plan.json'));status['completed'].append(job['id']);status['formal_training_started']=True
        else:pending.append(job)
    write(output/'queue.json',status);events=queue.Queue();active={};position=0
    def emit(identity,event):events.put((identity,event))
    with ThreadPoolExecutor(max_workers=limit) as pool:
        while position<len(pending) or active:
            while not status['failed_jobs'] and len(active)<limit and position<len(pending):
                job=pending[position];position+=1;status['active_jobs'][job['id']]=dict(stage='starting',child_pid=None);active[pool.submit(execute_job,output,plan,job,emit)]=job
            try:
                identity,event=events.get(timeout=.5);apply_event(status,identity,event)
            except queue.Empty:pass
            while not events.empty():
                identity,event=events.get_nowait();apply_event(status,identity,event)
            for future,job in list(active.items()):
                if not future.done():continue
                try:
                    future.result();status['completed'].append(job['id'])
                except Exception as error:status['failed_jobs'].append(dict(job=job['id'],error=repr(error)))
                status['active_jobs'].pop(job['id'],None);del active[future]
            status['status']='finishing_active_after_failure' if status['failed_jobs'] else 'running';write(output/'queue.json',status)
            if status['failed_jobs'] and not active:break
    status.update(status='failed' if status['failed_jobs'] else 'completed',finished_at=time.time(),active_jobs={});write(output/'queue.json',status)
    if status['failed_jobs']:raise RuntimeError('Formal job failure retained in queue; completed models are preserved')
    require(len(status['completed'])==36,'Formal matrix incomplete');return status

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,choices=[1,2,3]);a=p.parse_args();run_parallel(a.output,a.workers)
