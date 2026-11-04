import argparse,json,os,subprocess,sys,time
from pathlib import Path
from flood1km.data import sha
from flood1km.learners import ALGORITHMS
from train_with_service import write
def run(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();stages=[]
    write(args.output/'protocol.json',dict(user_request='All six to total256 first; eventual1024 or more',total=256,original_episodes=64,additional_per_algorithm=192,
        algorithms=list(ALGORITHMS),groups={'A':['happo','hatrpo'],'B':['hasac','maac'],'C':['mat','coma']},training_seed=1001,
        common_execution='service singleton masks before actor sampling; actual executed actions learned',
        comparison='paired 64/256 with same execution semantics, each service-only and with coverage watchdog; no sealed50 test',
        fairness='same episodes/layouts/reward/physics/CPU2; off-policy and on-policy update counts differ by algorithm as before',
        MAT_note='Matched64 re-evaluation required: causal later-agent decoder now conditions on forced earlier-agent service action, unlike old post-hoc override.',
        replay='clear old off-policy replay once at service-phase start; persist new replay for future resume',
        future='save full model/optimizers/RNG/replay every32 episodes; no automatic1024 run in this queue',
        stop='STOP_AFTER_CHECKPOINT file stops current algorithm at next saved boundary; failures stop queue',
        source_hashes={n:sha(Path(__file__).with_name(n)) for n in ['launch_six_service.py','train_with_service.py','coverage_controller.py','service_commitment.py']}))
    for a in ALGORITHMS:
        command=[sys.executable,'train_with_service.py','--algorithm',a,'--baseline',str(args.baseline),'--output',str(args.output/a),'--total','256']
        with (args.output/(a+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            write(args.output/'queue-status.json',dict(status='running',algorithm=a,pid=os.getpid(),child_pid=proc.pid,completed=stages))
            rc=proc.wait()
        stages.append(dict(algorithm=a,returncode=rc))
        if rc:
            write(args.output/'queue-status.json',dict(status='stopped',algorithm=a,completed=stages));return rc
    results={a:json.loads((args.output/a/'status.json').read_text(encoding='utf-8')) for a in ALGORITHMS}
    write(args.output/'results.json',results);write(args.output/'queue-status.json',dict(status='completed',completed=stages,elapsed_s=time.perf_counter()-start))
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();sys.exit(run(args))
