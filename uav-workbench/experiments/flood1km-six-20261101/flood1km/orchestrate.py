"""Prepared serial workflow; never automatically launched by installation or import."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main(args):
    root=Path(__file__).parents[1];args.output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['OMP_NUM_THREADS']='2'
    env['PYTHONUTF8']='1';env['PYTHONPATH']=str(root)
    log=[]
    def launch(name,command):
        start=time.perf_counter()
        with (args.output/f'{name}.log').open('w',encoding='utf-8') as out:
            child=subprocess.run(command,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
        log.append(dict(stage=name,returncode=child.returncode,elapsed_s=time.perf_counter()-start))
        (args.output/'workflow.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
        if child.returncode:raise RuntimeError(f'{name} failed/stopped; no automatic retry or next training')
    # A new invocation gets a new output directory. No existing data is deleted.
    launch('tests',[sys.executable,'-m','unittest','discover','-s','tests','-v'])
    launch('dataset',[sys.executable,'-m','flood1km.data','--output',str(args.output/'data'),'--terrain',str(args.terrain)])
    launch('benchmark',[sys.executable,'-m','flood1km.benchmark','--data',str(args.output/'data'),'--output',str(args.output/'benchmark.json')])
    for algorithm in ('happo','hatrpo','hasac','maac','mat','coma'):
        launch(algorithm,[sys.executable,'-m','flood1km.train','--algorithm',algorithm,'--data',str(args.output/'data'),
                          '--output',str(args.output/algorithm),'--episodes','64','--seed','701','--device','cuda','--execute'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:p.error('Preparation is complete only as a draft. Execution was paused by the user.')
    main(args)
