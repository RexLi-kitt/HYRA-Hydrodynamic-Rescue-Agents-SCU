"""Resume the already generated pilot, with serial, failure-stopping stages."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--attempt',default='01');a=p.parse_args()
    root=Path(__file__).parent;out=a.output;out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',PYTHONUTF8='1',PYTHONPATH=str(root))
    stages=[]
    def write(stage,status,**extra):
        (out/'queue-status.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage,status=status,stages=stages,**extra),indent=2),encoding='utf-8')
    def run(name,argv):
        write(name,'running');started=time.time()
        with (out/f'{name}-{a.attempt}.log').open('x',encoding='utf-8') as stream:
            proc=subprocess.Popen([sys.executable]+argv,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
            write(name,'running',child_pid=proc.pid);code=proc.wait()
        stages.append(dict(stage=name,returncode=code,seconds=time.time()-started))
        if code:raise RuntimeError(f'{name} exited {code}; queue stopped without retry')
        write(name,'completed')
    try:
        run('acceptance',['-m','unittest','discover','-s','tests','-v'])
        run('runtime',['-c','import torch,json; x=torch.ones(64,64); assert (x@x==64).all(); print(json.dumps(dict(torch=torch.__version__,device="cpu",matmul_passed=True)))'])
        run('benchmark',['-m','flood1km.benchmark','--data',str(out/'data'),'--output',str(out/'benchmark.json')])
        for name in ('happo','hatrpo','hasac','maac','mat','coma'):
            run(name,['-m','flood1km.train','--algorithm',name,'--data',str(out/'data'),'--output',str(out/name),'--episodes','64','--seed','701','--device','cpu','--execute'])
        write('all','completed')
    except BaseException as error:
        write('failed','failed',error=str(error));traceback.print_exc();raise

if __name__=='__main__':main()
