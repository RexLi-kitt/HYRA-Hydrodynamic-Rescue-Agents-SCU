import subprocess,json
from pathlib import Path
import numpy as np


class Deep:
    def __init__(self,edge=0.,width=28,depth=12,commit=5):
        self.options=dict(edge=edge,width=width,depth=depth,commit=commit)
        self.seen=np.zeros(10000,bool)
        self.proc=subprocess.Popen(['node',str(Path(__file__).with_suffix('.js'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    def actions(self,env,fixed):
        obs=env.stamps>=0
        message=dict(step=env.step_no,p=env.p.tolist(),fixed=fixed.tolist(),targets=env.targets[env.found & ~env.rescued].tolist(),
                     newIds=np.flatnonzero(obs & ~self.seen).tolist(),**self.options)
        if hasattr(self,'prior') and env.step_no%10==0:
            t=min(30,(env.step_no+4)//10)
            risk=(self.prior[t]-self.prior[min(30,t+10)])/24
            values=self.prior[t]/24+6*risk
            message['weights']=[[int(i),float(values[i])] for i in np.flatnonzero(values!=1)]
        self.seen=obs
        self.proc.stdin.write(json.dumps(message)+'\n');self.proc.stdin.flush()
        return np.array(json.loads(self.proc.stdout.readline()),int)
    def close(self):
        self.proc.stdin.close();self.proc.wait()

