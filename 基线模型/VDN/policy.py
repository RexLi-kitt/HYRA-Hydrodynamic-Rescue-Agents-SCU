import hashlib,importlib.util,os,random,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
NAME='VDN'
package='hyra_pack_'+NAME.lower()+'_'+hashlib.sha256(str(HERE).encode('utf8')).hexdigest()[:16]
if package not in sys.modules:
    spec=importlib.util.spec_from_file_location(package,HERE/'src'/'__init__.py',submodule_search_locations=[str(HERE/'src')])
    module=importlib.util.module_from_spec(spec);sys.modules[package]=module;spec.loader.exec_module(module)
from importlib import import_module
r=import_module(package+'.rescue')
if NAME!='MAPPO':
    p=import_module(package+'.dqn_core');replay=import_module(package+'.replay')

class Frozen:
    def __init__(self,checkpoint,mode,device,seed):
        self.mode=mode;self.device=device
        ck=torch.load(checkpoint,map_location=device,weights_only=True)
        if ck.get('format')!='HYRA_PORTABLE_WEIGHTS_V1' or ck.get('model')!=NAME:raise ValueError('wrong checkpoint format or model')
        if NAME=='MAPPO':
            if mode not in ('sampled','greedy'):raise ValueError('invalid MAPPO mode')
            self.model=r.Policy(device);self.model.actor.load_state_dict(ck['actor']);self.model.critic.load_state_dict(ck['critic']);self.model.actor.eval();self.model.critic.eval()
        else:
            if mode not in ('epsilon_0.05','greedy'):raise ValueError('invalid Q mode')
            self.model=p.RecurrentQ().to(device);self.model.load_state_dict(ck['online']);self.model.eval();self.hidden=self.model.init_hidden(4,device);self.previous=np.full(4,-1,np.int64)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    def act(self,obs):
        if NAME=='MAPPO':
            with torch.no_grad():a,_,_=self.model.act(r.encode(obs),deterministic=self.mode=='greedy')
            z=a.cpu().numpy();return z[:,0],z[:,1]-1
        context,self.hidden=replay.capture(self.model,obs,self.previous,self.hidden)
        a=replay.choose(self.model,context,self.device,.05 if self.mode=='epsilon_0.05' else 0.)
        self.previous=a.copy();return a//65,a%65-1

def make_policy():
    default_seed='77126' if NAME=='MAPPO' else '88126'
    seed=os.environ.get('HYRA_MODEL_SEED',default_seed)
    cp=Path(os.environ.get('HYRA_CHECKPOINT',str(HERE/'weights'/f'{seed}.pt')))
    default_mode='sampled' if NAME=='MAPPO' else 'epsilon_0.05'
    mode=os.environ.get('HYRA_MODE',default_mode)
    device=os.environ.get('HYRA_DEVICE','cuda:0' if torch.cuda.is_available() else 'cpu')
    return Frozen(cp,mode,device,int(os.environ.get('HYRA_SEED','581000')))
