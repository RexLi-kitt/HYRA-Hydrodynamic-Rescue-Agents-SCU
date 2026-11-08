"""Search-only additions around the preserved MAPPO policy and referee."""
import copy
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HERE=Path(__file__).resolve().parent
ALIGN=Path(r'C:\Users\李\Documents\Codex\2026-11-05\task\hyra_20261106_alignment')
OLD=ALIGN/'mappo_short_seed2_v1'
NEAR=ALIGN/'near_target_experiment_v1'
SDK=Path(r'C:\Users\李\Desktop\hyra\HYRA-referee-20261106.1')
PANEL=ALIGN/'local_training_candidate_panel_0_2400'
RULE=Path(r'D:\HYRA-MAPPO-20261031\experiments\rule-search-v2-20261103\policy.py')
UPSTREAM=Path(r'D:\HYRA-MAPPO-20261031\upstream\on-policy')
sys.path.insert(0,str(OLD));sys.path.insert(0,str(ALIGN))
import model as original_model
import trainer as original_trainer
from referee_bridge import RefereeBridge


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


sdk=module('search_official_referee',SDK/'run.py')
geometry=module('search_public_boundary',ALIGN/'dqn_boundary_mask_v1'/'boundary_geometry.py')
TRAIN=[41001,42002,41002,42004]
DEV=[41004,41005,42012,42015,51001,51002,41006,41007,42017,42018,51003,51004]
ARMS=('original','mask','local_mask')
SEEDS=(77126,77127,77128)
LOCAL_SIDE=15
LOCAL_DIM=2*LOCAL_SIDE**2+2


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def json_write(path,data):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def checkpoint_write(path,data):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def source_checkpoint(seed):return NEAR/'long_stage1_v1'/f'pure_{seed}_total_102400'/'checkpoint-total-102400.pt'
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()])
def set_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def local_coverage(o):
    """North/east aligned 10m cells around this UAV, never current hidden water."""
    pos=np.asarray(o.pos,dtype=float);last=np.asarray(o.last)
    if pos.shape!=(4,2) or last.shape!=(100,100):raise ValueError('Public geometry shape')
    offset=np.arange(-7,8);out=[]
    for p in pos:
        col,row=np.floor(p/10).astype(int)
        rr=row+offset[:,None];cc=col+offset[None,:]
        valid=(rr>=0)&(rr<100)&(cc>=0)&(cc<100)
        known=valid&(last[np.clip(rr,0,99),np.clip(cc,0,99)]>=0)
        subcell=p/10-np.floor(p/10)-.5
        out.append(np.r_[known.ravel(),valid.ravel(),subcell])
    return np.asarray(out,np.float32)


def encode(o,arm):
    if arm not in ARMS:raise ValueError(arm)
    x=original_model.encode(o)
    if arm=='local_mask':x=np.concatenate((x,local_coverage(o)),axis=1)
    if arm=='original':
        mask=np.tile(np.r_[True,(np.asarray(o.seen_at)>=0)&np.asarray(o.active,bool)],(4,9))
    else:mask=geometry.action_mask(o)
    # Exact Boolean masks travel with rollout samples; they are not neural features.
    return np.concatenate((x,mask.astype(np.float32)),axis=1)


def residual():
    net=nn.Sequential(nn.Linear(LOCAL_DIM,64),nn.ReLU(),nn.Linear(64,64))
    nn.init.orthogonal_(net[0].weight,gain=2**.5);nn.init.zeros_(net[0].bias)
    nn.init.zeros_(net[2].weight);nn.init.zeros_(net[2].bias)
    return net


class Actor(original_model.Actor):
    def __init__(self,use_local):
        super().__init__();self.use_local=use_local
        if use_local:self.local=residual()
    def distributions(self,x):
        old=x[:,:998]
        if not self.use_local:return super().distributions(old)
        h=self.net(old)+self.local(x[:,998:998+LOCAL_DIM])
        return Categorical(logits=self.move(h)),Categorical(logits=self.target(h).masked_fill(~(old[:,-65:]>.5),-1e9))


class Critic(original_model.Critic):
    def __init__(self,use_local):
        super().__init__();self.use_local=use_local
        if use_local:self.local=residual()
    def forward(self,x):
        old=x[:,:998]
        if not self.use_local:return super().forward(old)
        h=self.net[:-1](old)+self.local(x[:,998:998+LOCAL_DIM])
        return self.net[-1](h)


class Policy:
    def __init__(self,arm,device='cuda:0'):
        self.arm=arm;self.device=device;local=arm=='local_mask'
        self.actor=Actor(local).to(device);self.critic=Critic(local).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=.0005,eps=1e-5)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=.0005,eps=1e-5)
    def tensor(self,x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)
    def joint(self,x):
        m,t=self.actor.distributions(x)
        logits=(m.logits[:,:,None]+t.logits[:,None,:]).reshape(-1,585)
        mask=x[:,-585:]>.5
        if not bool(mask[:,520].all()):raise ValueError('Safe no-target hover absent')
        return Categorical(logits=logits.masked_fill(~mask,-torch.inf))
    def act(self,x,deterministic=False):
        x=self.tensor(x)
        if self.arm=='original':
            m,t=self.actor.distributions(x)
            a=m.probs.argmax(-1) if deterministic else m.sample()
            b=t.probs.argmax(-1) if deterministic else t.sample()
            actions=torch.stack((a,b),dim=-1);logp=m.log_prob(a)+t.log_prob(b)
        else:
            d=self.joint(x);packed=d.probs.argmax(-1) if deterministic else d.sample()
            actions=torch.stack((packed//65,packed%65),dim=-1);logp=d.log_prob(packed)
        return actions,logp.unsqueeze(-1),self.critic(x)
    def evaluate_actions(self,shared,obs,rnn,rnn_critic,actions,masks,available,active):
        x=self.tensor(obs);a=torch.as_tensor(actions,dtype=torch.long,device=self.device)
        if self.arm=='original':
            m,t=self.actor.distributions(x);logp=m.log_prob(a[:,0])+t.log_prob(a[:,1]);entropy=m.entropy()+t.entropy()
        else:
            d=self.joint(x);packed=a[:,0]*65+a[:,1]
            if not bool((x[:,-585:]>.5).gather(1,packed[:,None]).all()):raise ValueError('PPO action illegal in its rollout mask')
            logp=d.log_prob(packed);entropy=d.entropy()
        active=self.tensor(active);entropy=(entropy[:,None]*active).sum()/active.sum()
        return self.critic(self.tensor(shared)),logp[:,None],entropy


def make(arm,device='cuda:0'):
    args=original_trainer.get_config().parse_args([])
    args.use_recurrent_policy=False;args.use_naive_recurrent_policy=False
    args.ppo_epoch=5;args.num_mini_batch=1;args.lr=.0005;args.critic_lr=.0005
    args.gamma=.99;args.gae_lambda=.95;args.clip_param=.2;args.entropy_coef=.01
    p=Policy(arm,device);return p,original_trainer.R_MAPPO(args,p,device=torch.device(device))


def extend_optimizer(opt,old):
    new=opt.state_dict();ids=new['param_groups'][0]['params'];oldids=old['param_groups'][0]['params']
    assert len(ids)>=len(oldids)
    new['state']={ids[n]:copy.deepcopy(old['state'][oldid]) for n,oldid in enumerate(oldids) if oldid in old['state']}
    group=copy.deepcopy(old['param_groups'][0]);group['params']=ids;new['param_groups']=[group]
    opt.load_state_dict(new)


def initialize(arm,seed,device='cuda:0'):
    seed_all(seed)
    ck=torch.load(source_checkpoint(seed),map_location=device,weights_only=False)
    assert ck['config']['model_sha256']==sha(OLD/'model.py')
    assert ck['config']['trainer_sha256']==sha(OLD/'trainer.py')
    p,tr=make(arm,device)
    for name in ('actor','critic'):
        result=getattr(p,name).load_state_dict(ck[name],strict=False)
        assert not result.unexpected_keys and all(k.startswith('local.') for k in result.missing_keys)
    extend_optimizer(p.actor_optimizer,ck['actor_optimizer']);extend_optimizer(p.critic_optimizer,ck['critic_optimizer'])
    original_trainer.load_norm_state(tr,ck['value_norm'])
    return p,tr,ck


class Dataset:
    def __init__(self):
        self.rows=read(PANEL/'panel.json');self.by={(r['parent'],r['roster']):r for r in self.rows}
        self.manifest=read(PANEL/'SHA256.json');self.cache=OrderedDict();self.checked=set()
    def load(self,parent):
        if parent not in TRAIN+DEV:raise ValueError('Scene outside declared training/development sets')
        if parent in self.cache:self.cache.move_to_end(parent);return self.cache[parent]
        row=self.by[parent,0]
        for field in ('water','population'):
            path=PANEL/row[field]
            if path not in self.checked:
                assert sha(path)==self.manifest[row[field]];self.checked.add(path)
        pair=sdk.load_trial(PANEL,row);self.cache[parent]=pair
        while len(self.cache)>4:self.cache.popitem(last=False)
        return pair
    def bridge(self,parent):return RefereeBridge(sdk.DeliveryMission(*self.load(parent),(64,720,2400)))


def env_state(bridge):return {k:copy.deepcopy(v) for k,v in vars(bridge.env).items() if k not in ('water','truth')}
def restore_bridge(ds,parent,state,done=False):
    b=ds.bridge(parent);vars(b.env).update(copy.deepcopy(state));b.done=done;return b


def step(b,actions,arm):
    actions=np.asarray(actions);o=b.observation();packed=actions[:,0]*65+actions[:,1]
    mask=encode(o,arm)[:,-585:].astype(bool)
    if not mask[np.arange(4),packed].all():raise ValueError('Illegal behavior action')
    targets=actions[:,1]-1;old_pos=b.env.pos.copy();sup=b.env.supplied.copy();prog=b.env.progress.copy()
    oldconf=b.env.conflicts;oldbound=b.env.boundary;oldseen=int((o.seen_at>=0).sum())
    close=np.array([o.visible&(np.linalg.norm(o.seen_xy-o.pos[i],axis=1)<=1+1e-9) for i in range(4)])
    correct=sum(actions[i,0]==8 and targets[i]>=0 and close[i,targets[i]] for i in range(4))
    after,done,metrics=b.step(actions[:,0],targets,local_action_order=False)
    delivered=int(((b.env.supplied>=0)&(sup<0)).sum());discovered=int((after.seen_at>=0).sum())-oldseen
    conflicts=b.env.conflicts-oldconf;boundary=b.env.boundary-oldbound
    if arm!='original' and boundary:raise AssertionError('Boundary mask violated')
    parts=dict(delivery=10.*delivered,discovery=.25*discovered,coverage=.002*len(after.new_cells),collision=-.1*conflicts,time=-.005)
    rec=dict(t=int(o.t),moves=actions[:,0].tolist(),targets=targets.tolist(),position=old_pos.tolist(),
        displacement=np.linalg.norm(b.env.pos-old_pos,axis=1).tolist(),reward_parts=parts,reward=sum(parts.values()),
        new_cells=len(after.new_cells),discoveries=discovered,delivered=delivered,conflicts=int(conflicts),boundary=int(boundary),
        near=int(close.any(axis=1).sum()),correct_hover=int(correct),service_starts=int(((b.env.progress>0)&(prog==0)).sum()),
        selected_visible=int(sum(j>=0 and o.visible[j] for j in targets)),selected=int((targets>=0).sum()))
    return after,done,metrics,rec


def source_lock():
    paths=[HERE/n for n in ('core.py','carpet.py','train_round.py','validate.py')]
    paths += [OLD/'model.py',OLD/'trainer.py',ALIGN/'referee_bridge.py',geometry.__file__,RULE,SDK/'run.py',SDK/'SHA256.json',PANEL/'panel.json',PANEL/'SHA256.json']
    paths += list((SDK/'engine').glob('*.py'))
    paths += [UPSTREAM/'onpolicy/algorithms/r_mappo/r_mappo.py',UPSTREAM/'onpolicy/utils/valuenorm.py',UPSTREAM/'onpolicy/config.py']
    return {str(p):sha(p) for p in paths}


def check_lock(lock):
    for p,h in lock.items():
        if sha(p)!=h:raise RuntimeError('Source/input metadata changed: '+p)
