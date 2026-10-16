"""PyMARL core reuse with a paper-informed controller and common rescue environment."""


import sys
from types import SimpleNamespace
import torch
from ..artifacts import ROOT
from ..envs import RescueEnv
from ..agents import SearchMAC


def build_environment(environment_config):
    return RescueEnv(**environment_config)


def components():
    sys.path.insert(0,str(ROOT/"third_party/pymarl/src"))
    from components.episode_buffer import EpisodeBatch,ReplayBuffer
    from learners.q_learner import QLearner
    # Old PyMARL returns list slices; modern PyTorch requires tuple indices.
    # Adapt the imported class in memory, never alter warehouse source files.
    if not getattr(EpisodeBatch,"_hyra_tuple_indices",False):
        original=EpisodeBatch._parse_slices
        def tuple_indices(self,items):
            return tuple(original(self,items))
        EpisodeBatch._parse_slices=tuple_indices
        EpisodeBatch._hyra_tuple_indices=True
    return EpisodeBatch,ReplayBuffer,QLearner


def build_learner(env,config,logger,device="cuda",enhancement=True):
    EpisodeBatch,ReplayBuffer,QLearner = components()
    obs_shape = env.get_obs().shape[1:]
    scheme = dict(obs=dict(vshape=obs_shape,group="agents"),
        state=dict(vshape=len(env.get_state())),actions=dict(vshape=(1,),group="agents",dtype=torch.long),
        avail_actions=dict(vshape=(8,),group="agents",dtype=torch.int),reward=dict(vshape=(1,)),
        terminated=dict(vshape=(1,),dtype=torch.uint8))
    groups = dict(agents=env.n_agents)
    tc = config["training"]
    args = SimpleNamespace(n_agents=env.n_agents,state_shape=len(env.get_state()),mixer="qmix",
        mixing_embed_dim=32,hypernet_layers=2,hypernet_embed=64,lr=tc["lr"],gamma=tc["gamma"],
        optim_alpha=.99,optim_eps=1e-5,grad_norm_clip=10,double_q=True,
        target_update_interval=tc["target_update_interval"],learner_log_interval=1)
    mac = SearchMAC(obs_shape,env.n_agents,tc,device,enhancement)
    learner = QLearner(mac,scheme,logger,args)
    if device=="cuda":
        learner.cuda()
    # Targets use the execution representation; no virtual gradient computation.
    learner.target_mac.enable_enhancement = False
    buffer = ReplayBuffer(scheme,groups,tc["buffer_size"],config["environment"]["episode_limit"]+1,device="cpu")
    return mac,learner,buffer,scheme,groups
