"""Planning remains the exact default; learn task deviations, never idle by default."""
from pathlib import Path
import sys,copy
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-dispatch-v1'))
from dispatch_env import DispatchEnv

class ResidualEnv(DispatchEnv):
    def choices(self):
        mask,expert,options,features=super().choices()
        for i,a in enumerate(expert):
            options[i][8]=copy.deepcopy(options[i][a])
            features[i,8]=features[i,a].copy();mask[i,8]=True
        return mask,np.full(4,8,dtype=np.int64),options,features

def initialize_planning_default(model,bias=5.):
    """Deterministic policy is exactly planning at initialization; disclose this."""
    heads=[model.policy.head] if model.algorithm=='mat' else [actor[-1] for actor in model.actors]
    with torch.no_grad():
        for head in heads:
            head.weight.zero_();head.bias.zero_();head.bias[8]=bias

