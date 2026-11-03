"""Exact nearest4 planning default; unchanged physical executor, no extra reservations."""
from pathlib import Path
import sys,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-variable-v1'))
from variable_dispatch import DispatchEnv
from variable_environment import Env

class ResidualEnv(DispatchEnv):
 def choices(self):
  mask,expert,options,features=super().choices()
  for i,a in enumerate(expert):
   options[i][8]=copy.deepcopy(options[i][a]);features[i,8]=features[i,a].copy();mask[i,8]=True
  return mask,np.full(self.cfg.uavs,8,dtype=np.int64),options,features

 def observe(self):
  base=Env.observe(self);features=self.choices()[3]
  if self.cfg.people>80:raise ValueError('At most80 people')
  records=base[0,500:500+8*self.cfg.people].reshape(self.cfg.people,8)
  ordered=sorted(records.tolist(),key=lambda r:(-r[7],r[0],r[1],r[3],r[2],r[4],r[5],r[6]))
  padded=np.zeros((80,8),np.float32);padded[:len(ordered)]=ordered
  rest=base[:,500+8*self.cfg.people:]
  return np.concatenate((base[:,:500],np.tile(padded.ravel(),(self.cfg.uavs,1)),rest,features.reshape(self.cfg.uavs,-1)),axis=1).astype(np.float32)
