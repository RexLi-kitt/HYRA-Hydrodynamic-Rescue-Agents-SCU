"""Geometry-conditioned autoregressive MAT; official transformer blocks preserved."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .upstream import load_transformer

official = load_transformer()

class GeometryMAT(official.MultiAgentTransformer):
 def __init__(self,device='cpu',embedding=128,blocks=2,heads=4):
  super().__init__(37,1450,585,4,blocks,embedding,heads,encode_state=False,device=torch.device(device),action_type='Discrete',dec_actor=False,share_actor=False)
  self.geometry_head=nn.Sequential(nn.Linear(embedding+19+12+9,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
  nn.init.zeros_(self.geometry_head[-1].weight);nn.init.zeros_(self.geometry_head[-1].bias);self.to(device)
 def logits(self,shifted,obs,person,geometry,representation,mask):
  b=obs.shape[0];ctx=self.decoder.ln(self.decoder.action_encoder(shifted))
  for block in self.decoder.blocks:ctx=block(ctx,representation)
  base=self.decoder.head(ctx).reshape(b*4,9,65)
  people=person.reshape(b*4,64,19);geo=geometry.reshape(b*4,64,9,12);h=ctx.reshape(b*4,-1)
  target_valid=mask.reshape(b*4,9,65).any(1)[:,1:];ids=torch.nonzero(target_valid,as_tuple=False)
  if ids.numel():
   sample,person_id=ids[:,0],ids[:,1];k=len(sample);move=torch.eye(9,device=obs.device,dtype=obs.dtype).unsqueeze(0).expand(k,9,9)
   z=torch.cat((h[sample,None,:].expand(k,9,-1),people[sample,person_id,None,:].expand(k,9,19),geo[sample,person_id],move),-1)
   score=self.geometry_head(z).squeeze(-1);delta=torch.zeros((b*4,64,9),device=obs.device,dtype=obs.dtype).index_put((sample,person_id),score)
   base=base+torch.cat((torch.zeros_like(delta[:,:1]),delta),1).transpose(1,2)
  return base.reshape(b,4,585)
 def parallel(self,obs,person,geometry,actions,mask):
  if not mask[:,:,520].all():raise ValueError('safe hover absent')
  values,rep=self.encoder(None,obs);shift=torch.zeros((*obs.shape[:2],586),device=obs.device,dtype=obs.dtype);shift[:,0,0]=1
  shift[:,1:,1:]=F.one_hot(actions[:,:-1].long(),585).to(obs.dtype)
  logits=self.logits(shift,obs,person,geometry,rep,mask).masked_fill(~mask.bool(),-1e10);dist=Categorical(logits=logits)
  return dist.log_prob(actions.long()),dist.entropy(),values.squeeze(-1),logits
 @torch.no_grad()
 def sequential(self,obs,person,geometry,mask,deterministic=False):
  if not mask[:,:,520].all():raise ValueError('safe hover absent')
  values,rep=self.encoder(None,obs);shift=torch.zeros((*obs.shape[:2],586),device=obs.device,dtype=obs.dtype);shift[:,0,0]=1
  actions=[];logps=[]
  for i in range(4):
   logits=self.logits(shift,obs,person,geometry,rep,mask)[:,i,:].masked_fill(~mask[:,i,:].bool(),-1e10);dist=Categorical(logits=logits)
   action=dist.probs.argmax(-1) if deterministic else dist.sample();actions.append(action);logps.append(dist.log_prob(action))
   if i<3:shift[:,i+1,1:]=F.one_hot(action,585).to(obs.dtype)
  return torch.stack(actions,1),torch.stack(logps,1),values.squeeze(-1)
