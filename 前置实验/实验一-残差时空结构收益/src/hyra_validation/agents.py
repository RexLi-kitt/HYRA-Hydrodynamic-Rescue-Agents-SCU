"""Paper-informed CNN/action/time/GRU network with training-only enhancement."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SearchAgent(nn.Module):
    def __init__(self, obs_shape, embed=32, hidden=64):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.hidden_dim,self.embed = hidden,embed
        self.conv = nn.Sequential(nn.Conv2d(obs_shape[0],16,3,padding=1),nn.ReLU(),
                                  nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Flatten())
        self.encoder = nn.Linear(16*obs_shape[1]*obs_shape[2],embed)
        self.action_embed = nn.Linear(8,embed)
        self.gru = nn.GRUCell(3*embed,hidden)
        self.output = nn.Linear(hidden,8)
        self.register_buffer("direction",torch.ones(embed)/math.sqrt(embed))

    def time_embed(self,t,batch,device):
        frequency = torch.exp(torch.arange(0,self.embed,2,device=device)*(-math.log(10000.)/self.embed))
        value = float(t)*frequency
        return torch.stack([value.sin(),value.cos()],dim=-1).flatten()[None].expand(batch,-1)

    def forward(self,obs,last_action,t,hidden,enhance=False):
        z = F.relu(self.encoder(self.conv(obs.reshape(-1,*self.obs_shape))))
        a = self.action_embed(last_action)
        time = self.time_embed(t,len(z),z.device)
        if enhance:
            if not self.training or not torch.is_grad_enabled():
                raise RuntimeError("Enhancement is training-only")
            virtual = self.output(self.gru(torch.cat([z,a,time],dim=-1),hidden))
            gradient = torch.autograd.grad(virtual.max(dim=-1).values.sum(),z,retain_graph=True,create_graph=False)[0]
            norm = gradient.detach().norm(dim=-1)
            span = norm.max()-norm.min()
            surprise = (norm-norm.min())/span.clamp_min(1e-12)
            z = z+surprise[:,None]*self.direction
        h = self.gru(torch.cat([z,a,time],dim=-1),hidden)
        return self.output(h),h


class SearchMAC:
    """PyMARL learner controller. Actor API takes observations only."""
    def __init__(self,obs_shape,n_agents,config,device,enhancement=True):
        self.n_agents,self.device = n_agents,device
        self.agent = SearchAgent(obs_shape,config["embed_dim"],config["hidden_dim"]).to(device)
        self.enable_enhancement = enhancement
        self.hidden = None

    def init_hidden(self,batch_size):
        self.hidden = torch.zeros(batch_size*self.n_agents,self.agent.hidden_dim,device=self.device)

    def forward(self,batch,t):
        obs = batch["obs"][:,t].reshape(-1,*self.agent.obs_shape)
        previous = torch.zeros(len(obs),8,device=obs.device) if t==0 else F.one_hot(batch["actions"][:,t-1].reshape(-1),8).float()
        q,self.hidden = self.agent(obs,previous,t,self.hidden,
            enhance=self.enable_enhancement and self.agent.training)
        return q.reshape(batch.batch_size,self.n_agents,8)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self,other):
        self.agent.load_state_dict(other.agent.state_dict())

    def cuda(self):
        self.device = "cuda"
        self.agent.cuda()

    def save_models(self,path):
        torch.save(self.agent.state_dict(),str(path)+"/agent.th")

    def act(self,obs,previous,t,hidden,epsilon,rng):
        self.agent.eval()
        with torch.no_grad():
            q,h = self.agent(torch.as_tensor(obs,device=self.device),
                torch.as_tensor(previous,device=self.device),t,hidden,enhance=False)
        actions = q.argmax(dim=-1).cpu().numpy()
        random = rng.random(len(actions))<epsilon
        actions[random] = rng.integers(0,8,int(random.sum()))
        return actions,h
