from concurrent.futures import ProcessPoolExecutor
import torch
from torch import nn
from .sim import *
from release_utils import atomic_torch, model_file
from release_utils import write_rows
ARMS=('L0','LG','T0','TG')
SEEDS=(11,29,47)
def produce(job):
    p,n=job;dest=OUT/'samples'/f'{p}-{n}.npz'
    if dest.exists():return
    rec,x,y=run(p,n,collect=True)
    np.savez_compressed(dest,x=np.array(x,'f'),y=np.array(y,np.int16),hash=rec['row']['scenario_hash'])
def collect():
    assert json.loads((OUT/'A-gate.json').read_text())['passed']
    (OUT/'samples').mkdir(exist_ok=True)
    with ProcessPoolExecutor(4) as pool:list(pool.map(produce,[(p,n) for p in range(21001,21017) for n in range(4)]))
class Net(nn.Module):
    def __init__(self,d,rec):
        super().__init__();self.gru=nn.GRU(d,32,batch_first=True) if rec else None
        self.head=nn.Linear(32 if rec else d*12,1)
    def forward(self,x):
        hidden=self.gru(x)[0][:,-1] if self.gru is not None else x.flatten(1)
        return self.head(hidden).flatten().clamp(-12,5)
def dataset():
    xs=[];ys=[];seqids=[]
    for p in range(21001,21013):
        for n in range(4):
            with np.load(OUT/'samples'/f'{p}-{n}.npz') as f:x,y=f['x'],f['y']
            offset=sum(len(a) for a in xs)
            xs.append(np.concatenate((np.zeros_like(x[:11]),x)));ys.append(y[::6]);seqids.extend(offset+11+np.arange(0,len(x),6))
    mu=np.concatenate([x[11:] for x in xs]).mean((0,1),dtype=np.float64).astype('f')
    sd=np.concatenate([x[11:] for x in xs]).std((0,1),dtype=np.float64).astype('f').clip(.001)
    return np.concatenate(xs),np.concatenate(ys).reshape(-1),np.array(seqids),mu,sd
def train():
    torch.set_num_threads(1);x,y,ti,mu,sd=dataset()
    X=torch.from_numpy(((x-mu)/sd).clip(-10,10)).cuda();Y=torch.from_numpy(y.astype('f')).cuda()
    ti=torch.tensor(ti,device='cuda');steps=torch.arange(-11,1,device='cuda')
    for arm in ARMS:
        for seed in SEEDS if arm.startswith('T') else (11,):
            file=OUT/f'{arm}-{seed}.pt'
            if file.exists():continue
            torch.manual_seed(seed);rng=np.random.default_rng(seed);d=len(mu) if arm.endswith('G') else len(mu)-12
            net=Net(d,arm.startswith('T')).cuda();opt=torch.optim.Adam(net.parameters(),lr=.001)
            curve=[]
            for ep in range(32):
                losses=[]
                for batch in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/1024))):
                    ids=torch.as_tensor(batch,device='cuda')
                    seq=X[ti[ids//100,None]+steps, (ids%100)[:,None],:d]
                    v=net(seq);loss=(v.exp()-Y[ids]*v).mean()
                    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();losses.append(loss.item())
                curve.append(dict(epoch=ep+1,loss=np.mean(losses)))
                write_rows(OUT/f'{arm}-{seed}-curve.csv',curve)
            atomic_torch(file,dict(weights=net.cpu().state_dict(),mu=torch.from_numpy(mu[:d].copy()),sd=torch.from_numpy(sd[:d].copy())))
@lru_cache(maxsize=12)
def predictor(arm,seed):
    torch.set_num_threads(1)
    ck=torch.load(model_file(OUT,f'{arm}-{seed}.pt'),weights_only=True);mu=ck['mu'];sd=ck['sd']
    net=Net(len(mu),arm.startswith('T'));net.load_state_dict(ck['weights']);net.eval()
    @torch.inference_mode()
    def call(seq):
        x=torch.as_tensor(seq[:,:,:len(mu)]);return net(((x-mu)/sd).clamp(-10,10)).exp().numpy()
    return call
if __name__=='__main__':collect();train()
