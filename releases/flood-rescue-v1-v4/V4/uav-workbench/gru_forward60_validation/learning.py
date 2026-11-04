"""Supervised GRU for public regional histories, trained only on P0 trajectories."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from time import perf_counter
import numpy as np
import torch
from torch import nn
from flood_dqn import maps
from release_utils import atomic_torch, model_file
from release_utils import write_rows

OUT=maps.ROOT/'generated'/'gru-forward60-validation-v1'
SEEDS=(11,29,47)


class DemandNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(14,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(32,1),nn.Softplus())

    def forward(self,x):
        hidden=self.gru(x)[1][-1]
        return self.head(hidden).squeeze(-1)


def collect_case(job):
    from .scenario import make
    from .env import run
    p,n=job
    file=OUT/'samples'/f'{p}-{n}.npz'
    if file.exists():return
    _,_,X,Y=run(make(p,n),'P0',collect=True)
    # A fixed 30-second training stride; frame offsets match per-second inference.
    x=np.asarray(X,dtype=np.float32)[::30]
    y=np.asarray(Y,dtype=np.float32)[::30]
    assert x.shape==(50,100,14) and y.shape==(50,100) and np.isfinite(x).all()
    tmp=file.with_suffix('.part.npz')
    np.savez_compressed(tmp,x=x,y=y)
    tmp.replace(file)


def dataset(parents):
    xs=[];ys=[]
    for p in parents:
        for n in range(4):
            with np.load(OUT/'samples'/f'{p}-{n}.npz') as pack:x,y=pack['x'],pack['y']
            pad=np.concatenate((np.zeros_like(x[:3]),x))
            seq=np.stack([pad[i:i+50] for i in range(4)],axis=2)
            xs.append(seq.reshape(-1,4,14));ys.append(y.flatten())
    return np.concatenate(xs),np.concatenate(ys)


@lru_cache(maxsize=3)
def predictor(seed):
    torch.set_num_threads(1)
    ck=torch.load(model_file(OUT,f'M-{seed}.pt'),map_location='cpu',weights_only=True)
    net=DemandNet();net.load_state_dict(ck['weights']);net.eval()
    @torch.inference_mode()
    def predict(x):
        v=torch.from_numpy(np.asarray(x,dtype=np.float32))
        return net(((v-ck['mean'])/ck['scale']).clamp(-10,10)).numpy()
    return predict


def prepare_train():
    (OUT/'samples').mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(6) as pool:list(pool.map(collect_case,[(p,n) for p in range(21001,21017) for n in range(4)]))
    torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    x,y=dataset(range(21001,21013))
    z,v=dataset(range(21013,21017))
    mean=x[:,-1].mean(0,dtype=np.float64).astype('f');scale=x[:,-1].std(0,dtype=np.float64).astype('f').clip(.001)
    def normalize(x):return torch.from_numpy(((x-mean)/scale).clip(-10,10)).to(device)
    X=normalize(x);Z=normalize(z);Y=torch.from_numpy(y).to(device)
    criterion=nn.PoissonNLLLoss(log_input=False)
    metrics=[]
    for seed in SEEDS:
        file=OUT/f'M-{seed}.pt'
        if not file.exists():
            start=perf_counter()
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            net=DemandNet().to(device);opt=torch.optim.Adam(net.parameters(),lr=1e-3)
            curve=[]
            for epoch in range(32):
                ids=torch.as_tensor(rng.permutation(len(y)),device=device)
                loss_sum=torch.zeros((),device=device)
                for batch in ids.split(1024):
                    loss=criterion(net(X[batch]),Y[batch])
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    loss_sum+=loss.detach()*len(batch)
                curve.append(dict(epoch=epoch+1,poisson=float(loss_sum.item()/len(y))))
            write_rows(OUT/f'M-{seed}-curve.csv',curve)
            seconds=perf_counter()-start
            atomic_torch(file,dict(weights=net.cpu().state_dict(),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale),seconds=seconds))
        ck=torch.load(file,map_location='cpu',weights_only=True)
        net=DemandNet().to(device);net.load_state_dict(ck['weights']);net.eval()
        with torch.inference_mode():
            pred=torch.cat([net(batch) for batch in Z.split(1024)]).cpu().numpy()
        assert np.allclose(predictor(seed)(z[:100]),pred[:100],rtol=1e-4,atol=1e-5)
        metrics.append(dict(seed=seed,seconds=ck['seconds'],sha256=maps.digest(file),cal_mae=float(np.mean(abs(pred-v))),cal_poisson=float(np.mean(pred-v*np.log(pred.clip(1e-8))))))
    maps.save_json(OUT/'training.json',dict(device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,fit_rows=len(y),cal_rows=len(v),stride_seconds=30,epochs=32,models=metrics))
