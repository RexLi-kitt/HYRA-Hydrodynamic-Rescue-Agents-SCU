"""Fit-only fixed-feature head capacity/optimizer check, never a policy trial."""
import argparse,json,time
from pathlib import Path
from copy import deepcopy
from scale100_train import *
from scale100_aux_tuning import tuned_sources
from scale100_rules import water_aware

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    meta=json.loads((a.model/'actor.json').read_text());assert meta['sources']==tuned_sources() and digest(a.model/'actor.pt')==meta['actor_sha256']
    actor=Agent();actor.load_state_dict(torch.load(a.model/'actor.pt',weights_only=True));actor.eval();manifest=json.loads(DATA.read_text());xs=[];beds=[];targets=[]
    for row in manifest['cases']:
        if row['split']!='fit':continue
        case=FloodCase(row['path']);env=ScaleEnv(case,config(meta['task'],'D',11));view=env.reset();hidden=actor.hidden(2);previous=torch.zeros(2,8)
        for t in range(181):
            with torch.no_grad():_,hidden,_=actor(torch.from_numpy(view.images.copy()),previous,t,hidden,False)
            if t in (0,60,120,180):
                xs.append(hidden.detach().clone());beds.append(torch.tensor(case.bed,dtype=torch.float32)[None].expand(2,-1,-1).clone());targets.append(torch.tensor(np.stack((case.probes['before'][t],case.probes['after'][t])),dtype=torch.float32)[None].expand(2,-1,-1,-1,-1).clone())
            if t<180:
                actions=water_aware(env);view,_,_=env.step(actions);previous=F.one_hot(torch.tensor(actions),8).float()
    x=torch.cat(xs);bed=torch.cat(beds);target=torch.cat(targets);scale=torch.tensor([.5,.25,.25])[None,None,:,None,None];rows=[]
    for name in ('rmsprop_1e-5','rmsprop_1e-8','adam_1e-8'):
        model=deepcopy(actor);model.requires_grad_(False);model.aux.requires_grad_(True)
        optimizer=torch.optim.Adam(model.aux.parameters(),lr=1e-3,eps=1e-8) if name.startswith('adam') else torch.optim.RMSprop(model.aux.parameters(),lr=1e-3,alpha=.99,eps=1e-5 if name.endswith('1e-5') else 1e-8)
        start=time.perf_counter()
        for update in range(257):
            pred=model.predict(x,bed);loss=((pred-target)/scale).square().mean()
            if update in (0,16,64,128,256):rows.append(dict(optimizer=name,update=update,data_normalized_mse=float(loss.detach()),h_mae_m=float((pred[:,:,0]-target[:,:,0]).detach().abs().mean()),elapsed_s=time.perf_counter()-start))
            if update<256:optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    report=dict(schema='hyra-scale100-fit-only-head-optimizer-probe-v1',source_sha256=digest(__file__),starting_actor_sha256=meta['actor_sha256'],fit_parent_count=3,fixed_encoded_samples=len(x),optimization_steps=256,shared_encoder_frozen=True,policy_training=False,development_scored=False,holdout_scored=False,model_weights_exported=False,rows=rows,interpretation='Training-only capacity/conditioning diagnostic. A lower loss here is not generalization or task benefit.')
    write(a.output,report)
    for row in rows:
        if row['update'] in (0,256):print(json.dumps(row),flush=True)
if __name__=='__main__':main()
