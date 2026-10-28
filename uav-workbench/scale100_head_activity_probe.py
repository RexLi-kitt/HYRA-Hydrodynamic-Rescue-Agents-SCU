"""Fit-only decoder activity and latent-gradient diagnostic; no policy update."""
import argparse,json
from pathlib import Path
from scale100_train import *
from scale100_aux_precision import precision_sources
from scale100_rules import water_aware

ROOT=Path(__file__).parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=ROOT/'generated/scale100-aux-precision-20261026');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);entries=[r for r in json.loads(DATA.read_text())['cases'] if r['split']=='fit'];rows=[]
    for method in ('D','P'):
        model=a.folder/f'E1-{method}-s11';meta=json.loads((model/'actor.json').read_text());assert meta['sources']==precision_sources() and digest(model/'actor.pt')==meta['actor_sha256'];actor=Agent();actor.load_state_dict(torch.load(model/'actor.pt',weights_only=True));actor.eval()
        for entry in entries:
            case=FloodCase(entry['path']);env=ScaleEnv(case,config('E1',method,11));view=env.reset();hidden=actor.hidden(2);previous=torch.zeros(2,8)
            for t in range(181):
                with torch.no_grad():_,hidden,_=actor(torch.from_numpy(view.images.copy()),previous,t,hidden,False)
                if t in (0,60,120,180):
                    latent=hidden.detach().clone().requires_grad_(True);bed=torch.tensor(case.bed,dtype=torch.float32)[None].expand(2,-1,-1);pre=actor.aux[0](torch.cat((latent,bed.flatten(1)),dim=-1));pred=actor.predict(latent,bed);before=torch.tensor(case.probes['before'][t],dtype=torch.float32)[None];after=torch.tensor(case.probes['after'][t],dtype=torch.float32)[None];scale=torch.tensor([.5,.25,.25])[None,:,None,None]
                    data=(((pred[:,0]-before)/scale).square().mean()+((pred[:,1]-after)/scale).square().mean())/2
                    stepped=SWEOperator().step(pred[:,0],bed,float(case.probes['dt'][t]),source=torch.tensor(case.probes['source'][t],dtype=torch.float32));physics=((stepped-pred[:,1])/scale).square().mean();gd=torch.autograd.grad(data,latent,retain_graph=True)[0];gp=torch.autograd.grad(physics,latent)[0]
                    rows.append(dict(method=method,parent=entry['parent_terrain'],time_s=t*5,positive_hidden_fraction=float((pre>0).float().mean()),preactivation_min=float(pre.detach().min()),preactivation_max=float(pre.detach().max()),data_to_latent_gradient_norm=float(gd.norm()),physics_to_latent_gradient_norm=float(gp.norm()),data_loss=float(data.detach()),physics_loss=float(physics.detach())))
                if t<180:
                    actions=water_aware(env);view,_,_=env.step(actions);previous=F.one_hot(torch.tensor(actions),8).float()
    summary={}
    for method in ('D','P'):
        selected=[r for r in rows if r['method']==method];summary[method]=dict(states=len(selected),states_with_zero_active_units=sum(r['positive_hidden_fraction']==0 for r in selected),states_with_zero_data_gradient=sum(r['data_to_latent_gradient_norm']==0 for r in selected),states_with_zero_physics_gradient=sum(r['physics_to_latent_gradient_norm']==0 for r in selected),mean_positive_fraction=float(np.mean([r['positive_hidden_fraction'] for r in selected])))
    write(a.output,dict(schema='hyra-scale100-fit-only-decoder-activity-v1',source_sha256=digest(__file__),split='fit',policy_updated=False,rows=rows,summary=summary));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
