"""Fit P0/P2/P3 and capacity-matched P4/P5 without opening validation labels."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from hyra_validation.artifacts import new_run,write_json
from hyra_validation.metric_features import raw_features,empirical_features,physical_grid
from hyra_validation.metric_readers import fit_ridge,fit_pointwise,pointwise_extras


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train-count',type=int,default=24);parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--skip-neural',action='store_true');args=parser.parse_args()
    design=json.loads((ROOT/'configs/metric_development.design.v1.json').read_text(encoding='utf-8'))
    cache=Path(json.loads((ROOT/'results/latest_metric_feature_cache.json').read_text(encoding='utf-8'))['folder'])
    paths=[cache/f'metric_v2_train_{i:03}.npz' for i in range(args.train_count)]
    for p in paths:assert p.exists() and p.with_suffix('.json').exists(),str(p)
    cfg=dict(stage='training_diagnostic' if args.train_count<24 else 'development_training',train_count=args.train_count,
        cache=str(cache),train_reference_mesh=80,validation_opened=False,pilot_or_confirmation_opened=False,
        alpha_grid=[.1,1.,10.,100.,1000.],empirical_width_s=[5.,10.,20.],pointwise_alpha=10.,
        neural_kinds=['mlp','gru'],neural_losses=['mse','threshold_mse'],neural_checkpoints=[20,60,120],neural_epochs=args.epochs,
        hybrid_grid=4,hybrid_groups=['P4','P5'],hybrid_policy='same full raw MLP/GRU encoder; own-lead physical prior plus learned correction; identical P4/P5 parameter count seed batches losses and checkpoints',
        physical_grid=physical_grid(design),pointwise_policy='same-horizon physical scalar plus global scalar and source event probability; common nominal context shared',
        source_reference='pinned FloodGNN-GRU inspected; independent native torch temporal control; no reproduction claim')
    run=new_run('metric_model_training',cfg);(run/'models').mkdir();print('RUN '+str(run),flush=True)
    # Store actual source copies so subsequent implementation changes cannot erase provenance.
    for rel in json.loads((run/'provenance.json').read_text(encoding='utf-8'))['files']:
        target=run/'frozen_source'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/rel).read_bytes())
    receipts=[];arrays={k:[] for k in ('common','history','base','truth','residual','physical','global_component','event_probability','weight')}
    for path in paths:
        receipt=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['cache_sha256'];receipts.append(receipt)
        with np.load(path,allow_pickle=False) as z:
            selected=z['metadata'][:,0]==80
            for key in arrays:
                a=z['metadata'][selected,8] if key=='weight' else z[key][selected]
                # Arrays have block,time,... except physical/global/prob with grid axes first.
                if key=='physical':a=a.transpose(0,3,1,2,4,5).reshape(-1,9,2,12,26)
                elif key=='global_component':a=a.transpose(0,2,1,3,4).reshape(-1,9,12,26)
                elif key=='event_probability':a=a.transpose(0,2,1).reshape(-1,9)
                elif key=='weight':a=np.repeat(a,6)
                else:a=a.reshape(-1,*a.shape[2:])
                arrays[key].append(a)
        print('LOADED '+path.stem,flush=True)
    arrays={k:np.concatenate(v) for k,v in arrays.items()};w=arrays['weight'];y=arrays['residual'];common=arrays['common'];history=arrays['history']
    write_json(run/'training_inputs.json',dict(families=receipts,context_time_rows=len(y),target_rows=len(y)*12,
        shapes={k:list(v.shape) for k,v in arrays.items()},target_labels_used_only_for_training=True))
    raw=raw_features(history).astype(np.float32);empirical={width:empirical_features(history,width).astype(np.float32) for width in cfg['empirical_width_s']}
    registry=[]
    def save(name,group,kind,model,**spec):
        path=run/'models'/(name+'.pkl');path.write_bytes(pickle.dumps(model,protocol=5))
        registry.append(dict(name=name,group=group,kind=kind,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**spec))
        write_json(run/'registry.json',registry)
    for group,extra in [('P0',None),('P2',raw)]+[(f'P3_w{width:g}',empirical[width]) for width in cfg['empirical_width_s']]:
        for alpha in cfg['alpha_grid']:
            models=[fit_ridge(common[:,target] if extra is None else np.c_[common[:,target],extra],y[:,target],w,alpha) for target in range(12)]
            save(f'{group}_ridge_a{alpha:g}',group.split('_')[0],'ridge',models,alpha=alpha,features='common' if extra is None else ('raw' if group=='P2' else 'empirical'),width_s=None if not group.startswith('P3') else float(group.split('w')[1]))
        print('FITTED '+group,flush=True)
    for g in range(9):
        for mode,group in enumerate(('P4','P5')):
            models=[]
            for target in range(12):
                e=pointwise_extras(arrays['physical'][:,g,mode,target],arrays['global_component'][:,g,target],arrays['event_probability'][:,g])
                models.append(fit_pointwise(common[:,target],e,y[:,target],w,cfg['pointwise_alpha']))
            save(f'{group}_pointwise_g{g}',group,'pointwise',models,grid=g,mode=mode)
        print('FITTED PHYSICAL_GRID '+str(g),flush=True)
    if not args.skip_neural:
        import torch
        from hyra_validation.metric_nn import SourceTargetNetwork,normalization
        assert torch.cuda.is_available();torch.set_num_threads(1)
        normal=normalization(common,history,w);device='cuda'
        c=torch.as_tensor((common-normal['common_mean'])/normal['common_scale'],dtype=torch.float32,device=device)
        h=torch.as_tensor((history-normal['history_mean'])/normal['history_scale'],dtype=torch.float32,device=device)
        target=torch.as_tensor(y,dtype=torch.float32,device=device);weight=torch.as_tensor(w,dtype=torch.float32,device=device)
        threshold=1+4*torch.exp(-((torch.as_tensor(arrays['truth'],dtype=torch.float32,device=device)-.8)/.05)**2)
        for kind in cfg['neural_kinds']:
            for loss_kind in cfg['neural_losses']:
                seed=2026101609+(0 if kind=='mlp' else 100)+(0 if loss_kind=='mse' else 1);torch.manual_seed(seed)
                model=SourceTargetNetwork(common.shape[-1],kind).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                started=time.perf_counter();losslog=[]
                for epoch in range(1,args.epochs+1):
                    model.train();permutation=torch.randperm(len(y),device=device);epochloss=0.
                    for selection in permutation.split(256):
                        optimizer.zero_grad(set_to_none=True);prediction=model(c[selection],h[selection]);error=((prediction-target[selection])/.05)**2
                        if loss_kind=='threshold_mse':error=error*threshold[selection]
                        loss=(error.mean(dim=(1,2))*weight[selection]).sum()/weight[selection].sum();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                        epochloss+=float(loss.detach())*len(selection)
                    losslog.append(dict(epoch=epoch,loss=epochloss/len(y)))
                    if epoch in cfg['neural_checkpoints'] or epoch==args.epochs:
                        name=f'P2_{kind}_{loss_kind}_e{epoch}';path=run/'models'/(name+'.pt')
                        payload=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},normalization={k:torch.tensor(v) for k,v in normal.items()},kind=kind,common_size=common.shape[-1],seed=seed)
                        torch.save(payload,path);registry.append(dict(name=name,group='P2',kind='neural',network_kind=kind,loss=loss_kind,epoch=epoch,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                        write_json(run/'registry.json',registry);write_json(run/f'loss_{kind}_{loss_kind}.json',losslog)
                        print(f'NEURAL {kind} {loss_kind} epoch={epoch} loss={losslog[-1]["loss"]:.6g} seconds={time.perf_counter()-started:.1f}',flush=True)
        from hyra_validation.metric_nn import PhysicalSourceTargetNetwork
        g=cfg['hybrid_grid'];glob=torch.as_tensor(arrays['global_component'][:,g],dtype=torch.float32,device=device)
        probability=torch.as_tensor(arrays['event_probability'][:,g],dtype=torch.float32,device=device)
        for mode,group in enumerate(('P4','P5')):
            physical=torch.as_tensor(arrays['physical'][:,g,mode],dtype=torch.float32,device=device)
            for kind in cfg['neural_kinds']:
                for loss_kind in cfg['neural_losses']:
                    # Deliberately identical random seeds and batches for the P4/P5 pair.
                    seed=2026101611+(0 if kind=='mlp' else 100)+(0 if loss_kind=='mse' else 1);torch.manual_seed(seed)
                    model=PhysicalSourceTargetNetwork(common.shape[-1],kind).to(device)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);started=time.perf_counter();losslog=[]
                    for epoch in range(1,args.epochs+1):
                        model.train();permutation=torch.randperm(len(y),device=device);epochloss=0.
                        for selection in permutation.split(128):
                            optimizer.zero_grad(set_to_none=True)
                            prediction=model(c[selection],h[selection],physical[selection],glob[selection],probability[selection])
                            error=((prediction-target[selection])/.05)**2
                            if loss_kind=='threshold_mse':error=error*threshold[selection]
                            loss=(error.mean(dim=(1,2))*weight[selection]).sum()/weight[selection].sum()
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();epochloss+=float(loss.detach())*len(selection)
                        losslog.append(dict(epoch=epoch,loss=epochloss/len(y)))
                        if epoch in cfg['neural_checkpoints'] or epoch==args.epochs:
                            name=f'{group}_{kind}_{loss_kind}_g{g}_e{epoch}';path=run/'models'/(name+'.pt')
                            payload=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},normalization={k:torch.tensor(v) for k,v in normal.items()},kind=kind,common_size=common.shape[-1],seed=seed)
                            torch.save(payload,path);registry.append(dict(name=name,group=group,kind='neural_physical',network_kind=kind,loss=loss_kind,epoch=epoch,grid=g,mode=mode,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                            write_json(run/'registry.json',registry);write_json(run/f'loss_{group}_{kind}_{loss_kind}.json',losslog)
                            print(f'HYBRID {group} {kind} {loss_kind} epoch={epoch} loss={losslog[-1]["loss"]:.6g} seconds={time.perf_counter()-started:.1f}',flush=True)
    write_json(run/'completion.json',dict(completed=True,models=len(registry),training_families=args.train_count,validation_opened=False,pilot_or_confirmation_opened=False))
    write_json(ROOT/'results/latest_metric_model_training.json',dict(run=str(run),stage=cfg['stage'],completed=True))
    print('DONE '+str(run),flush=True)


if __name__=='__main__':main()
