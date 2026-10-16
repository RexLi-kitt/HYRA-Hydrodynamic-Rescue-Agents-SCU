"""Predictor registry. This module accepts public features, never reference labels."""
import hashlib
import pickle
from pathlib import Path
import numpy as np
from .lead import XY,SOURCES,TARGETS
from .full_upstream import ordinary_full
from .upstream_baselines import spatial_weights
from .metric_features import raw_features,empirical_features
from .metric_readers import predict_ridge,predict_pointwise,pointwise_extras


class MetricCandidates:
    def __init__(self,registry,ordinary_models):
        self.learned=registry;self.ordinary_models=ordinary_models;self.loaded={}
        empty=dict(values=np.zeros((1,6,12,6)))
        temporal=[key for key in ordinary_full(empty,ordinary_models) if key!='nominal']
        self.specs=[dict(name='P0_nominal',group='P0',kind='nominal')]
        self.specs.extend(dict(name=f'P1_{name}_{space}_g{gain:g}',group='P1',kind='ordinary',temporal=name,space=space,gain=gain)
            for name in temporal for space in ('uniform','idw1','idw2') for gain in (.25,.5,1.))
        self.specs.extend(registry)
        self.specs.extend(dict(name=f'{group}_direct_g{g}',group=group,kind='direct',grid=g,mode=mode)
            for g in range(9) for mode,group in enumerate(('P4','P5')))
        names=[s['name'] for s in self.specs]
        if len(set(names))!=len(names):raise ValueError('Duplicate candidates')

    def model(self,spec):
        if spec['name'] not in self.loaded:
            path=Path(spec['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
            if spec['kind'] in ('neural','neural_physical'):
                import torch
                from .metric_nn import SourceTargetNetwork,PhysicalSourceTargetNetwork
                payload=torch.load(path,map_location='cpu',weights_only=True)
                cls=SourceTargetNetwork if spec['kind']=='neural' else PhysicalSourceTargetNetwork
                model=cls(payload['common_size'],payload['kind']).to('cuda');model.load_state_dict(payload['state_dict'])
                self.loaded[spec['name']]=(model,{k:v.numpy() for k,v in payload['normalization'].items()})
            else:self.loaded[spec['name']]=pickle.loads(path.read_bytes())
        return self.loaded[spec['name']]

    def prepare(self,common,history,physical,global_component,event_probability):
        """No mesh, condition, center, anchor, labels, or future observations accepted."""
        updates=ordinary_full(dict(values=history),self.ordinary_models)
        available=np.any(history[:,:,:,5]>0,axis=2)
        return dict(common=common,history=history,physical=physical,global_component=global_component,
            event_probability=event_probability,raw=raw_features(history),
            empirical={width:empirical_features(history,width) for width in (5.,10.,20.)},
            updates=updates,spatial={space:spatial_weights(XY[SOURCES],XY[TARGETS],available,space) for space in ('uniform','idw1','idw2')})

    def predict(self,spec,p):
        kind=spec['kind'];common=p['common']
        if kind=='nominal':return np.zeros((len(common),12,26))
        if kind=='ordinary':return np.einsum('nts,nsq->ntq',p['spatial'][spec['space']],p['updates'][spec['temporal']])*spec['gain']
        if kind=='direct':return p['physical'][spec['grid'],spec['mode']]
        models=self.model(spec)
        if kind=='neural':
            from .metric_nn import predict_network
            return predict_network(*models,common,p['history'])
        if kind=='neural_physical':
            from .metric_nn import predict_physical_network
            g=spec['grid'];mode=spec['mode']
            return predict_physical_network(*models,common,p['history'],p['physical'][g,mode],p['global_component'][g],p['event_probability'][g])
        if kind=='ridge':
            extra=None if spec['features']=='common' else (p['raw'] if spec['features']=='raw' else p['empirical'][spec['width_s']])
            return np.array([predict_ridge(model,common[:,j] if extra is None else np.c_[common[:,j],extra]) for j,model in enumerate(models)]).transpose(1,0,2)
        if kind=='pointwise':
            g=spec['grid'];mode=spec['mode'];physical=p['physical'][g,mode];glob=p['global_component'][g];prob=p['event_probability'][g]
            return np.array([predict_pointwise(model,common[:,j],pointwise_extras(physical[:,j],glob[:,j],prob)) for j,model in enumerate(models)]).transpose(1,0,2)
        raise ValueError(kind)
