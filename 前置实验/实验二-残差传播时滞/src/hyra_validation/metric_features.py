"""Causal feature cache; offline scenario columns never enter predictor features."""
from itertools import product
import numpy as np
from .nominal_response import NominalResponseLibrary,response_predictions
from .metric_dataset import schedules

MODES=('physical','recency')
METADATA=('mesh','anchor','condition','center','phase','noise_std','delay','replicate','weight')


def physical_grid(design):
    h=design['proposed_physical_hyperparameters']
    return [dict(noise_std=s,beta_std=h['global_relative_forcing_prior_std'],amplitude_std=a)
            for s,a in product(h['noise_std_m'],h['pulse_coefficient_prior_std'])]


def build_family_features(data,progress=None):
    design=data.design;library=NominalResponseLibrary(data.nominal);grid=physical_grid(design)
    outputs={k:[] for k in ('common','history','base','truth','residual','physical','global_component','event_probability','metadata','source_checksum')}
    templates={};labels={};total=0
    # E0 is exactly shared between both centers: retain one copy with weight two.
    scenarios=[('E0',design['centers_y_m'][0],2.)]+[('E1',c,1.) for c in design['centers_y_m']]
    for mesh,anchor in product(design['reference_meshes'],design['anchors_s']):
        for condition,center,multiplicity in scenarios:
            key=(mesh,condition,anchor,center)
            labels[key]=data.labels(*key)
            for setting in schedules(design):
                s={k:v for k,v in setting.items() if k!='weight'}
                inp=data.inputs(mesh,condition,anchor,center,**s);history=inp['history']['values']
                physical=[];global_component=[];probabilities=[]
                for t,now in enumerate(inp['now_s']):
                    template_key=(anchor,s['phase_index'],s['delay'],t)
                    if template_key not in templates:
                        templates[template_key]=library.source_problem(history[t],now,inp['nominal_mean'][t],inp['nominal_drift'][t])
                    template=templates[template_key];active=history[t,:,:,5].ravel()>0
                    problem=dict(template,y=history[t,:,:,2].ravel()[active])
                    predictions=[response_predictions(problem,**g) for g in grid]
                    physical.append(np.array([[p[mode] for mode in MODES] for p in predictions]))
                    global_component.append(np.array([p['global_component'] for p in predictions]))
                    probabilities.append([p['event_probability'] for p in predictions])
                outputs['physical'].append(np.asarray(physical,dtype=np.float32).transpose(1,2,0,3,4))
                outputs['global_component'].append(np.asarray(global_component,dtype=np.float32).transpose(1,0,2,3))
                outputs['event_probability'].append(np.asarray(probabilities,dtype=np.float32).T)
                outputs['common'].append(inp['common'].astype(np.float32));outputs['history'].append(history.astype(np.float32))
                outputs['base'].append(inp['base']);outputs['truth'].append(labels[key]['truth']);outputs['residual'].append(labels[key]['residual'])
                outputs['source_checksum'].append(inp['observation_sha256'])
                outputs['metadata'].append([mesh,anchor,int(condition=='E1'),center,s['phase_index'],s['noise_std'],s['delay'],s['replicate'],setting['weight']*multiplicity])
                total+=1
        if progress:progress(dict(completed_blocks=total,mesh=mesh,anchor=anchor))
    return {k:np.array(v) for k,v in outputs.items()}


def raw_features(history):
    """All 432 values, with time units scaled before training-only normalization."""
    h=np.array(history,float,copy=True);h[...,3]/=60.;h[...,4]/=30.
    return h.reshape(*h.shape[:-3],-1)


def empirical_features(history,width_s):
    """Unsupervised age filters and common-mode detrending, with no hydraulic links."""
    h=np.asarray(history);mask=h[...,5];res=h[...,2];age=h[...,3]
    count=mask.sum(axis=-2,keepdims=True)
    trend=np.divide((res*mask).sum(axis=-2,keepdims=True),count,out=np.zeros_like(count,dtype=float),where=count>0)
    local=(res-trend)*mask;features=[]
    for center in np.arange(0,60,10):
        kernel=np.exp(-.5*((age-center)/width_s)**2)*mask;den=kernel.sum(axis=-1)
        for values in (res,local):
            features.append(np.divide((kernel*values).sum(axis=-1),den,out=np.zeros_like(den),where=den>0))
        features.append(den/12.)
    features.extend([trend[...,0,:].mean(axis=-1)[...,None],mask.mean(axis=(-2,-1))[...,None]])
    return np.concatenate([f if f.shape[-1]==1 else f for f in features],axis=-1)
