"""Training-only normalizations and matched pointwise physical readers."""
import numpy as np


def normalized_weights(weight):
    weight=np.asarray(weight,float)
    if weight.ndim!=1 or np.any(weight<0) or weight.sum()<=0:raise ValueError('Invalid training weights')
    return weight*len(weight)/weight.sum()


def standardize_train(x,weight):
    mean=np.average(x,axis=0,weights=weight);variance=np.average((x-mean)**2,axis=0,weights=weight)
    scale=np.sqrt(variance);scale=np.where(scale>1e-7,scale,1.)
    return mean,scale


def fit_ridge(x,y,weight,alpha):
    x=np.asarray(x,float);y=np.asarray(y,float);w=normalized_weights(weight)
    mean,scale=standardize_train(x,w);z=np.c_[np.ones(len(x)),(x-mean)/scale]
    penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
    gram=z.T@(w[:,None]*z)+penalty;rhs=z.T@(w*y if y.ndim==1 else w[:,None]*y)
    return dict(kind='ridge',mean=mean,scale=scale,coef=np.linalg.solve(gram,rhs),alpha=alpha)


def predict_ridge(model,x):
    z=np.c_[np.ones(len(x)),(np.asarray(x)-model['mean'])/model['scale']]
    return z@model['coef']


def fit_pointwise(x,extra,y,weight,alpha=10.):
    """One common reader plus SAME-HORIZON extras only. No lag undoing by shift."""
    x=np.asarray(x,float);extra=np.asarray(extra,float);y=np.asarray(y,float);w=normalized_weights(weight)
    if extra.shape[:2]!=y.shape or len(x)!=len(y):raise ValueError('Pointwise reader shape mismatch')
    mean,scale=standardize_train(x,w);z=np.c_[np.ones(len(x)),(x-mean)/scale]
    emean,escale=standardize_train(extra,w);e=(extra-emean)/escale
    penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
    gram=z.T@(w[:,None]*z)+penalty
    rhs=z.T@(w[:,None]*y);base=np.linalg.solve(gram,rhs)
    cross=np.einsum('nf,n,nqe->fqe',z,w,e)
    projected=np.linalg.solve(gram,cross.reshape(z.shape[1],-1)).reshape(cross.shape)
    schur=np.einsum('nqe,n,nqk->qek',e,w,e)-np.einsum('fqe,fqk->qek',cross,projected)
    schur+=alpha*np.eye(e.shape[-1])[None]
    erhs=np.einsum('nqe,n,nq->qe',e,w,y)-np.einsum('fqe,fq->qe',cross,base)
    ecoef=np.linalg.solve(schur,erhs[...,None])[...,0]
    xcoef=base-np.einsum('fqe,qe->fq',projected,ecoef)
    return dict(kind='pointwise',mean=mean,scale=scale,extra_mean=emean,extra_scale=escale,
        coef=xcoef,extra_coef=ecoef,alpha=alpha,physical_input_policy='same_horizon_only')


def predict_pointwise(model,x,extra):
    z=np.c_[np.ones(len(x)),(np.asarray(x)-model['mean'])/model['scale']]
    e=(np.asarray(extra)-model['extra_mean'])/model['extra_scale']
    return z@model['coef']+np.einsum('nqe,qe->nq',e,model['extra_coef'])


def pointwise_extras(physical,global_component,probability):
    physical=np.asarray(physical);global_component=np.asarray(global_component)
    probability=np.broadcast_to(np.asarray(probability)[...,None],physical.shape)
    return np.stack([physical,global_component,probability],axis=-1)
