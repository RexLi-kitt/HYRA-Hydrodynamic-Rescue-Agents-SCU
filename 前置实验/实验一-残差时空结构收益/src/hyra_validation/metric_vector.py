"""Vectorized exact piecewise-linear full-window scores at fractional travel times."""
import numpy as np


def interval_maximum_float(curves,begin,duration):
    curves=np.asarray(curves);begin=np.broadcast_to(np.asarray(begin,float),curves.shape[:-1]);end=begin+duration
    if np.any(begin<0) or np.any(end>2*(curves.shape[-1]-1)+1e-9):raise ValueError('Window outside curve')
    def endpoint(t):
        lo=np.floor(t/2).astype(int);hi=np.minimum(lo+1,curves.shape[-1]-1);weight=t/2-lo
        a=np.take_along_axis(curves,lo[...,None],axis=-1)[...,0]
        b=np.take_along_axis(curves,hi[...,None],axis=-1)[...,0]
        return a+(b-a)*weight
    indices=np.floor(begin/2).astype(int)[...,None]+np.arange(int(duration//2)+2)
    valid=(indices*2>=begin[...,None]) & (indices*2<=end[...,None])
    gathered=np.take_along_axis(curves,np.clip(indices,0,curves.shape[-1]-1),axis=-1)
    internal=np.max(np.where(valid,gathered,-np.inf),axis=-1)
    return np.maximum(np.maximum(endpoint(begin),endpoint(end)),internal)


def metric_vector_scores(curves,cases,row_times,row_targets):
    curves=np.asarray(curves)
    if curves.ndim==2:curves=curves[None]
    lookup={(float(t),int(r)):i for i,(t,r) in enumerate(zip(row_times,row_targets))}
    counts=[];margins=[];starts=[];maxima=[];common=[]
    for order_index in range(2):
        indices=np.array([[lookup[float(c['now_s']),r] for r in c['orders'][order_index][0]] for c in cases])
        travel=np.array([c['orders'][order_index][1] for c in cases])
        first=curves[:,indices[:,0]];second=curves[:,indices[:,1]]
        s1=np.broadcast_to(travel[:,0],first.shape[:-1])
        f6=interval_maximum_float(first,s1,6);f12=interval_maximum_float(first,s1,12)
        s2_abort=s1+6+travel[:,1];s2_full=s1+12+travel[:,1]
        a12=interval_maximum_float(second,s2_abort,12);b12=interval_maximum_float(second,s2_full,12)
        abort=f6>.8;s2=np.where(abort,s2_abort,s2_full);s12=np.where(abort,a12,b12)
        counts.append((f12<=.8).astype(np.int8)+(s12<=.8).astype(np.int8))
        margins.append(np.minimum(.8-f12,.8-s12));starts.append(np.stack([s1,s2],axis=-1))
        maxima.append(np.stack([f12,s12],axis=-1));common.append(np.stack([f12,a12,b12],axis=-1))
    count=np.stack(counts,axis=-1);margin=np.stack(margins,axis=-1)
    prefer1=np.array([c['orders'][1][0]<c['orders'][0][0] for c in cases])
    choose=((count[:,:,1]>count[:,:,0]) | ((count[:,:,1]==count[:,:,0]) &
        ((margin[:,:,1]>margin[:,:,0]+1e-12) | ((abs(margin[:,:,1]-margin[:,:,0])<=1e-12)&prefer1))))
    return dict(count=count,margin=margin,starts=np.stack(starts,axis=2),maxima=np.stack(maxima,axis=2),
                common_maxima=np.stack(common,axis=2),chosen=choose.astype(np.int8))
