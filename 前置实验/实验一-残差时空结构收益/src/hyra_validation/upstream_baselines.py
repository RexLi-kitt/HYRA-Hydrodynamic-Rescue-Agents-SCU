"""Common-source ordinary updates and spatial transfer for hidden targets.

The temporal update is the existing vector implementation independently checked
against the pinned FilterPy source. Only these new spatial weights are added.
No target residual or future reference value is an input.
"""
from __future__ import annotations

import numpy as np

from .hidden_target_inputs import causal_history, SOURCES
from .local_service import conventional_predictions


def source_histories(records, decision_times):
    histories = [causal_history(records,float(now)) for now in decision_times]
    values = np.stack([h['values'] for h in histories])
    conventional = values[:, :, :, [2,3,4,5]].copy()
    conventional[:, :, :, 1] /= 60.
    conventional[:, :, :, 2] /= 30.
    return dict(raw=values, conventional=conventional.reshape(-1,6,4),
                record_ids=np.stack([h['record_ids'] for h in histories]),
                available_sources=np.any(values[:, :, :, 5]>0,axis=2),
                source_ids=np.array(SOURCES))


def spatial_weights(source_xy,target_xy,available,kind):
    source_xy=np.asarray(source_xy,float);target_xy=np.asarray(target_xy,float)
    available=np.asarray(available,bool)
    if source_xy.shape!=(6,2) or target_xy.ndim!=2 or target_xy.shape[1]!=2 or available.ndim!=2 or available.shape[1]!=6:
        raise ValueError('Invalid common-source spatial grid')
    if not np.all(np.isfinite(source_xy)) or not np.all(np.isfinite(target_xy)):
        raise ValueError('Nonfinite spatial location')
    distance=np.linalg.norm(target_xy[:,None,:]-source_xy[None,:,:],axis=-1)
    if kind=='uniform':base=np.ones_like(distance)
    elif kind=='idw1':base=1/np.maximum(distance,1.)
    elif kind=='idw2':base=1/np.maximum(distance,1.)**2
    else:raise ValueError('Unknown spatial baseline')
    weights=base[None,:,:]*available[:,None,:]
    total=weights.sum(axis=-1,keepdims=True)
    return np.divide(weights,total,out=np.zeros_like(weights),where=total>0)


def ordinary_target_corrections(histories,source_xy,target_xy,config,gains=(.25,.5,1.)):
    if not np.array_equal(histories['source_ids'],SOURCES):raise ValueError('Non-common source network')
    updates=conventional_predictions(histories['conventional'],config)
    count=len(histories['raw']);result={}
    for kind in ('uniform','idw1','idw2'):
        weights=spatial_weights(source_xy,target_xy,histories['available_sources'],kind)
        for update,values in updates.items():
            if update=='nominal':continue
            source_curves=values.reshape(count,6,-1)
            mixed=np.einsum('nts,nsq->ntq',weights,source_curves)
            for gain in gains:
                if not np.isfinite(gain) or gain<0:raise ValueError('Invalid transfer gain')
                result[f'{update}_{kind}_g{gain:g}']=mixed*gain
    return result
