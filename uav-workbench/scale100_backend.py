"""Explicit registry for training variants that share the same21-channel actor.

Inference is deliberately unchanged. Metadata/source verification chooses the
variant; tensor shapes alone never identify the training algorithm.
"""
from pathlib import Path
import scale100_train as base

BACKENDS=('absolute_head_v1','aux_optimization_v2','aux_precision_v3','aux_resilient_v4')

def source_hashes(backend):
    if backend=='absolute_head_v1':return base.sources()
    if backend=='aux_optimization_v2':
        from scale100_aux_tuning import tuned_sources
        return tuned_sources()
    if backend=='aux_precision_v3':
        from scale100_aux_precision import precision_sources
        return precision_sources()
    if backend=='aux_resilient_v4':
        from scale100_aux_resilient import resilient_sources
        return resilient_sources()
    raise ValueError('Unregistered training backend')

def validate_meta(meta,backend):
    family=meta.get('model_family','absolute_head_v1')
    expected={'absolute_head_v1':'absolute_head_v1','aux_optimization_v2':'same_actor_aux_optimization_v2','aux_precision_v3':'same_actor_aux_precision_v3','aux_resilient_v4':'same_actor_aux_resilient_v4'}.get(backend)
    if expected is None:raise ValueError('Unregistered training backend')
    if family!=expected or meta['sources']!=source_hashes(backend):raise ValueError('Training backend/source metadata mismatch')
    if backend=='aux_resilient_v4' and meta.get('auxiliary_activation')!='leaky_relu_0.05':raise ValueError('Missing resilient decoder activation contract')

def make_actor(backend):
    if backend not in BACKENDS:raise ValueError('Unregistered training backend')
    actor=base.Agent()
    if backend=='aux_resilient_v4':actor.aux[1]=base.nn.LeakyReLU(.05)
    return actor

def train_backend(backend,dataset,output,task,method,seed,episodes,stop_after,resume,phase='formal'):
    if backend=='absolute_head_v1':function=base.train
    elif backend=='aux_optimization_v2':
        from scale100_aux_tuning import train_tuned
        function=train_tuned
    elif backend=='aux_precision_v3':
        from scale100_aux_precision import train_precise
        function=train_precise
    elif backend=='aux_resilient_v4':
        from scale100_aux_resilient import train_resilient
        function=train_resilient
    else:raise ValueError('Unregistered training backend')
    return function(dataset,output,task,method,seed,episodes,stop_after,resume,phase)
