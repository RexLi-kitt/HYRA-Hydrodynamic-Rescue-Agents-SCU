"""Bounded spatial correction for an unchanged strong-planner score.
Isolated candidate, not yet wired into training or production.
"""
import numpy as np

def neutral_residual_score(base, prediction=None, *, alpha=0., transport_fn=None):
    if not np.isfinite(alpha) or not 0 <= alpha <= .25:
        raise ValueError('alpha must be finite in [0, .25]')
    b=np.asarray(base)
    if b.shape != (100,100) or not np.isfinite(b).all() or (b<0).any():
        raise ValueError('base must be a finite nonnegative 100x100 score')
    if alpha == 0 or prediction is None:
        return b.copy()
    p=np.asarray(prediction,dtype=float)
    if p.shape not in ((100,), (10,10)) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('prediction must contain 100 finite nonnegative entries')
    p=p.reshape(10,10)
    # Neutral BEFORE transport: open-edge mass loss must not create model evidence.
    if np.ptp(p) == 0:
        return b.copy()
    q=p.copy() if transport_fn is None else np.asarray(transport_fn(p.copy()),dtype=float)
    if q.shape != (10,10) or not np.isfinite(q).all() or (q<0).any():
        raise ValueError('transport output invalid')
    delta=q-np.median(q)
    scale=np.max(np.abs(delta))
    if scale == 0: return b.copy()
    residual=np.repeat(np.repeat(delta/scale,10,axis=0),10,axis=1)
    return b*(1+alpha*residual)

