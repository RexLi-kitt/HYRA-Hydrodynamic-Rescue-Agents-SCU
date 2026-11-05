"""Isolated continuous attenuation of weak model contrast; unchanged physics."""
import numpy as np
from train import FineEnv

def contrast(prediction):
    p=np.asarray(prediction,dtype=float)
    if p.size!=100 or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Expected100 finite nonnegative counts')
    return float(np.max(np.abs(p-np.median(p))))

def strength(prediction,reference):
    if not np.isfinite(reference) or reference<=0:
        raise ValueError('Positive FIT-only reference required')
    return min(1.,contrast(prediction)/reference)

class GatedEnv(FineEnv):
    def __init__(self,water,scenario,predict,reference):
        self.reference=float(reference);self.gate_trace=[]
        super().__init__(water,scenario,predict,.25)

    def search_score(self,stale,risk):
        x=self.history.get(self)
        if self.prediction is None or self.predicted_at!=self.history.time:
            self.prediction=np.asarray(self.predict(x)).reshape(10,10)
            self.predicted_at=self.history.time
            self.gate_trace.append(dict(time=int(self.t),contrast=contrast(self.prediction),
                                       strength=strength(self.prediction,self.reference)))
        # Calculate strength BEFORE existing flow transport: transport edge loss
        # must not turn almost-uniform model output into confident evidence.
        gain=strength(self.prediction,self.reference)
        original=self.alpha
        try:
            self.alpha=original*gain
            return super().search_score(stale,risk)
        finally:self.alpha=original
