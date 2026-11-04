"""Isolated residual adapter. Original planner and executor are unchanged."""
from pathlib import Path
import sys,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-convlstm20-v1'))
from long_flow import FlowFogEnv,transport
from history_model import pool
from neutral_residual import neutral_residual_score

class ResidualEnv(FlowFogEnv):
    def __init__(self,water,scenario,predict=None,alpha=0.):
        self.alpha=alpha
        super().__init__(water,scenario,'residual',predict)
    def search_score(self,stale,risk):
        baseline=copy.copy(self);baseline.method='flow_coverage'
        score=FlowFogEnv.search_score(baseline,stale,risk)
        if self.alpha==0 or self.predict is None:
            return neutral_residual_score(score,alpha=self.alpha)
        x=self.history.get(self)
        if self.prediction is None or self.predicted_at!=self.history.time:
            self.prediction=np.asarray(self.predict(x)).reshape(10,10)
            self.predicted_at=self.history.time
        fresh=self.known&(self.t-self.last<=120)
        u=pool(np.where(fresh,self.measured[1],0));v=pool(np.where(fresh,self.measured[2],0))
        return neutral_residual_score(score,self.prediction,alpha=self.alpha,
            transport_fn=lambda p:.5*p+.5*transport(p,u,v,60))

