"""Use calibrated expected counts without max-contrast normalization or a 25% cap."""
import copy
import numpy as np
from joint_avoidance import SafeFineEnv
from adapter import FlowFogEnv,transport,pool

def count_score(base,p):
    p=np.asarray(p,float)
    if p.shape!=(10,10) or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid expected counts')
    if np.ptp(p)<1e-12:return base.copy()
    return base*(1+8*np.repeat(np.repeat(p,10,0),10,1))

class CountCoupledEnv(SafeFineEnv):
    def search_score(self,stale,risk):
        baseline=copy.copy(self);baseline.method='flow_coverage'
        base=FlowFogEnv.search_score(baseline,stale,risk)
        x=self.history.get(self)
        if self.prediction is None or self.predicted_at!=self.history.time:
            self.prediction=np.asarray(self.predict(x)).reshape(10,10)
            self.predicted_at=self.history.time
        fresh=self.known&(self.t-self.last<=120)
        u=pool(np.where(fresh,self.measured[1],0));v=pool(np.where(fresh,self.measured[2],0))
        p=.5*self.prediction+.5*transport(self.prediction,u,v,60)
        return count_score(base,p)

class OracleCoupledEnv(CountCoupledEnv):
    """Explicit privileged diagnostic, never a deployable model."""
    def __init__(self,w,s):super().__init__(w,s,lambda _:self.sector_counts(100).reshape(-1),.25)

if __name__=='__main__':
    b=np.arange(10000,dtype=float).reshape(100,100)+1
    np.testing.assert_array_equal(count_score(b,np.zeros((10,10))),b)
    np.testing.assert_array_equal(count_score(b,np.ones((10,10))),b)
    p=np.zeros((10,10));p[3,4]=1
    q=count_score(b,p);np.testing.assert_array_equal(q[30:40,40:50],9*b[30:40,40:50])
    np.testing.assert_array_equal(q[:20],b[:20])
    print('3 count-coupling invariants passed')
