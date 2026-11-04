import numpy as np
from .risk import Risk
from flood_dqn.env import DIRECTIONS,close
from coordinated_search_dqn.planner import geometry


class Guard(Risk):
    def __init__(self):
        super().__init__()
        self.options.update(width=34,depth=15,commit=5,cooperate=True)

    def actions(self,e,fixed):
        intended=super().actions(e,fixed)
        targets=e.targets[e.found&~e.rescued]
        ends,_,ids=geometry(e.p[:,None],e.heading[:,None],DIRECTIONS,targets)
        lens=np.linalg.norm(ends-e.p[:,None],axis=2)
        fresh=np.r_[e.stamps<0,False]
        published={i:int(a) for i,a in enumerate(fixed) if a>=0}
        for k in range(4):
            i=(e.step_no//6+k)%4
            if i in published:continue
            def clashes(a):
                return any(close(e.p[i],DIRECTIONS[a],lens[i,a],e.p[j],DIRECTIONS[b],lens[j,b]) for j,b in published.items())
            a=int(intended[i])
            if clashes(a):
                opts=[v for v in range(9) if not clashes(v)]
                if opts:a=max(opts,key=lambda v:fresh[ids[i,v]].sum()+DIRECTIONS[v]@DIRECTIONS[intended[i]])
            intended[i]=a;published[i]=a
        return intended

