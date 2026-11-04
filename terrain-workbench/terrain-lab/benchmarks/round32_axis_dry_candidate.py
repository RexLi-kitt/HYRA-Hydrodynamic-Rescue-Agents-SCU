"""Experimental axis-wise dry fallback; never installed as production default."""
import numpy as np
from refined_solver import RefinedFloodSolver
class AxisDryCandidate(RefinedFloodSolver):
    def _reconstruct(self):
        m=self.mesh;h=self.leaf['h'];u,v=self.leaf_velocity()
        if self.spatial_order==1:
            gradients=[(np.zeros(m.count),np.zeros(m.count)) for _ in range(4)]
        else:
            dh=h[self.ir]-h[self.il]
            gradients=[self._gradient(h,dh),self._gradient(h,dh+(self.leaf['bed'][self.ir]-self.leaf['bed'][self.il])),
                self._gradient(u),self._gradient(v)]
            dry_neighbor=(h[self.il]<=self.velocity_depth)|(h[self.ir]<=self.velocity_depth)
            axes=m.axis[self.neighbors]
            for axis in (1,0):
                near=h<=self.velocity_depth
                selected=dry_neighbor & (axes==axis)
                near[self.il[selected]]=True;near[self.ir[selected]]=True
                for gx,gz in gradients:
                    (gx if axis==1 else gz)[near]=0
            # Positivity at every subface, including hanging-node interfaces.
            factor=np.ones(m.count);gx,gz=gradients[0]
            for valid,cell,ox,oz in ((m.lv,m.l,m.ox_l,m.oz_l),(m.rv,m.r,m.ox_r,m.oz_r)):
                delta=gx[cell]*ox+gz[cell]*oz
                ratio=np.divide(h[cell],-delta,out=np.ones_like(delta),where=delta<0)
                np.minimum.at(factor,cell[valid],np.clip(ratio[valid],0,1))
            gradients[0]=(gx*factor,gz*factor)
        gh,geta,gu,gv=gradients
        gb=(geta[0]-gh[0],geta[1]-gh[1])
        sides=[]
        for cell,ox,oz in ((m.l,m.ox_l,m.oz_l),(m.r,m.ox_r,m.oz_r)):
            extrapolate=lambda pair:pair[0][cell]*ox+pair[1][cell]*oz
            sides.append([np.maximum(0,h[cell]+extrapolate(gh)),extrapolate(gb),
                u[cell]+extrapolate(gu),v[cell]+extrapolate(gv)])
        return sides

