"""Past-only observations; simulated fields are never labeled field truth."""
import numpy as np
from .contracts import load_run

class FloodFeed:
    def __init__(self,folder):self.metadata,self.arrays=load_run(folder)
    def observe(self,time_s,visible,wet_threshold_m,uncertainty_m=0.):
        a=self.arrays;t=a['time_s'];mask=np.asarray(visible)
        if mask.dtype!=bool or mask.shape!=a['depth_m'].shape[1:]:raise ValueError('Explicit observation footprint required')
        if not np.isfinite([time_s,wet_threshold_m,uncertainty_m]).all() or wet_threshold_m<=0 or uncertainty_m<0:raise ValueError('Threshold, uncertainty, time required')
        if time_s<t[0] or time_s>t[-1]+1e-10:raise ValueError('Outside known simulation horizon')
        i=int(np.searchsorted(t,time_s,side='right')-1);h=a['depth_m'][i]
        cls=np.full(h.shape,-1,dtype=np.int8);cls[mask]=(h[mask]>=wet_threshold_m).astype(np.int8)
        if uncertainty_m>0:cls[mask&(abs(h-wet_threshold_m)<=uncertainty_m)]=2
        return {'request_time_s':float(time_s),'source_frame_time_s':float(t[i]),'age_s':float(time_s-t[i]),
            'depth_m':np.where(mask,h,np.nan),'qx_m2_s':np.where(mask,a['qx_m2_s'][i],np.nan),
            'qy_m2_s':np.where(mask,a['qy_m2_s'][i],np.nan),'water_class':cls,'observed':mask.copy(),
            'evidence':self.metadata['case']['evidence'],'use':'research simulation observation, not personnel/vehicle safety decision'}
