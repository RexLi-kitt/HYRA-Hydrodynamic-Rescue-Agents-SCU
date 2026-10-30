"""Explicit-support depth reader. No clipping, wetness or velocity inference."""
from pathlib import Path
import numpy as np

class DepthReader:
    def __init__(self, depth, sx, sy, header):
        self.h=np.asarray(depth);self.sx=np.asarray(sx);self.sy=np.asarray(sy);self.header=dict(header)
        expected=(int(header['nrows']),int(header['ncols']))
        if any(a.shape!=expected for a in (self.h,self.sx,self.sy)):raise ValueError('shape mismatch')
        if not all(np.isfinite(a).all() for a in (self.h,self.sx,self.sy)):raise ValueError('nonfinite depth coefficients')
        if (self.h<0).any():raise ValueError('negative mean depth')
        if header['cellsize']<=0:raise ValueError('invalid cell size')
        self.corner_min=self.h-np.sqrt(3)*(abs(self.sx)+abs(self.sy))

    @classmethod
    def read(cls,path):
        p=Path(path)
        def load(q):
            lines=q.read_text().splitlines();header={k.lower():float(v) for k,v in (s.split() for s in lines[:6])}
            a=np.loadtxt(q,skiprows=6,ndmin=2)
            if (a==header.get('nodata_value',-9999)).any():raise ValueError('NODATA unsupported; must remain unknown')
            return header,a
        hd,h=load(p);hx,x=load(Path(str(p)+'1x'));hy,y=load(Path(str(p)+'1y'))
        if hd!=hx or hd!=hy:raise ValueError('coefficient coordinate headers differ')
        return cls(h,x,y,hd)

    def sample(self,x,y,mode='cell_mean'):
        if mode not in ('cell_mean','raw_polynomial'):raise ValueError('explicit supported mode required')
        g=self.header;dx=g['cellsize'];nx=int(g['ncols']);ny=int(g['nrows']);xl=g['xllcorner'];yl=g['yllcorner']
        if not (np.isfinite(x) and np.isfinite(y) and xl<=x<xl+nx*dx and yl<=y<yl+ny*dx):raise ValueError('outside half-open model domain')
        col=int(np.floor((x-xl)/dx));row=ny-1-int(np.floor((y-yl)/dx))
        valid=bool(self.corner_min[row,col]>=0)
        if mode=='raw_polynomial' and not valid:raise ValueError('inadmissible raw polynomial: negative corner depth; no silent clipping')
        z=float(self.h[row,col])
        if mode=='raw_polynomial':
            xc=xl+(col+.5)*dx;yc=yl+(ny-row-.5)*dx
            z+=np.sqrt(3)*(self.sx[row,col]*2*(x-xc)/dx+self.sy[row,col]*2*(y-yc)/dx)
        return dict(depth_m=float(z),support=mode,row=row,col=col,raw_polynomial_admissible=valid,wetness_classification='not_inferred',velocity='not_available')
