"""Analytic spatial-residual audit, not a long-time inlet-boundary benchmark."""
import argparse,json,sys
from pathlib import Path
import numpy as np
LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
from flood_solver import FloodSolver,G

def exact_depth(x,slope,q=.01,h0=.02):
    energy=h0+q*q/(2*G*h0*h0)
    lo=np.full_like(x,1e-12);hi=np.full_like(x,(q*q/G)**(1/3))
    for _ in range(80):
        mid=(lo+hi)/2
        residual=mid+q*q/(2*G*mid*mid)-slope*x-energy
        lo=np.where(residual>0,mid,lo);hi=np.where(residual>0,hi,mid)
    return (lo+hi)/2

def steady(slope,dx,order,flux):
    x=(np.arange(round(10/dx))+.5)*dx
    nodes,weights=np.polynomial.legendre.leggauss(8)
    h=.5*(exact_depth(x[:,None]+nodes*dx/2,slope)*weights).sum(axis=1)
    bed=np.tile(-slope*x,(4,1));depth=np.tile(h,(4,1))
    s=FloodSolver(bed,depth,dx,boundary='open',manning=0,spatial_order=order,flux_method=flux)
    s.mx[:]=.01
    reconstruction=s._reconstruction()
    sh,sb,_,_=reconstruction[1]
    hl,hr=s._faces(s.h,sh,1);bl,br=s._faces(s.bed,np.zeros_like(s.h),1)
    ol,orr=s._faces(np.zeros_like(s.h),sb,1);jump=(br-bl)+(orr-ol)
    clip=((hl>0)&(jump>=hl))|((hr>0)&(-jump>=hr))
    face_window=(np.arange(len(x)+1)*dx>=1)&(np.arange(len(x)+1)*dx<=3)
    dt=1e-6
    hn,mxn,_,_,_,_=s._transport_euler(dt,s.boundary_forcing()[0])
    region=(x>=1)&(x<=3)
    dh=(hn-s.h)/dt;dq=(mxn-s.mx)/dt
    residuals=dict(depth_mean_abs_m_s=float(abs(dh[:,region]).mean()),discharge_mean_abs_m2_s2=float(abs(dq[:,region]).mean()),
        clipped_face_fraction=float(clip[:,face_window].mean()),min_depth_m=float(h[region].min()))
    return dict(slope=slope,dx=dx,order=order,flux=flux,**residuals)

def uniform(slope,dx,order,flux):
    x=(np.arange(round(10/dx))+.5)*dx;h=np.full((4,len(x)),.002)
    s=FloodSolver(np.tile(-slope*x,(4,1)),h,dx,boundary='open',manning=0,spatial_order=order,flux_method=flux)
    s.mx[:]=h*.5;dt=1e-6
    hn,mxn,_,_,_,_=s._transport_euler(dt,s.boundary_forcing()[0]);region=(x>=1)&(x<=3)
    expected=h*.5+dt*G*h*slope
    return dict(slope=slope,dx=dx,order=order,flux=flux,
        depth_change=float(abs(hn[:,region]-h[:,region]).max()),
        acceleration_relative_error=float(abs((mxn[:,region]-s.mx[:,region])/dt-G*h[:,region]*slope).max()/(G*.002*slope)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={'steady':[],'uniform':[]}
    for slope in (.05,.13,.18):
      for dx in (.1,.05,.025):
       for order in (1,2):
        for flux in ('rusanov','knp'):
         result['steady'].append(steady(slope,dx,order,flux))
         result['uniform'].append(uniform(slope,dx,order,flux))
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for r in result['steady']:
      if r['flux']=='rusanov':print(json.dumps(r))
