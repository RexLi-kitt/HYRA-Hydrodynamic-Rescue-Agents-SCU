"""Optional compiled kernels for the native SWE scheme (no fast-math).

The NumPy solver remains the reference/fallback. Local optional packages may be
installed under .runtime/acceleration-packages without modifying a shared venv.
"""
from pathlib import Path
import sys

local = Path(__file__).resolve().parent/'.runtime/acceleration-packages'
if local.is_dir():
    sys.path.insert(0, str(local))

import numpy as np
from numba import njit

G = 9.81


@njit(cache=True, nogil=True)
def minmod(a, b, theta):
    centre = .5*(a+b)
    magnitude = min(theta*abs(a), theta*abs(b), abs(centre))
    return np.sign(centre)*magnitude if a*b > 0 else 0.


@njit(cache=True, nogil=True)
def reconstruction(h, bed, u, v, solid, wall_x, wall_y, periodic, order, theta, dry):
    nz, nx = h.shape
    result = np.zeros((2, 4, nz, nx))
    if order == 1:
        return result
    for axis in range(2):
        for z in range(nz):
            for x in range(nx):
                zm, zp, xm, xp = z, z, x, x
                if axis == 1:
                    xm, xp = (x-1) % nx, (x+1) % nx
                    boundary = x == 0 or x == nx-1
                    blocked = wall_x[z,x] or wall_x[z,x+1]
                else:
                    zm, zp = (z-1) % nz, (z+1) % nz
                    boundary = z == 0 or z == nz-1
                    blocked = wall_y[z,x] or wall_y[z+1,x]
                if (not periodic and boundary) or blocked or solid[z,x] or solid[zm,xm] or solid[zp,xp]:
                    continue
                if h[z,x] <= dry or h[zm,xm] <= dry or h[zp,xp] <= dry:
                    continue
                a, b = h[z,x]-h[zm,xm], h[zp,xp]-h[z,x]
                sh = minmod(a,b,theta)
                sh *= min(1., 2*max(h[z,x],0.)/max(abs(sh),1e-100))
                seta = minmod(a+(bed[z,x]-bed[zm,xm]), b+(bed[zp,xp]-bed[z,x]), theta)
                result[axis,0,z,x] = sh
                result[axis,1,z,x] = seta-sh
                result[axis,2,z,x] = minmod(u[z,x]-u[zm,xm],u[zp,xp]-u[z,x],theta)
                result[axis,3,z,x] = minmod(v[z,x]-v[zm,xm],v[zp,xp]-v[z,x],theta)
    return result


@njit(cache=True, nogil=True)
def flux_axis(h, bed, u, v, sh, sb, su, sv, blocked, axis, boundary,
              sea_level, sea0, sea1, fraction0, fraction1, q0, q1, ax0, ax1, az0, az1, method, low_froude, audit=None):
    nz, nx = h.shape
    shape = (nz,nx+1) if axis == 1 else (nz+1,nx)
    mass=np.empty(shape);advx=np.empty(shape);advz=np.empty(shape)
    pressure=np.empty(shape);cl_out=np.empty(shape);cr_out=np.empty(shape);speeds=np.empty(shape)
    for z in range(shape[0]):
        for x in range(shape[1]):
            along = x if axis == 1 else z
            extent = nx if axis == 1 else nz
            offset = z if axis == 1 else x
            edge = along == 0 or along == extent
            if axis == 1:
                zl=zr=z;xl=x-1;xr=x
            else:
                xl=xr=x;zl=z-1;zr=z
            if boundary == 3:
                zl%=nz;zr%=nz;xl%=nx;xr%=nx
            else:
                zl=max(0,min(nz-1,zl));zr=max(0,min(nz-1,zr))
                xl=max(0,min(nx-1,xl));xr=max(0,min(nx-1,xr))
            hl=h[zl,xl]+.5*sh[zl,xl];hr=h[zr,xr]-.5*sh[zr,xr]
            bl=.5*sb[zl,xl];br=-.5*sb[zr,xr]
            ul=u[zl,xl]+.5*su[zl,xl];ur=u[zr,xr]-.5*su[zr,xr]
            vl=v[zl,xl]+.5*sv[zl,xl];vr=v[zr,xr]-.5*sv[zr,xr]
            sea = sea0[offset] if along == 0 else sea1[offset] if along == extent else False
            if edge and boundary != 3:
                if boundary == 2 and sea:
                    if along == 0:
                        hl=max(0.,sea_level-bed[zl,xl]);ul=vl=0.
                    else:
                        hr=max(0.,sea_level-bed[zr,xr]);ur=vr=0.
                elif boundary == 0:
                    if along == 0:
                        if axis == 1: ul=-ul
                        else: vl=-vl
                    else:
                        if axis == 1: ur=-ur
                        else: vr=-vr
            jump=(bed[zr,xr]-bed[zl,xl])+(br-bl)
            dl=max(0.,hl-max(jump,0.));dr=max(0.,hr-max(-jump,0.))
            # Use the same normal/tangent convention as flood_numerics.face_flux.
            unl,unr=(ul,ur) if axis == 1 else (vl,vr)
            utl,utr=(vl,vr) if axis == 1 else (-ul,-ur)
            mnl,mnr=dl*unl,dr*unr;mtl,mtr=dl*utl,dr*utr
            cl,cr=np.sqrt(G*dl),np.sqrt(G*dr)
            correction=1.
            if low_froude:
                fr2=(unl*unl+utl*utl+unr*unr+utr*utr)/(G*(dl+dr)+1e-10)
                if low_froude == 1: correction=np.sqrt(min(1.,max(.001,fr2)))
                else: correction=np.sqrt(min(1.,.01+max(np.sqrt(fr2)-.01,0.)))
            if method == 0:
                speed=max(abs(unl)+cl,abs(unr)+cr)
                q=.5*(mnl+mnr)-.5*speed*(dr-dl)
                fn=.5*(mnl*unl+mnr*unr)-.5*speed*correction*(mnr-mnl)
                ft=.5*(mnl*utl+mnr*utr)-.5*speed*correction*(mtr-mtl)
                p=.25*G*(dl*dl+dr*dr)
            elif method == 1:
                sp=max(unl+cl,unr+cr,0.);sm=min(unl-cl,unr-cr,0.)
                denom=sp-sm;inverse=1./denom if denom>1e-14 else 0.
                q=(sp*mnl-sm*mnr+sp*sm*(dr-dl))*inverse
                fn=(sp*mnl*unl-sm*mnr*unr+correction*sp*sm*(mnr-mnl))*inverse
                ft=(sp*mnl*utl-sm*mnr*utr+correction*sp*sm*(mtr-mtl))*inverse
                p=.5*G*(sp*dl*dl-sm*dr*dr)*inverse if denom>1e-14 else .25*G*(dl*dl+dr*dr)
                speed=max(sp,-sm)
            elif method == 2:
                if low_froude != 0:
                    raise ValueError('Experimental HLLC requires low_froude=0')
                if dl == 0.: unl=utl=0.
                if dr == 0.: unr=utr=0.
                mnl,mnr=dl*unl,dr*unr;mtl,mtr=dl*utl,dr*utr
                sl=min(unl-cl,unr-cr);sr=max(unl+cl,unr+cr)
                if dl == 0.: sl=unr-2.*cr;sr=unr+cr
                if dr == 0.: sl=unl-cl;sr=unl+2.*cl
                sp=max(sr,0.);sm=min(sl,0.)
                denom=sp-sm;inverse=1./denom if denom>1e-14 else 0.
                q=(sp*mnl-sm*mnr+sp*sm*(dr-dl))*inverse
                fn=(sp*mnl*unl-sm*mnr*unr+sp*sm*(mnr-mnl))*inverse
                p=.5*G*(sp*dl*dl-sm*dr*dr)*inverse if denom>1e-14 else .25*G*(dl*dl+dr*dr)
                ft=(sp*mnl*utl-sm*mnr*utr+sp*sm*(mtr-mtl))*inverse
                hstar=(sp*dr-sm*dl-(mnr-mnl))*inverse
                reason=0
                if dl<=1e-8 or dr<=1e-8: reason=1
                elif hstar<=1e-8: reason=2
                elif unr-unl>=2.*(cl+cr): reason=3
                if reason == 0:
                    ft=q*(utl if q>=0. else utr)
                # Only unblocked interior faces; edge forcing is audited in
                # the existing boundary budget instead.
                if audit is not None and not edge and not blocked[z,x]:
                    audit[reason,z,x]+=1
                speed=max(sp,-sm)
            else:
                raise ValueError('Unsupported flux method code')
            ax,az=(fn,ft) if axis == 1 else (-ft,fn)
            if edge:
                iz,ix=(zr,xr) if along == 0 else (zl,xl)
                sign=-1. if along == 0 else 1.
                if boundary in (1,2) and not sea:
                    normal=u[iz,ix] if axis == 1 else v[iz,ix]
                    q=sign*max(0.,sign*h[iz,ix]*normal)
                    ax=q*u[iz,ix];az=q*v[iz,ix];p=.5*G*h[iz,ix]**2
                fraction=min(1.,fraction0[offset] if along == 0 else fraction1[offset])
                prescribed=q0[offset] if along == 0 else q1[offset]
                prescribed_x=ax0[offset] if along == 0 else ax1[offset]
                prescribed_z=az0[offset] if along == 0 else az1[offset]
                keep=1.-fraction
                q=q*keep-sign*prescribed
                ax=ax*keep+prescribed_x
                az=az*keep+prescribed_z
                p=p*keep+.5*G*h[iz,ix]**2*fraction
            corr_l=.5*G*(hl-dl)*(hl+dl);corr_r=.5*G*(hr-dr)*(hr+dr)
            if blocked[z,x]:
                q=ax=az=p=0.
                corr_l=.5*G*hl**2;corr_r=.5*G*hr**2
            mass[z,x]=q;advx[z,x]=ax;advz[z,x]=az;pressure[z,x]=p
            cl_out[z,x]=corr_l;cr_out[z,x]=corr_r;speeds[z,x]=speed
    return mass,advx,advz,pressure,cl_out,cr_out,speeds


@njit(cache=True, nogil=True)
def transport(h,mx,mz,bedx,bedz,fx,fz,dt,dx,periodic):
    nz,nx=h.shape;theta=np.ones(h.shape)
    for z in range(nz):
        for x in range(nx):
            outgoing=max(fx[0][z,x+1],0.)+max(-fx[0][z,x],0.)+max(fz[0][z+1,x],0.)+max(-fz[0][z,x],0.)
            requested=dt*outgoing;available=h[z,x]*dx
            if requested>available:theta[z,x]=available/requested
    limited=0
    for axis in range(2):
        f=fx if axis == 1 else fz
        for z in range(f[0].shape[0]):
            for x in range(f[0].shape[1]):
                zl=z;zr=z;xl=x;xr=x
                if axis == 1:xl-=1
                else:zl-=1
                if periodic:
                    donor_l=theta[zl%nz,xl%nx];donor_r=theta[zr%nz,xr%nx]
                else:
                    donor_l=theta[zl,xl] if 0<=zl<nz and 0<=xl<nx else 1.
                    donor_r=theta[zr,xr] if 0<=zr<nz and 0<=xr<nx else 1.
                factor=donor_l if f[0][z,x]>=0 else donor_r
                if factor<1.-1e-12 and f[0][z,x]!=0.:limited+=1
                for k in range(3):f[k][z,x]*=factor
    hn=np.empty_like(h);mxn=np.empty_like(h);mzn=np.empty_like(h)
    adjustment=0.;minimum=np.inf;finite=True;factor=dt/dx
    for z in range(nz):
        for x in range(nx):
            a=h[z,x]-factor*(fx[0][z,x+1]-fx[0][z,x]+fz[0][z+1,x]-fz[0][z,x])
            b=mx[z,x]-factor*(fx[1][z,x+1]+fx[3][z,x+1]+fx[4][z,x+1]
                -(fx[1][z,x]+fx[3][z,x])-fx[5][z,x]+fz[1][z+1,x]-fz[1][z,x])
            c=mz[z,x]-factor*(fx[2][z,x+1]-fx[2][z,x]
                +fz[2][z+1,x]+fz[3][z+1,x]+fz[4][z+1,x]
                -(fz[2][z,x]+fz[3][z,x])-fz[5][z,x])
            b-=factor*G*h[z,x]*bedx[z,x];c-=factor*G*h[z,x]*bedz[z,x]
            finite=finite and np.isfinite(a) and np.isfinite(b) and np.isfinite(c)
            minimum=min(minimum,a)
            adjustment-=min(a,0.)*dx**2
            hn[z,x]=max(a,0.);mxn[z,x]=b;mzn[z,x]=c
    return hn,mxn,mzn,adjustment,minimum,limited,finite


@njit(cache=True, nogil=True)
def timestep_bounds(h,u,v,solid,wall_x,wall_y,periodic,order,theta,dry):
    nz,nx=h.shape;maxx=maxz=0.;maximum=-1.;iz=ix=0
    for z in range(nz):
        for x in range(nx):
            bx=bz=0.
            for axis in range(2):
                zm,zp,xm,xp=z,z,x,x
                if axis == 1:
                    xm,xp=(x-1)%nx,(x+1)%nx
                    blocked=wall_x[z,x] or wall_x[z,x+1]
                    boundary=x==0 or x==nx-1
                    velocity=u
                else:
                    zm,zp=(z-1)%nz,(z+1)%nz
                    blocked=wall_y[z,x] or wall_y[z+1,x]
                    boundary=z==0 or z==nz-1
                    velocity=v
                sh=sv=0.
                unavailable=blocked or solid[z,x] or solid[zm,xm] or solid[zp,xp]
                if order==2 and not unavailable and (periodic or not boundary) and min(h[z,x],h[zm,xm],h[zp,xp])>dry:
                    sh=minmod(h[z,x]-h[zm,xm],h[zp,xp]-h[z,x],theta)
                    sh*=min(1.,2*max(h[z,x],0.)/max(abs(sh),1e-100))
                    sv=minmod(velocity[z,x]-velocity[zm,xm],velocity[zp,xp]-velocity[z,x],theta)
                bound=abs(velocity[z,x])+.5*abs(sv)+np.sqrt(G*max(h[z,x]+.5*abs(sh),0.))
                if axis==1:bx=bound
                else:bz=bound
            maxx=max(maxx,bx);maxz=max(maxz,bz)
            if bx+bz>maximum:maximum=bx+bz;iz=z;ix=x
    return maxx+maxz,iz,ix


@njit(cache=True, nogil=True)
def water_sources(h,mx,mz,solid,baseflow,scheduled,external,drain,dt,rain,infiltration,dx,inflow,sz,sx):
    removed_i=np.empty_like(h);removed_d=np.empty_like(h)
    total_i=total_d=0.;fluid=0
    for z in range(h.shape[0]):
        for x in range(h.shape[1]):
            local_rain=0. if solid[z,x] else rain
            if not solid[z,x]:fluid+=1
            value=h[z,x]+dt*(local_rain+baseflow[z,x]+scheduled[z,x]+external[z,x])
            if inflow and z==sz and x==sx:value+=inflow*dt/dx**2
            a=min(value,infiltration*dt)
            keep=(value-a)/value if value>0 else 0.
            mx[z,x]*=keep;mz[z,x]*=keep;value-=a
            b=min(value,drain[z,x]*dt/dx**2)
            keep=(value-b)/value if value>0 else 0.
            mx[z,x]*=keep;mz[z,x]*=keep;value-=b
            h[z,x]=value;removed_i[z,x]=a;removed_d[z,x]=b
            total_i+=a;total_d+=b
    return removed_i,removed_d,total_i*dx**2,total_d*dx**2,fluid


@njit(cache=True, nogil=True)
def complete_sinks(h,mx,mz,drain,applied_i,applied_d,dt,infiltration,dx):
    total_i=total_d=0.
    for z in range(h.shape[0]):
        for x in range(h.shape[1]):
            value=h[z,x]
            a=min(value,max(infiltration*dt-applied_i[z,x],0.))
            keep=(value-a)/value if value>0 else 0.
            mx[z,x]*=keep;mz[z,x]*=keep;value-=a
            b=min(value,max(drain[z,x]*dt/dx**2-applied_d[z,x],0.))
            keep=(value-b)/value if value>0 else 0.
            mx[z,x]*=keep;mz[z,x]*=keep;value-=b
            h[z,x]=value;total_i+=a;total_d+=b
    return total_i*dx**2,total_d*dx**2


@njit(cache=True, nogil=True)
def drag(h,mx,mz,manning,dt):
    for z in range(h.shape[0]):
        for x in range(h.shape[1]):
            if mx[z,x]==0. and mz[z,x]==0.:
                continue
            divisor=1.+dt*G*manning[z,x]**2*np.hypot(mx[z,x],mz[z,x])/max(h[z,x],1e-8)**(7./3.)
            mx[z,x]/=divisor;mz[z,x]/=divisor
