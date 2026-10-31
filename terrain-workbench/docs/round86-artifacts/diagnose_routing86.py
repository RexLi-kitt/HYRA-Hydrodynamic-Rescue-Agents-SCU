from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs'); R=O/'round86';R.mkdir(exist_ok=False)
B=O/'round83/carlisle_2400'; C=O/'round85/carlisle_2400_cfl_half'
SRC=O/'round83/lisflood8-rk-retry/swe/dg2/friction.cpp'
protocol={'purpose':'Locate stored early routing differences and test fixed-depth isotropic-friction direction invariance. No solver rerun, calibration or production change.',
 'times_s':[300,600],'thresholds_m':[1e-6,.001,.01],
 'groups':'all cells assigned to nearest original source in Euclidean distance; this is diagnostic grouping, not traced river catchments. Additional archived width>0 mask is nominal SGC channel metadata, not observed water.',
 'friction_probe':'Extract unchanged dg2::friction function, compile CPU64 minimal harness; apply one actual baseline post-output timestep versus two half steps and exact constant-depth Manning ODE on every active saved cell. Do not claim isolated response explains full routing.',
 'invariants':'zero roughness identity, isotropic drag preserves momentum direction; exact fixed-depth solution has semigroup property; finite dissipative outputs.',
 'decision':'If invariants fail, report reproducible operator defect with magnitude, then propose bounded isolated candidate. If no material effect, retain failure evidence and inspect source/limiter coupling instead. No automatic full run.',
 'cost':'two existing fields per model, tiny harness build/run <=120s each; no heavy hydrodynamic run','field_passes_added':0}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2))
hashes={}
def read(p):
    hashes[str(p.relative_to(O))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return np.loadtxt(p,skiprows=6)
source_text=SRC.read_text(); body=source_text[source_text.index('void dg2::friction\n'):]
assert body.count('void dg2::friction')==1
prefix='''#include <cmath>
#include <iostream>
#include <iomanip>
#define NUMERIC_TYPE double
#define C(x) (x)
#define FABS std::abs
#define SQRT std::sqrt
struct Solver {double DepthThresh=1e-3,SpeedThresh=1e-9,g=9.80655,Tstep;};
namespace dg2 {void friction(Solver*,double,double&,double&,double);}
'''
main='''
int main(){ double h,x,y,n,dt; std::cout<<std::setprecision(17);
while(std::cin>>h>>x>>y>>n>>dt){ Solver s;s.Tstep=dt;dg2::friction(&s,h,x,y,n);std::cout<<x<<" "<<y<<"\\n"; } }
'''
(R/'friction_probe.cpp').write_text(prefix+body+main)
(R/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\nproject(friction_probe LANGUAGES CXX)\nadd_executable(friction_probe friction_probe.cpp)\n')
cmake=r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
with (R/'build.log').open('w') as log:
    for args in [[cmake,'-S',str(R),'-B',str(R/'build'),'-G','Visual Studio 17 2022','-A','x64'],[cmake,'--build',str(R/'build'),'--config','Release','--parallel','1']]:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
exe=R/'build/Release/friction_probe.exe'
def actual(h,x,y,dt,n=.06):
    dat=''.join(f'{a:.17g} {b:.17g} {c:.17g} {n:.17g} {dt:.17g}\n' for a,b,c in zip(h,x,y))
    p=subprocess.run([str(exe)],input=dat,capture_output=True,text=True,check=True,timeout=30)
    return np.array([[float(x) for x in line.split()] for line in p.stdout.splitlines()])
def exact(h,q,dt,n=.06):
    return q/(1+9.80655*n*n*np.linalg.norm(q,axis=1)*dt/h**(7/3))[:,None]
def angle(q,z):
    return np.abs(np.degrees(np.arctan2(q[:,0]*z[:,1]-q[:,1]*z[:,0],np.sum(q*z,axis=1))))
width=read(O/'round73/original/Carlisle/width.asc'); channel=width>0
ny,nx=width.shape; yy,xx=np.indices(width.shape); e=338490+(xx+.5)*10; north=554700+(ny-yy-.5)*10
points=np.array([[342685,557505],[341396,554704],[339955,554704]])
dist=np.stack([(e-x)**2+(north-y)**2 for x,y in points]); nearest=dist.argmin(axis=0)
steps=np.loadtxt(B/'accepted_timesteps.txt');frames=[]; probes=[]; arrays={}
for idx,t in enumerate([300,600]):
    h0=read(B/f'results/check-{idx:04d}.wd'); h1=read(C/f'results/check-{idx:04d}.wd'); delta=h1-h0
    rows=[]
    for k in range(3):
        group=nearest==k
        rows.append({'nearest_source':k+1,'baseline_volume_m3':float(h0[group].sum()*100),'candidate_volume_m3':float(h1[group].sum()*100),'L1_depth_volume_m3':float(abs(delta[group]).sum()*100),'thresholds':[
            {'h_m':th,'changed':int((((h0>th)!=(h1>th))&group).sum()),'changed_in_nominal_channel':int((((h0>th)!=(h1>th))&group&channel).sum()),'changed_within_30m_of_nearest_source':int((((h0>th)!=(h1>th))&group&(dist[k]<=30**2)).sum())} for th in protocol['thresholds_m']]})
    j,i=np.unravel_index(abs(delta).argmax(),delta.shape)
    frames.append({'time_s':t,'groups':rows,'maximum_difference':{'row0':int(j),'col0':int(i),'easting':float(e[j,i]),'northing':float(north[j,i]),'h_baseline':float(h0[j,i]),'h_candidate':float(h1[j,i]),'nearest_source':int(nearest[j,i]+1),'distance_to_nearest_source_m':float(np.sqrt(dist[:,j,i].min())),'nominal_channel':bool(channel[j,i])}})
    qx=read(B/f'results/check-{idx:04d}.Qx');qy=read(B/f'results/check-{idx:04d}.Qy')
    mask=(h0>.001)&((abs(qx/h0.clip(1e-20))>1e-9)|(abs(qy/h0.clip(1e-20))>1e-9))
    h=h0[mask];q=np.stack([qx[mask],qy[mask]],axis=1)
    at=np.flatnonzero(np.isclose(steps[:,0],t,rtol=0,atol=1e-9));assert len(at)==1
    dt=float(steps[at[0],1]);one=actual(h,q[:,0],q[:,1],dt);half=actual(h,q[:,0],q[:,1],dt/2);two=actual(h,half[:,0],half[:,1],dt/2);ex=exact(h,q,dt)
    zero=actual(h,q[:,0],q[:,1],dt,n=0); assert np.array_equal(zero,q)
    assert np.allclose(exact(h,exact(h,q,dt/2),dt/2),ex,atol=1e-14,rtol=1e-13)
    assert np.isfinite(one).all() and np.all(np.linalg.norm(one,axis=1)<=np.linalg.norm(q,axis=1)*(1+1e-14))
    angles=angle(q,one);k=int(angles.argmax());coords=np.argwhere(mask)
    probes.append({'time_s':t,'fixed_depth_dt_s':dt,'active_cells':len(h),'maximum_direction_rotation_deg':float(angles[k]),'median_direction_rotation_deg':float(np.median(angles)),'exact_rotation_max_deg':float(angle(q,ex).max()),'max_one_vs_two_momentum_difference_m2_s':float(np.linalg.norm(one-two,axis=1).max()),'max_one_vs_exact_momentum_difference_m2_s':float(np.linalg.norm(one-ex,axis=1).max()),'max_rotation_cell':{'row0':int(coords[k,0]),'col0':int(coords[k,1]),'H':float(h[k]),'Q_before':q[k].tolist(),'Q_one':one[k].tolist(),'Q_two':two[k].tolist(),'Q_exact':ex[k].tolist()},'zero_n_identity':True,'exact_semigroup':True,'finite_dissipative':True})
    arrays[f'friction_t{t}']=np.column_stack([coords,h,q,one,two,ex,angles]);arrays[f'depth_difference_t{t}']=delta
    arrays[f'mask_difference_t{t}']=(h0>.001).astype('uint8')+2*(h1>.001).astype('uint8')
# Explicit simple numerical example, not a physical experiment.
h=np.array([.002]);q=np.array([[.0004,.0002]]);one=actual(h,q[:,0],q[:,1],.4);ex=exact(h,q,.4)
summary={'frames':frames,'fixed_depth_friction_probes':probes,'synthetic_example':{'h':.002,'q':q.tolist(),'dt':.4,'implemented':one.tolist(),'exact':ex.tolist(),'rotation_deg':float(angle(q,one)[0])},'caveat':'Qx/Qy exports are cell mean HU/HV per round80 reader. Fixed H local operator probe excludes gravity, fluxes, slope updates and source interaction; it cannot assign the cause of whole-domain errors. Nominal channel and nearest-source groups are not reality labels.','production_changed':False,'field_passes_added':0}
hashes[str(SRC.relative_to(O))]=hashlib.sha256(SRC.read_bytes()).hexdigest()
(R/'source-and-field-hashes.json').write_text(json.dumps(hashes,indent=2))
(R/'diagnosis.json').write_text(json.dumps(summary,indent=2)); np.savez_compressed(R/'full-diagnostics.npz',**arrays)
print(json.dumps(summary),flush=True)
