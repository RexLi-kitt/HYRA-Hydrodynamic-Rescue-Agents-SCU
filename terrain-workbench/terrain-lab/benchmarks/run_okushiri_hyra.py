"""Uncalibrated laboratory baseline. SI units, original time, all three gauges."""
import argparse,json,time,sys,hashlib,os
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from okushiri_boundary import OkushiriBoundarySolver
from benchmark_restart import restore_audit_metadata
from benchmark_step_audit import AcceptedStepAudit

def sample_bed(source,xx,yy,dx,quadrature=1,coordinate_origin="asc_center"):
    if coordinate_origin not in ("asc_center","xya_node"):raise ValueError("Unknown coordinate origin")
    offset=.5 if coordinate_origin=="asc_center" else 0.
    if quadrature not in (1,4,8):raise ValueError('quadrature must be 1, 4 or 8')
    nodes,weights=np.polynomial.legendre.leggauss(quadrature)
    result=np.zeros_like(xx)
    for a,wa in zip(nodes,weights):
        for b,wb in zip(nodes,weights):
            result+=wa*wb*.25*map_coordinates(source,[(source.shape[0]-1+offset)-(yy+b*dx/2)/.014,(xx+a*dx/2)/.014-offset],order=1,mode='nearest')
    return result

def run(data,out,dx,end,cfl,quadrature=1,coordinate_origin="xya_node",east_columns=0,exact_domain=False,resume=False):
    data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((data/'provenance.json').read_text())
    for f in prov['files']:assert hashlib.sha256((data/f['file']).read_bytes()).hexdigest()==f['sha256']
    source=np.loadtxt(data/'Benchmark_2_Bathymetry.asc',skiprows=6)
    wave=np.loadtxt(data/'Benchmark_2_input.txt',skiprows=1)
    obs=np.loadtxt(data/'output_ch5-7-9.txt',skiprows=1);obs[:,1:]/=100
    # .054/.027 share domain 5.454 x 3.402m; east wall +.006m is explicit.
    width=5.448 if exact_domain else 5.454
    if exact_domain and east_columns:raise ValueError("Exact domain cannot shift east wall")
    nx=int(round(width/dx));ny=int(round(3.402/dx))
    if not (np.isclose(nx*dx,width,rtol=0,atol=1e-12) and np.isclose(ny*dx,3.402,rtol=0,atol=1e-12)):
        raise ValueError("Square spacing must divide requested physical domain")
    if east_columns not in (-1,0,1):raise ValueError("east_columns must be -1, 0 or 1")
    nx+=east_columns
    x=(np.arange(nx)+.5)*dx;y=ny*dx-(np.arange(ny)+.5)*dx
    xx,yy=np.meshgrid(x,y)
    bed=sample_bed(source,xx,yy,dx,quadrature,coordinate_origin)
    s=OkushiriBoundarySolver(bed,np.maximum(-bed,0),dx,wave=wave,manning=.0025,cfl=cfl,max_timestep=.01)
    coords=np.array([[ny-.5-y0/dx,4.521/dx-.5] for y0 in [1.196,1.696,2.196]]).T
    indices=np.flatnonzero(obs[:,0]<=end+1e-12);pred=[];start=time.perf_counter();last=-1
    steps_dt=[]
    meta=dict(dx=dx,cfl=cfl,shape=[ny,nx],domain_m=[nx*dx,ny*dx],reference_domain_m=[5.448,3.402],east_wall_offset_m=nx*dx-5.448,east_columns=east_columns,exact_domain=exact_domain,bed_sampling='ASC bilinear field with tensor Gauss cell averaging; north-first rows',bed_quadrature=quadrature,coordinate_origin=coordinate_origin,manning=.0025,source_commit=prov['commit'],evidence='laboratory experiment',end_s=end)
    checkpoint=out/'checkpoint.npz'
    identity=dict(meta);identity.pop('end_s')
    identity['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity['boundary_sha256']=hashlib.sha256((Path(__file__).parent/'okushiri_boundary.py').read_bytes()).hexdigest()
    identity['step_audit_sha256']=hashlib.sha256((Path(__file__).resolve().parents[1]/'benchmark_step_audit.py').read_bytes()).hexdigest()
    identity['solver_sha256']=hashlib.sha256((Path(__file__).resolve().parents[1]/'flood_solver.py').read_bytes()).hexdigest()
    elapsed_before=0.;completed=-1
    if checkpoint.exists():
        if not resume:raise ValueError('Checkpoint exists; explicitly use --resume or a new output directory')
        with np.load(checkpoint,allow_pickle=False) as ck:
            info=json.loads(str(ck['metadata']))
            if info['identity']!=identity:raise ValueError('Checkpoint configuration/code mismatch')
            if not np.array_equal(ck['bed'],bed):raise ValueError('Checkpoint geometry mismatch')
            for field in ('h','mx','mz','maximum'):setattr(s,field,ck[field].copy())
            restore_audit_metadata(s,info['audit']);s._wave_time=s.time
            pred=ck['pred'].tolist();steps_dt=ck['steps_dt'].tolist()
            completed=info['completed_index'];elapsed_before=info['elapsed_s']
            if s.time>end:raise ValueError('Checkpoint time exceeds requested end')
    elif resume:raise ValueError('No checkpoint to resume')
    observer=AcceptedStepAudit(s,steps_dt)
    def save_checkpoint(i):
        info=dict(identity=identity,audit=s.summary(),completed_index=int(i),elapsed_s=elapsed_before+time.perf_counter()-start)
        temporary=out/'checkpoint.next.npz'
        np.savez_compressed(temporary,metadata=json.dumps(info),bed=bed,h=s.h,mx=s.mx,mz=s.mz,maximum=s.maximum,pred=np.asarray(pred),steps_dt=np.asarray(steps_dt))
        os.replace(temporary,checkpoint)
    (out/'configuration.json').write_text(json.dumps(meta,indent=2))
    for i in indices:
        if i<=completed:continue
        t=float(obs[i,0]);s.advance(t,observer=observer)
        pred.append(map_coordinates(s.h+s.bed,coords,order=1,mode='nearest'))
        if i%10==0 or i==indices[-1]:save_checkpoint(i)
        if int(t)//5!=last:
            last=int(t)//5;print(json.dumps(dict(time_s=t,wall_s=elapsed_before+time.perf_counter()-start,steps=s.steps)),flush=True)
    pred=np.array(pred);reference=obs[indices,1:];ts=obs[indices,0]
    np.savez_compressed(out/'timeseries.npz',time_s=ts,predicted_m=pred,observed_m=reference)
    np.savez_compressed(out/'final_state.npz',h=s.h,bed=bed,mx=s.mx,mz=s.mz,maximum=s.maximum)
    metrics=[]
    for j,name in enumerate(['ch5','ch7','ch9']):
        diff=pred[:,j]-reference[:,j]
        metrics.append(dict(gauge=name,rmse_m=float(np.sqrt(np.mean(diff**2))),mae_m=float(np.mean(abs(diff))),bias_m=float(diff.mean()),observed_peak_m=float(reference[:,j].max()),predicted_peak_m=float(pred[:,j].max()),observed_peak_s=float(ts[reference[:,j].argmax()]),predicted_peak_s=float(ts[pred[:,j].argmax()])))
    result=dict(**meta,wall_s=elapsed_before+time.perf_counter()-start,steps=s.steps,min_dt_s=min(steps_dt),max_dt_s=max(steps_dt),balance=s.summary()['balance'],diagnostics=s.diagnostics,metrics=metrics)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for j,ax in enumerate(axs):
        ax.plot(ts,reference[:,j],label='Observed');ax.plot(ts,pred[:,j],label='HYRA');ax.set_ylabel(['ch5','ch7','ch9'][j]+' stage (m)');ax.grid(alpha=.3);ax.legend()
    axs[-1].set_xlabel('Original time (s)');fig.suptitle(f'Okushiri uncalibrated baseline: dx={dx} m, east wall offset={(nx*dx-5.448)*1000:.1f} mm')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('out');p.add_argument('--dx',type=float,default=.054);p.add_argument('--end',type=float,default=22.5);p.add_argument('--cfl',type=float,default=.4);p.add_argument('--quadrature',type=int,choices=[1,4,8],default=1);p.add_argument('--coordinate-origin',choices=['asc_center','xya_node'],default='xya_node');p.add_argument('--east-columns',type=int,choices=[-1,0,1],default=0);p.add_argument('--exact-domain',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.data,a.out,a.dx,a.end,a.cfl,a.quadrature,a.coordinate_origin,a.east_columns,a.exact_domain,a.resume)
