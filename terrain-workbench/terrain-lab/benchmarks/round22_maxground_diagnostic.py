"""Bounded independent physical-model validation on imported float-metre DEM.

The measured vectors are prototype-scale maxima, not a hydrograph. A partial
run is saved but is explicitly ineligible for full-window validation claims.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
import merewether
import server
import map_sessions
from multilayer import json_safe
from benchmark_restart import restore_audit_metadata


def write(path,value):path.write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(args,source,provenance,factor,n):
    started=time.perf_counter();expired=lambda:time.perf_counter()-started>args.wall_budget
    tid,scene=merewether.make_case(LAB,source,factor,n,args.aggregation,layered=args.layered)
    scene['flood'].update(duration_s=args.duration,frames=9)
    variant='-layered-maxground-diagnostic' if args.layered else ('' if args.aggregation=='max' else '-'+args.aggregation)
    folder=args.output/f'f{factor}-n{n:g}{variant}';folder.mkdir(parents=True,exist_ok=True)
    world=map_sessions.create(LAB,server.describe(tid),scene);solver=world.solver
    points=merewether.observations(source);ymax=solver.h.shape[0]*solver.dx
    iz,ix=merewether.measurement_indices([(p['x_m'],p['y_m']) for p in points],solver.h.shape,solver.dx)
    stage=np.atleast_2d(np.loadtxt(source/'merewether.stage',skiprows=1))
    sz,sx=merewether.measurement_indices(stage[:,:2],solver.h.shape,solver.dx)
    maxima=np.zeros(len(points));umax=np.zeros(len(points));vmax=umax.copy();max_at=umax.copy()
    max_h=umax.copy();records=[];frames=[];previous_h=None;checks=[];steady_count=0
    checkpoint_step=max(1,args.duration/8);next_frame=0;last_print=0;last_checkpoint=0
    sid=(f'merewether-f{factor}-n{n:g}'+variant).replace('.','p')
    frame_dir=LAB/'generated'/sid/'maps';frame_dir.mkdir(parents=True,exist_ok=True)
    def frame():
        u,v=solver.velocity();filename=f'frame-{len(frames):03d}.f32'
        np.stack([solver.h,u,v,solver.maximum]).astype('<f4').tofile(frame_dir/filename)
        frames.append(dict(index=len(frames),url=f'/generated/{sid}/maps/{filename}',**solver.summary()))
    config=dict(terrain_id=tid,scenario=scene,source=provenance,comparison='independent measured maximum velocity vectors',
        sampling=dict(interval_s=args.sample_interval,velocity_minimum_depth_m=.01,method='containing-cell conserved state',
                      direction='observed Mod_Angle in x/y; solver z points south, so v_north=-mz/h'),
        roughness=world.meta.get('imported',{}).get('roughness'),wall_budget_s=args.wall_budget,
        terrain_representation='separate ground and inferred impermeable obstacles' if args.layered else args.aggregation+' elevation',
        obstacle_layers=world.meta.get('imported',{}).get('obstacle_layers'),
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),LAB/'raster_import.py',LAB/'merewether.py',LAB/'obstacle_layers.py',LAB/'flood_solver.py',LAB/'flood_acceleration.py']})
    prior_wall=0.
    if args.resume:
        previous=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        old_config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        if previous['complete']:raise ValueError('Saved observation window is already complete')
        if previous['requested_duration_s']!=args.duration or previous['n']!=n or previous['factor']!=factor:
            raise ValueError('Resume requires identical time window, roughness and grid')
        for name in ['flood_solver.py','flood_acceleration.py']:
            if old_config['code_sha256'][name]!=config['code_sha256'][name]:raise ValueError('Cannot resume after numerical solver changes')
        if old_config['sampling']!=config['sampling'] or old_config['source']['files']!=config['source']['files']:
            raise ValueError('Cannot resume with different source data or observation sampling')
        if old_config.get('terrain_representation')!=config['terrain_representation']:
            # Old archived max/mean runs predate the representation field.
            if args.layered or old_config.get('terrain_representation') is not None:
                raise ValueError('Cannot resume with a different terrain representation')
        if args.layered and old_config.get('obstacle_layers')!=config['obstacle_layers']:
            raise ValueError('Cannot resume after obstacle extraction changes')
        with np.load(folder/'final.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['bed'],solver.bed)
            np.testing.assert_array_equal(saved['n'],solver.manning)
            if args.layered:
                np.testing.assert_array_equal(saved['solid'],solver.solid)
                np.testing.assert_array_equal(saved['wall_x'],solver.wall_x)
                np.testing.assert_array_equal(saved['wall_y'],solver.wall_y)
            for name,key in [('h','depth'),('mx','mx'),('mz','mz'),('maximum','maximum_depth')]:
                getattr(solver,name)[:]=saved[key]
        replay=json.loads((frame_dir/'replay.json').read_text(encoding='utf-8'))
        frames=replay['process']['frames'];last=frames[-1]
        if last['time_s'] != previous['duration_s']:
            raise ValueError('Checkpoint frame time does not match saved result')
        restore_audit_metadata(solver, last)
        records=json.loads((folder/'series.json').read_text(encoding='utf-8'))
        maxima=np.array([p['model_max_speed_m_s'] for p in previous['points']])
        umax=np.array([p['model_u_at_max_m_s'] for p in previous['points']])
        vmax=np.array([p['model_v_at_max_m_s'] for p in previous['points']])
        max_at=np.array([p['time_of_model_max_s'] for p in previous['points']])
        max_h=np.array([p['max_depth_m'] for p in previous['points']])
        checks=previous['steady_checks'];last_checkpoint=solver.time
        next_frame=(np.floor(solver.time/checkpoint_step)+1)*checkpoint_step
        prior_wall=previous['wall_s']
        config['segments']=old_config.get('segments',[])+[dict(continued_from_s=solver.time,previous_wall_s=prior_wall,
                                                               previous_code_sha256=old_config['code_sha256'])]
    write(folder/'config.json',config)
    sources=folder/'source-code';sources.mkdir(exist_ok=True)
    for name in config['code_sha256']:
        path=Path(__file__) if name==Path(__file__).name else LAB/name
        (sources/name).write_bytes(path.read_bytes())
    print(json.dumps(dict(stage='start',factor=factor,n=n,shape=list(solver.h.shape),dx_m=solver.dx)),flush=True)
    if not args.resume:frame();next_frame=checkpoint_step
    budget_hit=False
    start=(np.floor(solver.time/args.sample_interval)+1)*args.sample_interval
    for end in np.arange(start,args.duration+args.sample_interval/2,args.sample_interval):
        try:world.advance_to(float(min(end,args.duration)),cancelled=expired)
        except InterruptedError:budget_hit=True;break
        h=solver.h[iz,ix];safe=np.maximum(h,1e-12)
        u=np.where(h>=.01,solver.mx[iz,ix]/safe,0.)
        v=np.where(h>=.01,-solver.mz[iz,ix]/safe,0.)
        speed=np.hypot(u,v);greater=speed>maxima
        maxima[greater]=speed[greater];umax[greater]=u[greater];vmax[greater]=v[greater];max_at[greater]=solver.time
        max_h=np.maximum(max_h,h)
        records.append(dict(time_s=solver.time,stages_m=(solver.bed[sz,sx]+solver.h[sz,sx]).tolist(),
                            speeds_m_s=speed.tolist(),depths_m=h.tolist()))
        if solver.time+1e-9>=next_frame:
            frame();next_frame+=checkpoint_step
        if solver.time-last_checkpoint>=20-1e-9:
            wet=solver.h>=.01
            change=float(np.abs(solver.h-previous_h)[wet].mean()) if previous_h is not None and wet.any() else None
            recent=records[-min(len(records),int(20/args.sample_interval))]
            gauge_change=max(abs(a-b) for a,b in zip(records[-1]['stages_m'],recent['stages_m']))
            steady_count=steady_count+1 if change is not None and change<.001 and gauge_change<.01 and solver.time>=300 else 0
            checks.append(dict(time_s=solver.time,wet_mean_depth_change_m=change,gauge_max_change_m=gauge_change,
                               consecutive_windows=steady_count))
            previous_h=solver.h.copy();last_checkpoint=solver.time
        if time.perf_counter()-last_print>8:
            print(json.dumps(dict(stage='simulate',factor=factor,n=n,time_s=solver.time,wall_s=time.perf_counter()-started)),flush=True)
            last_print=time.perf_counter()
    if not frames or frames[-1]['time_s']!=solver.time:frame()
    rows=[]
    for i,p in enumerate(points):
        angle=float(np.degrees(np.arctan2(vmax[i],umax[i]))%360)
        delta=float((angle-p['angle_xy_deg']+180)%360-180) if maxima[i]>=.05 and p['speed_m_s']>=.05 else None
        rows.append(dict(**p,model_max_speed_m_s=float(maxima[i]),model_u_at_max_m_s=float(umax[i]),
            model_v_at_max_m_s=float(vmax[i]),speed_error_m_s=float(maxima[i]-p['speed_m_s']),
            direction_error_deg=delta,time_of_model_max_s=float(max_at[i]),max_depth_m=float(max_h[i]),
            sampled_bed_m=float(solver.bed[iz[i],ix[i]]),grid_row=int(iz[i]),grid_col=int(ix[i])))
    err=np.array([r['speed_error_m_s'] for r in rows]);directions=[abs(r['direction_error_deg']) for r in rows if r['direction_error_deg'] is not None]
    balance=solver.summary()['balance'];complete=bool(abs(solver.time-args.duration)<1e-7)
    result=dict(factor=factor,dx_m=solver.dx,shape=list(solver.h.shape),n=n,duration_s=solver.time,requested_duration_s=args.duration,
        complete=complete,budget_hit=budget_hit,wall_s=prior_wall+time.perf_counter()-started,
        segment_wall_s=time.perf_counter()-started,quasi_steady=steady_count>=3,
        metrics=dict(speed_rmse_m_s=float(np.sqrt(np.mean(err**2))),speed_mae_m_s=float(abs(err).mean()),speed_bias_m_s=float(err.mean()),
            direction_mae_deg=float(np.mean(directions)) if directions else None,direction_points=len(directions),
            measured_points=len(points),model_dry_points=int(np.count_nonzero(max_h<.01))),
        metrics_status='full 800 s observed-maximum comparison' if complete and args.duration==800 else 'partial-window diagnostic; not full benchmark validation',
        balance=balance,finite=bool(np.isfinite(solver.h).all() and np.isfinite(solver.mx).all() and np.isfinite(solver.mz).all()),
        nonnegative=bool((solver.h>=0).all()),steady_checks=checks,points=rows,
        replay_id=sid,terrain_id=tid,water_level_validation='not scored: observation values unavailable, only sampling coordinates supplied')
    write(folder/'result.json',result);write(folder/'series.json',records)
    with (folder/'points.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(folder/'final.npz',bed=solver.bed,depth=solver.h,mx=solver.mx,mz=solver.mz,
                        maximum_depth=solver.maximum,n=solver.manning,solid=solver.solid,
                        wall_x=solver.wall_x,wall_y=solver.wall_y)
    map_sessions.save_process(LAB,sid,world,frames)
    render(folder,result,solver,points,records,ymax)
    print(json.dumps(dict(stage='done',**{k:result[k] for k in ['factor','n','duration_s','complete','wall_s','metrics','balance']})),flush=True)
    return result


def render(folder,result,solver,points,records,ymax):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,7),layout='constrained')
    extent=(0,solver.dx*solver.h.shape[1],0,ymax)
    im=axes[0].imshow(np.ma.masked_less(solver.h,.01),extent=extent,origin='upper',cmap='Blues',vmin=0)
    x=[p['x_m'] for p in points];y=[p['y_m'] for p in points]
    axes[0].quiver(x,y,[p['u_m_s'] for p in points],[p['v_m_s'] for p in points],color='#c66716',scale=12,width=.005,label='Measured maximum')
    rows=result['points'];axes[0].quiver(x,y,[p['model_u_at_max_m_s'] for p in rows],[p['model_v_at_max_m_s'] for p in rows],color='#294750',scale=12,width=.003,label='Model maximum')
    axes[0].set(xlabel='Local prototype X (m)',ylabel='Local prototype Y (m)',title=f'Maximum velocity vectors | dx={solver.dx:g} m')
    axes[0].legend(loc='upper left',fontsize=8);fig.colorbar(im,ax=axes[0],label='Final water depth (m)',shrink=.7)
    obs=np.array([p['speed_m_s'] for p in points]);model=np.array([p['model_max_speed_m_s'] for p in rows]);lim=max(obs.max(),model.max())*1.05
    axes[1].scatter(obs,model,color='#246e8a');axes[1].plot([0,lim],[0,lim],color='#555',ls='--')
    axes[1].set(xlabel='Measured maximum speed (m/s)',ylabel='Model maximum speed (m/s)',xlim=(0,lim),ylim=(0,lim),
                title=f'40 independent measured points\nRMSE={result["metrics"]["speed_rmse_m_s"]:.3f} m/s')
    axes[1].set_aspect('equal');axes[1].grid(alpha=.2)
    fig.suptitle(f'Merewether physical model | n={result["n"]:g} | t={result["duration_s"]:g} s\n'+
        ('Complete observation window' if result['complete'] else 'PARTIAL WINDOW: validation incomplete'),fontsize=13)
    fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for i in range(4):ax.plot([r['time_s'] for r in records],[r['stages_m'][i] for r in records],label=f'P{i+1}')
    ax.set(xlabel='Prototype time (s)',ylabel='Water surface elevation (m)',title='Computed water levels; no measured stage values in downloaded bundle')
    ax.legend();ax.grid(alpha=.2);fig.savefig(folder/'water-levels.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=LAB/'.runtime/merewether/source')
    p.add_argument('--output',type=Path,default=LAB/'.runtime/merewether/results')
    p.add_argument('--factor',type=int,default=4)
    p.add_argument('--manning',type=float,default=.042)
    p.add_argument('--aggregation',choices=['mean','max'],default='max')
    p.add_argument('--layered',action='store_true',help='Separate inferred obstacle footprint from fluid ground (experimental v1)')
    p.add_argument('--duration',type=float,default=800)
    p.add_argument('--sample-interval',type=float,default=.5)
    p.add_argument('--wall-budget',type=float,default=600)
    p.add_argument('--activate',action='store_true')
    p.add_argument('--resume',action='store_true',help='Continue the saved physical state and measured maxima after a budget stop')
    args=p.parse_args()
    if not args.layered: p.error('Diagnostic requires --layered')
    import obstacle_layers, raster_import
    original_write=raster_import.write
    def isolated_write(root,name,*a,**kw):return original_write(root,'round22-'+name,*a,**kw)
    raster_import.write=isolated_write
    original_aggregate=obstacle_layers.aggregate
    def maxground(layers,factor,native_dx=.175):
        coarse=original_aggregate(layers,factor,native_dx)
        surface=layers['obstacle_top']
        coarse['ground']=raster_import.coarsen(surface,np.ones(surface.shape,bool),factor,'max')[0]
        coarse['audit']['ground_diagnostic']='max original surface; same layered solid'
        return coarse
    obstacle_layers.aggregate=maxground
    if not 1<=args.duration<=3600 or not .01<=args.manning<=.2 or args.sample_interval<=0 or args.duration%args.sample_interval:
        p.error('Invalid time window, sampling interval or roughness')
    args.output.mkdir(parents=True,exist_ok=True);os.environ['HYRA_FLOOD_BACKEND']='numba'
    source,provenance=merewether.fetch(args.source)
    result=run(args,source,provenance,args.factor,args.manning)
    if args.activate:
        previous=LAB/'current.json'
        if previous.exists() and not (args.output/'previous-current.json').exists():(args.output/'previous-current.json').write_bytes(previous.read_bytes())
        write(previous,dict(directory=result['terrain_id']))


if __name__=='__main__':main()
