"""Geometry audit, bounded native-resolution patch, and paired flood report.

Patch boundaries are a synthetic closed-box dam break, not experimental data.
Only the unchanged full-domain 800 s forcing can produce benchmark scores.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy import ndimage

LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
import merewether
import obstacle_layers
import raster_import
from multilayer import json_safe
from flood_solver import FloodSolver


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def connected(mask,dx,points):
    """Four-neighbour geometric reachability, independent of elevation/water."""
    labels,_=ndimage.label(~mask)
    x=(np.arange(mask.shape[1])+.5)*dx
    inlet=(x>=67.4375)&(x<105.4375)
    starts=np.unique(labels[-1,inlet]);starts=starts[starts>0]
    yy=mask.shape[0]*dx
    return {p['id']:bool(labels[int((yy-p['y_m'])/dx),int(p['x_m']/dx)] in starts) for p in points}


def geometry(out,surface,layers,source):
    points=merewether.observations(source)
    stage=np.loadtxt(source/'merewether.stage',skiprows=1)
    probes=[dict(id=f'P{i+1}',x_m=float(p[0]),y_m=float(p[1])) for i,p in enumerate(stage)]+points
    reference=connected(layers['obstacle'],.175,probes)
    rows=[];audits={};checks={}
    for factor in [1,2,4,8]:
        coarse=obstacle_layers.aggregate(layers,factor);dx=.175*factor
        reachable=connected(coarse['solid'],dx,probes)
        old,_,_=raster_import.coarsen(surface,np.ones(surface.shape,bool),factor,'max')
        for p in probes:
            z,col=int((old.shape[0]*dx-p['y_m'])/dx),int(p['x_m']/dx)
            rows.append(dict(id=p['id'],factor=factor,dx_m=dx,x_m=p['x_m'],y_m=p['y_m'],
                old_max_bed_m=float(old[z,col]),**obstacle_layers.probe(layers,coarse,p['x_m'],p['y_m'],dx),
                native_inlet_connected=reference[p['id']],coarse_inlet_connected=reachable[p['id']]))
        labels,n=ndimage.label(coarse['solid'],structure=np.ones((3,3),bool))
        native_labels,n_native=ndimage.label(layers['obstacle'],structure=np.ones((3,3),bool))
        top=coarse['audit']['padded_north_cells'];lost_components=[];split_components=[]
        for j,sl in enumerate(ndimage.find_objects(native_labels),start=1):
            zz,xx=np.nonzero(native_labels[sl]==j);zz+=sl[0].start;xx+=sl[1].start
            hits=np.unique(labels[(zz+top)//factor,xx//factor]);hits=hits[hits>0]
            if not hits.size:lost_components.append(j)
            if len(hits)>1:split_components.append(j)
        audits[str(factor)]=dict(**coarse['audit'],coarse_components=int(n),
            vanished_native_components=lost_components,split_native_components=split_components,
            native_inlet_connected_points=sum(reference.values()),coarse_inlet_connected_points=sum(reachable.values()),
            newly_disconnected_points=[p['id'] for p in probes if reference[p['id']] and not reachable[p['id']]],
            note='2D obstacle-only topology; no water level, capacity or velocity implication')
        checks[f'factor{factor}_all_components_represented']=not lost_components
    checks['known_ground_bitwise_unchanged']=bool(np.array_equal(layers['ground'][~layers['obstacle']],surface[~layers['obstacle']]))
    checks['native_footprint_identity']=bool(np.array_equal(obstacle_layers.aggregate(layers,1)['solid'],layers['obstacle']))
    checks['finite_ground']=bool(np.isfinite(layers['ground']).all())
    write(out/'geometry.json',dict(extraction=layers['metadata'],grids=audits,checks=checks))
    table(out/'geometry-probes.csv',rows)
    render_geometry(out,surface,layers,rows)
    if not all(checks.values()):raise AssertionError('Geometry audit failure')
    return rows,audits


def render_geometry(out,surface,layers,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=obstacle_layers.aggregate(layers,4)
    fig,axes=plt.subplots(1,3,figsize=(13,6),layout='constrained')
    extent=(0,surface.shape[1]*.175,0,surface.shape[0]*.175)
    old,_,_=raster_import.coarsen(surface,np.ones(surface.shape,bool),4,'max')
    coarse_extent=(0,c['ground'].shape[1]*.7,0,c['ground'].shape[0]*.7)
    axes[0].imshow(surface,extent=extent,origin='upper',cmap='Greys',vmin=18,vmax=26)
    axes[1].imshow(old,extent=coarse_extent,origin='upper',cmap='Greys',vmin=18,vmax=26)
    axes[2].imshow(c['ground'],extent=coarse_extent,origin='upper',cmap='terrain',vmin=18,vmax=22)
    axes[2].imshow(np.ma.masked_where(~c['solid'],np.ones_like(c['solid'])),extent=coarse_extent,origin='upper',cmap='Greys',vmin=0,vmax=1)
    selected=[p for p in rows if p['factor']==4 and p['id'] in ['P3','20-F','19-J']]
    for ax,title in zip(axes,['Original combined DEM, 0.175 m','Previous maximum DEM, 0.7 m','Ground + independent solid mask, 0.7 m']):
        for p in selected:
            ax.scatter([p['x_m']],[p['y_m']],c='#c93d2e',s=28,marker='+')
            ax.annotate(p['id'],(p['x_m'],p['y_m']),xytext=(4,3),textcoords='offset points',fontsize=8)
        ax.set(xlim=(15,105),ylim=(125,225),xlabel='Prototype X (m)',ylabel='Prototype Y (m)',title=title)
    fig.suptitle('Inferred obstacles remain explicit; coarse footprint errors are audited',fontsize=13)
    fig.savefig(out/'layered-map.png',dpi=160);plt.close(fig)

    coords=[p for p in rows if p['factor']==4 and p['id'] in ['P1','P2','P3','20-F']]
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,p in zip(axes.flat,coords):
        x,y=p['x_m'],p['y_m'];yy=c['ground'].shape[0]*.7
        z0=int((yy-y-2.8)/.7);z1=int((yy-y+2.8)/.7)+1;x0=int((x-2.8)/.7);x1=int((x+2.8)/.7)+1
        ex=(x0*.7,x1*.7,yy-z1*.7,yy-z0*.7)
        im=ax.imshow(c['ground'][z0:z1,x0:x1],extent=ex,origin='upper',cmap='Blues',vmin=p['native_ground_m']-.2,vmax=p['native_ground_m']+.2)
        ax.imshow(np.ma.masked_where(~c['solid'][z0:z1,x0:x1],np.ones_like(c['solid'][z0:z1,x0:x1])),extent=ex,origin='upper',cmap='Greys',vmin=0,vmax=1)
        ax.scatter([x],[y],marker='+',c='#e22b30',s=130,lw=2)
        ax.set(title=f"{p['id']}: old {p['old_max_bed_m']:.2f} -> ground {p['coarse_ground_m']:.2f} m",xlabel='X (m)',ylabel='Y (m)')
        fig.colorbar(im,ax=ax,label='Ground elevation (m)',shrink=.65)
    fig.suptitle('Fixed probes on ground; black cells are independent obstacles',fontsize=13)
    fig.savefig(out/'probe-ground.png',dpi=160);plt.close(fig)


def patch(out,cache,surface,layers,budget):
    """Fine native terrain in an explicitly synthetic hydraulic diagnostic."""
    cache.mkdir(parents=True,exist_ok=True)
    # Aligned with the full-domain 0.7 m grid; native world extent retained.
    crop=(slice(1029,1125),slice(232,344));native=surface[crop];ground=layers['ground'][crop];mask=layers['obstacle'][crop]
    configs=[('combined-native',native,np.zeros_like(mask),.175),('layered-native',ground,mask,.175)]
    co=obstacle_layers.aggregate(layers,4)
    old,_,_=raster_import.coarsen(surface,np.ones(surface.shape,bool),4,'max')
    coarse_crop=(slice(258,282),slice(58,86))
    configs += [('combined-coarse',old[coarse_crop],np.zeros((24,28),bool),.7),
                ('layered-coarse',co['ground'][coarse_crop],co['solid'][coarse_crop],.7)]
    outputs=[];start=time.perf_counter()
    for label,bed,solid,dx in configs:
        # A fixed head at the left reservoir; no measured boundary values used.
        xx=(np.arange(bed.shape[1])+.5)*dx
        initial=np.where((xx[None,:]<2.8)&~solid,np.maximum(19.8-bed,0),0.)
        solver=FloodSolver(bed,initial,dx,boundary='closed',manning=.042,max_steps=500000)
        solver.set_environment(solid=solid)
        records=[];frames=[];cancelled=lambda:time.perf_counter()-start>budget
        for end in [0,5,10,20,40]:
            try:solver.advance(float(end),cancelled=cancelled)
            except InterruptedError:break
            frames.append(solver.h.copy());records.append(solver.summary())
        u,v=solver.velocity();summary=solver.summary()
        np.savez_compressed(cache/(label+'.npz'),bed=bed,solid=solid,depth=solver.h,u=u,v=v,dx=dx,
            maximum_depth=solver.maximum,frames=np.array(frames),times=[r['time_s'] for r in records])
        outputs.append(dict(label=label,dx_m=dx,shape=list(bed.shape),complete=solver.time==40,
            duration_s=solver.time,wet_cells=int((solver.h>=.01).sum()),wet_area_m2=float((solver.h>=.01).sum()*dx**2),
            maximum_speed_m_s=float(np.hypot(u,v).max()),initial_volume_m3=summary['balance']['initial_m3'],
            volume_m3=summary['balance']['current_m3'],residual_m3=summary['balance']['residual_m3'],
            numerical_adjustment_m3=summary['balance']['numerical_adjustment_m3'],
            solid_storage_m3=float(solver.h[solid].sum()*dx**2),finite=bool(np.isfinite(solver.h).all() and np.isfinite(u).all() and np.isfinite(v).all()),
            nonnegative=bool((solver.h>=0).all())))
        if cancelled():break
    result=dict(kind='synthetic closed-box dam-break on real terrain crop; not Merewether validation',
        setup=dict(source_row_range=[1029,1125],source_col_range=[232,344],extent_xy_m=[40.6,60.2,187.6,204.4],
            reservoir_x_width_m=2.8,initial_water_surface_m=19.8,duration_s=40,all_boundaries='closed',n=.042,
            numerical_method='unchanged Rusanov/MUSCL/RK2/CFL0.4'),
        wall_budget_s=budget,wall_s=time.perf_counter()-start,results=outputs)
    if len(outputs)==4 and all(r['complete'] for r in outputs):
        a=dict(np.load(cache/'combined-native.npz'));b=dict(np.load(cache/'layered-native.npz'))
        result['native_equivalence']=dict(max_depth_difference_m=float(np.abs(a['depth']-b['depth']).max()),
            max_u_difference_m_s=float(np.abs(a['u']-b['u']).max()),max_v_difference_m_s=float(np.abs(a['v']-b['v']).max()),
            note='Same native terrain and initial head; engineering consistency, not experimental accuracy')
        for r in outputs:
            a=dict(np.load(cache/(r['label']+'.npz')));dd=r['dx_m']
            # Subtract global indices, rather than rounding again in the crop.
            # 20-F is exactly on a coarse y face at supplied coordinate precision.
            if dd==.175:
                z=int((surface.shape[0]*dd-195.30)/dd)-1029;col=int(47.86/dd)-232
            else:
                z=int((old.shape[0]*dd-195.30)/dd)-258;col=int(47.86/dd)-58
            r['depth_at_20F_location_m']=float(a['depth'][z,col])
            r['ground_or_surface_at_20F_m']=float(a['bed'][z,col])
        result['initial_volume_note']='Identical initial head, not equalized volume; coarse representations change initial storage.'
        result['probe_indexing']='Global containing-cell indices minus crop offset; no independent crop rounding at exact grid faces.'
    write(out/'patch.json',result)
    if len(outputs)!=4 or not all(r['complete'] and r['finite'] and r['nonnegative'] and abs(r['residual_m3'])<1e-7 and r['solid_storage_m3']==0 for r in outputs):
        raise RuntimeError('Patch diagnostic incomplete or failed physical invariant')
    render_patch(out,cache,outputs)
    return result


def render_patch(out,cache,outputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for ax,r in zip(axes.flat,outputs):
        ar=dict(np.load(cache/(r['label']+'.npz')));h=ar['depth'];solid=ar['solid'];dd=r['dx_m']
        extent=(40.6,60.2,187.6,204.4)
        ax.imshow(ar['bed'],extent=extent,origin='upper',cmap='Greys',vmin=19,vmax=25)
        im=ax.imshow(np.ma.masked_less(h,.01),extent=extent,origin='upper',cmap='Blues',vmin=0,vmax=.8)
        ax.imshow(np.ma.masked_where(~solid,np.ones_like(solid)),extent=extent,origin='upper',cmap='Greys',vmin=0,vmax=1)
        yy,xx=np.indices(h.shape);step=max(1,int(1.4/dd));choose=(h>=.01)&(xx%step==0)&(yy%step==0)
        ax.quiver(40.6+(xx[choose]+.5)*dd,204.4-(yy[choose]+.5)*dd,ar['u'][choose],-ar['v'][choose],scale=10,width=.003,color='#1b424e')
        ax.scatter([47.86],[195.30],c='#e22b30',s=90,marker='+');ax.annotate('20-F location (no observed time series)',(47.86,195.30),xytext=(3,4),textcoords='offset points',fontsize=7)
        ax.set(title=f"{r['label']} | dx={dd:g} m",xlabel='Prototype X (m)',ylabel='Prototype Y (m)')
    fig.colorbar(im,ax=list(axes.flat),label='Water depth at 40 s (m)',shrink=.65)
    fig.suptitle('Synthetic dam break on original terrain crop\nFine patches check obstacle flow and storage; not experimental boundary conditions',fontsize=13)
    fig.savefig(out/'native-patch.png',dpi=160);plt.close(fig)


def paired_report(out,full,baseline):
    current=read(full/'result.json');old=read(baseline/'result.json')
    cfg=read(full/'config.json');oldcfg=read(baseline/'config.json')
    ar=dict(np.load(full/'final.npz'));series=read(full/'series.json');times=np.array([r['time_s'] for r in series])
    speeds=np.array([r['speeds_m_s'] for r in series]);late=speeds[times>=750].mean(axis=0)
    checks=dict(complete_800s=current['complete'] and current['duration_s']==800,
        same_forcing=cfg['scenario']['external_inflows']==oldcfg['scenario']['external_inflows'],
        same_flood_settings=cfg['scenario']['flood']==oldcfg['scenario']['flood'],
        same_numerical_kernel=all(cfg['code_sha256'][k]==oldcfg['code_sha256'][k] for k in ['flood_solver.py','flood_acceleration.py']),
        same_source=cfg['source']['files']==oldcfg['source']['files'],
        fixed_measurements=len(current['points'])==len(old['points'])==40 and all(all(a[k]==b[k] for k in ['id','x_m','y_m','speed_m_s','angle_xy_deg']) for a,b in zip(current['points'],old['points'])),
        mass=abs(current['balance']['residual_m3'])<1e-6 and current['balance']['numerical_adjustment_m3']==0,
        forcing_volume=abs(current['balance']['external_inflow_m3']-19.6992*800)<1e-6,
        solid_stays_dry=bool((ar['depth'][ar['solid']]==0).all()),
        finite=all(np.isfinite(ar[k]).all() for k in ar),nonnegative=bool((ar['depth']>=0).all()),
        same_roughness=bool(np.all(ar['n']==.042)),
        maxima_recomputed=bool(np.allclose(speeds.max(axis=0),[r['model_max_speed_m_s'] for r in current['points']],rtol=1e-12,atol=1e-12)))
    errors=np.array([r['model_max_speed_m_s']-r['speed_m_s'] for r in current['points']])
    checks['rmse_recomputed']=bool(np.isclose(np.sqrt(np.mean(errors**2)),current['metrics']['speed_rmse_m_s'],rtol=1e-12))
    checks={k:bool(v) for k,v in checks.items()};write(out/'validation.json',checks)
    if not all(checks.values()):raise AssertionError('Paired comparison audit failed: '+str(checks))
    rows=[]
    for i,(p,b) in enumerate(zip(current['points'],old['points'])):
        rows.append(dict(id=p['id'],x_m=p['x_m'],y_m=p['y_m'],measured_max_m_s=p['speed_m_s'],
            old_max_m_s=b['model_max_speed_m_s'],layered_max_m_s=p['model_max_speed_m_s'],layered_late_mean_m_s=float(late[i]),
            old_error_m_s=b['speed_error_m_s'],layered_error_m_s=p['speed_error_m_s'],
            absolute_error_change_m_s=abs(p['speed_error_m_s'])-abs(b['speed_error_m_s']),
            old_direction_error_deg=b['direction_error_deg'],layered_direction_error_deg=p['direction_error_deg'],
            old_max_depth_m=b['max_depth_m'],layered_max_depth_m=p['max_depth_m'],
            old_bed_m=b['sampled_bed_m'],layered_ground_m=p['sampled_bed_m']))
    table(out/'paired-points.csv',rows)
    stages=[];stagexy=np.loadtxt(LAB/'.runtime/merewether/source/experiment/merewether.stage',skiprows=1)
    for i,(x,y) in enumerate(stagexy):
        z=int((ar['bed'].shape[0]*.7-y)/.7);col=int(x/.7)
        stages.append(dict(id=f'P{i+1}',x_m=float(x),y_m=float(y),ground_m=float(ar['bed'][z,col]),solid=bool(ar['solid'][z,col]),
            final_depth_m=float(ar['depth'][z,col]),final_stage_m=float(ar['bed'][z,col]+ar['depth'][z,col])))
    table(out/'stage-results.csv',stages)
    write(out/'summary.json',dict(baseline=old['metrics'],layered=current['metrics'],
        improved_points=sum(p['absolute_error_change_m_s']<0 for p in rows),wall_s=current['wall_s'],
        common_direction_cohort=dict(points=sum(p['old_direction_error_deg'] is not None and p['layered_direction_error_deg'] is not None for p in rows),
            baseline_mae_deg=float(np.mean([abs(p['old_direction_error_deg']) for p in rows if p['old_direction_error_deg'] is not None and p['layered_direction_error_deg'] is not None])),
            layered_mae_deg=float(np.mean([abs(p['layered_direction_error_deg']) for p in rows if p['old_direction_error_deg'] is not None and p['layered_direction_error_deg'] is not None]))),
        balance=current['balance'],stages=stages,scope='full-domain 800 s paired diagnostic; inferred geometry, no acceptance threshold'))
    compact=out/'full-run';compact.mkdir(exist_ok=True)
    for name in ['result.json','config.json','points.csv']:shutil.copyfile(full/name,compact/name)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    write(out/'provenance.json',dict(script_sha256=sha(Path(__file__)),
        layered={name:sha(full/name) for name in ['result.json','config.json','series.json','final.npz']},
        baseline={name:sha(baseline/name) for name in ['result.json','config.json']},source=cfg['source']))
    render_pair(out,current,old,ar,rows,series,stages)
    text_report(out,current,old,rows,stages)
    print(json.dumps(dict(stage='paired-complete',checks=len(checks),metrics=current['metrics'],wall_s=current['wall_s'])))


def render_pair(out,current,old,ar,rows,series,stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,8),layout='constrained')
    extent=(0,ar['bed'].shape[1]*.7,0,ar['bed'].shape[0]*.7)
    ax=axes[0];ax.imshow(np.ma.masked_less(ar['depth'],.01),extent=extent,origin='upper',cmap='Blues',vmin=0,vmax=1)
    ax.imshow(np.ma.masked_where(~ar['solid'],np.ones_like(ar['solid'])),extent=extent,origin='upper',cmap='Greys',vmin=0,vmax=1)
    sc=ax.scatter([p['x_m'] for p in rows],[p['y_m'] for p in rows],c=[p['layered_error_m_s'] for p in rows],s=40,cmap='RdBu_r',vmin=-2,vmax=2,edgecolors='white',linewidths=.5)
    for p in rows:
        if p['id'] in ['14-F','14-H','15-J','19-J','20-F']:ax.annotate(p['id'],(p['x_m'],p['y_m']),xytext=(3,3),textcoords='offset points',fontsize=8)
    ax.set(xlim=(10,110),ylim=(125,225),xlabel='X (m)',ylabel='Y (m)',title='Layered final water and peak-speed errors')
    fig.colorbar(sc,ax=ax,label='Model minus measured maximum speed (m/s)',shrink=.6)
    ranked=sorted(rows,key=lambda p:abs(p['old_error_m_s']),reverse=True);y=np.arange(40)
    ax=axes[1];ax.scatter([p['measured_max_m_s'] for p in ranked],y,c='black',s=17,label='Measured maximum')
    ax.scatter([p['old_max_m_s'] for p in ranked],y,c='#b6782b',s=19,marker='x',label='Previous maximum DEM')
    ax.scatter([p['layered_max_m_s'] for p in ranked],y,c='#147e8f',s=19,label='Layered ground / obstacles')
    ax.set(yticks=y,yticklabels=[p['id'] for p in ranked],xlabel='Maximum speed (m/s)',title='All 40 fixed measurement locations');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.09))
    fig.suptitle('Same 0.7 m grid, forcing, n=0.042 and numerical kernel | 800 s',fontsize=14)
    fig.savefig(out/'paired-flood.png',dpi=165);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    times=[r['time_s'] for r in series]
    for i,(ax,p) in enumerate(zip(axes.flat,stages)):
        ax.plot(times,[r['stages_m'][i] for r in series],c='#147e8f',label='Layered computed water surface')
        ax.axhline(p['ground_m'],ls=':',color='#75634d',label='Fluid ground')
        ax.set(title=p['id'],xlabel='Time (s)',ylabel='Elevation (m)');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Computed stages after separation | no raw measured stage values supplied',fontsize=13)
    fig.savefig(out/'layered-stages.png',dpi=160);plt.close(fig)


def text_report(out,current,old,rows,stages):
    geo=read(out/'geometry.json');pa=read(out/'patch.json');g=geo['grids']['4']
    chosen=['14-F','14-H','15-H','17-E','15-J','19-J','20-F']
    rr=[next(p for p in rows if p['id']==key) for key in chosen]
    rtable='\n'.join(f"| {p['id']} | {p['measured_max_m_s']:.2f} | {p['old_max_m_s']:.2f} | {p['layered_max_m_s']:.2f} | {p['layered_late_mean_m_s']:.2f} | {p['layered_max_depth_m']:.3f} |" for p in rr)
    stable='\n'.join(f"| {p['id']} | {p['ground_m']:.3f} | {p['final_depth_m']:.3f} | {p['final_stage_m']:.3f} |" for p in stages)
    a,b=old['metrics'],current['metrics'];improved=sum(p['absolute_error_change_m_s']<0 for p in rows)
    cohort=[p for p in rows if p['old_direction_error_deg'] is not None and p['layered_direction_error_deg'] is not None]
    paired_direction_before=np.mean([abs(p['old_direction_error_deg']) for p in cohort])
    paired_direction_after=np.mean([abs(p['layered_direction_error_deg']) for p in cohort])
    text=f'''# Merewether 地面与障碍分层：第一版实现与同条件对照

完成了地面／障碍分层导入、几何检查、0.175米局部绕流检查及0.7米全域800秒对照。原始DEM、旧基线、回放和求解器方程保留。当前版本使用独立整格不透水占地，**没有实现粗格内部的部分通水、子网格阻力或全域局部加密耦合**。

## 数据表示与来源

原包只提供叠加障碍的DEM，没有独立裸地或墙／墩轮廓。另一份已下载DEM与本包逐像素一致，不能作为独立裸地。第一版按DEM中反复出现的30米人工平台及9×9像素开运算残差>0.75米提取障碍；平台边缘两像素范围作同阈值补充。阈值和规则在运行水流前固定，未使用40点速度或水位来拟合。

保留原始表面、推断障碍掩膜、地面高程、估计地面标记。障碍下面采用最近已知地面估计，**不是实测裸地**；未标为障碍的原始地面保持逐位不变。不能把此规则用于任意DEM：30米是本实验数据特征，其他地图应提供独立障碍掩膜。尚未自动区分墙、建筑和桥墩的语义。

粗格地面只平均原始非障碍地面；阻水占地单独按覆盖过半取整格，每个将消失的小障碍保留最大重叠格。这保证每个连通障碍至少有表示，但不保证原始宽度、空隙和连通性。覆盖比例单独保存供诊断，求解器不使用比例通量。

独立wall_x／wall_y接口仍可导入已知墙线，本包没有可靠墙线，所以没有从建筑边缘擅自生成新的墙。墙及占地的阻水守恒有独立测试。

## 几何检查

原始提取障碍面积{g['native_obstacle_area_m2']:.2f} m²，共{g['native_components']}个8邻域连通对象。0.7米整格表示额外封闭原始地面{g['extra_blocked_native_area_m2']:.2f} m²，遗漏原始障碍占地{g['lost_obstacle_native_area_m2']:.2f} m²；{g['minimum_component_cells_added']}个细对象依靠保留一个粗格避免消失。所有对象至少有格表示，并不等于几何精确。

0.175／0.35／0.7／1.4米的面积、对象丢失／分裂、原始与粗格测点落障碍及南侧入口的4邻域几何可达性见geometry.json及geometry-probes.csv。可达性只检查障碍拓扑，不代表给定水位下能进水或具有正确过流能力。已知测点不平移、不挖槽。

![分层地图](layered-map.png)
![关键点恢复地面](probe-ground.png)

在0.7米分层地图中P1、P2、P3、20-F均位于可通水格。0.35米的P2、1.4米的P2／P3仍落入整格障碍；这说明网格相位与障碍取整仍重要，不能只按格宽排序。再次合并此地图会破坏已审计的占地，因此导入流程要求按目标格宽重新生成，阻止工作台直接二次合并。

## 原始格宽局部检查

在原始坐标x=[40.6,60.2]、y=[187.6,204.4]米裁剪地形，0.175米为96×112格。使用封闭边界、左侧2.8米宽初始水面19.8米的人工溃坝，计算40秒；比较原始叠加高程与分层障碍，同时运行0.7米对照。总计算预算{pa['wall_budget_s']:g}秒，实际{pa['wall_s']:.1f}秒。

四组均保持非负、有限和水量守恒，固体格储水为零。这个检查验证原始地面和桥墩占地可直接参与细格计算；**裁剪边界为人工条件，不能用这些结果给Merewether实验精度打分，也没有与全域粗格双向耦合**。没有启用复制父格高程的旧加密路径。

原始0.175米两组最终水深最大差为{pa['native_equivalence']['max_depth_difference_m']:.3g}米，说明在这组未漫顶条件下，显式固体障碍能保持原高程障碍的细格水流行为。四组固定相同初始水面，粗格初始水量随地形表示变化，未人为缩放水量来拼成相同起点；实际初始体积和20-F坐标处水深都保存在patch.json。该检查不能证明其他水位下等价，尤其不覆盖漫顶。

![原始格宽局部水流](native-patch.png)

## 全域800秒同条件对照

使用同一0.7米格宽、95,700格、n=0.042、南侧38米来水、总流量19.6992 m³/s、北侧自由出流及其余封闭边界；同一Rusanov/MUSCL/RK2/CFL0.4内核，内核哈希与基线一致。完整跑到800秒，主计算{current['wall_s']:.1f}秒，预算900秒。配置、实际固体占地、阻水边及运行源代码快照随本地结果保存。主比较仍使用40个固定点0–800秒最大速度，不以局部检查或后期均速替换。

| 指标 | 原最高值DEM | 分层地面／障碍 |
|---|---:|---:|
| 最大速度RMSE(m/s) | {a['speed_rmse_m_s']:.3f} | {b['speed_rmse_m_s']:.3f} |
| 最大速度MAE(m/s) | {a['speed_mae_m_s']:.3f} | {b['speed_mae_m_s']:.3f} |
| 最大速度偏差(m/s) | {a['speed_bias_m_s']:+.3f} | {b['speed_bias_m_s']:+.3f} |
| 方向MAE(°) | {a['direction_mae_deg']:.1f} | {b['direction_mae_deg']:.1f} |
| 有效方向点数 | {a['direction_points']} | {b['direction_points']} |
| 最大水深未达1cm点数 | {a['model_dry_points']} | {b['model_dry_points']} |

{improved}/40点的绝对速度误差减小，其他点增大或持平。没有删掉干点和方向无定义点。方向只对实验和模型速度都≥0.05 m/s者评分，样本身份见paired-points.csv。模型后期均速取750–800秒，仅供诊断。

为避免方向有效点数改变影响解释，另对两组共同有效的{len(cohort)}点比较：方向MAE从{paired_direction_before:.2f}°降至{paired_direction_after:.2f}°。新增有效方向点不会被当成同一固定样本的误差下降。

这组地形表示变更使速度RMSE降低约{100*(1-b['speed_rmse_m_s']/a['speed_rmse_m_s']):.1f}%，MAE降低约{100*(1-b['speed_mae_m_s']/a['speed_mae_m_s']):.1f}%。14-F从2.721降到0.930 m/s，接近实测0.8；20-F恢复有效进水，最大速度0.288 m/s、实测0.4。19-J仍未达到1厘米水深；14-H和15-H仍持续过快。14-G反而从1.776升到2.229 m/s（实测0.9），17-E的起涨峰值也略增。因此只支持部分局部流路改善，不把全域误差归因于单一问题。此对照同时改变了地面合并和障碍栅格表示，尚未单独分离两者的贡献。

| 点位 | 实测最大(m/s) | 旧最大(m/s) | 分层最大(m/s) | 分层后期均速(m/s) | 分层最大水深(m) |
|---|---:|---:|---:|---:|---:|
{rtable}

![全40点同条件对比](paired-flood.png)

| 水位点 | 地面高程(m) | 800秒水深(m) | 800秒计算水面(m) |
|---|---:|---:|---:|
{stable}

这些水位是计算值，数据包未给出精确实测水位数值，仍不算水位RMSE。模型入水和峰值时刻也没有实验逐时真值可评分。

![分层后的计算水位](layered-stages.png)

## 当前完成范围与后续

数据分层、独立不透水障碍接入、几何检查、原始格宽局部计算和完整粗格对照已完成。分层使地面与阻水作用能够分别检查，但推断轮廓、整格取整和遗漏开口仍限制精度。改善幅度只能支持这组配置，不能据此认定模拟器已通过真实洪水验证。

如果继续，优先取得独立墙／墩轮廓，修正几何中实际发现的封口、漏口；再在计算预算允许时做真实地形的全域细格或守恒局部加密。粗格部分通水属于储水／通量／阻力模型变更，应另做守恒、静水、窄通道与同条件实验，不把当前覆盖比例直接乘到通量上。

复现命令见上级merewether.md。validation.json检查完整窗口、固定测点、同一来水、参数、来源与内核、重新提取最大值、RMSE、水量、糙率及固体无储水。provenance.json记录源数据和输入结果哈希；full-run/归档紧凑结果与配置；原始DEM、大型场、局部原始场和回放留在本地忽略目录。

本地工作台已检查800秒回放、n=0.042、95,700格、固定格宽提示以及地形页独立“阻水占地”切换。界面截图见ui-replay.png、ui-obstacles.png。
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['geometry','patch','report','all'],default='all')
    p.add_argument('--output',type=Path,default=LAB.parent/'docs/experiments/2026-10-20/merewether/layered')
    p.add_argument('--patch-budget',type=float,default=120)
    p.add_argument('--full',type=Path,default=LAB/'.runtime/merewether/layered-results/f4-n0.042-layered-v1')
    p.add_argument('--baseline',type=Path,default=LAB/'.runtime/merewether/results/f4-n0.042')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    os.environ['HYRA_FLOOD_BACKEND']='numba'
    source=LAB/'.runtime/merewether/source/experiment'
    surface,valid,header=raster_import.read_ascii(source/'merewether-0p175m.dem')
    layers=obstacle_layers.merewether_layers(surface)
    if args.phase in ['geometry','all']:geometry(args.output,surface,layers,source)
    if args.phase in ['patch','all']:patch(args.output,LAB/'.runtime/merewether/layered-patch',surface,layers,args.patch_budget)
    if args.phase in ['report','all']:paired_report(args.output,args.full,args.baseline)


if __name__=='__main__':main()
