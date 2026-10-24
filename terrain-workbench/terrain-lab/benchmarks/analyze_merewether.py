"""Reproducible point, geometry and temporal diagnosis of saved Merewether runs.

Primary scores retain all 40 fixed measurement locations and the full 800 s
maximum-speed definition. Late speeds and neighbourhoods are diagnostics only.
No observed time history or measured depth is invented.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
import raster_import


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def load_run(folder):
    result = read(folder/'result.json')
    series = read(folder/'series.json')
    arrays = dict(np.load(folder/'final.npz', allow_pickle=False))
    return dict(label=folder.name, result=result, arrays=arrays, series=series,
                config=read(folder/'config.json'),
                times=np.array([p['time_s'] for p in series]),
                speeds=np.array([p['speeds_m_s'] for p in series]),
                depths=np.array([p['depths_m'] for p in series]))


def diagnose(run, native):
    r, a = run['result'], run['arrays']
    assert r['complete'] and r['duration_s'] == 800
    dx, factor = r['dx_m'], r['factor']
    ymax = a['bed'].shape[0]*dx
    native_ymax = native.shape[0]*.175
    observed = np.array([p['speed_m_s'] for p in r['points']])
    late = run['speeds'][run['times'] >= 750].mean(axis=0)
    late_range = np.ptp(run['speeds'][run['times'] >= 750], axis=0)
    err = np.array([p['speed_error_m_s'] for p in r['points']])
    total_sse = float(np.sum(err**2))
    speed_final = np.where(a['depth'] >= .01, np.hypot(a['mx'], a['mz'])/np.maximum(a['depth'], 1e-12), 0.)
    rows=[]
    for i, p in enumerate(r['points']):
        z, x = p['grid_row'], p['grid_col']
        rz, rx = int((native_ymax-p['y_m'])/.175), int(p['x_m']/.175)
        arrival=np.flatnonzero(run['depths'][:,i] >= .01)
        z0,z1=max(0,z-int(np.ceil(2/dx))),min(a['bed'].shape[0],z+int(np.ceil(2/dx))+1)
        x0,x1=max(0,x-int(np.ceil(2/dx))),min(a['bed'].shape[1],x+int(np.ceil(2/dx))+1)
        zz,xx=np.mgrid[z0:z1,x0:x1]
        disk=(xx*dx+dx/2-p['x_m'])**2+(ymax-(zz+.5)*dx-p['y_m'])**2 <= 2**2
        wet=disk & (a['depth'][z0:z1,x0:x1] >= .01)
        neighbours=speed_final[z0:z1,x0:x1][wet]
        vector_error=np.hypot(p['model_u_at_max_m_s']-p['u_m_s'],p['model_v_at_max_m_s']-p['v_m_s'])
        rows.append(dict(**p, late_mean_speed_m_s=float(late[i]), late_speed_range_m_s=float(late_range[i]),
            late_speed_error_m_s=float(late[i]-observed[i]), peak_minus_late_m_s=float(p['model_max_speed_m_s']-late[i]),
            vector_error_m_s=float(vector_error), squared_error_share_pct=100*err[i]**2/total_sse,
            model_first_1cm_s=float(run['times'][arrival[0]]) if arrival.size else None,
            native_point_bed_m=float(native[rz,rx]), bed_uplift_at_point_m=float(a['bed'][z,x]-native[rz,rx]),
            cell_centre_distance_m=float(np.hypot((x+.5)*dx-p['x_m'],ymax-(z+.5)*dx-p['y_m'])),
            final_depth_m=float(a['depth'][z,x]), final_speed_m_s=float(speed_final[z,x]),
            final_2m_wet_neighbour_count=int(neighbours.size),
            final_2m_neighbour_min_speed_m_s=float(neighbours.min()) if neighbours.size else None,
            final_2m_neighbour_max_speed_m_s=float(neighbours.max()) if neighbours.size else None))
    order=np.argsort(err**2)[::-1]
    nonzero=observed >= .05
    metrics=dict(primary=r['metrics'], late_diagnostic_rmse_m_s=rms(late-observed),
        late_diagnostic_bias_m_s=float(np.mean(late-observed)),
        vector_rmse_m_s=rms([p['vector_error_m_s'] for p in rows]),
        top5_squared_error_share_pct=float(sum(rows[i]['squared_error_share_pct'] for i in order[:5])),
        top10_squared_error_share_pct=float(sum(rows[i]['squared_error_share_pct'] for i in order[:10])),
        points_abs_error_over_05=int(np.count_nonzero(abs(err)>.5)),
        points_abs_error_under_025=int(np.count_nonzero(abs(err)<=.25)),
        transient_excess_over_05=int(sum(p['peak_minus_late_m_s']>.5 for p in rows)),
        late_speed_range_over_01=int(np.count_nonzero(late_range>.1)),
        largest_late_speed_range_m_s=float(late_range.max()),
        measured_flow_model_dry=[p['id'] for p in rows if p['speed_m_s']>=.05 and p['max_depth_m']<.01],
        direction_over_45deg=[p['id'] for p in rows if p['direction_error_deg'] is not None and abs(p['direction_error_deg'])>45],
        normalized_rmse_by_observed_mean=float(rms(err)/observed[nonzero].mean()),
        measured_mean_nonzero_m_s=float(observed[nonzero].mean()))
    groups=[]
    for name, mask in [('upstream_y135_160',np.array([p['y_m']<160 for p in rows])),
                       ('middle_y160_190',np.array([160<=p['y_m']<190 for p in rows])),
                       ('downstream_y190_216',np.array([p['y_m']>=190 for p in rows]))]:
        groups.append(dict(region=name,points=int(mask.sum()),rmse_m_s=rms(err[mask]),bias_m_s=float(err[mask].mean()),
                           squared_error_share_pct=float(100*np.sum(err[mask]**2)/total_sse)))
    # Verify the saved grid and repeat its padding/alignment, never resample by eye.
    method='mean' if run['label'].endswith('-mean') else 'max'
    coarse,solid,meta=raster_import.coarsen(native,np.ones(native.shape,bool),factor,method)
    np.testing.assert_array_equal(coarse,a['bed'])
    reconstructed=np.repeat(np.repeat(coarse,factor,axis=0),factor,axis=1)
    top=meta['padded_north_cells']
    uplift=reconstructed[top:top+native.shape[0],:native.shape[1]]-native
    yy=native_ymax-(np.arange(native.shape[0])+.5)*.175
    region=uplift[(yy>=125)&(yy<=225)]
    geometry=dict(aggregation=method,native_dx_m=.175,native_shape=list(native.shape),
        cell_width_to_side_pier_ratio=dx/.35,cell_width_to_corner_pier_ratio=dx/.52,
        native_area_raised_over_05_m2=float(np.count_nonzero(uplift>.5)*.175**2),
        pier_region_area_raised_over_05_m2=float(np.count_nonzero(region>.5)*.175**2),
        pier_region_fraction_raised_over_05_pct=float(np.mean(region>.5)*100),
        max_point_uplift_m=float(max(p['bed_uplift_at_point_m'] for p in rows)))
    return dict(rows=rows,metrics=metrics,groups=groups,geometry=geometry)


def stage_audit(runs,native,source):
    coords=np.loadtxt(source/'merewether.stage',skiprows=1)
    nymax=native.shape[0]*.175
    rows=[]
    for i,(x,y) in enumerate(coords):
        raw=float(native[int((nymax-y)/.175),int(x/.175)])
        for run in runs:
            a=run['arrays'];dx=run['result']['dx_m'];z=int((a['bed'].shape[0]*dx-y)/dx);col=int(x/dx)
            rows.append(dict(point=f'P{i+1}',run=run['label'],x_m=float(x),y_m=float(y),native_bed_m=raw,
                model_bed_m=float(a['bed'][z,col]),uplift_m=float(a['bed'][z,col]-raw),
                final_depth_m=float(a['depth'][z,col]), final_surface_m=float(a['bed'][z,col]+a['depth'][z,col]),
                note='No raw measured stage supplied; no stage error scored'))
    return rows


def render(out,runs,diag,native,stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    main=runs[0];points=diag[main['label']]['rows'];a=main['arrays'];dx=main['result']['dx_m']
    fig,axes=plt.subplots(1,2,figsize=(12,8),layout='constrained')
    ax=axes[0];ax.imshow(a['bed'],extent=(0,a['bed'].shape[1]*dx,0,a['bed'].shape[0]*dx),origin='upper',cmap='Greys',vmin=18,vmax=26)
    x=[p['x_m'] for p in points];y=[p['y_m'] for p in points]
    sc=ax.scatter(x,y,c=[p['speed_error_m_s'] for p in points],s=70,cmap='RdBu_r',vmin=-2,vmax=2,edgecolor='white',lw=.5)
    worst={p['id'] for p in sorted(points,key=lambda p:p['squared_error_share_pct'],reverse=True)[:8]}
    for p in points:
        if p['id'] in worst or p['max_depth_m']<.01:ax.annotate(p['id'],(p['x_m'],p['y_m']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlim=(10,110),ylim=(125,225),xlabel='Prototype X (m)',ylabel='Prototype Y (m)',title='Where the peak-speed errors occur\nRed: too fast | blue: too slow / dry')
    fig.colorbar(sc,ax=ax,label='Model minus measured maximum speed (m/s)',shrink=.7)
    ax=axes[1];ranked=sorted(points,key=lambda p:p['squared_error_share_pct'],reverse=True)
    ids=[p['id'] for p in ranked];pos=np.arange(len(ids))
    ax.hlines(pos,[p['speed_m_s'] for p in ranked],[p['model_max_speed_m_s'] for p in ranked],color='#aeb9c2',lw=1)
    ax.scatter([p['speed_m_s'] for p in ranked],pos,c='#151b20',s=18,label='Measured maximum')
    ax.scatter([p['model_max_speed_m_s'] for p in ranked],pos,c='#c44737',s=20,label='Model maximum 0-800 s')
    ax.scatter([p['late_mean_speed_m_s'] for p in ranked],pos,c='#167c91',s=17,marker='x',label='Model mean 750-800 s (diagnostic)')
    ax.set(yticks=pos,yticklabels=ids,xlabel='Speed (m/s)',title='All 40 points, ordered by squared error')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.065))
    fig.suptitle('Merewether detailed diagnosis | dx=0.7 m, n=0.042',fontsize=15)
    fig.savefig(out/'point-errors.png',dpi=170);plt.close(fig)

    selected=['14-F','14-H','15-H','17-E','19-J','16-K']
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    ids=[p['id'] for p in points]
    for ax,key in zip(axes.flat,selected):
        j=ids.index(key);p=points[j]
        ax.plot(main['times'],main['speeds'][:,j],c='#167c91',label='Model speed')
        ax.axhline(p['speed_m_s'],c='#c44737',ls='--',label='Measured maximum (not a time series)')
        ax.axvspan(750,800,alpha=.12,color='#167c91')
        ax.set(title=f"{key} | peak {p['model_max_speed_m_s']:.2f}, late {p['late_mean_speed_m_s']:.2f} m/s",xlabel='Model time (s)',ylabel='Speed (m/s)',ylim=(0,None))
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Persistent speed errors, transient peaks, and missed wetting',fontsize=14)
    fig.savefig(out/'speed-histories.png',dpi=160);plt.close(fig)

    coords=np.loadtxt(LAB/'.runtime/merewether/source/experiment/merewether.stage',skiprows=1)
    fig,axes=plt.subplots(3,2,figsize=(11,12),layout='constrained')
    for i in range(3):
        x,y=coords[i];lims=(x-2.8,x+2.8,y-2.8,y+2.8)
        rz=int((native.shape[0]*.175-y)/.175);rx=int(x/.175)
        for j,(bed,dd) in enumerate([(native,.175),(a['bed'],dx)]):
            yy=bed.shape[0]*dd;z0=max(0,int((yy-lims[3])/dd));z1=min(bed.shape[0],int((yy-lims[2])/dd)+1)
            x0=max(0,int(lims[0]/dd));x1=min(bed.shape[1],int(lims[1]/dd)+1)
            ax=axes[i,j];im=ax.imshow(bed[z0:z1,x0:x1],extent=(x0*dd,x1*dd,yy-z1*dd,yy-z0*dd),origin='upper',cmap='terrain',vmin=float(native[rz,rx]-.5),vmax=float(native[rz,rx]+5))
            ax.scatter([x],[y],marker='+',s=120,c='black',lw=2)
            if j:
                from matplotlib.patches import Rectangle
                zz=int((yy-y)/dd);xx=int(x/dd);ax.add_patch(Rectangle((xx*dd,yy-(zz+1)*dd),dd,dd,fill=False,edgecolor='#e12632',lw=2))
            ax.set(xlim=lims[:2],ylim=lims[2:],xlabel='X (m)',ylabel='Y (m)',title=f'P{i+1}: '+('native 0.175 m' if j==0 else 'max aggregation 0.7 m'))
            fig.colorbar(im,ax=ax,label='Bed elevation (m)',shrink=.7)
    fig.suptitle('Water-level probes intersect enlarged obstacles after max aggregation\nBlack cross: fixed source coordinate; red box: sampled coarse cell',fontsize=13)
    fig.savefig(out/'stage-geometry.png',dpi=155);plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for i,key in enumerate(['20-F','19-J']):
        p=next(p for p in points if p['id']==key);x,y=p['x_m'],p['y_m'];radius=3.5
        for j,(field,dd,title) in enumerate([(native,.175,'Native bed'),(a['bed'],dx,'Coarse maximum bed'),(a['depth'],dx,'Final model depth')]):
            ymax=field.shape[0]*dd;z0=max(0,int((ymax-y-radius)/dd));z1=min(field.shape[0],int((ymax-y+radius)/dd)+1)
            x0=max(0,int((x-radius)/dd));x1=min(field.shape[1],int((x+radius)/dd)+1)
            ax=axes[i,j]
            vals=field[z0:z1,x0:x1]
            if j==2:vals=np.ma.masked_less(vals,.01)
            im=ax.imshow(vals,extent=(x0*dd,x1*dd,ymax-z1*dd,ymax-z0*dd),origin='upper',
                cmap='Blues' if j==2 else 'Greys',vmin=0 if j==2 else p['native_point_bed_m']-.5,
                vmax=1 if j==2 else p['native_point_bed_m']+4)
            ax.scatter([x],[y],c='#e12632',marker='+',s=130,lw=2)
            ax.set(xlim=(x-radius,x+radius),ylim=(y-radius,y+radius),xlabel='X (m)',ylabel='Y (m)',title=key+' | '+title)
            fig.colorbar(im,ax=ax,label='Depth (m)' if j==2 else 'Elevation (m)',shrink=.65)
    fig.suptitle('Two missed-flow points with different local geometry\n20-F: obstacle expands into probe cell | 19-J: broad local flow deficit',fontsize=13)
    fig.savefig(out/'dry-point-geometry.png',dpi=155);plt.close(fig)

    fig,axes=plt.subplots(1,len(runs),figsize=(4*len(runs),7),layout='constrained',squeeze=False)
    for ax,run in zip(axes.flat,runs):
        ar=run['arrays'];dd=run['result']['dx_m'];yy=ar['bed'].shape[0]*dd
        ax.imshow(ar['bed'],extent=(0,ar['bed'].shape[1]*dd,0,yy),origin='upper',cmap='Greys',vmin=18,vmax=26)
        h=ax.imshow(np.ma.masked_less(ar['depth'],.01),extent=(0,ar['bed'].shape[1]*dd,0,yy),origin='upper',cmap='Blues',vmin=0,vmax=1)
        ax.scatter(x=[p['x_m'] for p in points],y=[p['y_m'] for p in points],s=8,c='#d24936')
        for p in diag[run['label']]['rows']:
            if p['speed_m_s']>.05 and p['max_depth_m']<.01:ax.annotate(p['id'],(p['x_m'],p['y_m']),fontsize=8,color='#a00000')
        ax.set(xlim=(10,110),ylim=(125,225),xlabel='X (m)',ylabel='Y (m)',title=run['label']+'\nFinal depth; red: fixed probes')
    fig.colorbar(h,ax=list(axes.flat),label='Final depth (m)',shrink=.5)
    fig.savefig(out/'wetting-comparison.png',dpi=155);plt.close(fig)


def report(out,runs,diag,stages,checks):
    main=diag[runs[0]['label']];m=main['metrics'];ranked=sorted(main['rows'],key=lambda p:p['squared_error_share_pct'],reverse=True)
    table=[]
    for p in ranked[:12]:
        angle='—' if p['direction_error_deg'] is None else f"{p['direction_error_deg']:+.1f}"
        table.append(f"| {p['id']} | {p['speed_m_s']:.2f} | {p['model_max_speed_m_s']:.2f} | {p['late_mean_speed_m_s']:.2f} | {p['speed_error_m_s']:+.2f} | {angle} | {p['max_depth_m']:.3f} | {p['time_of_model_max_s']:g} | {p['squared_error_share_pct']:.1f}% |")
    direction_table=[]
    for pp in sorted((p for p in main['rows'] if p['direction_error_deg'] is not None),key=lambda p:abs(p['direction_error_deg']),reverse=True)[:6]:
        model_angle=np.degrees(np.arctan2(pp['model_v_at_max_m_s'],pp['model_u_at_max_m_s']))%360
        direction_table.append(f"| {pp['id']} | {pp['speed_m_s']:.2f} | {pp['model_max_speed_m_s']:.2f} | {pp['angle_xy_deg']:.1f} | {model_angle:.1f} | {pp['direction_error_deg']:+.1f} |")
    control=diag['f8-n0.042-mean'];control_base=diag['f8-n0.042']
    control_better=sum(abs(p['speed_error_m_s'])<abs(q['speed_error_m_s']) for p,q in zip(control['rows'],control_base['rows']))
    summary=[]
    for run in runs:
        d=diag[run['label']];s=d['metrics'];r=run['result']
        summary.append(f"| {run['label']} | {s['primary']['speed_rmse_m_s']:.3f} | {s['primary']['speed_mae_m_s']:.3f} | {s['primary']['speed_bias_m_s']:+.3f} | {s['primary']['direction_mae_deg']:.1f} | {s['late_diagnostic_rmse_m_s']:.3f} | {s['primary']['model_dry_points']} |")
    st=[]
    for p in stages:
        if p['run']==runs[0]['label']:st.append(f"| {p['point']} | {p['native_bed_m']:.3f} | {p['model_bed_m']:.3f} | {p['uplift_m']:+.3f} | {p['final_depth_m']:.3f} |")
    regions=[f"| {g['region']} | {g['points']} | {g['rmse_m_s']:.3f} | {g['bias_m_s']:+.3f} | {g['squared_error_share_pct']:.1f}% |" for g in main['groups']]
    geo=main['geometry']
    text=f'''# Merewether：逐测点差距与误差来源诊断（2026-10-20）

本报告拆分已有三组800秒计算，并增加一组仅改变地形合并规则的对照。**已确认“最高高程合并”会把部分水位测点所在的低地变成建筑高程；流速还同时存在持续过快、起涨峰值过高和局部漏淹。当前不能把全部偏差归结为糙率或求解器。**

## 1. 比较口径与资料边界

40个固定位置的实验最大流速／方向来自 [Zenodo 6914888 Test 4](https://zenodo.org/records/6914888)。主评分沿用完整0–800秒最大速度矢量，与[原论文第22–23页、图12](https://eprints.whiterose.ac.uk/id/eprint/197348/1/2210.16972.pdf)的提取方法对应。最大速度发生时取同一对u/v，不拼接不同时间的分量。方向误差为模型减实测，包裹到±180°，只对双方速度≥0.05 m/s的点计算。

750–800秒模型速度平均值用于区分持续误差和起涨瞬态；实验没有逐时速度记录，不能把它替换为最大值主评分。水深、入水时间和峰值时间均是模型诊断量，**没有相应实验逐时真值，不能称为水深误差、到达时间误差**。没有使用论文其他求解器的预测值作为观测。

| 配置 | 最大速度RMSE | 最大速度MAE | 最大速度偏差 | 方向MAE(°) | 后期均速诊断RMSE | 未达1cm点数 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary)}

速度单位均为m/s。主配置观测非零速度均值为{m['measured_mean_nonzero_m_s']:.3f} m/s；速度矢量RMSE为{m['vector_rmse_m_s']:.3f} m/s。单看速度大小会忽略水流走错方向的问题。所有40点仍进入主速度评分，未剔除干点或异常点。

## 2. 差距集中在哪些点

主配置误差最大的5点贡献了总平方误差的 **{m['top5_squared_error_share_pct']:.1f}%**，前10点贡献{m['top10_squared_error_share_pct']:.1f}%。绝对速度误差≤0.25 m/s的点有{m['points_abs_error_under_025']}个，>0.5 m/s的点有{m['points_abs_error_over_05']}个。因此优先检查少数局部流路比统一调整全域参数更有针对性。

| 点位 | 实测最大 | 模型最大 | 模型后期均速 | 最大速度误差 | 方向误差(°) | 最大水深(m) | 模型峰值时刻(s) | 平方误差占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

“—”表示方向无意义，未参与方向指标。无有效进水点的速度保持为零，不将0秒解释成实际到达或峰值时刻。完整40点见 [points-f4-n0.042.csv](points-f4-n0.042.csv)，含原始／粗格高程、速度分量、方向、模型入水时刻、最后50秒速度波动和误差贡献。其他配置也有同结构CSV，另有所有配置的逐点差值表 [paired-point-changes.csv](paired-point-changes.csv)。

![误差位置与40点排序](point-errors.png)

按南向北的测点区域分组（地理分组固定，不按误差选区）：

| 区域 | 点数 | 最大速度RMSE | 偏差 | 平方误差占比 |
|---|---:|---:|---:|---:|
{chr(10).join(regions)}

## 3. 三类不同的流速问题

**持续过快。** 14-F实测0.8 m/s，模型最大及后期均速都约2.72 m/s；14-H、15-H也在后期保留明显高估。这些点不能用“只是在涨水时出现一次尖峰”解释。14-F的采样格高程只比原始测点高约1.3厘米，说明该处的直接高程抬升解释不了整个速度误差；上游分流、附近障碍开口、动量和数值处理仍需分离验证。

**起涨阶段的局部峰值过高。** 17-E最大约1.54 m/s、后期约0.72 m/s，实测最大0.6 m/s；22-D最大约1.52 m/s、后期约0.66 m/s，实测0.6 m/s。主配置有{m['transient_excess_over_05']}点的最大速度超过后期均速0.5 m/s以上。但论文同样对全时段最大值评分，不能因后期更接近就把起涨峰值删除。

**有实验流速的位置没有形成有效水流。** 主配置漏淹非零速度点为{', '.join(m['measured_flow_model_dry'])}；方向误差绝对值>45°的点为{', '.join(m['direction_over_45deg'])}。19-J的实测为1.6 m/s，模型始终未达到1厘米水深。该点原始地面已经较高，合并只额外抬高约2厘米，所以不能仅凭干点认定是该格的最大值合并直接封住了它；也可能是邻近通道的连通与分流失真。必须检查其周围流路，而不是自动移动测点。

20-F则不同：它的原始测点地面19.291米、粗格高程21.978米，被抬高2.687米，最后干燥，但2米邻域仍有湿格。均值合并对照中该点恢复进水、最大流速约1.211 m/s，高于实测0.4 m/s。几何表示对该点有直接影响，但恢复进水后仍有速度误差。19-J在均值组仍不进水，不能用同一解释概括两者。

![两类漏淹点的局部几何与水深](dry-point-geometry.png)

最后50秒有{m['late_speed_range_over_01']}点速度极差超过0.1 m/s，40点中最大的极差为{m['largest_late_speed_range_m_s']:.6f} m/s；这些测点后期已基本稳定，持续速度误差无法解释为“还没跑到稳定”。这只核对了40个位置，不代替全域收敛。逐时图中的实测虚线只是一个最大值水平，不代表实测时间曲线。

![持续偏差、瞬态与漏淹](speed-histories.png)

**方向错误不能被速度大小接近掩盖。** 例如15-J的速度大小只差约0.085 m/s，最大速度矢量的方向却相差77.8°。以下角度均在原数据提供的旋转局部x/y坐标系中，以+x为0°、逆时针增加；不是地理正北方位角。无法从现有最大值记录判断方向误差是否在整个时段持续。

| 点位 | 实测速度 | 模型速度 | 实测方向(°) | 模型方向(°) | 有符号方向误差(°) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(direction_table)}

## 4. 已确认的地形表示问题

四个水位点的位置来自同一数据包，原始0.175米DEM的采样高度与0.7米粗格逐格对应；没有人为平移或寻找更低的邻格。结果是前三点被同一粗格内的高障碍抬高：

| 水位点 | 原始测点地面(m) | 粗格高程(m) | 抬升(m) | 800秒水深(m) |
|---|---:|---:|---:|---:|
{chr(10).join(st)}

这证明前三点的干燥首先受粗格表示破坏影响，不能将“干地的地面高程”作为水面与论文曲线直接相减。它也不证明全部坐标和原始观测已经充分核验。原包没有精确实测水位数值，本报告仍不算水位RMSE。

0.7米网格宽度是0.35米侧墩的2倍、0.52米角墩的1.35倍；1.4米则为4倍和2.69倍。相邻墩和窄缝很容易在合并后变粗或闭合。对原始125≤y≤225米的桥墩区域逐原始像素回投核算：0.7米最高值合并将 **{geo['pier_region_area_raised_over_05_m2']:.1f} m²（{geo['pier_region_fraction_raised_over_05_pct']:.2f}%）** 的地面抬高超过0.5米。这是地形变化量，不是实验淹没面积误差，也不能当作全部新增建筑面积。

![三个水位点原始和粗格地形](stage-geometry.png)

## 5. 单因素对照与可以得出的结论

原三组把网格宽度、糙率分别改变；新组保持1.4米、n=0.042、同一来水和求解器，仅由最高高程改成块内平均高程。新增算例计算预算240秒，实际时间、完整性、输入与求解器哈希在 configuration-comparison.json 中记录。均值可能降低／淹没细桥墩，**用于检验合并规则的敏感性，不作为正式修复**。完整结果汇总见上表及summary.json。

均值组完整跑完800秒，实际主计算{runs[-1]['result']['wall_s']:.1f}秒。与同宽最高值组相比，RMSE约降低4.0%，MAE却从0.559升到0.612 m/s；40点中{control_better}点绝对速度误差减小、{40-control_better}点增大或持平。14-F从3.519降到2.652 m/s，仍明显高于实测0.8；20-F从干燥变为进水，但19-J反而从0.346 m/s变为干燥。说明合并方式会改变分流和局部进水，整体指标小幅下降不代表问题已解决。P1–P3在均值组也仍干燥，均值不能恢复贴近墙边的真实地面。

同宽n从0.042降到0.02使最大速度RMSE从0.764升到1.335 m/s，说明减小阻力会加重本轮总体高估；不能据此推出所有点都该增加糙率。0.7米相对1.4米的RMSE只小降、MAE却上升，而且漏淹点身份改变，尚未达到网格收敛。

![四组进水与干点位置](wetting-comparison.png)

## 6. 下一步优先级与可区分的假设

1. **先解决细障碍与地面不能共用一个粗格最高值的问题。** 优先保留可通水地面，再用已有独立阻水边或子网格开口表示墙和桥墩；入口、通道有效宽度和测点落格应自动检查。不能简单改成均值来抹掉障碍。
2. **做原生0.175米与0.35米的几何体检，然后做有预算的精度对照。** 原生完整域约153万格，不能沿用当前95,700格的耗时预期。若只截取桥墩区，上游水位／动量需由独立核验的全域边界提供，不能拿现有错误边界截图作为证明。短窗口只能验证局部数值行为，不可冒称完成800秒最大速度验证。
3. **在几何明确后检查持续过快的点和方向错误。** 保持地形、n、边界不变，对照Rusanov通量与更低扩散的实现、局部重构、静水保持和窄通道测试。当前没有独立试验能定量分摊通量算法、湍流缺失、入流动量闭合的各自贡献，不直接判定需要更换算法。
4. **最后再考虑率定空间糙率。** 用一部分独立水位／流速资料率定，保留另一部分验证；本实验路基材料不应套用自然群系分类来填平所有误差。补齐原始水位数值和测量不确定性后再给通过阈值。

## 7. 证据与复现

`benchmarks/analyze_merewether.py`只读取已保存结果和原始DEM。用原始像素重建粗格并严格核对数组、从逐时记录重新提取40点最大值、检查速度口径与方向样本、核验时间窗口与水量、固定点身份与坐标；全部检查见validation.json。源文件哈希和各组场景差异见configuration-comparison.json。

输出summary.json、stage-geometry.csv、各配置逐点CSV、paired-point-changes.csv与五张图。evidence-provenance.json记录分析脚本、原始DEM、观测CSV、各组result/series/final的SHA-256；新增对照的紧凑原始结果及配置保存在control-mean/。2米邻域速度范围只作网格落点敏感性诊断，不改变主评分、不选择最接近观测的邻格。未对图12进行精确数字化，未把文献L1-norm直接与本项目速度MAE相比。
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=LAB/'.runtime/merewether/results')
    p.add_argument('--extra',type=Path,default=LAB/'.runtime/merewether/diagnostic-runs/f8-n0.042-mean')
    p.add_argument('--output',type=Path,default=LAB.parent/'docs/experiments/2026-10-20/merewether/diagnostics')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=LAB/'.runtime/merewether/source/experiment'
    native,valid,header=raster_import.read_ascii(source/'merewether-0p175m.dem')
    assert valid.all() and header['cellsize_m']==.175
    runs=[load_run(args.results/name) for name in ['f4-n0.042','f8-n0.042','f8-n0.02']]
    runs.append(load_run(args.extra))
    diag={};checks={};configs={}
    for run in runs:
        label=run['label'];r=run['result'];diag[label]=diagnose(run,native)
        csv_write(out/f'points-{label}.csv',diag[label]['rows'])
        obs=np.array([p['speed_m_s'] for p in r['points']]);maximum=run['speeds'].max(axis=0)
        checks[label+'_max_from_series']=bool(np.allclose(maximum,[p['model_max_speed_m_s'] for p in r['points']],rtol=1e-12,atol=1e-12))
        checks[label+'_rmse']=bool(np.isclose(rms(maximum-obs),r['metrics']['speed_rmse_m_s']))
        checks[label+'_sampling_every_half_second']=bool(np.allclose(np.diff(run['times']),.5))
        checks[label+'_complete_800s']=bool(r['complete'] and run['times'][-1]==800)
        checks[label+'_mass']=bool(abs(r['balance']['residual_m3'])<1e-6 and r['balance']['numerical_adjustment_m3']==0)
        checks[label+'_forty_fixed_points']=bool(len(r['points'])==40 and all(all(p[k]==q[k] for k in ['id','x_m','y_m','speed_m_s','angle_xy_deg']) for p,q in zip(r['points'],runs[0]['result']['points'])))
        checks[label+'_geometry_exact']=True  # diagnose asserts byte-equivalent float arrays.
        cfg=run['config'];configs[label]=dict(terrain_id=cfg['terrain_id'],flood=cfg['scenario']['flood'],
            external_inflows=cfg['scenario']['external_inflows'],code_sha256=cfg['code_sha256'],
            sampling=cfg['sampling'],source_files=cfg['source']['files'],wall_s=r['wall_s'],complete=r['complete'])
    if len(runs)==4:
        a,b=runs[1]['config'],runs[3]['config']
        checks['aggregation_control_same_solver']=all(a['code_sha256'][k]==b['code_sha256'][k] for k in ['flood_solver.py','flood_acceleration.py'])
        checks['aggregation_control_same_forcing']=a['scenario']['external_inflows']==b['scenario']['external_inflows']
        checks['aggregation_control_same_flood_parameters']=a['scenario']['flood']==b['scenario']['flood']
        checks['aggregation_control_same_source']=a['source']['files']==b['source']['files']
    stages=stage_audit(runs,native,source);csv_write(out/'stage-geometry.csv',stages)
    paired=[]
    for i,base in enumerate(diag[runs[0]['label']]['rows']):
        row=dict(id=base['id'],x_m=base['x_m'],y_m=base['y_m'],measured_max_speed_m_s=base['speed_m_s'])
        for run in runs:
            pp=diag[run['label']]['rows'][i]
            for field in ['model_max_speed_m_s','speed_error_m_s','direction_error_deg','late_mean_speed_m_s','max_depth_m']:
                row[run['label']+'__'+field]=pp[field]
        paired.append(row)
    csv_write(out/'paired-point-changes.csv',paired)
    write(out/'summary.json',{label:{k:v for k,v in d.items() if k!='rows'} for label,d in diag.items()})
    write(out/'validation.json',checks);write(out/'configuration-comparison.json',configs)
    if not all(checks.values()):raise AssertionError('Diagnosis audit failed: '+str(checks))
    def fingerprint(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    write(out/'evidence-provenance.json',dict(analysis_sha256=fingerprint(Path(__file__)),
        source_dem_sha256=fingerprint(source/'merewether-0p175m.dem'),
        source_observations_sha256=fingerprint(source/'Location_of_velocity_sampling_points.csv'),
        inputs={folder.name:{name:fingerprint(folder/name) for name in ['result.json','series.json','final.npz']}
                for folder in [*(args.results/name for name in ['f4-n0.042','f8-n0.042','f8-n0.02']),args.extra]}))
    target=out/'control-mean';target.mkdir(exist_ok=True)
    for name in ['result.json','config.json','points.csv']:shutil.copyfile(args.extra/name,target/name)
    render(out,runs,diag,native,stages);report(out,runs,diag,stages,checks)
    print(json.dumps(dict(checks=len(checks),passed=all(checks.values()),summary={k:v['metrics'] for k,v in diag.items()}),ensure_ascii=False))


if __name__=='__main__':
    main()
