"""Compare saved 1 m preview baseline with full-surface uniform/classified n.

Keep original probes, boundary faces, hydrograph, numerics and physical extent.
Two new paired-control runs isolate surface generation from roughness changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB))
import server
import map_sessions
from grid_coarsening import blocks


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terrain-id',required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-budget',type=float,default=360)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    base=args.baseline.resolve();cfg=read(base/'config.json')
    preview=server.describe(cfg['terrain_id']);full=server.describe(args.terrain_id)
    assert full['full'] and full['shape']==preview['shape'] and full['origin']==preview['origin'] and full['step']==1
    assert read((LAB/full['id']).parent/'preset.json')==read((LAB/preview['id']).parent/'preset.json')
    checks={}
    for name in ['flood_solver.py','flood_acceleration.py']:
        checks[name]=digest(LAB/name)==cfg['source_sha256'][name]
    if not all(checks.values()):raise RuntimeError('Solver changed since baseline; a fresh matched baseline is required')
    raw_checks={}
    for name in ['height','terrain','riverWaterLevel','riverZone','riverMask','temperature','moisture']:
        oldgrid=read(LAB/preview['id']/'grid.json');newgrid=read(LAB/full['id']/'grid.json')
        raw_checks[name]=digest(LAB/preview['id']/oldgrid['fields'][name]['file'])==digest(LAB/full['id']/newgrid['fields'][name]['file'])
    if not all(raw_checks.values()):raise RuntimeError('Native generator outputs differ; cannot claim same original terrain')
    # Freeze source geometry and orientation even if surface-derived inlet
    # inference differs. Do not change the published full map's own annotations.
    terrain=dict(full,river_inlets=preview['river_inlets'])
    os.environ['HYRA_FLOOD_BACKEND']='numba'
    options=dict(cfg['options'],grid_factor=1)
    def create(mode,curve):
        return map_sessions.create(LAB,terrain,dict(mode='current',terrain_id=terrain['id'],
            flood=dict(options,roughness_mode=mode),external_inflows=[dict(id=cfg['inlet_id'],kind='river',enabled=True,
                inlet_direction_mode='auto',inlet_velocity_m_s=0,hydrograph=curve)]))
    started=time.perf_counter();expired=lambda:time.perf_counter()-started>args.wall_budget
    sample_times=np.arange(0,cfg['duration_s']+1e-9,cfg['sample_interval_s'])
    variants={};metadata={};records={};frames={}
    for key,mode in [('full_uniform','uniform'),('full_classified','landcover')]:
        tick=time.perf_counter();warm=create(mode,cfg['control_hydrograph'])
        warm.advance_to(cfg['warmup_s'],cancelled=expired)
        branches=[create(mode,cfg['control_hydrograph']),create(mode,cfg['pulse_hydrograph'])]
        for branch in branches:
            for field in ['h','mx','mz']:getattr(branch.solver,field)[:]=getattr(warm.solver,field)
            branch.solver.start_scenario()
        variants[key]=branches;records[key]=[];frames[key]=[[],[]]
        metadata[key]=dict(mode=mode,warmup_wall_s=time.perf_counter()-tick,roughness=branches[0].meta['roughness'])
        del warm
        print(json.dumps(dict(stage='warmed',variant=key,elapsed_s=time.perf_counter()-started)),flush=True)
    first=variants['full_uniform'][0];second=variants['full_classified'][0]
    np.testing.assert_array_equal(first.base.z,second.base.z)
    # State at t=0 differs after n-dependent warmup; raw initial surface and all
    # forcing are identical. Each variant has its own matched control branch.
    with np.load(base/'grid-1m.npz',allow_pickle=False) as a:
        old_bed=a['bed'].copy();old_times=a['time_s'].copy()
        baseline_h=a['flood'].copy();baseline_control=a['control'].copy()
    bed_diff=first.base.z-old_bed
    provenance=dict(source_preview=preview['id'],full_surface=full['id'],native_fields_identical=raw_checks,
        solver_unchanged=checks,original_config=cfg,wall_budget_s=args.wall_budget,
        boundary_geometry='original preview boundary faces and inferred momentum direction, frozen across variants',
        source_sha256={p.name:digest(p) for p in [Path(__file__),LAB/'surface_roughness.py',LAB/'map_sessions.py',LAB/'flood_run.py']},
        bed_difference=dict(mae_m=float(np.abs(bed_diff).mean()),max_abs_m=float(np.abs(bed_diff).max()),
            changed_cells=int(np.count_nonzero(bed_diff))),
        comparison='preview uniform -> full uniform isolates surface pipeline; full uniform -> full classified isolates n including its warmup effect')
    write(out/'comparison-config.json',provenance)
    elapsed_by_variant={key:metadata[key]['warmup_wall_s'] for key in variants}
    last_print=time.perf_counter();budget_hit=False
    for at in sample_times:
        try:
            for key,branches in variants.items():
                tick=time.perf_counter()
                for branch in branches:branch.advance_to(float(at),cancelled=expired)
                elapsed_by_variant[key]+=time.perf_counter()-tick
        except InterruptedError:
            budget_hit=True;break
        # Retain only timestamps reached by all four new branches.
        for key,branches in variants.items():
            h0,h1=[branch.solver.h for branch in branches]
            for i,h in enumerate([h0,h1]):frames[key][i].append(h.astype(np.float32).copy())
            c,f=blocks(h0,4),blocks(h1,4)
            row=dict(time_s=float(at),additional_wet_area_m2=int(((f>=.05)&(c<.05)).sum())*16,probes={})
            for p in cfg['probes']:
                z,x=int(p['z_m']),int(p['x_m'])
                row['probes'][p['name']]=dict(control_stage_m=float(branches[0].base.z[z,x]+h0[z,x]),
                    flood_stage_m=float(branches[1].base.z[z,x]+h1[z,x]),pulse_rise_m=float(h1[z,x]-h0[z,x]))
            records[key].append(row)
        if time.perf_counter()-last_print>8:
            print(json.dumps(dict(stage='simulate',time_s=float(at),elapsed_s=time.perf_counter()-started)),flush=True)
            last_print=time.perf_counter()
    count=len(records['full_uniform'])
    if count<2:raise RuntimeError('No comparable window within budget')
    times=sample_times[:count]
    assert np.array_equal(old_times[:count],times)
    runs={'preview_uniform':dict(series=read(base/'grid-1m.json')['series'][:count],
        h=baseline_h[:count],control=baseline_control[:count],bed=old_bed,
        n=np.full(old_bed.shape,.035),wall_s=read(base/'grid-1m.json')['summary']['wall_s'],historical=True)}
    for key,branches in variants.items():
        runs[key]=dict(series=records[key],h=np.stack(frames[key][1]),control=np.stack(frames[key][0]),
            bed=branches[0].base.z,n=branches[0].base.n,wall_s=elapsed_by_variant[key],historical=False)
        # Endpoint budget cancellation may leave solver ahead of retained fields;
        # mark this explicitly and report its actual time in the budget record.
        metadata[key]['branches']=[branch.solver.summary() for branch in branches]
        np.savez_compressed(out/f'{key}.npz',time_s=times,bed=runs[key]['bed'],n=runs[key]['n'],
                            control=runs[key]['control'],flood=runs[key]['h'])
        write(out/f'{key}.json',dict(metadata=metadata[key],series=records[key]))
    summaries={}
    for key,run in runs.items():
        row=dict(wall_s=run['wall_s'],historical=run['historical'],probes={},
            max_additional_wet_area_m2=max(r['additional_wet_area_m2'] for r in run['series']),
            final_wet_area_m2=int((blocks(run['h'][-1],4)>=.05).sum())*16,
            initial_volume_m3=float(run['h'][0].astype(float).sum()),
            minimum_n=float(run['n'].min()),maximum_n=float(run['n'].max()))
        for p in cfg['probes']:
            values=np.array([r['probes'][p['name']]['pulse_rise_m'] for r in run['series']])
            hits=np.flatnonzero((values[:-1]>=.05)&(values[1:]>=.05))
            row['probes'][p['name']]=dict(peak_pulse_rise_m=float(values.max()),
                arrival_s=float(times[hits[0]]) if len(hits) else None)
        summaries[key]=row
    deltas={}
    for reference,estimate in [('preview_uniform','full_uniform'),('full_uniform','full_classified'),('preview_uniform','full_classified')]:
        a,b=[blocks(runs[k]['h'][-1],4) for k in [reference,estimate]]
        wet=(a>=.05)|(b>=.05)
        da,db=[blocks(runs[k]['h'][-1]-runs[k]['control'][-1],4) for k in [reference,estimate]]
        deltas[reference+'__'+estimate]=dict(final_wet_depth_mae_m=float(np.abs(b-a)[wet].mean()),
            final_pulse_depth_mae_m=float(np.abs(db-da)[wet].mean()),
            final_wet_iou=float(((a>=.05)&(b>=.05)).sum()/wet.sum()))
    pulse_reached=np.any(runs['full_classified']['h']-runs['full_classified']['control']>=.05,axis=0)
    n=runs['full_classified']['n']
    rough_effect=dict(pulse_reached_cells=int(pulse_reached.sum()),
        reached_cells_with_nonuniform_n=int(np.count_nonzero(pulse_reached & (np.abs(n-.035)>1e-12))))
    summary=dict(duration_s=float(times[-1]),sample_interval_s=cfg['sample_interval_s'],budget_hit=budget_hit,
        simulation_wall_s=time.perf_counter()-started,runs=summaries,differences=deltas,
        roughness_on_pulse_footprint=rough_effect,provenance=provenance)
    write(out/'summary.json',summary)
    render(out,cfg,runs,summary)
    print(json.dumps(dict(stage='done',duration_s=float(times[-1]),wall_s=time.perf_counter()-started,
                         summary=summaries,differences=deltas),ensure_ascii=True),flush=True)


def render(out,cfg,runs,summary):
    # Static scientific comparison: common physical probe coordinates and times,
    # ordered terrain/roughness interventions, common depth scale and map extent.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'preview_uniform':('#777777',':','Preview / uniform n'),
            'full_uniform':('#24659c','--','Full surface / uniform n'),
            'full_classified':('#cc7a18','-','Full surface / classified n')}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.6),layout='constrained')
    for ax,name in zip(axes.flat,['upstream','middle','downstream','area']):
        for key,run in runs.items():
            x=[r['time_s'] for r in run['series']]
            y=[r['additional_wet_area_m2'] if name=='area' else r['probes'][name]['pulse_rise_m'] for r in run['series']]
            color,dash,label=styles[key];ax.plot(x,y,color=color,ls=dash,label=label,lw=2)
        ax.set_title('Additional wet area: paired control, common 4 m cells' if name=='area' else name.capitalize()+' probe')
        ax.set_xlabel('Simulation time (s)');ax.set_ylabel('Area (m²)' if name=='area' else 'Pulse stage rise (m)')
        ax.grid(alpha=.17);ax.legend(frameon=False,fontsize=8)
        if name!='area':ax.axhline(.05,color='#444444',ls='-.',lw=.8)
    fig.suptitle(f'Full-surface and roughness effects | all 1 m grids | {summary["duration_s"]:g} s',fontsize=15)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
    vmax=max(float(run['h'][-1].max()) for run in runs.values())
    for ax,(key,run) in zip(axes,runs.items()):
        image=ax.imshow(np.ma.masked_less(run['h'][-1],.05),cmap='Blues',vmin=0,vmax=vmax,
                        extent=(0,256,256,0),interpolation='nearest')
        ax.set_title(styles[key][2]);ax.set_xlabel('Local X (m)');ax.set_ylabel('Local Z (m)')
    fig.colorbar(image,ax=axes,label='Final flood depth (m)',shrink=.8)
    fig.suptitle(f'Same seed, crop and boundary | t = {summary["duration_s"]:g} s | dry < 0.05 m',fontsize=14)
    fig.savefig(out/'depth-maps.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    image=ax.imshow(runs['full_classified']['n'],extent=(0,256,256,0),cmap='YlGn',vmin=.015,vmax=.12)
    ax.set_title('Static classified Manning n: complete surface');ax.set_xlabel('Local X (m)');ax.set_ylabel('Local Z (m)')
    fig.colorbar(image,ax=ax,label='Manning n')
    fig.savefig(out/'roughness.png',dpi=160);plt.close(fig)
    lines=['# 完整地表与静态糙率的1米洪水对照','',
        f'同一原始地图：种子3346354700，起点(8512,9536)，256×256米。所有计算格宽1米，20秒预热，正式共同观察{summary["duration_s"]:g}秒，每2秒记录。',
        '','三组为上次预览地形统一n=0.035、完整地表统一n=0.035、完整地表分类n。新两组均包含10 m³/s配对对照和最高200 m³/s脉冲。固定原入口几何/方向、原探针位置、时间过程及数值方法。',
        '','| 组别 | 中游脉冲峰值(m) | 下游到达(s) | 最大新增湿区(m²) | 预热与两分支耗时(s) |',
        '|---|---:|---:|---:|---:|']
    for key,row in summary['runs'].items():
        arrival=row['probes']['downstream']['arrival_s']
        lines.append(f'| {styles[key][2]} | {row["probes"]["middle"]["peak_pulse_rise_m"]:.4f} | {arrival if arrival is not None else "未达到阈值"} | {row["max_additional_wet_area_m2"]} | {row["wall_s"]:.2f} |')
    lines+=['','到达指相对本组对照水位连续两帧抬升5厘米。新增湿区统一按4米格平均水深≥5厘米、且本组对照干燥计算。',
        '','预览→完整地表同时改变河床和初始水体，不能把总差异都归因于糙率。完整地表统一→分类只改变糙率及其预热影响。20秒不保证稳态，1米并非实测真值，本结果不证明分类参数已经率定。',
        '','![响应](comparison.png)','','![水深](depth-maps.png)','','![糙率](roughness.png)']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
