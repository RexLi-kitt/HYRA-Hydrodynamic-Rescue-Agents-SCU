"""Bounded 1/2/4 m flood observation on the current native 1 m terrain.

Paired constant-inflow controls distinguish pulse response from initial relaxation.
No terrain, defaults, current-map pointer or saved interactive process is modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
import server
import map_sessions
import river_network
from grid_coarsening import blocks


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-budget', type=float, default=240)
    parser.add_argument('--max-duration', type=float, default=120)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    expired = lambda: time.perf_counter() - started > args.wall_budget
    os.environ['HYRA_FLOOD_BACKEND'] = 'numba'
    terrain = server.describe(json.loads((LAB/'current.json').read_text())['directory'])
    if terrain['step'] != 1 or max(terrain['shape']) > 256:
        raise ValueError('This bounded observation requires a native 1 m map no larger than 256 x 256')
    ports = terrain['river_inlets']['crossings']
    inlet = max((p for p in ports if p['role'] == 'inlet'), key=lambda p: p['width_blocks'])
    mask, bed, water = river_network.read_native(LAB, terrain)
    nz, nx = bed.shape
    # Deep channel points at fixed physical locations, selected before any runs.
    probes = []
    for name, fraction in [('upstream', .1), ('middle', .5), ('downstream', .9)]:
        z = min(nz-1, round((nz-1)*fraction))
        candidates = np.flatnonzero(mask[z] & np.isfinite(water[z]))
        if not len(candidates):
            raise ValueError(f'No river at probe row {z}; define probes for this geometry explicitly')
        x = int(candidates[np.argmax((water-bed)[z, candidates])])
        probes.append(dict(name=name, x_m=x+.5, z_m=z+.5))

    options = dict(metres_per_block=1, boundary='open', initial_water=True, baseflow=False,
                   warmup_s=0, rain_mm_h=0, infiltration_mm_h=0, inflow_m3_s=0,
                   flux_method='rusanov', spatial_order=2, time_integrator='rk2',
                   low_froude=1, cfl=.4, manning=.035, roughness_mode='uniform', max_timestep=.5, max_steps=30000)
    def world(factor, hydrograph):
        return map_sessions.create(LAB, terrain, dict(mode='current', terrain_id=terrain['id'],
            flood=dict(options, grid_factor=factor), external_inflows=[dict(id=inlet['id'], kind='river',
                enabled=True, inlet_direction_mode='auto', inlet_velocity_m_s=0, hydrograph=hydrograph)]))

    # Compile only on a tiny grid; count its wall time against the same budget.
    from flood_solver import FloodSolver
    tiny = FloodSolver(np.zeros((4,4)), np.ones((4,4))*.1, 1, boundary='closed')
    tiny.advance(.01, cancelled=expired)
    print(json.dumps(dict(stage='compiled', wall_s=time.perf_counter()-started)), flush=True)
    pilot = world(1, [[0,10]])
    tick = time.perf_counter(); pilot.advance_to(1, cancelled=expired)
    rate = time.perf_counter()-tick
    # Three warmups + paired branches: fine grid dominates. Reserve >30% for
    # slowing flow and report generation; duration remains identical for all grids.
    remaining = args.wall_budget-(time.perf_counter()-started)
    duration = min(args.max_duration, math.floor(max(0,(remaining-25)/(max(rate,.01)*3.2)-20)/2)*2)
    if duration < 30:
        dump(output/'budget_stop.json',dict(pilot_wall_s=rate,remaining_s=remaining,proposed_duration_s=duration))
        raise RuntimeError('Budget does not support at least 30 s of paired comparison; stopped after pilot')
    warmup = 20.
    # Same time-normalized piecewise-constant pulse at every grid width.
    pulse = [[0,10], [duration/12,100], [duration/6,200],
             [duration/3,200], [duration/2,50], [2*duration/3,10]]
    times = np.arange(0, duration+1e-8, 2.)
    config = dict(terrain_id=terrain['id'], native_shape=terrain['shape'], origin=terrain['origin'],
        source_kind=terrain['full'] and 'full_surface' or 'continuous_preview',
        factors=[1,2,4], options=options, inlet_id=inlet['id'], inlet_width_m=inlet['width_blocks'],
        pulse_hydrograph=pulse, control_hydrograph=[[0,10]], warmup_s=warmup, duration_s=duration,
        sample_interval_s=2, wall_budget_s=args.wall_budget, pilot_1_sim_s_wall_s=rate,
        arrival_definition='first of two consecutive samples with flood-control stage >= 0.05 m; null = not detected within window',
        inundation_definition='depth >= 0.05 m; flood wet / control dry; all areas compared on common 4 m cell averages',
        reference='1 m numerical comparison only; not measured ground truth', probes=probes,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [LAB/'flood_solver.py',LAB/'flood_acceleration.py',Path(__file__)]},
        python=sys.version,numpy=np.__version__)
    dump(output/'config.json',config)
    print(json.dumps(dict(stage='pilot',wall_per_sim_s=rate,duration_s=duration,remaining_wall_s=remaining)),flush=True)
    all_runs = {}; summaries = []
    try:
        for factor in [4,2,1]:
            tick = time.perf_counter()
            warmed = world(factor, [[0,10]])
            warmed.advance_to(warmup, cancelled=expired)
            branches = [world(factor, [[0,10]]), world(factor, pulse)]
            for branch in branches:
                for field in ['h','mx','mz']:
                    getattr(branch.solver,field)[:] = getattr(warmed.solver,field)
                branch.solver.start_scenario()
            del warmed
            depths = [[],[]]; series = []; last_print=time.perf_counter()
            for at in times:
                for index, branch in enumerate(branches):
                    branch.advance_to(float(at), cancelled=expired)
                    depths[index].append(branch.solver.h.astype(np.float32).copy())
                h0,h1=[b.solver.h for b in branches]
                common0,common1=[blocks(h,4//factor) for h in (h0,h1)]
                row=dict(time_s=float(at), additional_wet_area_m2=int(((common1>=.05)&(common0<.05)).sum())*16,
                         probes={})
                for p in probes:
                    z,x=int(p['z_m']//factor),int(p['x_m']//factor)
                    row['probes'][p['name']]=dict(control_stage_m=float(branches[0].base.z[z,x]+h0[z,x]),
                        flood_stage_m=float(branches[1].base.z[z,x]+h1[z,x]),
                        pulse_rise_m=float(h1[z,x]-h0[z,x]))
                series.append(row)
                if time.perf_counter()-last_print>5:
                    print(json.dumps(dict(stage='simulate',factor=factor,time_s=float(at),wall_s=time.perf_counter()-started)),flush=True)
                    last_print=time.perf_counter()
            summary=dict(factor=factor,cells=branches[0].solver.h.size,wall_s=time.perf_counter()-tick,
                         control=branches[0].solver.summary(),flood=branches[1].solver.summary(),
                         grid_check=branches[0].meta['river_network_grid_check'],probes={})
            for p in probes:
                values=np.array([row['probes'][p['name']]['pulse_rise_m'] for row in series])
                hits=np.flatnonzero((values[:-1]>=.05)&(values[1:]>=.05))
                summary['probes'][p['name']]=dict(peak_pulse_rise_m=float(values.max()),
                    arrival_s=float(times[hits[0]]) if len(hits) else None)
            summary['max_additional_wet_area_m2']=max(row['additional_wet_area_m2'] for row in series)
            dump(output/f'grid-{factor}m.json',dict(summary=summary,series=series))
            np.savez_compressed(output/f'grid-{factor}m.npz',time_s=times,bed=branches[0].base.z,
                control=np.stack(depths[0]),flood=np.stack(depths[1]))
            all_runs[factor]=dict(depths=depths,series=series,bed=branches[0].base.z)
            summaries.append(summary)
            print(json.dumps(dict(stage='complete',factor=factor,wall_s=summary['wall_s'],steps=summary['flood']['steps'])),flush=True)
    except InterruptedError:
        dump(output/'budget_stop.json',dict(completed_factors=list(all_runs),wall_s=time.perf_counter()-started))
        raise RuntimeError('Wall budget reached; partial grid results retained, no full comparison claim')

    reference=blocks(all_runs[1]['depths'][1][-1],4)
    reference_wet=reference>=.05
    for summary in summaries:
        factor=summary['factor']; estimate=blocks(all_runs[factor]['depths'][1][-1],4//factor)
        wet=estimate>=.05;union=(wet|reference_wet).sum()
        summary['final_depth_mae_vs_1m_common4_m']=float(np.abs(estimate-reference).mean())
        summary['final_depth_mae_wet_union_m']=float(np.abs(estimate-reference)[wet|reference_wet].mean())
        summary['final_wet_iou_vs_1m_common4']=float((wet&reference_wet).sum()/union) if union else 1.
    dump(output/'summary.json',dict(config=config,runs=sorted(summaries,key=lambda r:r['factor']),
                                   experiment_wall_s=time.perf_counter()-started))
    render(output,config,all_runs,summaries)
    print(json.dumps(dict(stage='done',output=str(output),wall_s=time.perf_counter()-started)),flush=True)


def render(output,config,runs,summaries):
    # Chart contract: paired pulse response over common 2 s timestamps; all
    # factors have identical physical extent. Static scientific evidence, PNG.
    # No ranking until results exist; categorical lines also differ by dash.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={1:('#24659c','-'),2:('#cc7a18','--'),4:('#4e7b57',':')}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
    fig.suptitle(f'Grid-width flood comparison | {config["duration_s"]:g} s pulse observation',fontsize=15)
    for ax,name in zip(axes.flat,['upstream','middle','downstream','area']):
        for factor in [1,2,4]:
            rows=runs[factor]['series'];t=[r['time_s'] for r in rows]
            y=[r['additional_wet_area_m2'] if name=='area' else r['probes'][name]['pulse_rise_m'] for r in rows]
            color,dash=styles[factor];ax.plot(t,y,color=color,ls=dash,lw=2,label=f'{factor} m')
        ax.set_title('Additional wet area (common 4 m cells)' if name=='area' else f'{name.capitalize()} channel probe')
        ax.set_xlabel('Simulation time (s)');ax.set_ylabel('Area (m²)' if name=='area' else 'Flood minus control stage (m)')
        ax.grid(alpha=.17);ax.legend(frameon=False)
        if name!='area':ax.axhline(.05,color='#666666',lw=.8,ls='-.')
    fig.savefig(output/'response.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
    vmax=max(float(np.max(runs[f]['depths'][1][-1])) for f in runs)
    for ax,factor in zip(axes,[1,2,4]):
        h=runs[factor]['depths'][1][-1]; shown=np.ma.masked_less(h,.05)
        image=ax.imshow(shown,extent=(0,config['native_shape'][1],config['native_shape'][0],0),
                        cmap='Blues',vmin=0,vmax=vmax,interpolation='nearest')
        for p in config['probes']:ax.plot(p['x_m'],p['z_m'],'o',ms=4,color='#b15f16')
        ax.set_title(f'{factor} m grid');ax.set_xlabel('Local X (m)');ax.set_ylabel('Local Z (m)')
    fig.colorbar(image,ax=axes,label='Final flood depth (m)',shrink=.8)
    fig.suptitle(f'Same terrain and inflow | t = {config["duration_s"]:g} s | dry < 0.05 m',fontsize=14)
    fig.savefig(output/'depth-maps.png',dpi=160);plt.close(fig)
    lines=['# 短时洪水网格对照','',
        f'当前原始地图 {config["native_shape"]}，每格1米。正式观察 {config["duration_s"]:g} 秒，每2秒记录；每档先预热20秒，再从相同预热状态分别运行10 m³/s对照和洪水脉冲。',
        '','峰值来水200 m³/s是合成压力输入，不对应某一重现期。无降雨、入渗、额外地物或人员；边界为开边界。',
        '','1米仅为数值比较基准，不是实测真值。初始地图为连续地形预览，20秒预热不代表达到稳态；配对对照用于分离共同的初始调整。',
        '','| 格宽 | 两分支及预热耗时(s) | 中游脉冲最大抬升(m) | 下游到达(s) | 最大新增湿面积(m²) | 末时刻湿区深度MAE(m) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in sorted(summaries,key=lambda r:r['factor']):
        arrival=r['probes']['downstream']['arrival_s']
        lines.append(f'| {r["factor"]} m | {r["wall_s"]:.2f} | {r["probes"]["middle"]["peak_pulse_rise_m"]:.4f} | {arrival if arrival is not None else "窗口内未达阈值"} | {r["max_additional_wet_area_m2"]} | {r["final_depth_mae_wet_union_m"]:.4f} |')
    lines+=['','到达：洪水减对照水位连续两帧达到5厘米；采样间隔2秒。新增湿面积：同一时刻洪水有水而对照干燥，统一按4米格平均水深>=5厘米计算。MAE同样在共同4米格上计算。',
            '','小地图和短时单脉冲结果不能证明任意地图或长时间过程的精度。见 config.json、summary.json、各档时间序列及NPZ原始场。',
            '','![脉冲响应](response.png)','','![最终水深](depth-maps.png)']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
