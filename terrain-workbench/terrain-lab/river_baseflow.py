"""Synthetic daily river inflow: upstream sources and an explicit spin-up.

No water-level restoration is used. Discharge is a scenario parameter, not
an estimate of real catchment runoff. Source detection uses native river
connectivity and water levels before the hydraulic grid is coarsened.
"""
from collections import deque
import numpy as np


def plan(river, water, bed, factor, total_discharge):
    river = np.asarray(river, dtype=bool)
    nz, nx = river.shape
    seen = np.zeros_like(river)
    groups = []
    for z, x in zip(*np.nonzero(river)):
        if seen[z, x]: continue
        queue = deque([(int(z), int(x))]); seen[z, x] = True; cells = []
        while queue:
            cz, cx = queue.popleft(); cells.append((cz, cx))
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    zz, xx = cz+dz, cx+dx
                    if 0 <= zz < nz and 0 <= xx < nx and river[zz, xx] and not seen[zz, xx]:
                        seen[zz, xx] = True; queue.append((zz, xx))
        zz, xx = np.asarray(cells).T
        wet = np.isfinite(water[zz, xx]) & (water[zz, xx] > bed[zz, xx]+.01)
        if not wet.any(): continue
        zz, xx = zz[wet], xx[wet]
        # The highest native river water surface supplies this connected reach.
        # At equal stages prefer a shallow head instead of a deep downstream pool.
        order = np.lexsort((xx, zz, -(bed[zz, xx]), -(water[zz, xx])))
        hz, hx = int(zz[order[0]]), int(xx[order[0]])
        near = (np.abs(zz-hz) <= factor) & (np.abs(xx-hx) <= factor)
        chosen = list(zip(zz[near], xx[near]))
        coarse = sorted({(int(z)//factor, int(x)//factor) for z, x in chosen})
        groups.append(dict(native_head=[hz, hx], cells=coarse, native_cells=len(cells)))
    sources = {}
    if groups and total_discharge > 0:
        # Share a specified total across reaches by their native river area.
        weight = sum(g['native_cells'] for g in groups)
        for group in groups:
            discharge = total_discharge*group['native_cells']/weight
            group['discharge_m3_s'] = discharge
            for cell in group['cells']:
                sources[cell] = sources.get(cell, 0.0)+discharge/len(group['cells'])
    return groups, [dict(z=z, x=x, discharge_m3_s=q) for (z,x), q in sorted(sources.items())]


def spin_up(solver, limit, infiltration, update, cancelled):
    """Report finite warm-up honestly; convergence is never assumed from elapsed time."""
    elapsed = 0.0; stable_windows = 0; checks = []
    depth=lambda:solver.leaf['h'] if hasattr(solver,'mesh') else solver.h
    velocity=solver.leaf_velocity if hasattr(solver,'mesh') else solver.velocity
    update(state='running',progress=0,message='正在准备基流预热，首次计算可能稍慢…')
    def progress(s):
        update(state='running',progress=0,warmup_time_s=float(s.time),steps=s.steps,
            dt_s=s.last_timestep.get('dt_s'),
            message=f'正在基流预热：{s.time:.2f} / {limit:.0f} 秒 · 已算 {s.steps} 步（洪水尚未开始）')
    while elapsed < limit-1e-9:
        before, bu, bv = depth().copy(), *velocity()
        end = min(limit, elapsed+60)
        solver.advance(end, infiltration=infiltration, cancelled=cancelled,progress=progress)
        u, v = velocity()
        change = float(np.max(np.abs(depth()-before)))
        velocity_change = float(np.max(np.hypot(u-bu, v-bv)))
        stable_windows = stable_windows+1 if change <= .01 and velocity_change <= .01 else 0
        checks.append(dict(time_s=end, max_depth_change_m=change,
                           max_velocity_change_m_s=velocity_change))
        elapsed = end
        update(state='running', progress=0,
               message=f'正在基流预热：{elapsed:.0f} / {limit:.0f} 秒（洪水尚未开始）')
        if elapsed >= 120 and stable_windows >= 2: break
    return dict(duration_s=elapsed, converged=bool(stable_windows >= 2 and elapsed >= 120),
                depth_tolerance_m=.01, velocity_tolerance_m_s=.01, window_s=60,
                checks=checks, balance=solver.summary()['balance'])
