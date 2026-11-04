"""Explicit-window summaries for one gauge; no time interpolation or imputation."""
import math
from collections import Counter


def summarize_gauge_window(samples, *, start_s, end_s):
    if not all(math.isfinite(t) for t in (start_s,end_s)) or start_s>end_s:
        raise ValueError('Invalid observation window')
    rows=list(samples)
    previous=-math.inf;identity=None
    for row in rows:
        t=row['time_s']
        if not math.isfinite(t) or t<=previous:raise ValueError('Sample times must strictly increase')
        previous=t
        key=(row['x_m'],row['z_m'],tuple(row['origin_center_m']),row['method'])
        if identity is None:identity=key
        if key!=identity:raise ValueError('Cannot combine different gauges or sampling methods')
        if row['status'] not in ('wet','dry','solid','outside'):raise ValueError('Unknown gauge status')
        speed=row['speed_m_s']
        if row['status']=='wet':
            if speed is None or not math.isfinite(speed) or speed<0:raise ValueError('Invalid wet speed')
        elif speed is not None:raise ValueError('Missing observation must not carry a speed')
    selected=[r for r in rows if start_s<=r['time_s']<=end_s]
    counts=Counter(r['status'] for r in selected)
    wet=[r for r in selected if r['status']=='wet']
    peak=max(wet,key=lambda r:r['speed_m_s']) if wet else None
    missing=len(selected)-len(wet)
    return dict(window_s=[start_s,end_s],sample_count=len(selected),wet_sample_count=len(wet),
        missing_sample_count=missing,status_counts=dict(counts),
        endpoint_samples_present=bool(selected and selected[0]['time_s']==start_s and selected[-1]['time_s']==end_s),
        max_sample_gap_s=max((b['time_s']-a['time_s'] for a,b in zip(selected,selected[1:])),default=None),
        sampled_peak_speed_m_s=peak['speed_m_s'] if peak else None,
        sampled_peak_time_s=peak['time_s'] if peak else None,
        interpretation='maximum of available wet samples in the closed window; no continuous-time or missing-value guarantee')
