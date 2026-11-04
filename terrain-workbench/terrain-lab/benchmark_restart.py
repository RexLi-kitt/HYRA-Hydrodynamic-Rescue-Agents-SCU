"""Restore audit metadata after native benchmark fields/configuration are checked.

This is not a complete checkpoint loader: callers restore conserved fields,
maximum depth and identical geometry/forcing before using this function.
"""
from copy import deepcopy
import math


def restore_audit_metadata(solver, frame):
    required = ('time_s', 'steps', 'balance', 'diagnostics',
                'external_inflow_by_id_m3', 'peak_speed_m_s', 'peak_speed_time_s')
    if any(key not in frame for key in required):
        raise ValueError('Checkpoint lacks cumulative audit metadata; rerun instead of resetting it')
    time = float(frame['time_s'])
    peak = float(frame['peak_speed_m_s'])
    peak_time = float(frame['peak_speed_time_s'])
    steps = frame['steps']
    volumes = deepcopy(frame['external_inflow_by_id_m3'])
    balance = {key: frame['balance'][key] for key in solver.balance}
    if (not all(math.isfinite(v) for v in (time, peak, peak_time))
            or time < 0 or peak < 0 or not 0 <= peak_time <= time
            or not isinstance(steps, int) or isinstance(steps, bool) or steps < 0
            or not isinstance(volumes, dict)
            or not all(math.isfinite(v) and v >= 0 for v in volumes.values())
            or not all(math.isfinite(v) for v in balance.values())
            or not set(solver.external_volume_by_id).issubset(volumes)):
        raise ValueError('Invalid checkpoint audit metadata')
    solver.time = time
    solver.steps = steps
    solver.balance = balance
    solver.diagnostics = deepcopy(frame['diagnostics'])
    solver.external_volume_by_id = volumes
    solver.peak_speed_m_s = peak
    solver.peak_speed_time_s = peak_time
