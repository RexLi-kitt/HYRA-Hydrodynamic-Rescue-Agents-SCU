"""The exact same full raw observation tensor for H/E/P and both ablations."""
import numpy as np
from .common_data import causal_frame

HISTORY_CHANNELS = 46  # 6 sufficient statistics + 20 raw depths + 20 raw masks.


def full_raw_frame(panel, observations, now, delay_s=0.):
    base = np.floor(now / 5.) * 5.
    fraction = now - base
    summary = causal_frame(panel, base, max(0., delay_s - fraction))
    summary[:, :, 2] += fraction / 60.
    expired = summary[:, :, 2] > 1. + 1e-7
    summary[:, :, 0] = np.where(expired, 0., summary[:, :, 0])
    summary[:, :, 3:6] = np.where(expired[:, :, None], 0., summary[:, :, 3:6])
    n = len(summary)
    raw = np.zeros((n, 13, 20), np.float32)
    mask = np.zeros_like(raw)
    sample = observations["sample_time_s"]
    # Collection records have zero delay; augment transport without changing
    # any sample or label. Live records use their actual arrival timestamps.
    arrival = observations["arrival_time_s"] + delay_s
    legal = (sample >= now - 60. - 1e-9) & (sample <= now + 1e-9) & (arrival <= now + 1e-9)
    chosen = np.flatnonzero(legal)
    slots = 12 - np.rint((base - sample[chosen]) / 5.).astype(int)
    if np.any((slots < 0) | (slots > 12)):
        raise ValueError("Observation lies outside shared history")
    side = int(round(n ** .5))
    cells = observations["row"][chosen] * side + observations["col"][chosen]
    drones = observations["drone"][chosen]
    if np.any((drones < 0) | (drones >= 20)):
        raise ValueError("Drone ID exceeds shared input capacity")
    raw[cells, slots, drones] = observations["depth_m"][chosen]
    mask[cells, slots, drones] = 1.
    return np.concatenate((summary, raw, mask), axis=-1)
