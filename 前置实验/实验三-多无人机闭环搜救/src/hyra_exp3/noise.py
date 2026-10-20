"""Observation noise fixed by world, sampling tick, platform and public cell.

Changing a route or the number/order of other observations cannot change the
noise of a shared observation. This is required for intervention comparisons.
"""
import hashlib
import struct
import numpy as np


def world_noise(seed,tick,drone,flat_cells,std_m=.02):
    if min(int(tick),int(drone))<0 or std_m<0:
        raise ValueError('Invalid noise key')
    cells=np.asarray(flat_cells,int).reshape(-1)
    if np.any(cells<0):
        raise ValueError('Negative cell key')
    digests=[hashlib.sha256(b'hyra3-observation-v1'+struct.pack('<QQQQ',int(seed)&((1<<64)-1),
        int(tick),int(drone),int(cell))).digest() for cell in cells]
    scale=float(2**64)
    u1=np.asarray([(int.from_bytes(d[:8],'little')+.5)/scale for d in digests])
    u2=np.asarray([(int.from_bytes(d[8:16],'little')+.5)/scale for d in digests])
    return std_m*np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
