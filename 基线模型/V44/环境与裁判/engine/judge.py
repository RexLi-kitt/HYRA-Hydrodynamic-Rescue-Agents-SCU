"""Horizontal separation at the six recorded instants of a five-second decision."""
import numpy as np

import os
DELIVERY_RADIUS_M = float(os.environ.get("HYRA_STUDY_RADIUS","3"))
if DELIVERY_RADIUS_M not in (1.,2.,3.):raise ValueError("Unregistered delivery radius")
INNER_SQ = (DELIVERY_RADIUS_M / 2.0) ** 2
ROUND_OFF_M = 1e-9

def within_radius(aircraft, recipient):
    difference = aircraft - recipient
    return bool(np.all(np.sum(difference * difference, axis=-1) <= (DELIVERY_RADIUS_M + ROUND_OFF_M) ** 2))
