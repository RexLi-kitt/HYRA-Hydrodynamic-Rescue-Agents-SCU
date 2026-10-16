"""Causal, matched-support controls for the exploratory lead-time audit."""
import numpy as np
from .data import observation
from .lead import (MONITORS, ROWS, SOURCES, available_history, features,
                   sensor_records)


METHODS = ("I1_common", "I2_unaligned", "I3_physical_lag", "N0_recent",
           "N1_physical_norm", "N2_recent_norm", "N3_uniform_norm", "N4_wrong_norm")
NORMALIZED = METHODS[4:]


def matched_features(event, records, region, now, tau, graph):
    """One shared causal base; vary only the source-history summary."""
    base = features(event, records, region, now, tau, graph, "I1_common")
    out = {m: base.copy() for m in METHODS}
    ages, residuals, weights, centers, sources = [], [], [], [], []
    for link in graph[str(region)]:
        for row in available_history(records, link["source"], now, 6):
            ages.append(now-row["valid_time_s"])
            residuals.append(row["residual_m"])
            weights.append(link["weight"]/6)
            centers.append(max(link["travel_s"]-tau, 0))
            sources.append(link["source"])
    if not ages:
        return out
    a, r, w, c = map(np.asarray, (ages, residuals, weights, centers))
    kernels = {
        "uniform": w,
        "physical": w*np.exp(-.5*((a-c)/10)**2),
        "recent": w*np.exp(-.5*(a/10)**2),
        "wrong": w*np.exp(-.5*((a-(c+20)%40)/10)**2),
    }
    for method, kernel in (("I2_unaligned", "uniform"),
                           ("I3_physical_lag", "physical"),
                           ("N0_recent", "recent")):
        kw = kernels[kernel]
        out[method][-3:] = [kw@r, kw.sum(), kw@a/60]
    # All normalized controls expose exactly the same support and mean age.
    # A kernel's own total mass is deliberately not an extra input.
    support, age = w.sum(), (w@a)/w.sum()/60
    for method, kernel in zip(NORMALIZED, ("physical", "recent", "uniform", "wrong")):
        kw = kernels[kernel]
        values, edge_weights = [], []
        for link in graph[str(region)]:
            mask = np.asarray(sources)==link['source']
            if mask.any():
                values.append((kw[mask]@r[mask])/kw[mask].sum())
                edge_weights.append(link['weight'])
        # Normalize each source's history first, preserving fixed spatial weights.
        value = np.average(values, weights=edge_weights)
        out[method][-3:] = [value, support, age]
    return out


def phased_records(event, noise, delay, moving, phase=0):
    """Phase zero reproduces the original experiment exactly."""
    if phase == 0:
        return sensor_records(event, noise, delay, moving)
    if not moving or phase not in (1, 2, 3) or noise != 0:
        raise ValueError("Additional phases are noiseless rotating controls only")
    groups = [SOURCES]+[[10*y+x for y in ROWS] for x in (3, 5, 7)]
    records = {i: [] for i in MONITORS}
    for t in np.arange(20, 301, 10):
        for region in groups[(int(t)//10+phase)%4]:
            value = np.interp(t, event["time_s"], event["truth_depth_m"][:, region])
            records[region].append(observation(event, region, t, t+delay, value, 20))
    return records


def timing_rows(records, graph, now, region, tau, wait=0):
    """Timing potential, not a bound on persistence-based predictability.

    Age is now minus valid time, already incorporating communication delay.
    Missing evidence is kept as missing, never a zero residual or a success.
    """
    output = []
    for link in graph[str(region)]:
        h = available_history(records, link["source"], now, 6)
        age = now-h[0]["valid_time_s"] if h else None
        lead = link["travel_s"]-age if h else None
        remaining = lead-wait if h else None
        desired_unclipped = link["travel_s"]-tau
        output.append(dict(time_s=now, region=region, source=link["source"],
            tau=tau, wait_s=wait, travel_s=link["travel_s"], age_s=age,
            nominal_lead_s=lead, nominal_remaining_s=remaining,
            horizon_margin_s=lead-tau if h else None, observed=bool(h),
            positive_remaining=bool(h and remaining>0),
            horizon_covered=bool(h and lead>=tau and remaining>0),
            post_action_horizon_covered=bool(h and remaining>=tau),
            desired_age_s=max(desired_unclipped, 0),
            desired_clipped=desired_unclipped<0,
            valid_time_s=h[0]["valid_time_s"] if h else None,
            arrival_time_s=h[0]["arrival_time_s"] if h else None))
    return output
