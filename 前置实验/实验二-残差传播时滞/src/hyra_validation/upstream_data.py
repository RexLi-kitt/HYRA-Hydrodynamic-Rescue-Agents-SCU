"""Actual published forecast versions and the common upstream-only records.

Reference target labels have no argument in this module. Paired events share
an explicitly supplied noise array rather than advancing an internal RNG.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .hidden_target_inputs import SourceRecord, SOURCES


@dataclass(frozen=True)
class ForecastTable:
    release_s: np.ndarray
    valid_s: np.ndarray
    depth_m: np.ndarray
    location_ids: tuple[int, ...]

    def __post_init__(self):
        release = np.array(self.release_s, dtype=float, copy=True)
        valid = np.array(self.valid_s, dtype=float, copy=True)
        values = np.array(self.depth_m, dtype=float, copy=True)
        if release.ndim != 1 or not len(release) or np.any(np.diff(release) <= 0):
            raise ValueError('Forecast issues must be unique and ordered')
        if valid.ndim != 2 or len(valid) != len(release) or valid.shape[1] == 0 or np.any(np.diff(valid, axis=1) <= 0):
            raise ValueError('Invalid version-specific valid grid')
        if values.shape != (*valid.shape, len(self.location_ids)) or len(set(self.location_ids)) != len(self.location_ids):
            raise ValueError('Forecast location/array mismatch')
        if not all(np.all(np.isfinite(a)) for a in (release,valid,values)):
            raise ValueError('Nonfinite forecast')
        if np.any(valid[:, 0] != release):
            raise ValueError('Each saved forecast must start at its actual issue')
        for name, value in [('release_s',release),('valid_s',valid),('depth_m',values)]:
            value.setflags(write=False); object.__setattr__(self, name, value)

    def at_issue(self, index, query_s):
        query = np.atleast_1d(np.asarray(query_s, dtype=float))
        valid, depth = self.valid_s[index], self.depth_m[index]
        if np.any(~np.isfinite(query)) or query.min() < valid[0] or query.max() > valid[-1]:
            raise ValueError('Query outside this published version')
        return np.column_stack([np.interp(query, valid, depth[:, j]) for j in range(depth.shape[1])])

    def at_decision(self, now_s, offsets_s):
        index = np.searchsorted(self.release_s, now_s, side='right')-1
        if index < 0: raise ValueError('No published version at decision')
        return float(self.release_s[index]), self.at_issue(index, now_s+np.asarray(offsets_s))


def source_records(times_s, depth_m, source_ids, forecasts, valid_s, delay_s, noise_m):
    """Build six-channel observations and match actual issue <= valid-20 s."""
    if tuple(source_ids) != SOURCES or tuple(forecasts.location_ids) != SOURCES:
        raise ValueError('Only the six common upstream channels are permitted')
    times = np.asarray(times_s, dtype=float); depth = np.asarray(depth_m, dtype=float)
    valid = np.asarray(valid_s, dtype=float); noise = np.asarray(noise_m, dtype=float)
    if times.ndim != 1 or len(times)<2 or np.any(np.diff(times)<=0) or depth.shape != (len(times),6):
        raise ValueError('Invalid upstream trajectory')
    if valid.ndim != 1 or not len(valid) or np.any(np.diff(valid)<=0) or noise.shape != (len(valid),6) or not np.isfinite(delay_s) or delay_s<0:
        raise ValueError('Invalid observation schedule or shared noise')
    if valid.min()<times[0] or valid.max()>times[-1] or not all(np.all(np.isfinite(x)) for x in (times,depth,valid,noise)):
        raise ValueError('Invalid observation times/values')
    records = []; audit = []
    for j,t in enumerate(valid):
        issue = int(np.searchsorted(forecasts.release_s,t-20,side='right')-1)
        if issue<0: raise ValueError('Historical matching issue was not saved')
        prior = forecasts.at_issue(issue,t)[0]
        observations = np.array([np.interp(t,times,depth[:,s]) for s in range(6)])+noise[j]
        for s, source in enumerate(SOURCES):
            identity = j*6+s
            record = SourceRecord(identity,source,float(t),float(t+delay_s),
                                  float(forecasts.release_s[issue]),float(observations[s]),float(prior[s]))
            records.append(record)
            audit.append(dict(record_id=identity,source=source,valid_s=float(t),arrival_s=float(t+delay_s),
                              prior_issue_index=issue,prior_release_s=record.prior_release_s,
                              actual_lead_s=float(t-record.prior_release_s)))
    checksum=hashlib.sha256(json.dumps([r.__dict__ for r in records],sort_keys=True).encode()).hexdigest()
    return dict(records=records,references=audit,observation_sha256=checksum)
