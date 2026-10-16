"""Signed sparse residual evidence; no reference truth in online interface."""
import numpy as np


def fit_structure(train_events, representation_config):
    config = representation_config
    if any(e["metadata"]["split"] != "train" for e in train_events):
        raise ValueError("Fit accepts training events only")
    n = train_events[0]["truth_depth_m"].shape[1]
    samples = []
    for event in train_events:
        for f, release in enumerate(event["release_s"]):
            valid = release + config["lead_time_s"]
            vi = np.flatnonzero(event["forecast_valid_s"][f] == valid)
            ti = np.flatnonzero(event["time_s"] == valid)
            if len(vi) and len(ti):
                samples.append(event["truth_depth_m"][ti[0]] - event["forecast_depth_m"][f, vi[0]])
    x = np.asarray(samples)
    cov = np.cov(x, rowvar=False)
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    xy = train_events[0]["xy_m"]
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    spacing = np.min(distance[distance > 0])
    candidates = distance <= spacing * 1.5
    weights = np.zeros((n, n))
    for j in range(n):
        others = np.flatnonzero(candidates[j] & (np.arange(n) != j))
        others = others[np.argsort(-np.abs(corr[j, others]), kind="stable")]
        selected = np.r_[j, others[:config["max_links_per_region"] - 1]]
        weights[j, selected] = corr[j, selected]
        if np.abs(weights[j]).sum() < 1e-12:
            weights[j, j] = 1
        weights[j] /= np.abs(weights[j]).sum()
    return dict(weights=weights.tolist(), correlations=corr.tolist(), frozen=True,
                fitted_event_ids=[e["metadata"]["event_id"] for e in train_events],
                candidate_rule="8-neighbour terrain grid including self")


def distance_structure(xy, config):
    xy = np.asarray(xy)
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    w = np.zeros_like(d)
    for j in range(len(xy)):
        selected = np.argsort(d[j], kind="stable")[:config["max_links_per_region"]]
        w[j, selected] = 1 / np.maximum(d[j, selected], np.min(d[d > 0]))
        w[j] /= w[j].sum()
    return dict(weights=w.tolist(), frozen=True)


def causal_history(records, now, config):
    histories = [[] for _ in range(config["n_regions"])]
    for r in records:
        age = now - r["valid_time_s"]
        if (r["arrival_time_s"] <= now and 0 <= age <= config["max_age_s"]
                and r.get("prior_release_s", float("inf")) <= r["valid_time_s"]
                and r.get("residual_m") is not None):
            histories[r["region_id"]].append(r)
    for i, h in enumerate(histories):
        unique = {r["valid_time_s"]: r for r in sorted(h, key=lambda r: r["arrival_time_s"], reverse=True)}
        histories[i] = sorted(unique.values(), key=lambda r: r["valid_time_s"], reverse=True)[:config["history_length"]]
    return histories


def build_representation(observation_records, available_forecasts, now_s, frozen_structure, config):
    if not frozen_structure.get("frozen"):
        raise ValueError("Unfrozen structure")
    histories = causal_history(observation_records, now_s, config)
    k, n = config["history_length"], config["n_regions"]
    method = config["method"]
    preserve_raw=config.get("preserve_raw_history",False)
    raw=np.zeros((n,4*k),dtype=np.float32)
    for i,h in enumerate(histories):
        for q,r in enumerate(h):
            raw[i,4*q:4*q+4]=(r["observed_depth_m"],r["prior_depth_m"],
                (now_s-r["valid_time_s"])/config["max_age_s"],1)
    features = np.zeros((n, 4*k), dtype=np.float32)
    evidence, coverage = np.zeros(n), np.zeros(n)
    if method == "B0_raw":
        for i, h in enumerate(histories):
            coverage[i] = len(h)/k
        features=np.c_[raw,np.zeros((n,3),dtype=np.float32)] if preserve_raw else raw
        return dict(features=features, evidence=evidence, mask=coverage>0, coverage=coverage)
    w = np.eye(n) if method=="B_local" else np.asarray(frozen_structure["weights"])
    for i, h in enumerate(histories):
        coverage[i] = len(h)/k
        for r in h:
            decay = 1 if method == "B2_uniform" else np.exp(-(now_s-r["valid_time_s"])/config["decay_time_s"])
            evidence[i] += decay*r["residual_m"]/k
    summary, support = w@evidence, np.abs(w)@coverage
    if preserve_raw:
        features=np.c_[raw,summary,support>0,support].astype(np.float32)
    else:
        features[:,0], features[:,1], features[:,2] = summary, support>0, support
    return dict(features=features, evidence=summary, mask=support>0, coverage=support)
