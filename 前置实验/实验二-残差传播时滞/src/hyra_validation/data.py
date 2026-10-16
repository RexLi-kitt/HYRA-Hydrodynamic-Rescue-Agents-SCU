"""Family-isolated events and causal, version-matched sensor records."""
import json
from pathlib import Path
import numpy as np


def load_events(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    seen, families, output = set(), {}, {}
    for split in ("train", "validation", "test"):
        if not manifest.get(split):
            raise ValueError(f"Empty {split} split")
        output[split] = []
        for filename in manifest[split]:
            event_path = (path.parent / filename).resolve()
            meta = json.loads(event_path.with_suffix(".json").read_text(encoding="utf-8"))
            family = meta["event_family_id"]
            if families.get(family, split) != split:
                raise ValueError("Event-family leakage")
            if meta["event_id"] in seen:
                raise ValueError("Duplicate event")
            families[family] = split
            seen.add(meta["event_id"])
            with np.load(event_path, allow_pickle=False) as archive:
                event = {k: archive[k] for k in archive.files}
            event["metadata"] = meta
            if np.any(np.diff(event["time_s"]) <= 0) or not np.isfinite(event["truth_depth_m"]).all():
                raise ValueError("Invalid times or depths")
            if meta["units"] != {"time": "s", "xy": "m", "depth": "m"}:
                raise ValueError("Invalid units")
            if np.any(event["forecast_valid_s"] < event["release_s"][:, None]):
                raise ValueError("Forecast valid before release")
            n = len(event["xy_m"])
            if event["truth_depth_m"].shape != (len(event["time_s"]), n):
                raise ValueError("Truth/grid shape mismatch")
            if event["forecast_depth_m"].shape != (*event["forecast_valid_s"].shape,n):
                raise ValueError("Forecast/grid shape mismatch")
            if np.any(event["truth_depth_m"] < 0) or not np.isfinite(event["forecast_depth_m"]).all():
                raise ValueError("Negative truth or nonfinite forecast")
            if np.any(np.diff(event["release_s"])<=0) or np.any(np.diff(event["forecast_valid_s"],axis=1)<=0):
                raise ValueError("Forecast time order")
            output[split].append(event)
    return output


def match_prior(event, region, valid_s, lead_s):
    matches = np.argwhere((event["forecast_valid_s"] == valid_s) &
                          (event["release_s"][:, None] == valid_s - lead_s))
    if len(matches) != 1:
        return None
    f, v = map(int, matches[0])
    return dict(matched_forecast_id=f, prior_release_s=float(event["release_s"][f]),
                prior_depth_m=float(event["forecast_depth_m"][f, v, region]))


def observation(event, region, valid_s, arrival_s, value, lead_s):
    prior = match_prior(event, region, valid_s, lead_s)
    record = dict(region_id=int(region), valid_time_s=float(valid_s),
                  arrival_time_s=float(arrival_s), observed_depth_m=float(value))
    if prior is not None:
        record.update(prior)
        record["residual_m"] = float(value - prior["prior_depth_m"])
    else:
        record.update(matched_forecast_id=None, prior_depth_m=None, residual_m=None)
    return record


def prepare_observations(event, config):
    rng = np.random.default_rng(config["seed"])
    records = []
    for index, t in enumerate(event["time_s"]):
        if t % config["sample_interval_s"] or t < config["lead_time_s"]:
            continue
        regions = config["route_regions"][(int(t)//config["sample_interval_s"]) % len(config["route_regions"])]
        for region in regions:
            value = event["truth_depth_m"][index, region] + rng.normal(0, config["noise_std_m"])
            records.append(observation(event, region, t, t + config["delay_s"], value, config["lead_time_s"]))
    return records


def current_forecast(event, now, horizon_s):
    available = np.flatnonzero(event["release_s"] <= now)
    if len(available) == 0:
        raise ValueError("No published forecast")
    f = int(available[-1])
    times = event["forecast_valid_s"][f]
    if not times[0] <= now+horizon_s <= times[-1]:
        raise ValueError("Future outside released forecast")
    return np.array([np.interp(now+horizon_s, times, event["forecast_depth_m"][f, :, r])
                     for r in range(event["truth_depth_m"].shape[1])])
