"""Materialize registered synthetic inputs and compare V44 deliveries against the archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE / "runtime"
MODEL = ROOT / "uav-downstream-meet-v44"
sys.path.insert(0, str(MODEL))
from study_terminal44 import NoveltyBeam, load_model, read, run, s, stop_at_edge, world

MANIFEST = read(ROOT / "inputs.json")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(row):
    seed, family = row["seed"], row["family"]
    path = ROOT / row["water"]
    if not path.is_file():
        example = BASE / "环境与裁判" / "examples" / "water" / f"{seed}-{family}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if example.is_file():
            with np.load(example, allow_pickle=False) as z:
                np.save(path, z["water"])
        elif family == "ordinary":
            s.scenes.original.OUT = path.parent.parent
            s.scenes.original.get_ordinary(seed, 1)
        else:
            raise FileNotFoundError(path)
    if digest(path) != row["meta"]["water_sha"]:
        raise ValueError(f"Water differs from archive: {seed} {family}")
    field = s.water(path)
    hist = ROOT / row["history"]
    if not hist.exists():
        ids = np.asarray(row["ids"], dtype=int)
        yy, xx = np.unravel_index(ids, (100, 100))
        samples = field[:, :, yy, xx].astype(float)
        delta = samples[1:] - samples[:-1]
        hazard = np.empty((480, 5, len(ids)))
        k = read(ROOT / "uav-pressure-search-v8" / "risk-lock.json")["k"]
        for sec in range(5):
            h, u, v = np.moveaxis(samples[:-1] + ((sec + .5) / 5) * delta, 1, 0)
            hazard[:, sec] = k * s.scenes.shape(h, np.hypot(u, v))
        clock = np.array(row["clock"])
        hits = hazard.reshape(2400, len(ids)).cumsum(0) >= clock
        fall = np.where(hits.any(0), hits.argmax(0) + 1, 2401)
        pop = s.simulate(field, ids, fall, seed, row["roster"], np.array(row["meta"]["traps"]))
        for key, expected in row["hashes"].items():
            if hashlib.sha256(np.ascontiguousarray(pop[key])).hexdigest() != expected:
                raise ValueError(f"Population regeneration differs: {seed} {key}")
        hist.parent.mkdir(parents=True, exist_ok=True)
        temp = hist.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **pop, ids=ids, clock=clock)
        os.replace(temp, hist)
    with np.load(hist, allow_pickle=False) as z:
        pop = {}
        for key, expected in row["hashes"].items():
            a = z[key]
            if hashlib.sha256(np.ascontiguousarray(a)).hexdigest() != expected:
                raise ValueError(f"Population differs from archive: {seed} {key}")
            pop[key] = a[:64] if key == "fall" else a[:, :64]
    directory = ROOT / row["scene"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input.json").write_text(json.dumps(dict(row["meta"], water=str(path))), encoding="utf8")
    return field, pop


def replay(row, field, pop):
    stage = row["phase"]
    if stage not in ("repeat", "mixed", "fresh"):
        raise ValueError("Use --prepare-only for fitting and selection inputs.")
    archive = read(MODEL / f"terminal-{stage}.json")
    record = next(r for r in archive["rows"] if
                  (r["seed"], r["family"], r["roster"]) == (row["seed"], row["family"], row["roster"]))
    population, _ = stop_at_edge(pop)
    for key, method in [("pure", "pure"), ("base", None), ("previous", "early_delivery"),
                        (str(archive["winner"]), "terminal_dispatch")]:
        policy = NoveltyBeam("service") if method == "pure" else load_model(fleet_policy=method)
        observed = run(world(field, population), policy)
        for metric in ("first", "delivery", "conflicts", "counts"):
            if observed[metric] != record["arms"][key][metric]:
                raise ValueError(f"{method} differs in {metric}")
        print(f"  {method or 'default'}: {observed['counts'][-1]}/64", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--case", choices=list(MANIFEST))
    choice.add_argument("--stage", choices=("fit", "select", "repeat", "mixed", "fresh", "all"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.stage in ("all", "fit", "select") and not args.prepare_only:
        parser.error("Use --prepare-only for all, fit or select.")
    for name, row in MANIFEST.items():
        if args.case and name != args.case:
            continue
        if args.stage and args.stage not in ("all", row["phase"]):
            continue
        field, pop = prepare(row)
        print(f"{name} inputs verified", flush=True)
        if not args.prepare_only:
            replay(row, field, pop)


if __name__ == "__main__":
    main()

