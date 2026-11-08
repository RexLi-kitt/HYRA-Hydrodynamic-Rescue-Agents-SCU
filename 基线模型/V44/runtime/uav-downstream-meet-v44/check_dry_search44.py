"""Paired development replay for V44's dry-water approach cost."""
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiment44 import (HERE, finish, jobs, load_scene, material, new_agent, quota,
                          read, runtime, save, sha)


def evaluate(item):
    stage, case, roster = item
    _, seed, family, _ = case
    field, people = (material if stage == "screen" else load_scene)(case, roster)
    judge, _ = runtime(1, "planning")
    arms = {}
    for mode in ("continuous_search", "dry_search"):
        env = judge(field, people, (64, 720, 2400))
        assert env.radius == 1
        policy = new_agent(mode)
        arm = finish(env, policy, np.full(64, -1))
        arm["delivered"] = sum(t >= 0 for t in arm["delivery"])
        arm["dry_searches"] = policy.deep.dry_search_calls
        arms[mode] = arm
    base, candidate = arms.values()
    if stage == "screen":
        old = next(r["revised"] for r in read(HERE/"search-integration.json")["rows"]
                   if (r["seed"], r["family"], r["roster"]) == (seed, family, roster))
        assert all(base[k] == old[k] for k in ("first", "delivery", "conflicts"))
    if family == "continuous":
        assert candidate == base
    return dict(seed=seed, family=family, roster=roster, arms=arms)


def main(stage):
    assert stage in ("screen", "branch")
    if stage == "branch":
        assert read(HERE/"dry-search-screen.json")["delta"] > 0
    items = jobs(stage)
    rows = []
    for offset in range(0, len(items), 4):
        quota()
        with ProcessPoolExecutor(max_workers=4) as workers:
            rows.extend(workers.map(evaluate, [(stage, c, r) for c, r in items[offset:offset+4]]))
    parents = defaultdict(int)
    for row in rows:
        if row["family"] == "rewet":
            a, b = row["arms"].values()
            parents[row["seed"]] += b["delivered"] - a["delivered"]
    version = {n: sha(HERE/n) for n in ("dry-search-protocol.md","experiment44.py","continuous44.py",
                "check_dry_search44.py","fraction.dll","shortlist44.py","tracker44.py")}
    delta = sum(parents.values())
    save(HERE/f"dry-search-{stage}.json", dict(version=version, delta=delta, parents=parents, rows=rows))
    print(stage, delta, dict(parents), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
