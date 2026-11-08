"""Paired integration replay of the original V44 pressure development set. No model selection."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiment44 import (HERE, code_key, finish, historical_record_matches, jobs,
                          material, new_agent, read, runtime, save, sha)
from model import load_model


def check(case, roster):
    _, seed, family, _ = case
    old_record = read(HERE / "screen" / f"{seed}-{family}-{roster}.json")
    assert historical_record_matches(old_record["code"], code_key())
    field, population = material(case, roster)
    judge, _ = runtime(1, "planning")
    rows = {}
    for label, agent in (("original", new_agent("extended")), ("revised", load_model())):
        env = judge(field, population, (64, 720, 2400))
        assert env.radius == 1
        row = finish(env, agent, np.full(64, -1))
        if label == "original":
            archived = old_record["arms"]["extended"]
            assert all(row[k] == archived[k] for k in ("first", "delivery", "conflicts"))
        else:
            assert agent.deep.checked == 2
            row.update(search_calls=agent.deep.search_calls, changed_first=agent.deep.changed_first)
        row["delivered"] = sum(t >= 0 for t in row["delivery"])
        rows[label] = row
    return dict(seed=seed, family=family, roster=roster, people=64, **rows)


if __name__ == "__main__":
    cases, rosters = zip(*jobs("screen"))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(check, cases, rosters))
    for result in rows:
        print(result["seed"], result["family"], result["original"]["delivered"],
              result["revised"]["delivered"], result["revised"]["changed_first"], flush=True)
    assert any(r["revised"]["changed_first"] for r in rows)
    files = ("experiment44.py", "tracker44.py", "model.py", "continuous44.py", "shortlist44.py",
             "fraction.cpp", "fraction.dll", "check_search44.py")
    save(HERE / "search-integration.json", dict(
        purpose="integration regression only; eight development parents, two pressure maps per parent",
        version={name: sha(HERE / name) for name in files}, rows=rows))
