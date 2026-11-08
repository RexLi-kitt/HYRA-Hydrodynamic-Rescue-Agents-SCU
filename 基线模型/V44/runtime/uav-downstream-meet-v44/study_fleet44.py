"""Fit on matched 120-second forks; replay complete missions on other maps."""
import copy
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from experiment44 import HERE, finish, material, load_scene, runtime, quota, read, save, sha, s, cloned_mission
from fit_route_value44 import training_cases
from model import load_model
from fleet_options44 import Fleet, NAMES

TIMES = (180, 540, 900, 1260, 1620)


def new_world(case, roster, loader):
    from boundary_trial44 import stop_at_edge
    field, people = loader(case, roster)
    people, _ = stop_at_edge(people)
    judge, _ = runtime(1, "planning")
    env = judge(field, people, (64,720,2400))
    assert env.radius == 1
    return env


def advance(env, agent, until):
    while env.t < until:
        env.step(*agent.act(env.observation()))


def train_case(job):
    quota()
    c, roster = job
    env = new_world(c, roster, material)
    policy = load_model(fleet_policy="base")
    rows = []
    for t in TIMES:
        advance(env, policy, t)
        xs, outcomes = [], []
        for name in NAMES:
            fork, agent = cloned_mission(env), copy.deepcopy(policy)
            agent.deep.selector = name
            advance(fork, agent, t+120)
            decision = agent.deep.decisions[-1]
            assert decision["t"] == t and decision["choice"] == NAMES.index(name)
            xs.append(decision["x"])
            outcomes.append(sum(int(t < d <= t+120) for d in fork.supplied))
        assert np.allclose(xs,xs[0])
        rows.append(dict(t=t,x=xs[0],delivered=outcomes))
    return dict(seed=c[1],family=c[2],roster=roster,rows=rows)


def check_case(job):
    quota()
    c, roster, ref = job
    env = new_world(c, roster, load_scene)
    arms = {}
    for mode in ("rule","learned"):
        agent = load_model(fleet_policy=mode)
        outcome = finish(cloned_mission(env),agent,np.full(64,-1))
        outcome["sent"] = sum(d>=0 for d in outcome["delivery"])
        outcome["decisions"] = agent.deep.decisions
        arms[mode] = outcome
        if c[2] == "ordinary" and agent.branch is agent.sparse:
            assert all(outcome[k] == ref["base"][k] for k in ("first","delivery","conflicts"))
    for mode in ("pure","base"):
        arms[mode] = {k:ref[mode][k] for k in ("sent","first","delivery","conflicts")}
    return dict(seed=c[1],family=c[2],roster=roster,arms=arms)


def main(stage):
    if stage == "train":
        with ProcessPoolExecutor(4) as pool:
            cases = list(pool.map(train_case, training_cases()))
        samples = [row for case in cases for row in case["rows"]]
        X = np.array([r["x"] for r in samples])
        Y = np.array([r["delivered"] for r in samples])
        mean,scale = X.mean(0),X.std(0)
        scale[scale<1e-6] = 1
        z = (X-mean)/scale
        coef = np.linalg.solve(z.T@z+8*np.eye(z.shape[1]), z.T@(Y-Y.mean(0)))
        save(HERE/"fleet-options-fit.json",dict(cases=cases,mean=mean.tolist(),scale=scale.tolist(),
             coef=coef.tolist(),intercept=Y.mean(0).tolist(),source=sha(HERE/"fleet_options44.py"),
             protocol=sha(HERE/"fleet-options-protocol.md")))
        print(len(samples),Y.sum(0),flush=True)
    else:
        assert stage == "evaluate"
        fit = read(HERE/"fleet-options-fit.json")
        assert fit["source"] == sha(HERE/"fleet_options44.py") and fit["protocol"] == sha(HERE/"fleet-options-protocol.md")
        archive = read(HERE/"edge-search-evaluate.json")
        refs = {(r["seed"],r["family"],r["roster"]):r["arms"] for r in archive["rows"]}
        work = [(c,r,refs[c[1],c[2],r]) for side,r in (("a",0),("b",1)) for c in s.cases(side)]
        assert not {c[1] for c,_,_ in work} & {r["seed"] for r in fit["cases"]}
        rows = []
        with ProcessPoolExecutor(4) as pool:
            for row in pool.map(check_case, work):
                rows.append(row)
                if len(rows)%8 == 0:
                    print(len(rows),len(work),flush=True)
        totals = {f:{m:sum(r["arms"][m]["sent"] for r in rows if f=="all" or r["family"]==f)
              for m in ("pure","base","rule","learned")} for f in ("all","ordinary","continuous","rewet")}
        save(HERE/"fleet-options-evaluate.json",dict(rows=rows,totals=totals,weights=sha(HERE/"fleet-options-fit.json"),
             controls=sha(HERE/"edge-search-evaluate.json"),runner=sha(HERE/"study_fleet44.py")))
        print(totals,flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
