"""Check shared judge provenance, count paired outcomes, and group by map parent."""
import numpy as np
from study_fleet44 import new_world
from experiment44 import HERE, read, save, sha, s, load_scene, finish, NoveltyBeam
from model import load_model

def main():
    rows = read(HERE/"fleet-options-evaluate.json")["rows"]
    assert len(rows) == len({(r["seed"],r["family"],r["roster"]) for r in rows}) == 40
    cases = {(c[1],c[2],n):c for side,n in (("a",0),("b",1)) for c in s.cases(side)}
    # Reproduce both cached controls in one example per regime per panel.
    checked = set()
    for r in rows:
        key = (r["family"],r["roster"])
        if key in checked:
            continue
        checked.add(key)
        case = cases[r["seed"],r["family"],r["roster"]]
        for m,agent in (("base",load_model()),("pure",NoveltyBeam("service"))):
            env = new_world(case,r["roster"],load_scene)
            data = finish(env,agent,np.full(64,-1))
            assert all(data[k] == r["arms"][m][k] for k in data)
    totals = {}
    for m in ("pure","base","rule","learned"):
        found = sent = conflicts = 0
        choices = np.zeros(3,int)
        for row in rows:
            arm = row["arms"][m]
            d,f = map(np.asarray,(arm["delivery"],arm["first"]))
            assert len(d)==len(f)==64 and ((d<0)|((f>=0)&(f<=d))).all()
            assert arm["sent"] == (d>=0).sum()
            found += int((f>=0).sum())
            sent += arm["sent"]
            conflicts += arm["conflicts"]
            for decision in arm.get("decisions",[]):
                choices[decision["choice"]] += 1
        totals[m] = dict(found=found,sent=sent,conflicts=conflicts,choices=choices.tolist())
    pairs = {}
    for ctrl in ("base","pure","rule"):
        rng = np.random.default_rng(4411)
        boot = np.zeros(10000)
        gain = lose = 0
        for ordinary in (True,False):
            parent = {}
            for r in rows:
                if (r["family"]=="ordinary") != ordinary:
                    continue
                a,b = (np.asarray(r["arms"][m]["delivery"])>=0 for m in ("learned",ctrl))
                gain += int((a&~b).sum())
                lose += int((b&~a).sum())
                key = r["seed"]
                parent[key] = parent.get(key,0)+int(a.sum()-b.sum())
            v = list(parent.values())
            boot += rng.choice(v,(len(boot),len(v))).sum(1)
        pairs[ctrl] = dict(gained=gain,lost=lose,ci_pp=np.quantile(boot*100/2560,[.025,.975]).tolist())
    save(HERE/"fleet-options-audit.json",dict(totals=totals,pairs=pairs,controls_checked=sorted(checked),
         source={n:sha(HERE/n) for n in ["fleet-oracle44.py","fleet-oracle-schedule.json","fleet-options-protocol.md",
             "fleet_options44.py","study_fleet44.py","fleet-options-fit.json","fleet-options-evaluate.json","boundary_trial44.py","model.py"]}))
    print(totals,pairs,flush=True)

if __name__ == "__main__":
    main()
