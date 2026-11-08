"""Paired development evaluation of the frozen route-prefix gate."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiment44 import HERE,ROOT,s,load_scene,material,runtime,NoveltyBeam,finish,quota,read,save,sha
from boundary_trial44 import stop_at_edge
from model import load_model


def replay(job):
    panel,case,rep,controls=job
    quota()
    field,pop=(material if panel=="pressure" else load_scene)(case,rep)
    pop,contact=stop_at_edge(pop)
    judge,_=runtime(1,"planning")
    arms={}
    for mode in ("base","pure","all","detour_gate") if controls else ("detour_gate",):
        agent=NoveltyBeam("service") if mode=="pure" else load_model(edge_search=None if mode=="base" else mode)
        env=judge(field,pop,(64,720,2400))
        assert env.radius==1
        row=finish(env,agent,np.full(64,-1))
        d=np.array(row["delivery"])
        row.update(sent=int((d>=0).sum()),found=sum(t>=0 for t in row["first"]),
                   after_contact=int(((contact>=0)&(d>=contact)).sum()))
        if mode=="detour_gate":
            row.update(compared=agent.deep.compared,accepted=agent.deep.accepted)
        arms[mode]=row
    return dict(panel=panel,seed=case[1],family=case[2],roster=rep,arms=arms)


def main():
    prior=read(HERE/"edge-search-evaluate.json")["rows"]
    cache={(r["panel"],r["seed"],r["family"],r["roster"]):r["arms"] for r in prior}
    work=[]
    for panel,rep in (("A",0),("B",1)):
        for c in s.cases(panel.lower()):
            work.append((panel,c,rep,c[3]==0 and c[2]=="continuous"))
    work.extend(("pressure",(str(ROOT/"uav-committed-views-v50"),2026110701+k,f,k),0,True)
                for k in range(8) for f in ("continuous","rewet"))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(replay,work):
            if row["panel"]!="pressure":
                old=cache[row["panel"],row["seed"],row["family"],row["roster"]]
                for mode in ("base","pure","all"):
                    if mode in row["arms"]:
                        assert all(row["arms"][mode][k]==old[mode][k] for k in ("first","delivery","conflicts"))
                    else:
                        row["arms"][mode]={**old[mode],"after_contact":old[mode]["at_edge"]}
            rows.append(row)
            if len(rows)%8==0:
                print(len(rows),len(work),flush=True)
    summary={}
    for group in ("A","B","mixed","pressure"):
        subset=[r for r in rows if (r["panel"] in ("A","B") if group=="mixed" else r["panel"]==group)]
        summary[group]={m:{k:sum(r["arms"][m][k] for r in subset) for k in ("sent","found","after_contact")}
                        for m in ("base","pure","all","detour_gate")}
    version={n:sha(HERE/n) for n in ("detour-gate-protocol.md","detour_gate44.py","edge_search44.py",
                                      "test_detour44.py","model.py","boundary_trial44.py","edge-search-evaluate.json")}
    save(HERE/"detour-gate-result.json",dict(version=version,summary=summary,rows=rows))
    print(summary,flush=True)


if __name__=="__main__":
    main()
