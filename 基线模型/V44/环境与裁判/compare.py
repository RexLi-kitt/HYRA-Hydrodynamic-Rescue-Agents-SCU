"""Paired map-block uncertainty for results produced by run.py eval."""
from pathlib import Path
import argparse,json
import numpy as np

def index(r):
    rows={t["id"]:t for t in r["trials"]}
    if len(rows)!=len(r["trials"]):raise ValueError("Duplicate trial IDs")
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("candidate",type=Path)
    parser.add_argument("baseline",type=Path)
    args=parser.parse_args()
    a,b=[json.loads(p.read_text(encoding="utf-8-sig")) for p in (args.candidate,args.baseline)]
    for key in ("version","panel_sha","panel_files_sha"):
        if a[key]!=b[key]:raise ValueError("Mismatched "+key)
    aa,bb=index(a),index(b)
    if not aa or aa.keys()!=bb.keys():raise ValueError("Unpaired trials")
    groups={};families={}
    for k in sorted(aa):
        x,y=aa[k],bb[k]
        if any(x[v]!=y[v] for v in ("map","family","parent","roster")):raise ValueError(k)
        if x["metrics"]["total"]!=64 or y["metrics"]["total"]!=64:raise ValueError(k)
        row=np.array([x["metrics"]["delivered"],y["metrics"]["delivered"],64],int)
        f=x["family"];families[f]=families.get(f,np.zeros(3,int))+row
        g=groups.setdefault("ordinary" if f=="ordinary" else "paired_pressure",{})
        g[x["parent"]]=g.get(x["parent"],np.zeros(3,int))+row
    total=sum(families.values());n,m,den=map(int,total)
    rng=np.random.default_rng(28114);boot=np.zeros((20000,3),int)
    for g in groups.values():
        blocks=np.array(list(g.values()))
        boot+=blocks[rng.integers(len(blocks),size=(20000,len(blocks)))].sum(1)
    diff=100*(boot[:,0]-boot[:,1])/boot[:,2]
    relative=None
    if np.all(boot[:,1]>0):
        relative=np.quantile(100*(boot[:,0]-boot[:,1])/boot[:,1],[.025,.975]).tolist()
    print(json.dumps(dict(trials=len(aa),candidate=n,baseline=m,total=den,candidate_RCR=n/den,baseline_RCR=m/den,
               difference_pp=100*(n-m)/den,relative_percent=100*(n-m)/m if m else None,
               paired_difference_pp_ci95=np.quantile(diff,[.025,.975]).tolist(),paired_relative_percent_ci95=relative,
               families={k:v.tolist() for k,v in families.items()}),indent=2))
if __name__=="__main__":main()
