"""Publish the preregistered P3 screening without selecting or refitting models."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hyra_uav.experiments import write_json


def publish(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/"summary.json").read_text(encoding="utf-8"))
    metrics=json.loads((source/"metrics.json").read_text(encoding="utf-8"))
    for name in ("plan.json","freeze.json","replay-index.json","graph-audit.json","fit-summary.json","summary.json"):
        shutil.copyfile(source/name,output/name)
    bands=[]
    for scene in ("p3-holdout-slow","p3-holdout-rapid","p3-holdout-async"):
        for group in ("B1","B2","B3","A1","A2"):
            for band in ("age_0_60","age_60_180","age_180_plus"):
                selected=[m for m in metrics if m["scene"]==scene and m["group"]==group and m["band"]==band and m["fraction"]==.1]
                route_means=[[m["precision"] for m in selected if m["route"]==route and m["precision"] is not None] for route in ("R0","R1","R2")]
                bands.append(dict(scene=scene,group=group,band=band,precision=float(np.mean([np.mean(r) for r in route_means if r])) if any(route_means) else None,
                                  empty_times=sum(m["n"]==0 for m in selected)))
    write_json(output/"age-bands.json",bands)
    colors={"B1":"#6b7075","B2":"#1684b5","B3":"#087f68","A1":"#bd7850","A2":"#805c9e"}
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),sharey=True)
    for ax,scene in zip(axes,("p3-holdout-slow","p3-holdout-rapid","p3-holdout-async")):
        for group,color in colors.items():
            rows=[r for r in summary["events"] if r["scene"]==scene and r["group"]==group]
            ax.plot([r["fraction"]*100 for r in rows],[r["precision"]*100 for r in rows],"o-",label=group,color=color)
        prevalence=next(r["prevalence"] for r in summary["events"] if r["scene"]==scene)
        ax.axhline(prevalence*100,color="#bbb",linestyle="--",label="Prevalence")
        ax.set_title(scene.replace("p3-holdout-",""));ax.set_xticks([5,10,20]);ax.set_xlabel("Selected old records (%)");ax.grid(alpha=.2)
    axes[0].set_ylabel("Review precision (%)");axes[-1].legend(fontsize=8)
    fig.suptitle("P3 linear screening / three held-out synthetic events / equal route-time weights")
    fig.tight_layout();fig.savefig(output/"ranking.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("source",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();publish(args.source,args.output)
