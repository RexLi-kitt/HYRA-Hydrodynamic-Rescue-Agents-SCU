"""Archive P2 scale selection evidence and state/observation sensitivity plots."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_uav.core import FloodProcess
from hyra_uav.experiments import FIXTURES,write_json


def publish(source,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    for name in ("plan.json","decision.json","index.json","comparison.json","audits.json"):
        shutil.copyfile(source/name,output/name)
    rows=json.loads((source/"index.json").read_text(encoding="utf-8"))
    sensitivity=[]
    for r in rows:
        if r["refinement"]==1 and r["output_s"]==5:continue
        base=next(b for b in rows if b["mechanism"]==r["mechanism"] and b["method"]==r["method"] and b["start"]==r["start"] and b["refinement"]==1 and b["output_s"]==5)
        sensitivity.append(dict(run_id=r["run_id"],quality_delta=r["time_average_correct_coverage"]-base["time_average_correct_coverage"],
                                first_coverage_delta=r["final"]["first_coverage"]-base["final"]["first_coverage"]))
    write_json(output/"policy-sensitivity.json",sensitivity)
    fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True,sharey=True)
    colors={"R0":"#95651d","R1":"#1684b5","R2":"#087f68"}
    for row,(refinement,interval) in enumerate(((1,5),(2,5),(1,2.5))):
        for col,mechanism in enumerate(("slow","rapid","async")):
            ax=axes[row,col]
            for method,color in colors.items():
                curves=[]
                for entry in rows:
                    if (entry["refinement"],entry["output_s"],entry["mechanism"],entry["method"])!=(refinement,interval,mechanism,method):continue
                    replay=json.loads((source/"runs"/entry["run_id"]/"replay.json").read_text(encoding="utf-8"))
                    times=[s["time_s"] for s in replay["snapshots"]]
                    curves.append([s["metrics"]["correct_coverage"] for s in replay["snapshots"]])
                ax.step(times,np.mean(curves,axis=0),where="post",color=color,label=method)
            ax.set_title(f"{mechanism} / dx={5/refinement:g}m / sample={interval:g}s");ax.grid(alpha=.2);ax.set_ylim(0,1)
            if col==0:ax.set_ylabel("Class-balanced correct coverage")
            if row==2:ax.set_xlabel("Physical time (s)")
    axes[0,2].legend();fig.suptitle("P2 scale audit / 81 runs / mean of 3 starts per event / synthetic scope")
    fig.tight_layout();fig.savefig(output/"scale-curves.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(9,9))
    for row,mechanism in enumerate(("slow","rapid","async")):
        process=FloodProcess(FIXTURES/mechanism/"map-result.zip")
        for col,index in enumerate((0,60,120)):
            ax=axes[row,col];ax.imshow(process.labels[index],cmap="Blues",vmin=0,vmax=1,origin="upper",extent=(-2.5,157.5,157.5,-2.5))
            ax.set_title(f"{mechanism} / {process.times[index]:g}s");ax.set_xlabel("x (m)");ax.set_ylabel("z (m)")
    fig.tight_layout();fig.savefig(output/"water-boundaries.png",dpi=140);plt.close(fig)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("source",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();publish(args.source,args.output)
