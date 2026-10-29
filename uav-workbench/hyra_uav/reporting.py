"""Scientific pilot plots and compact reviewable evidence, separate from run logs."""
import json
from pathlib import Path
import shutil
import numpy as np

from .core import FloodProcess
from .experiments import available_scenes, write_json


def report(pilot_folder, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pilot_folder, output = Path(pilot_folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((pilot_folder / "index.json").read_text(encoding="utf-8"))
    plan = json.loads((pilot_folder / "plan.json").read_text(encoding="utf-8"))
    successful = [r for r in rows if r["status"] == "completed"]
    aggregate = []
    for duration in plan["task_durations_s"]:
        for scene in ("slow", "rapid", "async"):
            for method in ("R0", "R1", "R2"):
                selected = [r for r in successful if r["duration_s"] == duration and r["scene"] == scene and r["method"] == method]
                aggregate.append(dict(duration_s=duration, scene=scene, method=method, completed=len(selected),
                    mean_quality=float(np.mean([r["time_average_correct_coverage"] for r in selected])) if selected else None,
                    mean_final_coverage=float(np.mean([r["final"]["first_coverage"] for r in selected])) if selected else None,
                    mean_final_error=float(np.mean([r["final"]["observed_error_rate"] for r in selected])) if selected else None))
    write_json(output / "pilot-summary.json", dict(plan=plan, runs=rows, aggregated=aggregate))
    colors = {"R0": "#95651d", "R1": "#1684b5", "R2": "#087f68"}
    for duration in plan["task_durations_s"]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex=True, sharey=True)
        for col, scene in enumerate(("slow", "rapid", "async")):
            for method in colors:
                selected = [r for r in successful if r["duration_s"] == duration and r["scene"] == scene and r["method"] == method]
                for row, key in enumerate(("correct_coverage", "first_coverage")):
                    curves = []
                    for r in selected:
                        data = json.loads((pilot_folder / r["run_id"] / "replay.json").read_text(encoding="utf-8"))
                        times = [s["time_s"] for s in data["snapshots"]]
                        curves.append([s["metrics"][key] for s in data["snapshots"]])
                    if curves:
                        axes[row, col].step(times, np.mean(curves, axis=0), where="post", color=colors[method], label=method)
            axes[0, col].set_title(scene+" / synthetic development event")
            axes[1, col].set_xlabel("Physical time (s)")
        axes[0, 0].set_ylabel("Class-balanced current correct coverage")
        axes[1, 0].set_ylabel("First coverage")
        for ax in axes.ravel():
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=.2)
        axes[0, 2].legend()
        fig.suptitle(f"HyRA v2 pilot / {duration}s budget / mean of 3 starts per event / no confidence claim")
        fig.tight_layout()
        fig.savefig(output / f"curves-{duration}s.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, scene in zip(axes, [s for s in available_scenes() if s["id"] in ("slow", "rapid", "async")]):
        process = FloodProcess(scene["path"])
        audit = process.audit()
        write_json(output / (scene["id"]+"-audit.json"), audit)
        ax.step(process.times, audit["flooded_area_m2"], where="post")
        ax.set_title(scene["id"])
        ax.set_xlabel("Physical time (s)")
        ax.set_ylabel("Flooded area (m²), depth >= 0.05m")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "flooded-area.png", dpi=160)
    plt.close(fig)
    for filename in ("plan.json", "index.json"):
        shutil.copyfile(pilot_folder / filename, output / filename)
    return aggregate
