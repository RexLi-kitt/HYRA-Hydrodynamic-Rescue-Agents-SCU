"""Plot post-execution service timing against complete-window deadlines."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def plot(source):
    groups = {}
    for path in sorted((source / "jobs").glob("*/condition_*/case_summary.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        for episode in case["episodes"]:
            bucket = groups.setdefault((case["side"], episode["delay_s"]), [])
            for target in episode["targets"]:
                start, margin = target["service_start_s"], target["safe_start_margin_after_actual_service_s"]
                if start is not None and margin is not None:
                    bucket.append((start, start + margin))
    if not groups:
        raise ValueError("No completed service-window diagnostics")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    cjk = font is not None
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, key in zip(axes.ravel(), ((32, 0.), (50, 0.), (50, 10.), (70, 0.))):
        data = np.asarray(groups.get(key, []), float).reshape(-1, 2)
        x = np.array([0, 1200.])
        axis.fill_between(x, 0, x, color="#f2e5dd", alpha=.7)
        axis.plot(x, x, color="#9a9690", linewidth=.8, linestyle="--")
        if len(data):
            axis.scatter(data[:, 0], data[:, 1], s=4, alpha=.12, color="#276e65", linewidths=0, rasterized=True)
            median = float(np.median(data[:, 1] - data[:, 0]))
            note = f"n={len(data):,}\n" + (f"余量中位数 {median:.0f}秒" if cjk else f"Median margin {median:.0f}s")
            axis.text(.97, .06, note, transform=axis.transAxes, ha="right", va="bottom", fontproperties=font)
        title = f"{key[0] ** 2}格 / 延迟{key[1]:g}秒" if cjk else f"{key[0] ** 2} cells / {key[1]:g}s delay"
        axis.set_title(title, fontproperties=font)
        axis.set_xlim(0, 1200)
        axis.set_ylim(0, 1200)
        axis.grid(alpha=.15)
        axis.set_xlabel("实际服务开始时间（秒）" if cjk else "Actual service start (s)", fontproperties=font)
        axis.set_ylabel("所在窗口最后合法开始时间（秒）" if cjk else "Last safe start in containing window (s)", fontproperties=font)
    figure.suptitle("设计集：实际服务与完整窗口的时序关系" if cjk else "Design set: service starts and complete-window deadlines",
                    fontproperties=font, fontsize=15)
    destination = ROOT / "docs" / "results" / "设计集服务时序.png"
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    plot(parser.parse_args().source_run.resolve())
