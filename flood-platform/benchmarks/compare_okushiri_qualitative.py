"""Frozen whole-window qualitative audit of existing Okushiri laboratory runs.

No solver is executed. No time shift, offset correction or parameter search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

STATIONS = ("ch5", "ch7", "ch9")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_series(times, predicted, observed, reference):
    if times.shape != (451,) or predicted.shape != (451, 3) or observed.shape != (451, 3):
        raise ValueError("incomplete 451 x 3 time series")
    if not all(np.isfinite(v).all() for v in (times, predicted, observed)):
        raise ValueError("nonfinite time series")
    if np.max(np.abs(times-reference[:, 0])) > 1e-12:
        raise ValueError("time shift or missing/duplicate/reordered sample")
    error = float(np.max(np.abs(observed-reference[:, 1:]/100)))
    if error > 1e-12:
        raise ValueError("observation unit/value differs from original cm record")
    return error


def phase(times, values, start, stop, resolution, fraction=0.25):
    inside = (times >= start-1e-12) & (times <= stop+1e-12)
    early = inside & (times <= start + fraction*(stop-start)+1e-12)
    late = inside & (times >= stop - fraction*(stop-start)-1e-12)
    a, b = values[early], values[late]
    if not len(a) or not len(b):
        raise ValueError("missing edge block")
    low, high = float(b.min()-a.max()), float(b.max()-a.min())
    label = "rising" if low > resolution else "falling" if high < -resolution else "unresolved"
    difference = np.diff(values[inside])
    return {"label": label, "change_interval_m": [low, high],
            "first_block_m": [float(a.min()), float(a.max())],
            "last_block_m": [float(b.min()), float(b.max())],
            "sample_count": int(inside.sum()), "edge_counts": [len(a), len(b)],
            "endpoint_change_m": float(values[inside][-1]-values[inside][0]),
            "internal_reversal": bool(np.any(difference > 0) and np.any(difference < 0))}


def compare(a, b):
    if "unresolved" in (a, b):
        return "unknown"
    return "consistent" if a == b else "inconsistent"


def classify_windows(times, observed, predicted, width, resolution, fraction):
    rows = []; start = 0.0
    while start < 22.5-1e-12:
        stop = min(start+width, 22.5)
        for j, station in enumerate(STATIONS):
            a = phase(times, observed[:, j], start, stop, resolution, fraction)
            b = phase(times, predicted[:, j], start, stop, resolution, fraction)
            rows.append({"station": station, "start_s": start, "end_s": stop,
                         "observed": a, "model": b, "comparison": compare(a["label"], b["label"])})
        start = stop
    return rows


def peak_interval(times, values, resolution, step):
    peak = float(values.max())
    indices = np.flatnonzero(values >= peak-resolution/2)
    return {"max_m": peak, "near_max_times_s": times[indices].tolist(),
            "sample_support_s": [float(times[indices[0]]-step/2), float(times[indices[-1]]+step/2)]}


def order(a, b):
    if a[1] < b[0]-1e-12:
        return "first_before_second"
    if b[1] < a[0]-1e-12:
        return "second_before_first"
    return "unresolved"


def count_rows(rows):
    return {name: sum(r["comparison"] == name for r in rows)
            for name in ("consistent", "inconsistent", "unknown")}


def audit(protocol_path, base, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches

    output.mkdir(exist_ok=True, parents=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "hyra.okushiri.qualitative.v1":
        raise ValueError("unregistered protocol")
    for item in protocol["inputs"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("input changed since preregistration: " + item["path"])
    raw = np.loadtxt(base/"round39/output_ch5-7-9.txt", skiprows=1)
    if raw.shape != (3992,4) or not np.isfinite(raw).all():
        raise ValueError("unexpected original observation archive")
    reference = raw[(raw[:,0] >= 0) & (raw[:,0] <= 22.5)]
    if reference.shape != (451,4):
        raise ValueError("missing original common-window sample")
    rules = protocol["phase_rules"]; resolution = rules["resolution_m"]
    width = rules["primary_window_s"]; fraction=rules["edge_block_fraction"]
    cases = [protocol["primary_case"], *protocol["diagnostic_cases"]]
    results = []; heat = []; heat_labels = []
    for case in cases:
        directory = base/case
        with np.load(directory/"timeseries.npz", allow_pickle=False) as data:
            t, observed, predicted = [data[key].copy() for key in ("time_s", "observed_m", "predicted_m")]
        obs_error = validate_series(t,predicted,observed,reference)
        rows = classify_windows(t,observed,predicted,width,resolution,fraction)
        sensitivities=[]
        for step in rules["sensitivity_windows_s"]:
            rr = classify_windows(t,observed,predicted,step,resolution,fraction)
            sensitivities.append({"window_s":step,"counts":count_rows(rr),"rows":rr})
        obs_peaks={}; model_peaks={}
        for j,s in enumerate(STATIONS):
            obs_peaks[s]=peak_interval(t,observed[:,j],resolution,protocol["sample_step_s"])
            model_peaks[s]=peak_interval(t,predicted[:,j],resolution,protocol["sample_step_s"])
        pairs=[]
        for first,second in protocol["peak_order"]["all_pairs"]:
            a=order(obs_peaks[first]["sample_support_s"],obs_peaks[second]["sample_support_s"])
            b=order(model_peaks[first]["sample_support_s"],model_peaks[second]["sample_support_s"])
            pairs.append({"pair":[first,second],"observed_order":a,"model_order":b,"comparison":compare(a,b)})
        complete = rows+pairs
        result={"case":case,"role":"primary_physical_domain" if case==cases[0] else "diagnostic_only",
                "configuration":json.loads((directory/"configuration.json").read_text()),
                "original_observation_max_difference_m":obs_error,
                "phase_counts":count_rows(rows),"station_counts":{s:count_rows([r for r in rows if r["station"]==s]) for s in STATIONS},
                "phase_rows":rows,"peak_order_rows":pairs,"peak_order_counts":count_rows(pairs),
                "observed_peaks":obs_peaks,"model_peaks":model_peaks,"sensitivity":sensitivities,
                "screening_all_rows_consistent":all(r["comparison"]=="consistent" for r in complete),
                "laboratory_accuracy_certified":False,
                "no_other_qualitative_claim":"No spatial inundation, local depth, current direction, route or UAV validation from three stage traces",
                "unchanged_quantitative_rmse_mm":(np.sqrt(np.mean((predicted-observed)**2,axis=0))*1000).tolist()}
        results.append(result)
        for station in STATIONS:
            heat.append([{"consistent":0,"unknown":1,"inconsistent":2}[r["comparison"]] for r in rows if r["station"]==station])
            heat_labels.append(case.split("/")[0]+" "+station)
        if case==cases[0]:
            fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True,layout="constrained")
            for j,ax in enumerate(axs):
                ax.plot(t,observed[:,j]*1000,label="Laboratory observation",color="black",lw=1.3)
                ax.plot(t,predicted[:,j]*1000,label="HYRA exact-domain 6 mm",color="#1568a3",lw=1.1)
                for r in rows:
                    if r["station"]==STATIONS[j] and r["comparison"]!="consistent":
                        ax.axvspan(r["start_s"],r["end_s"],alpha=.16,color="#df4c35" if r["comparison"]=="inconsistent" else "#8b8b8b")
                ax.set(ylabel=STATIONS[j]+" elevation (mm)");ax.grid(alpha=.15)
            axs[0].legend(fontsize=9,ncol=2)
            axs[-1].set(xlabel="Laboratory time (s); no time or level shift",xlim=(0,22.5))
            fig.suptitle("Fixed 1 s net-trend audit: red = opposite; grey = unresolved (never a pass)")
            fig.savefig(output/"primary-full-curves.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5),layout="constrained")
    edges=np.r_[np.arange(23),22.5]
    ax.pcolormesh(edges,np.arange(10),np.asarray(heat),cmap=ListedColormap(["#86cda0","#d4d4d4","#e66852"]),vmin=0,vmax=2,edgecolors="white",linewidth=.5)
    ax.set_yticks(np.arange(9)+.5,heat_labels);ax.invert_yaxis()
    ax.set(xlabel="Fixed laboratory time window (s)",title="All three gauges retained; offset-domain cases are diagnostics only")
    ax.legend(handles=[patches.Patch(color=c,label=n) for c,n in [("#86cda0","Same resolved trend"),("#d4d4d4","Unresolved; not passed"),("#e66852","Opposite resolved trends")]],loc="upper center",bbox_to_anchor=(.5,-.12),ncol=3)
    fig.savefig(output/"all-window-classifications.png",dpi=150);plt.close(fig)
    dump(output/"comparison.json",{"protocol_sha256":sha(protocol_path),"source_full_observation_end_s":float(raw[-1,0]),"cases":results})
    lines=["# Okushiri固定时窗定性对照", "", "实验室缩尺证据；同一0–22.5s时窗，全三站保留，无移时或去偏置。1s是预登记报告尺度，记录舍入带不是仪器误差，不能外推为现场认证。", "", "|配置|1s同趋势|反向趋势|无法判断|主峰先后同/反/未知|全部一致|", "|---|---:|---:|---:|---|---|"]
    for r in results:
        c=r["phase_counts"];p=r["peak_order_counts"]
        lines.append(f'|{r["case"]}|{c["consistent"]}|{c["inconsistent"]}|{c["unknown"]}|{p["consistent"]}/{p["inconsistent"]}/{p["unknown"]}|{r["screening_all_rows_consistent"]}|')
    for r in results:
        lines += ["", "## "+r["case"], "", "|测站/时窗s|观测类别|模拟类别|观测变化区间mm|模拟变化区间mm|对照|", "|---|---|---|---|---|---|"]
        for row in r["phase_rows"]:
            o=row["observed"];m=row["model"]
            interval=lambda z: f'{1000*z[0]:.3f}…{1000*z[1]:.3f}'
            lines.append(f'|{row["station"]} {row["start_s"]:g}–{row["end_s"]:g}|{o["label"]}|{m["label"]}|{interval(o["change_interval_m"])}|{interval(m["change_interval_m"])}|{row["comparison"]}|')
    (output/"all-rows.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    dump(output/"manifest.json",{"script_sha256":sha(__file__),"protocol_sha256":sha(protocol_path),
                                 "files":[{"path":str(f),"sha256":sha(f)} for f in output.iterdir() if f.is_file() and f.name!="manifest.json"]})
    return [{"case":r["case"],"phase_counts":r["phase_counts"],"peak_order_counts":r["peak_order_counts"],"all_consistent":r["screening_all_rows_consistent"]} for r in results]


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--base",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.protocol,args.base,args.output),ensure_ascii=False,indent=2))
