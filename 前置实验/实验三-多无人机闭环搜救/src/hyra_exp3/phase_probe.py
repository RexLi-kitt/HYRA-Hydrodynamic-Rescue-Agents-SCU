"""Unfrozen mission-phase sensitivity probe using independent physical burn-in."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .artifacts import finish_run, new_run, write_json
from .design import design_jobs, target_opportunities
from .diagnostics import load_field
from .protocol import initialize
from .simulation import stratified_starts


def mission_view(field, start_s, duration_s=1200.):
    times = np.asarray(field["times_s"], float)
    indices = np.flatnonzero((times >= start_s - 1e-9) & (times <= start_s + duration_s + 1e-9))
    if len(indices) < 2 or not np.isclose(times[indices[0]], start_s) or not np.isclose(times[indices[-1]], start_s + duration_s):
        raise ValueError("Mission phase endpoints require actual saved physical outputs, without padding")
    return {**field, "times_s": times[indices] - start_s, "depths_m": field["depths_m"][indices]}


def geometric_upper_bound(targets, map_shape, n_drones, speed_mps=5., cell_size_m=10.):
    """Optimistic individual feasibility, with target locations known only for diagnosis.

    Ignores search, route coupling and payload allocation; not a jointly feasible
    oracle policy and never used to select targets or guide the planner.
    """
    starts = np.array(stratified_starts(map_shape, n_drones), float)
    count, any_window = 0, 0
    for row in targets:
        intervals = row["truth_safe_start_intervals_s"]
        any_window += bool(intervals)
        travel = float(np.min(np.linalg.norm(starts - np.array(row["cell"]), axis=1))) * cell_size_m / speed_mps
        count += any(max(a, travel) <= b + 1e-8 for a, b in intervals)
    return dict(targets_with_any_complete_window=int(any_window), optimistic_individually_feasible_targets=int(count),
                ignores_search_joint_routes_and_resource_coupling=True)


def run_probe(root):
    from .pipeline import _episode, _generate
    root = Path(root)
    protocol = initialize(root)
    job = next(j for j in design_jobs(protocol) if j["side"] == 50)
    phases = list(range(0, 901, 150))
    specification = dict(kind="unfrozen_development_variant", formal_test=False, original_protocol_changed=False,
                         side=50, family=job["family"], rise_m=.9, rise_duration_s=600., perturbation_m=.1,
                         mission_start_physical_s=phases, mission_duration_s=1200., physical_end_s=2100.,
                         delays_s=[0., 10.], layout_seeds=job["layout_seeds"],
                         selection="equally_spaced_phase_grid_fixed_before_any_P_model_or_probe_results")
    run = new_run(root, "phase_probe", specification)
    write_json(run / "phase_spec.json", specification)
    print(f"Phase probe directory: {run}", flush=True)
    started = time.perf_counter()
    try:
        common = dict(duration_s=2100., output_interval_s=2., mesh_spacing_m=10.,
                      family_seed=job["family"]["family_seed"], rise_m=.9, rise_duration_s=600., perturbation_m=.1)
        truth_record, full_truth = _generate(run, 50, "truth", nominal=False, **common)
        nominal_record, full_nominal = _generate(run, 50, "nominal", nominal=True, **common)
        write_json(run / "physical_fields.json", dict(truth=truth_record, nominal=nominal_record))
        records = []
        for phase in phases:
            truth, nominal = mission_view(full_truth, phase), mission_view(full_nominal, phase)
            for layout_index, layout_seed in enumerate(job["layout_seeds"]):
                for delay in (0., 10.):
                    directory = run / f"phase{phase}" / f"layout{layout_index}_delay{delay:g}"
                    directory.mkdir(parents=True)
                    episode = _episode(directory, truth, nominal, delay, layout_seed)
                    targets = target_opportunities(truth, episode)
                    bounds = geometric_upper_bound(targets, (50, 50), 10)
                    completed = episode["summary"]["successful_services"]
                    if completed > bounds["optimistic_individually_feasible_targets"]:
                        raise AssertionError("Actual services exceed the individual geometry bound")
                    row = dict(phase_s=phase, layout_index=layout_index, layout_seed=layout_seed, delay_s=delay,
                               summary=episode["summary"], targets=targets, bounds=bounds,
                               optimistic_extra_service_upper_bound=bounds["optimistic_individually_feasible_targets"] - completed,
                               physical_to_mission_clock="physical_time_s = mission_time_s + phase_s",
                               pre_mission_truth_observations_given_to_planner=False)
                    write_json(directory / "phase_diagnosis.json", row)
                    records.append(row)
            print(f"Phase {phase}s complete: " + ", ".join(str(r['summary']['successful_services']) for r in records[-4:]) + " /30", flush=True)
        groups = []
        for phase in phases:
            for delay in (0., 10.):
                rows = [r for r in records if r["phase_s"] == phase and r["delay_s"] == delay]
                groups.append(dict(phase_s=phase, delay_s=delay, episodes=len(rows),
                    mean_completed=float(np.mean([r["summary"]["successful_services"] for r in rows])),
                    mean_any_window=float(np.mean([r["bounds"]["targets_with_any_complete_window"] for r in rows])),
                    mean_individual_upper_bound=float(np.mean([r["bounds"]["optimistic_individually_feasible_targets"] for r in rows])),
                    optimistic_extra_service_upper_bound=float(np.mean([r["optimistic_extra_service_upper_bound"] for r in rows])),
                    failed_service_attempts=sum(r["summary"]["failed_service_attempts"] for r in rows)))
        summary = dict(specification=specification, episodes=len(records), grouped=groups,
                       formal_test=False, global_phase_selected=False, trained_model_comparison=False)
        write_json(run / "phase_summary.json", summary)
        finish_run(root, run, elapsed_wall_s=time.perf_counter() - started)
        write_json(root / "results" / "latest_phase_probe.json", dict(run=str(run), summary=summary))
        _report(root, run, summary)
        return run
    except BaseException as error:
        import traceback
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(root, run, status="failed", error=repr(error), elapsed_wall_s=time.perf_counter() - started)
        raise


def _report(root, run, summary):
    lines = ["# 任务启动水情阶段：开发候选诊断", "",
             "本诊断在原始设计集之外另立开发变体，原方案未修改、未冻结任何新阶段。仅检查一个设计家族、2500格和一种水情，不能代表全部家族。", "",
             "阶段为水动力从初始状态先独立积分0／150／300／450／600／750／900秒后再派出无人机；每次任务仍完整运行1200秒。真值与名义场分别积分至2100秒，按实际输出截取，没有把末帧复制到未来。无人机没有获得任务开始前的真值观测。", "",
             "7个等间隔阶段在执行前固定；每阶段两套相同布局／噪声和0／10秒延迟，共28回合。场景为0.9米涨幅、600秒历时、0.1米扰动；仍使用名义预测与贪心规划，未训练P，也未按P表现选阶段。", "",
             "| 水动力先行时间 | 延迟 | 平均完成／30 | 存在任意完整窗口的目标 | 个体几何可行上界 | 乐观可挽回目标上界 | 失败尝试总数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["grouped"]:
        lines.append(f"| {row['phase_s']}s | {row['delay_s']:g}s | {row['mean_completed']:.1f} | {row['mean_any_window']:.1f} | "
                     f"{row['mean_individual_upper_bound']:.1f} | {row['optimistic_extra_service_upper_bound']:.1f} | {row['failed_service_attempts']} |")
    lines += ["", "上界是在执行后使用真值及目标位置计算的诊断，不参与目标抽样或规划。它忽略搜索、联合路径、机群分工与资源耦合，因此不保证任何方法能实现该额外完成数。", "",
              "应同时检查任务是否还有可完成窗口、名义基准是否出现损失，以及损失是否存在理论挽回空间。完成率下降本身不能作为有效设计的证据。", "",
              "后续若采用某个候选阶段，需要在全部独立设计家族、涨水条件和尺度中重新核验并登记新协议；本小样本诊断没有选择正式条件或开启测试集。", "",
              f"原始运行：`{run}`；固定候选为 `phase_spec.json`，汇总为 `phase_summary.json`，逐目标诊断为每回合的 `phase_diagnosis.json`。", ""]
    path = root / "docs" / "results" / "20261018_任务启动阶段候选诊断.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Phase probe report: {path}", flush=True)
