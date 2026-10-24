"""Event-driven continuous-flight engineering benchmark with causal shared sensing.

Physical execution is shared by explicitly supplied information modules and
controllers. The default controller remains the nominal-only preflight rule.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
import time

import numpy as np

from .windows import NominalPredictor
from .noise import world_noise


@dataclass
class EpisodeConfig:
    map_shape: tuple[int, int] = (50, 50)
    n_drones: int = 10
    n_targets: int = 30
    duration_s: float = 1200.
    cell_size_m: float = 10.
    speed_mps: float = 5.
    sensor_period_s: float = 5.
    sensor_radius_cells: int = 2
    planning_period_s: float = 10.
    service_duration_s: float = 12.
    max_service_depth_m: float = .8
    observation_noise_m: float = .02
    observation_delay_s: float = 0.
    payload: int = 4
    seed: int = 31001
    forecast_horizon_s: float = 60.
    planner_budget_s: float = .1
    retain_raw_observations: bool = True
    noise_mode: str = "legacy_stream"
    reassignment_policy: str = "full"

    def __post_init__(self):
        if self.noise_mode not in ("legacy_stream", "counter_keyed") or self.reassignment_policy not in ("full", "defer_new_evidence"):
            raise ValueError("Unknown observation noise or reassignment policy")
        if len(self.map_shape) != 2 or min(self.map_shape) <= 0:
            raise ValueError("Invalid map shape")
        if self.n_drones <= 0 or self.n_targets < 0 or self.payload < 0 or self.sensor_radius_cells < 0:
            raise ValueError("Invalid fleet/target/sensor count")
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.duration_s, self.cell_size_m, self.speed_mps, self.sensor_period_s,
                self.planning_period_s, self.service_duration_s, self.forecast_horizon_s, self.planner_budget_s)):
            raise ValueError("Invalid clock or geometry")
        if not math.isfinite(self.observation_delay_s) or self.observation_delay_s < 0 or self.observation_noise_m < 0:
            raise ValueError("Invalid observation noise or delay")


class DepthField:
    def __init__(self, times, depths):
        self.times, self.depths = np.asarray(times, float), np.asarray(depths)
        if self.depths.ndim != 3 or len(self.depths) != len(self.times) or len(self.times) < 2:
            raise ValueError("Expected depths[T,H,W]")
        if not np.all(np.isfinite(self.depths)) or not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0):
            raise ValueError("Invalid field")

    def at(self, now, cells):
        if not self.times[0] - 1e-9 <= now <= self.times[-1] + 1e-9:
            raise ValueError("No extrapolation of truth")
        upper = min(max(int(np.searchsorted(self.times, now, side="right")), 1), len(self.times) - 1)
        lower = upper - 1
        fraction = (now - self.times[lower]) / (self.times[upper] - self.times[lower])
        cells = np.asarray(cells, int).reshape(-1, 2)
        # Evaluate the saved piecewise-linear field in float64, like the
        # independent service auditor; float32 scalar promotion can otherwise
        # change a strict 0.8m decision at a fractional service endpoint.
        return ((1 - fraction) * self.depths[lower, cells[:, 0], cells[:, 1]].astype(float) +
                fraction * self.depths[upper, cells[:, 0], cells[:, 1]].astype(float))

    def service_succeeds(self, cell, start, duration, threshold):
        end = start + duration
        if start < self.times[0] or end > self.times[-1] + 1e-9:
            return False
        interior = self.times[(self.times > start) & (self.times < end)]
        values = [float(self.at(t, [cell])[0]) for t in np.r_[start, interior, end]]
        return max(values) <= threshold


def stratified_starts(shape, count):
    height, width = shape
    columns = int(math.ceil(math.sqrt(count * width / height)))
    rows = int(math.ceil(count / columns))
    cells = []
    for row in range(rows):
        n = min(columns, count - len(cells))
        for column in range(n):
            cells.append((min(height - 1, int((row + .5) * height / rows)),
                          min(width - 1, int((column + .5) * width / n))))
    return cells


def run_episode(truth_depths, config, *, times=None, nominal_depths=None, targets=None,
                starts=None, predictor=None, accessible=None, record_events=True, planner=None):
    c = config
    truth = np.asarray(truth_depths)
    if truth.shape[1:] != tuple(c.map_shape):
        raise ValueError("Map and field shapes differ")
    if times is None:
        raise ValueError("Explicit hydraulic output times required")
    field = DepthField(times, truth)
    if field.times[0] > 0 or field.times[-1] < c.duration_s - 1e-9:
        raise ValueError("Truth does not cover the full episode")
    if predictor is None:
        if nominal_depths is None:
            raise ValueError("Independent nominal forecast required; truth cannot stand in for prediction")
        predictor = NominalPredictor(times, nominal_depths, horizon_s=c.forecast_horizon_s,
                                     service_duration_s=c.service_duration_s, threshold_m=c.max_service_depth_m)
    height, width = c.map_shape
    allowed = np.ones(c.map_shape, bool) if accessible is None else np.asarray(accessible, bool)
    if allowed.shape != tuple(c.map_shape):
        raise ValueError("Invalid accessibility map")
    # Target selection uses geometry only, before any sensing or prediction.
    layout_rng = np.random.default_rng(np.random.SeedSequence([c.seed, 1]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([c.seed, 2]))
    if targets is None:
        candidates = np.argwhere(allowed)
        if c.n_targets > len(candidates):
            raise ValueError("Too many distinct targets")
        targets = candidates[layout_rng.choice(len(candidates), c.n_targets, replace=False)]
    targets = [tuple(map(int, cell)) for cell in targets]
    if len(targets) != c.n_targets or len(set(targets)) != len(targets):
        raise ValueError("Target count/uniqueness mismatch")
    if any(not (0 <= r < height and 0 <= col < width) or not allowed[r, col] for r, col in targets):
        raise ValueError("Inaccessible target")
    starts = stratified_starts(c.map_shape, c.n_drones) if starts is None else [tuple(map(int, cell)) for cell in starts]
    if len(starts) != c.n_drones or any(not (0 <= r < height and 0 <= col < width) or not allowed[r, col] for r, col in starts):
        raise ValueError("Invalid start geometry")
    def xy(cell):
        return np.array([(cell[1] + .5) * c.cell_size_m, (cell[0] + .5) * c.cell_size_m], float)
    robots = [dict(id=i, position=xy(cell), mode="idle", destination=None, target=None,
                   payload=c.payload, service_start=None, service_end=None, planned_start=None,
                   distance_m=0., prediction_version=None) for i, cell in enumerate(starts)]
    target_rows = [dict(id=i, cell=list(cell), discovered_at=None, first_sensed_at=None,
                        served_at=None, attempts=0, failures=0, planned_versions=[]) for i, cell in enumerate(targets)]
    target_by_cell = {cell: i for i, cell in enumerate(targets)}
    known = set()
    covered = np.zeros(c.map_shape, bool)
    visits = np.zeros(c.map_shape, np.int32)
    histories = {}
    history_limit = (int(math.ceil(60. / c.sensor_period_s) + 1) * c.n_drones
                     if getattr(predictor, "requires_full_raw_history", False) else 12)
    observation_queue = []
    raw_observations, events, trajectories = [], [], []
    queue_seq, sensor_batches, arrived_count, plan_timeouts, reassignments = 0, 0, 0, 0, 0
    planning_latencies = []
    now, last, next_sensor, next_plan = 0., 0., 0., 0.
    started_wall = time.perf_counter()
    def log(kind, **data):
        if record_events:
            events.append(dict(time_s=float(now), event=kind, **data))

    def assign(robot, cell, target_id=None, service_start=None, version=None):
        robot.update(destination=xy(cell), target=target_id, planned_start=service_start,
                     prediction_version=version, mode="fly")

    def plan(reason):
        nonlocal plan_timeouts, reassignments
        plan_start = time.perf_counter()
        recent = [row for history in histories.values() for row in history if row["sample_time"] >= now - 60.]
        predictor.issue(now, recent)
        free = [r for r in robots if r["mode"] != "service"]
        state_at_plan = [dict(drone=r["id"],position_m=r["position"].tolist(),payload=r["payload"],
            mode=r["mode"],target=r["target"],planned_start=r["planned_start"]) for r in robots]
        busy_targets = {r["target"] for r in robots if r["mode"] == "service"}
        task_ids = [i for i in sorted(known) if target_rows[i]["served_at"] is None and i not in busy_targets]
        def legacy_proposals():
            proposals = {}
            estimates = {i: predictor.estimate(targets[i], now) for i in task_ids}
            remaining = set(r["id"] for r in free if r["payload"] > 0)
            pending = set(task_ids)
            while remaining and pending:
                best = None
                for rid in sorted(remaining):
                    robot = robots[rid]
                    for tid in sorted(pending):
                        estimate = estimates[tid]
                        travel = float(np.linalg.norm(robot["position"] - xy(targets[tid]))) / c.speed_mps
                        start = estimate.earliest_start(now + travel)
                        if start is None or start + c.service_duration_s > c.duration_s + 1e-9:
                            continue
                        close = next(b for a, b in estimate.service_intervals if a - 1e-9 <= start <= b + 1e-9)
                        score = (close, start + c.service_duration_s, travel, rid, tid)
                        if best is None or score < best[0]:
                            best = (score, rid, tid, start, estimate.version)
                if best is None:
                    break
                _, rid, tid, start, version = best
                proposals[rid] = (targets[tid], tid, start, version)
                pending.remove(tid)
                remaining.remove(rid)
            # O(cells) integral image computes novel 5x5 footprint gain, without a
            # dense cell-to-cell matrix. At most O(drones*cells) coordinate work.
            unseen = (~covered & allowed).astype(np.int32)
            radius = c.sensor_radius_cells
            padded = np.pad(unseen, radius)
            integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            size = 2 * radius + 1
            gains = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
            candidate_cells = np.argwhere((gains > 0) & allowed)
            candidate_xy = np.column_stack(((candidate_cells[:, 1] + .5) * c.cell_size_m,
                                            (candidate_cells[:, 0] + .5) * c.cell_size_m))
            reserved = []
            for robot in free:
                rid = robot["id"]
                if rid in proposals or not len(candidate_cells):
                    continue
                # Zero-payload UAVs can continue useful exploration.
                distance = np.linalg.norm(candidate_xy - robot["position"], axis=1)
                scores = gains[candidate_cells[:, 0], candidate_cells[:, 1]] / (1 + distance / c.speed_mps)
                for position in reserved:
                    scores[np.linalg.norm(candidate_xy - position, axis=1) < size * c.cell_size_m] = -1
                index = int(np.argmax(scores))
                if scores[index] > 0:
                    cell = tuple(map(int, candidate_cells[index]))
                    proposals[rid] = (cell, None, None, None)
                    reserved.append(candidate_xy[index])
            return proposals
        if planner is None:
            proposals = legacy_proposals()
        if planner is not None:
            # Explicit information boundary: no truth, undiscovered target list,
            # or not-yet-arrived sensing queue is passed to the controller.
            proposals = planner(now=now,
                robots=[dict(id=r["id"], position=r["position"].copy(), payload=r["payload"]) for r in free],
                known_targets={tid: tuple(targets[tid]) for tid in task_ids},
                covered=covered.copy(), allowed=allowed.copy(), predictor=predictor, config=c)
        interventions = []
        # The intervention delays only a new-evidence switch of a currently
        # feasible target until the next ordinary periodic planning event.
        if c.reassignment_policy == "defer_new_evidence" and "periodic" not in reason:
            held = {}
            for robot in free:
                tid = robot["target"]
                proposal = proposals.get(robot["id"])
                if tid is None or tid in busy_targets or robot["payload"] <= 0 or target_rows[tid]["served_at"] is not None:
                    continue
                if proposal is not None and proposal[1] == tid:
                    continue
                eta = now + float(np.linalg.norm(robot["position"] - xy(targets[tid]))) / c.speed_mps
                start = max(eta, robot["planned_start"] if robot["planned_start"] is not None else eta)
                estimate = predictor.estimate(targets[tid], now)
                if estimate.can_start(start) is True and start + c.service_duration_s <= c.duration_s + 1e-9:
                    held[robot["id"]] = (targets[tid], tid, start, estimate.version)
                    interventions.append(dict(drone=robot["id"], held_target=tid,
                        displaced_target=proposal[1] if proposal else None))
            held_targets = {p[1] for p in held.values()}
            proposals = {rid:p for rid,p in proposals.items() if rid in held or p[1] not in held_targets}
            proposals.update(held)
            if interventions and planner is not None:
                diagnostics = dict(planner.last_diagnostics)
                diagnostics["unconstrained_selected_routes"] = diagnostics.get("selected_routes", [])
                diagnostics["allocation_intervened"] = True
                diagnostics["selected_routes"] = [dict(drone=robot["id"], targets=[proposals[robot["id"]][1]],
                    starts_s=[proposals[robot["id"]][2]], finish_s=proposals[robot["id"]][2] + c.service_duration_s,
                    version=proposals[robot["id"]][3], initial_position_m=robot["position"].tolist(), payload=robot["payload"])
                    for robot in free if robot["id"] in proposals and proposals[robot["id"]][1] is not None]
                planner.last_diagnostics = diagnostics
        elapsed = time.perf_counter() - plan_start
        planning_latencies.append(elapsed * 1000.)
        if elapsed > c.planner_budget_s:
            plan_timeouts += 1
            # Continue only currently feasible target plans or still-useful
            # exploration; invalid plans stop and await the next replan.
            for robot in free:
                tid = robot["target"]
                feasible = robot["destination"] is not None
                if tid is not None and feasible:
                    estimate = predictor.estimate(targets[tid], now)
                    eta = now + float(np.linalg.norm(robot["position"] - robot["destination"])) / c.speed_mps
                    feasible = estimate.can_start(max(eta, robot["planned_start"] or eta)) is True
                if not feasible:
                    robot.update(mode="idle", destination=None, target=None)
            log("planning_timeout", reason=reason, version=predictor.version, planning_state=state_at_plan,
                elapsed_ms=elapsed * 1000.)
            return
        changes = 0
        change_details = []
        for robot in free:
            previous = robot["target"]
            proposal = proposals.get(robot["id"])
            if previous is not None and (proposal is None or proposal[1] != previous):
                changes += 1
                change_details.append(dict(drone=robot["id"], previous_target=previous,
                    next_target=proposal[1] if proposal else None, previous_version=robot["prediction_version"],
                    next_version=proposal[3] if proposal else None))
            if proposal is None:
                robot.update(mode="idle", destination=None, target=None, planned_start=None)
            else:
                cell, tid, start, version = proposal
                assign(robot, cell, tid, start, version)
                if tid is not None:
                    target_rows[tid]["planned_versions"].append(version)
        reassignments += changes
        log("plan", reason=reason, version=predictor.version, task_assignments=sum(p[1] is not None for p in proposals.values()),
            reassignments=changes, reassignment_details=change_details, policy_interventions=interventions,
            planning_state=state_at_plan, elapsed_ms=elapsed * 1000.,
            **(dict(planner_diagnostics=planner.last_diagnostics) if planner is not None else {}))

    while now <= c.duration_s + 1e-9:
        delta = now - last
        for robot in robots:
            if robot["mode"] == "fly":
                vector = robot["destination"] - robot["position"]
                distance = float(np.linalg.norm(vector))
                advance = min(distance, c.speed_mps * delta)
                if distance > 1e-10:
                    robot["position"] += vector * (advance / distance)
                    robot["distance_m"] += advance
                if distance - advance <= 1e-7:
                    robot["position"] = robot["destination"].copy()
                    if robot["target"] is None:
                        robot.update(mode="idle", destination=None)
                    else:
                        robot["mode"] = "wait"
                        log("target_arrival", drone=robot["id"], target=robot["target"], position_m=robot["position"].tolist())
        failed = False
        for robot in robots:
            if robot["mode"] == "service" and robot["service_end"] <= now + 1e-9:
                tid = robot["target"]
                success = field.service_succeeds(targets[tid], robot["service_start"], c.service_duration_s, c.max_service_depth_m)
                row = target_rows[tid]
                if success:
                    row["served_at"] = float(now)
                    robot["payload"] -= 1
                else:
                    row["failures"] += 1
                    failed = True
                log("service_complete", drone=robot["id"], target=tid, start_s=robot["service_start"],
                    success=bool(success), payload_remaining=robot["payload"], prediction_version=robot["prediction_version"])
                robot.update(mode="idle", destination=None, target=None, service_start=None, service_end=None)
            if robot["mode"] == "wait" and robot["planned_start"] <= now + 1e-9:
                tid = robot["target"]
                if target_rows[tid]["served_at"] is None and now + c.service_duration_s <= c.duration_s + 1e-9:
                    robot.update(mode="service", service_start=float(now), service_end=now + c.service_duration_s)
                    target_rows[tid]["attempts"] += 1
                    log("service_start", drone=robot["id"], target=tid, prediction_version=robot["prediction_version"])
                else:
                    robot.update(mode="idle", destination=None, target=None)
        if now >= next_sensor - 1e-9:
            for robot in robots:
                col, row = np.floor(robot["position"] / c.cell_size_m).astype(int)
                row, col = int(min(height - 1, max(0, row))), int(min(width - 1, max(0, col)))
                cells = [(r, q) for r in range(max(0, row - c.sensor_radius_cells), min(height, row + c.sensor_radius_cells + 1))
                         for q in range(max(0, col - c.sensor_radius_cells), min(width, col + c.sensor_radius_cells + 1))]
                noise = (world_noise(c.seed, int(round(now / c.sensor_period_s)), robot["id"],
                    [r * width + col for r, col in cells], c.observation_noise_m)
                    if c.noise_mode == "counter_keyed" else noise_rng.normal(0., c.observation_noise_m, len(cells)))
                values = field.at(now, cells) + noise
                observations = [dict(cell=cell, depth_m=float(value), sample_time=float(now),
                                     arrival_time=float(now + c.observation_delay_s), drone=robot["id"])
                                for cell, value in zip(cells, values)]
                sensed_targets = [target_by_cell[cell] for cell in cells if cell in target_by_cell]
                for tid in sensed_targets:
                    if target_rows[tid]["first_sensed_at"] is None:
                        target_rows[tid]["first_sensed_at"] = float(now)
                heapq.heappush(observation_queue, (now + c.observation_delay_s, queue_seq, observations, sensed_targets))
                queue_seq += 1
                sensor_batches += 1
                for cell in cells:
                    visits[cell] += 1
                trajectories.append(dict(time_s=float(now), drone=robot["id"], position_m=robot["position"].tolist(),
                                         sensor_center_cell=[row, col], mode=robot["mode"]))
            next_sensor += c.sensor_period_s
        newly_found = False
        while observation_queue and observation_queue[0][0] <= now + 1e-9:
            _, _, observations, sensed_targets = heapq.heappop(observation_queue)
            arrived_count += len(observations)
            for observation in observations:
                cell = observation["cell"]
                covered[cell] = True
                histories.setdefault(cell, deque(maxlen=history_limit)).append(observation)
                if c.retain_raw_observations:
                    raw_observations.append(observation)
            for tid in sensed_targets:
                if tid not in known:
                    known.add(tid)
                    target_rows[tid]["discovered_at"] = float(now)
                    newly_found = True
                    log("target_discovered", target=tid, sampled_at=target_rows[tid]["first_sensed_at"])
        if now >= next_plan - 1e-9 or newly_found or failed:
            reasons = [label for flag, label in ((newly_found, "new_target"), (failed, "service_failure"), (now >= next_plan - 1e-9, "periodic")) if flag]
            plan("+".join(reasons))
            if now >= next_plan - 1e-9:
                next_plan += c.planning_period_s
        # Newly assigned zero-distance target actions begin at this same time,
        # without an artificial full timestep of delay.
        for robot in robots:
            if robot["mode"] == "fly" and np.linalg.norm(robot["destination"] - robot["position"]) <= 1e-8:
                robot["mode"] = "wait" if robot["target"] is not None else "idle"
            if robot["mode"] == "wait" and robot["planned_start"] <= now + 1e-9:
                robot.update(mode="service", service_start=float(now), service_end=now + c.service_duration_s)
                target_rows[robot["target"]]["attempts"] += 1
                log("service_start", drone=robot["id"], target=robot["target"], prediction_version=robot["prediction_version"])
        if now >= c.duration_s - 1e-9:
            break
        candidates = [c.duration_s, next_sensor, next_plan]
        if observation_queue:
            candidates.append(observation_queue[0][0])
        for robot in robots:
            if robot["mode"] == "fly":
                candidates.append(now + float(np.linalg.norm(robot["destination"] - robot["position"])) / c.speed_mps)
            elif robot["mode"] == "wait":
                candidates.append(robot["planned_start"])
            elif robot["mode"] == "service":
                candidates.append(robot["service_end"])
        last = now
        now = min(value for value in candidates if value > now + 1e-8)
    failures = dict(undiscovered=0, discovered_too_late=0, predicted_window_error_or_execution_failure=0,
                    resource_or_time_exhausted=0)
    for row in target_rows:
        if row["served_at"] is not None:
            continue
        if row["discovered_at"] is None:
            reason = "undiscovered"
        elif row["failures"]:
            reason = "predicted_window_error_or_execution_failure"
        else:
            cell = tuple(row["cell"])
            times_remaining = field.times[(field.times >= row["discovered_at"]) & (field.times + c.service_duration_s <= c.duration_s)]
            has_future_window = any(field.service_succeeds(cell, t, c.service_duration_s, c.max_service_depth_m) for t in times_remaining)
            reason = "resource_or_time_exhausted" if has_future_window else "discovered_too_late"
        row["failure_category"] = reason
        failures[reason] += 1
    served = sum(row["served_at"] is not None for row in target_rows)
    summary = dict(
        method=planner.name if planner is not None else "nominal_only_greedy_engineering_preflight", formal_test=False, trained_model=False,
        hidden_targets=c.n_targets, successful_services=served,
        success_rate=served / c.n_targets if c.n_targets else None,
        discovered_targets=len(known), failed_service_attempts=sum(r["failures"] for r in target_rows),
        last_successful_service_s=max((r["served_at"] for r in target_rows if r["served_at"] is not None), default=None),
        failure_categories=failures, all_hidden_targets_in_denominator=True,
        coverage_fraction=float(covered[allowed].mean()), repeated_sensor_cell_visits=int(np.maximum(visits - 1, 0).sum()),
        distance_m=sum(r["distance_m"] for r in robots), sensor_batches=sensor_batches,
        arrived_observations=arrived_count, not_arrived_batches_at_end=len(observation_queue),
        planning_calls=len(planning_latencies), planning_timeouts=plan_timeouts,
        planning_latency_ms=dict(median=float(np.median(planning_latencies)), p95=float(np.quantile(planning_latencies, .95)),
                                 maximum=float(max(planning_latencies))),
        task_reassignments=reassignments, causal_service_gain_from_reassignment=None,
        elapsed_wall_s=time.perf_counter() - started_wall, final_time_s=float(now),
        clock="continuous_event_driven_flight_arrivals_and_service_completion",
    )
    for robot in robots:
        robot["position"] = robot["position"].tolist()
        if robot["destination"] is not None:
            robot["destination"] = robot["destination"].tolist()
    return dict(summary=summary, targets=target_rows, drones=robots, events=events,
                trajectories=trajectories, raw_observations=raw_observations,
                limitations=["Bounded candidate-route rolling optimization, not a globally optimal planner" if planner is not None else "Greedy single-next-task assignment; formal multi-visit planner is not implemented",
                             "Failure causes combine forecast window and execution failure pending trained-model instrumentation",
                             "No counterfactual attribution or H/E/P effect estimate in engineering preflight"])
