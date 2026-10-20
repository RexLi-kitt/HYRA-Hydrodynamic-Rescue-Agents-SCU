"""Central rolling route selection and forecast-aware exploration.

The planner receives only public geometry, arrived coverage and known targets.
Route generation is bounded beam search; the integer program optimizes over
that candidate pool, not over every possible route. Only the first visit is
executed and all remaining visits are reconsidered at the next planning event.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


def footprint_gains(covered, allowed, radius):
    unseen = (~covered & allowed).astype(np.int32)
    integral = np.pad(np.pad(unseen, radius), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    size = 2 * radius + 1
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


@dataclass(frozen=True)
class Route:
    drone: int
    targets: tuple[int, ...]
    starts: tuple[float, ...]
    finish: float
    travel_s: float
    version: int
    coverage: int


class RollingPlanner:
    def __init__(self, *, forecast_exploration=True, multi_visit=True, beam_width=6, frontier_limit=16,
                 solver_budget_s=.02):
        if not isinstance(beam_width, int) or beam_width < 1 or not isinstance(frontier_limit, int) or frontier_limit < 1:
            raise ValueError("Candidate counts must be positive integers")
        if not np.isfinite(solver_budget_s) or solver_budget_s <= 0:
            raise ValueError("Positive solver budget required")
        self.forecast_exploration = forecast_exploration
        self.multi_visit = multi_visit
        self.beam_width, self.frontier_limit = beam_width, frontier_limit
        self.solver_budget_s = solver_budget_s
        self.last_diagnostics = {}

    @property
    def name(self):
        return f"rolling_routes{int(self.multi_visit)}_forecast_exploration{int(self.forecast_exploration)}"

    def __call__(self, *, now, robots, known_targets, covered, allowed, predictor, config):
        c = config
        def xy(cell):
            return np.asarray([(cell[1] + .5) * c.cell_size_m, (cell[0] + .5) * c.cell_size_m])
        cells = {tid: tuple(cell) for tid, cell in known_targets.items()}
        estimates = {tid: predictor.estimate(cell, now) for tid, cell in cells.items()}
        gain = footprint_gains(covered, allowed, c.sensor_radius_cells)
        proposals, routes = {}, []
        candidate_count, solver_optimal = 0, True
        if self.multi_visit:
            for robot in robots:
                rid = robot["id"]
                # (visits, starts, finish, position, travel time, minimum slack)
                beam = [((), (), now, robot["position"], 0., float("inf"))]
                for depth in range(min(robot["payload"], 4, len(cells))):
                    expanded = []
                    for visits, starts, finish, position, travel, slack in beam:
                        for tid, cell in cells.items():
                            if tid in visits:
                                continue
                            leg = float(np.linalg.norm(xy(cell) - position)) / c.speed_mps
                            start = estimates[tid].earliest_start(finish + leg)
                            if start is None or start + c.service_duration_s > c.duration_s + 1e-9:
                                continue
                            close = next(b for a, b in estimates[tid].service_intervals if a - 1e-9 <= start <= b + 1e-9)
                            expanded.append((visits + (tid,), starts + (start,), start + c.service_duration_s,
                                             xy(cell), travel + leg, min(slack, close - start)))
                    expanded.sort(key=lambda r: (r[4], r[2], r[5], r[0]))
                    # All single visits are retained; deeper branches are bounded.
                    retained = expanded if depth == 0 else expanded[:self.beam_width]
                    for visits, starts, finish, position, travel, slack in retained:
                        routes.append(Route(rid, visits, starts, finish, travel, predictor.version,
                                            int(gain[cells[visits[-1]]])))
                    beam = expanded[:self.beam_width]
                    if not beam:
                        break
            candidate_count = len(routes)
            selected = []
            if routes:
                robot_ids = [r["id"] for r in robots]
                task_ids = list(cells)
                matrix = np.zeros((len(robot_ids) + len(task_ids), len(routes)))
                for j, route in enumerate(routes):
                    matrix[robot_ids.index(route.drone), j] = 1
                    for tid in route.targets:
                        matrix[len(robot_ids) + task_ids.index(tid), j] = 1
                # Counts dominate all coverage and flight terms for protocol sizes.
                # Coverage is a route-end proxy; actual future exploration replans.
                primary_weight = 1 + len(robots) * (100 * (2 * c.sensor_radius_cells + 1) ** 2 + c.forecast_horizon_s + 1)
                objective = np.asarray([-primary_weight * len(r.targets) - 100 * r.coverage + r.travel_s +
                                        (r.finish - now) / (1000 * max(1, len(robots))) for r in routes])
                solution = milp(objective, integrality=np.ones(len(routes)), bounds=Bounds(0, 1),
                                constraints=LinearConstraint(csc_matrix(matrix), -np.inf, 1),
                                options={"time_limit": self.solver_budget_s, "mip_rel_gap": 0.})
                solver_optimal = solution.status == 0
                if solution.x is not None:
                    selected = [r for r, value in zip(routes, solution.x) if value > .5]
                else:
                    used, drones = set(), set()
                    for route in sorted(routes, key=lambda r: (-len(r.targets), -r.coverage, r.travel_s, r.drone, r.targets)):
                        if route.drone not in drones and not used.intersection(route.targets):
                            selected.append(route)
                            used.update(route.targets)
                            drones.add(route.drone)
                for route in selected:
                    tid = route.targets[0]
                    proposals[route.drone] = (cells[tid], tid, route.starts[0], route.version)
            selected_routes = [dict(drone=r.drone, targets=list(r.targets), starts_s=list(r.starts),
                                    finish_s=r.finish, version=r.version,
                                    initial_position_m=next(robot["position"].tolist() for robot in robots if robot["id"] == r.drone),
                                    payload=next(robot["payload"] for robot in robots if robot["id"] == r.drone)) for r in selected]
        else:
            # Preserve the historical single-next-task rule for the ablation.
            remaining = {r["id"]: r for r in robots if r["payload"] > 0}
            pending = set(cells)
            while remaining and pending:
                best = None
                for rid, robot in remaining.items():
                    for tid in sorted(pending):
                        travel = float(np.linalg.norm(robot["position"] - xy(cells[tid]))) / c.speed_mps
                        start = estimates[tid].earliest_start(now + travel)
                        if start is None or start + c.service_duration_s > c.duration_s + 1e-9:
                            continue
                        close = next(b for a, b in estimates[tid].service_intervals if a - 1e-9 <= start <= b + 1e-9)
                        item = (close, start + c.service_duration_s, travel, rid, tid, start)
                        if best is None or item < best:
                            best = item
                if best is None:
                    break
                _, _, _, rid, tid, start = best
                proposals[rid] = (cells[tid], tid, start, estimates[tid].version)
                pending.remove(tid)
                remaining.pop(rid)
            selected_routes = []
        explorers = [r for r in robots if r["id"] not in proposals]
        candidates = np.argwhere((gain > 0) & allowed)
        candidate_xy = np.column_stack(((candidates[:, 1] + .5) * c.cell_size_m, (candidates[:, 0] + .5) * c.cell_size_m))
        radius = c.sensor_radius_cells
        size = 2 * radius + 1
        def clip_point(row, col):
            return (max(0, min(allowed.shape[0] - 1, row)), max(0, min(allowed.shape[1] - 1, col)))
        reserved, query_count, decisions = [], 0, []
        if len(candidates):
            shortlist = {}
            for robot in explorers:
                distance = np.linalg.norm(candidate_xy - robot["position"], axis=1)
                base = gain[candidates[:, 0], candidates[:, 1]] / (1 + distance / c.speed_mps)
                # Spatial suppression supplies alternatives beyond one footprint.
                work, indexes = base.copy(), []
                for _ in range(min(self.frontier_limit, len(candidates))):
                    index = int(np.argmax(work))
                    if work[index] <= 0:
                        break
                    indexes.append(index)
                    work[np.linalg.norm(candidate_xy - candidate_xy[index], axis=1) < size * c.cell_size_m] = -1
                shortlist[robot["id"]] = (np.asarray(indexes, int), base, distance)
            forecast = {}
            if self.forecast_exploration:
                # Query center and footprint corners, all from public geometry.
                query_cells = set()
                for indexes, _, _ in shortlist.values():
                    for index in indexes:
                        row, col = map(int, candidates[index])
                        for dr, dc in ((0, 0), (-radius, -radius), (-radius, radius), (radius, -radius), (radius, radius)):
                            query_cells.add(clip_point(row + dr, col + dc))
                query_cells = sorted(query_cells)
                batch = getattr(predictor, "estimate_many", None)
                forecast = dict(zip(query_cells, batch(query_cells, now) if batch else [predictor.estimate(cell, now) for cell in query_cells]))
                query_count = len(query_cells)
            for robot in explorers:
                indexes, base, distance = shortlist[robot["id"]]
                if not self.forecast_exploration:
                    scores = base.copy()
                    for position in reserved:
                        scores[np.linalg.norm(candidate_xy - position, axis=1) < size * c.cell_size_m] = -1
                    index = int(np.argmax(scores))
                    if scores[index] > 0:
                        cell = tuple(map(int, candidates[index]))
                        proposals[robot["id"]] = (cell, None, None, None)
                        reserved.append(candidate_xy[index])
                    continue
                choices = []
                for index in indexes:
                    position, cell = candidate_xy[index], tuple(map(int, candidates[index]))
                    if any(np.linalg.norm(position - p) < size * c.cell_size_m for p in reserved):
                        continue
                    score, weight = float(base[index]), 1.
                    if self.forecast_exploration:
                        # A discovery needs the next sensor tick, communication,
                        # and local travel before a complete 12s service can start.
                        arrival = now + distance[index] / c.speed_mps + c.sensor_period_s + c.observation_delay_s + radius * c.cell_size_m / c.speed_mps
                        weights = []
                        for dr, dc in ((0, 0), (-radius, -radius), (-radius, radius), (radius, -radius), (radius, radius)):
                            q = clip_point(cell[0] + dr, cell[1] + dc)
                            if covered[q] or not allowed[q]:
                                continue
                            estimate = forecast[q]
                            start = estimate.earliest_start(arrival)
                            if start is not None:
                                close = next(b for a, b in estimate.service_intervals if a - 1e-9 <= start <= b + 1e-9)
                                # Urgency is used only for an actual closing in
                                # support; a clipped horizon is not a deadline.
                                closing = close + c.service_duration_s < estimate.valid_until - 1e-7
                                weights.append(1. + (20. / (5. + max(0., close - start)) if closing else 0.))
                            elif arrival + c.service_duration_s > estimate.valid_until + 1e-9:
                                weights.append(.25)  # Unknown beyond support.
                            else:
                                weights.append(.05)  # Coverage still has value.
                        weight = float(np.mean(weights)) if weights else .05
                        score *= weight
                    choices.append((score, -float(distance[index]), -int(index), index, weight))
                if choices:
                    score, _, _, index, weight = max(choices)
                    cell = tuple(map(int, candidates[index]))
                    proposals[robot["id"]] = (cell, None, None, predictor.version if self.forecast_exploration else None)
                    reserved.append(candidate_xy[index])
                    decisions.append(dict(drone=robot["id"], cell=list(cell), weight=weight))
        self.last_diagnostics = dict(controller=self.name, candidate_routes=candidate_count,
                                     settings=dict(beam_width=self.beam_width, frontier_limit=self.frontier_limit,
                                                   solver_budget_s=self.solver_budget_s),
                                     candidate_pool_optimal=solver_optimal, selected_routes=selected_routes,
                                     frontier_query_locations=query_count, exploration=decisions)
        return proposals
