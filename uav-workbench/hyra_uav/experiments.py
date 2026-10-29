"""Portable run records, physical development fixtures, and a paired pilot."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import uuid

import numpy as np
from .core import Config, FloodProcess, Pose, Task
from .policies import policy_for

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
TERRAIN = REPO / "terrain-workbench" / "terrain-lab"
FIXTURES = ROOT / "fixtures"
OUTPUT = ROOT / "generated"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def source_version():
    files = sorted(ROOT.glob("hyra_uav/*.py")) + [ROOT / "run.py"]
    hashes = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
    return dict(git_head=head.stdout.strip(), source_files_sha256=hashes,
                source_digest=sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                python=sys.version, numpy=np.__version__)


def available_scenes():
    entries = []
    candidate = TERRAIN / "generated/maps-dca2287a0ca2/maps/map-result.zip"
    if candidate.is_file():
        entries.append(dict(id="existing-demo", name="现有洪水联调过程", family="multilayer-demo",
                            kind="existing_synthetic_hydraulic_process", path=str(candidate)))
    for path in sorted(FIXTURES.glob("*/scene.json")):
        scene = json.loads(path.read_text(encoding="utf-8"))
        scene["path"] = str(path.parent / "map-result.zip")
        if Path(scene["path"]).is_file():
            entries.append(scene)
    for path in sorted((OUTPUT / "imports").glob("*/map-result.zip")):
        entries.append(dict(id="import-"+path.parent.name, name="导入过程 "+path.parent.name,
                            family="unclassified_import", kind="import_requires_physical_review", path=str(path)))
    return entries


def run_episode(scene, method="R0", config=None, pose=None, actions=None):
    started = time.perf_counter()
    process = FloodProcess(scene["path"])
    config = config or Config(duration_s=float(process.times[-1]))
    task = Task(process, config, pose)
    policy = policy_for(method, task.view(), config) if actions is None else None
    count = round(config.duration_s / config.action_s)
    if actions is not None and len(actions) != count:
        raise ValueError(f"固定动作必须恰好为 {count} 条")
    try:
        for i in range(count):
            action = policy.choose(task.view(), config) if policy else actions[i]
            task.step(action)
    except (Exception, KeyboardInterrupt) as error:
        partial = task.result()
        partial.update(scene={k:v for k,v in scene.items() if k!="path"},
                       method=method if policy else "fixed_actions", version=source_version(),
                       elapsed_s=time.perf_counter()-started, status="interrupted",
                       reason=type(error).__name__+": "+str(error),
                       last_valid_time_s=task.snapshots[-1]["time_s"], data_split="development")
        save_run(partial, OUTPUT/"interrupted"/uuid.uuid4().hex[:12])
        raise
    result = task.result()
    result.update(scene={k: v for k, v in scene.items() if k != "path"}, method=method if policy else "fixed_actions",
                  version=source_version(), elapsed_s=time.perf_counter()-started,
                  status="completed", data_split="development", independent_event_claim=False)
    return result


def save_run(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "replay.json", result)
    for name in ("actions", "trajectory", "observations", "changes"):
        write_json(folder / (name+".json"), result[name])
    write_json(folder / "audit.json", result["audit"])
    write_json(folder / "manifest.json", {k: v for k, v in result.items()
                                          if k not in ("snapshots", "truth", "trajectory", "observations", "changes")})


def build_fixtures():
    """Use the unchanged HyRA SWE solver. These are synthetic development events."""
    sys.path.insert(0, str(TERRAIN))
    from flood_solver import FloodSolver
    import flood_run
    from multilayer import MapWorld
    from map_sessions import save_process
    recipes = [
        dict(id="slow", name="缓变：均匀降雨积水", family="synthetic-shallow-basin",
             rain=600, infiltration=0, sources={}, drains=0),
        dict(id="rapid", name="局部快速变化：短时来水", family="synthetic-sloped-basin",
             rain=0, infiltration=0, sources={"0": [[0, 0], [100, 30], [220, 0]]}, drains=.04),
        dict(id="async", name="分区异步涨退：隔水双区", family="synthetic-divided-basin",
             rain=0, infiltration=0, sources={"0": [[0, 12], [180, 0]], "1": [[0, 0], [240, 12], [420, 0]]}, drains=.08),
    ]
    for recipe in recipes:
        folder = FIXTURES / recipe["id"]
        if (folder / "map-result.zip").exists():
            raise ValueError(f"拒绝覆盖已有过程：{folder}")
        shape, dx = (32, 32), 5.0
        zz, xx = np.indices(shape)
        if recipe["id"] == "slow":
            bed = .0005*((xx-15.5)**2 + (zz-15.5)**2)
        else:
            bed = .006*xx + .003*zz
        layers = dict(source_id=np.full(shape, -1), drain_capacity=np.full(shape, recipe["drains"]),
                      hydrographs=recipe["sources"])
        if recipe["sources"]:
            layers["source_id"][8, 4] = 0
        if recipe["id"] == "async":
            layers["source_id"][23, 26] = 1
            wall = np.zeros((32, 33), bool)
            wall[:, 16] = True
            layers["wall_x"] = wall
        config = flood_run.settings(dict(resolution=32, duration_s=600, rain_mm_h=recipe["rain"],
                                         rain_duration_s=600 if recipe["rain"] else 0,
                                         infiltration_mm_h=recipe["infiltration"], boundary="closed",
                                         baseflow=False, warmup_s=0), shape)
        config["frames"] = 121
        solver = FloodSolver(bed, np.zeros(shape), dx, boundary="closed", manning=.035)
        meta = dict(config=config, coarsening_factor=1, origin_blocks=[0, 0],
                    terrain_source="synthetic_uav_development", terrain_id=recipe["family"],
                    initial_conditions=dict(duration_s=0, converged=False, reason="no_active_baseflow"))
        world = MapWorld(solver, meta, layers)
        frame_folder = folder / "generated" / "event" / "maps"
        frame_folder.mkdir(parents=True)
        frames = []
        for i, t in enumerate(np.arange(0, 601, 5)):
            world.advance_to(float(t))
            u, v = solver.velocity()
            filename = f"frame-{i:03d}.f32"
            np.stack((solver.h, u, v, solver.maximum)).astype("<f4").tofile(frame_folder / filename)
            frames.append(dict(index=i, url="/generated/event/maps/"+filename, **solver.summary()))
        save_process(folder, "event", world, frames)
        (frame_folder / "map-result.zip").replace(folder / "map-result.zip")
        # Keep the compact portable process, remove only this builder's loose frames.
        for f in frame_folder.glob("frame-*.f32"):
            f.unlink()
        write_json(folder / "scene.json", dict(id=recipe["id"], name=recipe["name"], family=recipe["family"],
                                               kind="synthetic_hydraulic_development", recipe=recipe,
                                               note="合成浅水方程开发过程，未用真实洪水率定；三种机制不构成充分泛化样本"))
        print(f"生成 {recipe['id']}：121个实际计算帧", flush=True)


def build_logic_demo():
    """Tiny hand-authored sequence to expose stale-record/return semantics."""
    folder = FIXTURES / "logic-demo"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "map-result.zip"
    if path.exists():
        raise ValueError("拒绝覆盖已有逻辑演示")
    grid = dict(shape=[5, 9], axis_order=["z", "x"], origin_blocks=[0, 0], dx_m=1)
    labels = np.zeros((9, 5, 9))
    labels[1:, 2, 3] = 1
    frames = [dict(index=i, time_s=i, url=f"/logic/frame-{i}.f32") for i in range(9)]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("process.json", json.dumps(dict(grid=grid, config=dict(metres_per_block=1),
                          frames=frames, frame_fields=["depth", "u", "v", "maximum_depth"], frame_dtype="<f4")))
        archive.writestr("map_model.json", json.dumps(dict(grid=grid, M_base=dict(z=np.zeros((5, 9)).tolist()))))
        archive.writestr("data_dictionary.json", json.dumps(dict(frames=dict(order="[field,z,x]"))))
        for i, layer in enumerate(labels):
            depth = layer*.1
            archive.writestr(f"frame-{i}.f32", np.stack((depth, np.zeros_like(depth), np.zeros_like(depth), depth)).astype("<f4").tobytes())
    config = Config(speed_m_s=2, angular_speed_deg_s=90, range_m=1.1, fov_deg=60,
                    action_s=1, sample_s=1, duration_s=8)
    actions = ["forward", "right", "right", "hover", "hover", "hover", "hover", "hover"]
    write_json(folder / "scene.json", dict(id="logic-demo", name="逻辑验收：旧记录与返回纠正", family="hand-authored-logic",
               kind="hand_authored_not_hydraulic_evidence", recommended_config=asdict(config),
               demo_pose=dict(x_m=2, z_m=2, heading_deg=0), demo_actions=actions,
               note="人工改变(行2,列3)，t=1水情变化但仍保留t=0旧记录，t=3转回后纠正；不作为洪水机制证据"))


def pilot(folder, task_durations=(600,)):
    scenes = [s for s in available_scenes() if s["id"] in ("slow", "rapid", "async")]
    if len(scenes) != 3:
        raise ValueError("需要先生成三类物理开发过程")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    starts = [Pose(0, 0, 0), Pose(155, 155, 180), Pose(0, 155, 270)]
    plan = dict(stage="P2", data_split="development", independent_events=3,
                uncertainty="仅开发诊断，不估计泛化置信区间", methods=["R0", "R1", "R2"],
                starts=[asdict(p) for p in starts], task_durations_s=list(task_durations),
                config=asdict(Config()), version=source_version(), scenes=scenes,
                selection_rule="保留所有候选；依据覆盖/重访/变化时间选择，不依据B3胜出")
    write_json(folder / "plan.json", plan)
    rows = []
    for duration in task_durations:
        for scene in scenes:
            for method in plan["methods"]:
                for i, pose in enumerate(starts):
                    run_id = f"t{duration}-{scene['id']}-{method}-start{i}"
                    try:
                        result = run_episode(scene, method, Config(duration_s=duration), pose)
                        save_run(result, folder / run_id)
                        row = dict(run_id=run_id, scene=scene["id"], family=scene["family"], method=method,
                                   start=i, duration_s=duration, status="completed", elapsed_s=result["elapsed_s"], **result["summary"])
                    except Exception as error:
                        row = dict(run_id=run_id, scene=scene["id"], method=method, start=i,
                                   duration_s=duration, status="failed", error=str(error))
                    rows.append(row)
                    write_json(folder / "index.json", rows)
                    print(run_id, row["status"], flush=True)
    write_json(folder / "summary.json", dict(plan=plan, runs=rows))
    return rows
