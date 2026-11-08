"""Counterfactual flood motion is private to the simulator. All policies share the same draw."""
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
ROOT = EXPERIMENTS.parents[2]
sys.path.insert(0, str(HERE.parent / "speed_first"))
sys.path.insert(0, str(EXPERIMENTS / "flood-drop-convlstm20-v1"))
sys.path.insert(0, str(EXPERIMENTS / "flood1km-six-20261101"))
from fall_risk import response
from long_environment import Config, Env as Drift, sample_water
from flood1km.env import MOVES, sector, segment_circle

RULES = json.loads((HERE / "protocol.json").read_text())
DURATION = RULES["mission_s"]
COUNT = RULES["people"]
EPS = RULES["roundoff_m"]
START = np.array([[5,5],[995,5],[5,995],[995,995]], dtype=float)
OUT = ROOT / "uav-unified-supply-20261105"
ARCHIVE = ROOT / "uav-sector5-end-to-end-20261105/run01/scenarios.json"
FIELD_DIR = ROOT / "uav-speed-first-exposure-20261105/window1800/fields"

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def histories(water, baseline, case, seed):
    """Sample all 80 once; no rejection of people who never fall."""
    rng, clock = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    eligible = baseline[0,0] >= .01
    eligible[[0,0,99,99],[0,99,0,99]] = False
    chosen = rng.choice(np.flatnonzero(eligible), COUNT, replace=False)
    rows, cols = np.unravel_index(chosen,(100,100))
    xy0 = np.column_stack((cols, rows)) * 10. + 5
    threshold = clock.exponential(size=COUNT)
    dose = np.zeros(COUNT)
    fall = np.full(COUNT, DURATION+1, dtype=int)
    at = np.moveaxis(np.asarray(water[:DURATION//5+1,:,rows,cols], dtype=float),1,-1)
    assert at.shape == (DURATION//5+1,COUNT,3)
    for sec in range(DURATION):
        k, rem = divmod(sec,5)
        frame = at[k]+((rem+.5)/5)*(at[k+1]-at[k])
        dose += RULES["hazard_k"] * response(frame[:,0],np.hypot(frame[:,1],frame[:,2]))
        fall[(fall>DURATION) & (dose>=threshold)] = sec+1
    cfg = Config(**{**case["scenario"]["config"], "duration":DURATION, "people":COUNT,
                    "window":RULES["personal_window_after_fall_s"]})
    motion = SimpleNamespace(cfg=cfg, water=water, person=xy0.copy(),
            scenario={"seed":int(seed),"events":[{} for _ in range(COUNT)]},
            traps=np.array(case["scenario"]["traps"],float).reshape(-1,2),
            status=np.zeros(COUNT,int), stuck=np.zeros(COUNT,bool), stop_kind=np.zeros(COUNT,int),
            refloat_counts=np.zeros(COUNT,int),events=[],refloat_enabled=True)
    motion.refloat_draw = lambda j, now: Drift.refloat_draw(motion,j,now)
    points = np.empty((DURATION+1,COUNT,2),float)
    available = np.ones((DURATION+1,COUNT),bool)
    velocity = np.zeros((DURATION//5+1,COUNT,2),float)
    for sec in range(DURATION+1):
        motion.status[(motion.status==0) & (fall<=sec)]=1
        points[sec] = motion.person
        available[sec] = motion.status != 4
        if sec%5==0:
            drift = (motion.status==1) & ~motion.stuck
            if drift.any():
                sampled=sample_water(water,sec,motion.person[drift])
                velocity[sec//5,drift] = np.where(sampled[:,:1]>=cfg.wet_depth,sampled[:,1:],0)
        if sec < DURATION:
            Drift.move_people(motion,sec)
    assert np.isfinite(points).all() and np.isfinite(velocity).all()
    return dict(points=points, available=available, velocity=velocity, fall=fall,
                threshold=threshold, initial=xy0, seed=int(seed), wet_source=chosen)

