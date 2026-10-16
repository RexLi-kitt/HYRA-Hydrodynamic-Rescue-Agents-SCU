import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


@pytest.fixture
def config():
    return json.loads((ROOT/"configs/pilot.json").read_text(encoding="utf-8"))


@pytest.fixture
def simple_event():
    t = np.arange(0,361,2,dtype=float)
    releases = np.arange(0,301,10,dtype=float)
    valid = releases[:,None]+np.arange(0,61,2)
    return dict(time_s=t,truth_depth_m=np.full((len(t),100),.2),xy_m=np.array([(x*10,y*10) for y in range(10) for x in range(10)]),
        release_s=releases,forecast_valid_s=valid,forecast_depth_m=np.full((*valid.shape,100),.2),
        metadata=dict(event_id="fixture",event_family_id="fixture",split="train",units=dict(time="s",xy="m",depth="m")))
