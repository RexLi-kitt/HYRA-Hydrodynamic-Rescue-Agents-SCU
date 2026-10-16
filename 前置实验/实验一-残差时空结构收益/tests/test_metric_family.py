import json
from pathlib import Path
import numpy as np
import pytest
from hyra_validation.metric_family import draw_metric_family,independent_restart

DESIGN=json.loads((Path(__file__).resolve().parents[1]/'configs/metric_development.design.v1.json').read_text(encoding='utf-8'))


def test_fresh_family_streams_are_stable_and_split_separated():
    a=draw_metric_family(DESIGN,'train',0);assert a==draw_metric_family(DESIGN,'train',0)
    b=draw_metric_family(DESIGN,'validation',0)
    assert a['params']!=b['params'] and a['family_id']!=b['family_id']
    assert a['seed_identity'][0]!=2026101517


def test_nominal_restart_cannot_read_reference_state_and_time_must_be_saved():
    raw=dict(time_s=np.array([0,2]),**{k:np.array([[0,1],[2,3]]) for k in ('stage','xmomentum','ymomentum')})
    info=dict(kind='truth_base',raw_path='truth.npz',raw_sha256='sha')
    with pytest.raises(ValueError):independent_restart(info,raw,2,'same_mesh_independent_nominal_base')
    with pytest.raises(ValueError):independent_restart(info,raw,1,'reference_base_environment_only_not_forecast')
    got=independent_restart(info,raw,2,'reference_base_environment_only_not_forecast')
    np.testing.assert_array_equal(got['state']['stage'],[2,3])
