import json
import numpy as np
import pytest

from hyra_exp3.design import design_jobs, target_opportunities, run_design, _cached_field
from hyra_exp3.protocol import default_protocol, initialize


def test_design_matrix_keeps_all_conditions_and_excludes_non_design_families():
    jobs = design_jobs(default_protocol())
    assert len(jobs) == 18
    assert {r["family"]["split"] for r in jobs} == {"design"}
    assert len({r["family"]["family_id"] for r in jobs}) == 6
    assert sum(len(r["conditions"]) for r in jobs) == 216
    assert sum(len(r["conditions"]) * len(r["layout_seeds"]) * len(r["delays_s"]) for r in jobs) == 576
    assert all(len({tuple(sorted(c.items())) for c in r["conditions"]}) == 12 for r in jobs)
    # Layout/noise streams are matched across scales within a family and differ
    # between families. No choice depends on a water field or model outcome.
    for family in {r["family"]["family_id"] for r in jobs}:
        assert len({tuple(r["layout_seeds"]) for r in jobs if r["family"]["family_id"] == family}) == 1
    assert len({tuple(r["layout_seeds"]) for r in jobs}) == 6


def test_post_execution_margin_uses_complete_service_and_preserves_censoring():
    times = np.arange(31.)
    depth = np.full((31, 2), .2)
    depth[:, 0] = .2 + times * .03  # crosses .8 at t=20, last safe start at 8
    episode = dict(execution_record=dict(config=dict(map_shape=[1, 2])), targets=[
        dict(id=0, cell=[0, 0], discovered_at=0., served_at=15., attempts=1, failures=0),
        dict(id=1, cell=[0, 1], discovered_at=0., served_at=12., attempts=1, failures=0)])
    result = target_opportunities(dict(times_s=times, depths_m=depth), episode)
    assert result[0]["safe_start_margin_after_actual_service_s"] == pytest.approx(5.)
    assert not result[0]["closing_time_censored_by_episode_end"]
    assert result[1]["closing_time_censored_by_episode_end"]
    assert result[1]["safe_start_margin_after_actual_service_s"] == pytest.approx(18.)


def test_resume_refuses_changed_execution_specification_before_running_workers(tmp_path, monkeypatch):
    initialize(tmp_path)
    run = tmp_path / "results" / "runs" / "partial"
    run.mkdir(parents=True)
    (run / "design_spec.json").write_text(json.dumps(dict(changed=True)), encoding="utf-8")
    monkeypatch.setattr("hyra_exp3.design.execution_hashes", lambda root: {"simulation.py": "new_hash"})
    with pytest.raises(ValueError, match="specification or execution code changed"):
        run_design(tmp_path, run, workers=1)


def test_interrupted_field_is_preserved_and_recovery_cache_is_verified(tmp_path, monkeypatch):
    from hyra_exp3.artifacts import sha256, write_json
    directory = tmp_path / "truth"
    directory.mkdir()
    partial = directory / "field.npz"
    partial.write_bytes(b"incomplete original output")
    destinations = []
    def generate(side, destination, **kwargs):
        destinations.append(destination)
        destination.mkdir(parents=True)
        path = destination / "field.npz"
        np.savez_compressed(path, times_s=[0., 2.], depths_m=[[.2], [.3]])
        manifest = dict(field_path=str(path), field_sha256=sha256(path))
        write_json(destination / "manifest.json", manifest)
        return manifest
    monkeypatch.setattr("hyra_exp3.hydraulics.generate_field", generate)
    record, field = _cached_field(directory, 32, family_seed=1)
    assert partial.read_bytes() == b"incomplete original output"
    assert destinations[0].is_relative_to(directory / "recovery_attempts")
    _cached_field(directory, 32, family_seed=1)
    assert len(destinations) == 1
    with pytest.raises(ValueError, match="specification differs"):
        _cached_field(directory, 32, family_seed=2)
    np.testing.assert_array_equal(field["times_s"], [0., 2.])


def test_recovery_rejects_tampered_committed_field(tmp_path):
    from hyra_exp3.artifacts import write_json
    path = tmp_path / "field.npz"
    path.write_bytes(b"damaged output")
    write_json(tmp_path / "manifest.json", dict(field_path=str(path), field_sha256="wrong"))
    with pytest.raises(ValueError, match="checksum"):
        _cached_field(tmp_path, 32, family_seed=1)
