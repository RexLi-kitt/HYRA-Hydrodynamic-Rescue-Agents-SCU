import numpy as np
import pytest

from hyra_exp3.artifacts import write_json
from hyra_exp3.phase_matrix import PHASES, TruthWindowDiagnostic, oracle_episode, summarize
from hyra_exp3.simulation import EpisodeConfig


def test_truth_diagnostic_keeps_publication_support_and_labels_privileged_evidence():
    times = np.arange(41.)
    predictor = TruthWindowDiagnostic(times, np.full((41, 2, 2), .2), horizon_s=20)
    predictor.issue(5, [])
    window = predictor.estimate((0, 0), 5)
    assert window.valid_until == 25 and window.can_start(14) is None
    assert window.evidence == "privileged_future_truth_diagnostic_only"
    assert predictor.queries == [(5., 0, 0, 1)]
    assert PHASES == [450., 600.]


def test_oracle_episode_labels_data_and_queries_only_discovered_targets(tmp_path):
    times = np.arange(41.)
    truth = dict(times_s=times, depths_m=np.full((41, 25), .2))
    config = EpisodeConfig(map_shape=(5, 5), n_drones=1, n_targets=1, duration_s=40, observation_noise_m=0)
    result = oracle_episode(tmp_path, truth, config)
    assert result["summary"]["successful_services"] == 1
    assert result["execution_record"]["privileged_future_truth"]
    assert not result["execution_record"]["training_data_eligible"]
    assert result["execution_record"]["all_queries_after_discovery"]
    with np.load(result["execution_record"]["query_log"]["path"]) as log:
        discovered = result["targets"][0]["discovered_at"]
        assert np.all(log["time_s"] >= discovered)


def test_negative_oracle_pairs_are_kept_and_averaged_within_family(tmp_path):
    write_json(tmp_path / "phase_matrix_spec.json", dict(planned_phase_cases=432, planned_pairs=1152))
    for index, nominal in enumerate((6, 8)):
        pair = dict(layout_index=0, delay_s=0., nominal_summary=dict(hidden_targets=12, successful_services=nominal, planning_timeouts=0),
                    oracle_summary=dict(successful_services=7, planning_timeouts=0),
                    bounds=dict(optimistic_individually_feasible_targets=12, targets_with_any_complete_window=12),
                    achieved_oracle_minus_nominal_pp=100 * (7 - nominal) / 12)
        write_json(tmp_path / "jobs" / "family_side32" / f"condition_{index:02d}" / "phase450" / "phase_case.json",
                   dict(family=dict(family_id="exp3-design-000"), side=32, phase_s=450., condition=dict(index=index), pairs=[pair]))
    result = summarize(tmp_path)
    group = result["groups"][0]
    assert result["status"] == "partial"
    assert group["positive_oracle_pairs"] == 1 and group["negative_oracle_pairs"] == 1
    assert group["achieved_oracle_delta_pp"] == pytest.approx(0)
    assert len(result["family_means"]) == 1
