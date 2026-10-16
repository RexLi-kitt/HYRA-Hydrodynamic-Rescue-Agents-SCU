import pytest
from hyra_validation.evaluation import score_event,compare_paired_events


def test_equal_sample_scoring_required():
    with pytest.raises(ValueError):
        score_event([1,2],[1],dict(underestimation_threshold_m=.15))


def test_family_bootstrap_aggregates_variants():
    rows=[]
    for family in ["a","b"]:
        for method,value in [("B0_raw",1),("B1_distance",2),("B2_uniform",3),("B2_structure",4)]:
            for _ in range(3):
                rows.append(dict(event_family_id=family,method=method,metric=value))
    result=compare_paired_events(rows,dict(metric="metric"))
    assert result[0]["independent_families"]==2
    assert result[0]["difference_b2_minus_baseline"]==3


def test_unpaired_event_rejected():
    with pytest.raises(ValueError,match="Incomplete"):
        compare_paired_events([dict(event_family_id="a",method="B2_structure",metric=1)],dict(metric="metric"))


def test_seed_and_family_uncertainty_preserves_pairing():
    rows=[]
    for family in ("a","b"):
        for seed in (11,22,33):
            for method,value in [("B0_raw",1),("B1_distance",1),("B2_uniform",1),("B2_structure",2)]:
                rows.append(dict(event_family_id=family,training_seed=seed,method=method,metric=value))
    result=compare_paired_events(rows,dict(metric="metric"))
    assert result[0]["training_seeds"]==3
    assert result[0]["event_and_training_seed_interval_95"]==[1.,1.]
