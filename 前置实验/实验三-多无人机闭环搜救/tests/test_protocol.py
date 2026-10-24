from hyra_exp3.protocol import default_protocol, family_split, validate_protocol, initialize


def test_fresh_family_split_is_reproducible_and_disjoint():
    protocol = default_protocol()
    assert family_split(protocol) == family_split(protocol)
    rows = family_split(protocol)["families"]
    assert len(rows) == 78
    assert len({r["family_seed"] for r in rows}) == 78
    assert {s: sum(r["split"] == s for r in rows) for s in protocol["family_counts"]} == protocol["family_counts"]


def test_matrix_does_not_silently_drop_forcing_conditions():
    audit = validate_protocol(default_protocol())
    assert audit["structural_checks_passed"]
    assert audit["one_condition_per_family"]["total_with_three_delay_methods"] == 3024
    assert audit["all_conditions_per_family"]["total_with_three_delay_methods"] == 36288
    assert not audit["configuration_frozen"]
    assert not audit["formal_test_ready"]


def test_split_registration_cannot_be_silently_overwritten(tmp_path):
    initialize(tmp_path)
    initialize(tmp_path)
    assert (tmp_path / "manifests" / "family_split.json").is_file()
