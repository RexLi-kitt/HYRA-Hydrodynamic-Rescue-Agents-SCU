from collections import Counter
from hyra_exp3.execution_plan import balanced_allocation
from hyra_exp3.protocol import default_protocol, family_split

def test_formal_allocation_is_balanced_disjoint_and_independent_of_list_order():
    p = default_protocol()
    families = family_split(p)["families"]
    a = balanced_allocation(families, p["study_id"])
    assert a == balanced_allocation(list(reversed(families)), p["study_id"])
    for split, repeats in (("train", 2), ("validation", 1), ("test", 3)):
        selected = [r for r in a if r["split"] == split]
        assert set(Counter(r["condition_index"] for r in selected).values()) == {repeats}
        assert len({r["condition_index"] for r in selected}) == 12
    assert 36 * 2 * 3 * (3 + 5 + 3 + 3) == 3024
