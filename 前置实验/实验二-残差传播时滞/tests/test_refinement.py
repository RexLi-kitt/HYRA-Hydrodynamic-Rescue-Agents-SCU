import numpy as np
import pytest
from hyra_validation.refinement import comparison_summary, stable_controls


def test_threshold_equality_and_submillimetre_flip_are_retained():
    result = comparison_summary([.8, .7999, .9], [.8001, .8, .9001])
    assert result['windows'] == 3 and result['label_changes'] == 1
    assert result['max_window_change_m'] == pytest.approx(.0001)
    with pytest.raises(ValueError): comparison_summary([.8], [np.nan])
    with pytest.raises(ValueError): comparison_summary([.8], [.8, .9])


def test_control_selection_preserves_near_and_far_stable_and_excludes_disagreements():
    common = dict(family='train_0', anchor_s=180, physical_center_y_m=20, condition='E0', target=13)
    rows = [dict(common, lower_m=a, upper_m=b, label_disagrees=d)
            for a, b, d in [(.7, .9, True), (.79, .795, False), (.1, .2, False), (.79, .795, False)]]
    assert stable_controls(rows) == [1, 2]
