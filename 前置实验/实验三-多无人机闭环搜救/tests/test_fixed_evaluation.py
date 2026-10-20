import numpy as np
from hyra_exp3.evaluation import predicted_safe_windows
from hyra_exp3.windows import service_start_intervals


def test_vectorized_fixed_windows_check_interior_and_fractional_12s_endpoint():
    rng=np.random.default_rng(42)
    curves=rng.uniform(.5,1.1,(40,13))
    curves[0,:]=.2;curves[0,1]=1.
    times=np.arange(13)*5.
    actual=predicted_safe_windows(curves)
    assert not actual[0,0]
    for cell,curve in enumerate(curves):
        intervals=service_start_intervals(times,curve)
        expected=[any(a-1e-9<=start<=b+1e-9 for a,b in intervals) for start in range(0,41,10)]
        assert actual[cell].tolist()==expected
