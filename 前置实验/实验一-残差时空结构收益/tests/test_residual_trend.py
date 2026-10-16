from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_residual_trend import trend_prediction


def test_linear_residual_is_extrapolated_using_actual_age():
    x=np.zeros((1,134));h=x[:,:24].reshape(1,6,4)
    # Current-time error .03m, slope .001m/s, newest observation is 10s old.
    for i,age in enumerate((10,20,30)):
        h[0,i]=[.03-.001*age,0,age/60,1]
    p,detail=trend_prediction(x,40)
    np.testing.assert_allclose(p[0,[0,5,15]],[.03,.04,.06],atol=1e-12)
    np.testing.assert_allclose(detail['slope'],[.001])


def test_single_observation_holds_and_missing_is_zero():
    x=np.zeros((2,134));x[0,:4]=[-.01,0,20/60,1]
    p,detail=trend_prediction(x,40)
    np.testing.assert_allclose(p[0],-.01)
    np.testing.assert_allclose(p[1],0)
    np.testing.assert_array_equal(detail['slope'],[0,0])
