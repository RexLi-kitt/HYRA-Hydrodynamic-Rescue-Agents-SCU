import numpy as np
from hyra_validation.metric_readers import fit_ridge,predict_ridge,fit_pointwise,predict_pointwise,pointwise_extras
from hyra_validation.metric_features import raw_features,empirical_features


def test_pointwise_matches_independent_augmented_ridge():
    rng=np.random.default_rng(394);x=rng.normal(size=(45,8));x[:,3]=1.;y=rng.normal(size=(45,4))
    extra=rng.normal(size=(45,4,3));weight=rng.uniform(.3,2.,45);query=rng.normal(size=(7,8));query_extra=rng.normal(size=(7,4,3))
    joint=fit_pointwise(x,extra,y,weight,10.);actual=predict_pointwise(joint,query,query_extra)
    for q in range(4):
        direct=fit_ridge(np.c_[x,extra[:,q]],y[:,q],weight,10.)
        expected=predict_ridge(direct,np.c_[query,query_extra[:,q]])
        np.testing.assert_allclose(actual[:,q],expected,atol=1e-12,rtol=1e-12)
    # Altering another horizon's physical feature cannot change this horizon.
    changed=query_extra.copy();changed[:,2]+=100
    np.testing.assert_array_equal(actual[:,0],predict_pointwise(joint,query,changed)[:,0])


def test_all_raw_values_preserved_and_empirical_missingness_finite():
    h=np.zeros((3,6,12,6));h[0,4,11,2]=.2;h[0,4,11,5]=1
    raw=raw_features(h);assert raw.shape==(3,432) and .2 in raw[0]
    for width in (5.,10.,20.):
        f=empirical_features(h,width);assert f.shape==(3,110) and np.all(np.isfinite(f))
        np.testing.assert_array_equal(f[1:],0)


def test_pointwise_probability_broadcast():
    p=np.ones((9,12,26));e=pointwise_extras(p,2*p,np.arange(108).reshape(9,12))
    assert e.shape==(9,12,26,3);np.testing.assert_array_equal(e[:,:,0,2],np.arange(108).reshape(9,12))
