import numpy as np
import pytest
from scipy.stats import multivariate_normal,norm
from hyra_validation.nominal_response import half_gaussian_posterior,interpolate_nodes,public_nominal_view


def test_batched_half_gaussian_evidence_matches_direct_observation_covariance():
    rng=np.random.default_rng(18);x=rng.normal(size=(3,8,2));y=rng.normal(0,.1,8)
    sigma=.04;prior=np.diag([.02**2,2.**2]);out=half_gaussian_posterior(x,y,sigma,.02,2.)
    for i in range(3):
        covariance=x[i]@prior@x[i].T+sigma*sigma*np.eye(8)
        evidence=multivariate_normal.logpdf(y,mean=np.zeros(8),cov=covariance)
        mu=prior@x[i].T@np.linalg.solve(covariance,y)
        cov=prior-prior@x[i].T@np.linalg.solve(covariance,x[i]@prior)
        z=mu[1]/np.sqrt(cov[1,1]);expected=evidence+np.log(2)+norm.logcdf(z)
        np.testing.assert_allclose(out['untruncated_mean'][i],mu,atol=1e-10)
        np.testing.assert_allclose(out['log_evidence'][i],expected,atol=1e-8)
        assert out['mean'][i,1]>=0


def test_zero_pulse_column_has_no_extra_evidence_penalty():
    x=np.zeros((1,4,2));y=np.zeros(4);out=half_gaussian_posterior(x,y,.02,.02,2.)
    expected=4*norm.logpdf(0,scale=.02)
    assert out['log_evidence'][0]==pytest.approx(expected)


def test_response_interpolation_zero_before_event_and_no_future_extrapolation():
    times=np.array([0,2,4]);curves=np.array([[0,0],[2,4],[4,8]])
    np.testing.assert_allclose(interpolate_nodes(times,curves,[-1,1,3],[0,1,0],True),[0,2,3])
    with pytest.raises(ValueError):interpolate_nodes(times,curves,[5],[0],True)


def test_public_view_rejects_reference_calibration_even_if_key_looks_nominal():
    manifest=dict(nominal_base='nominal_base',nominal_sensitivities={'plus':'p','minus':'m'},nominal_calibration={},forecast_versions={},fields={
        'nominal_base':dict(kind='truth_base',projected_path='unused',projected_sha256='sha'),
        'p':dict(kind='nominal_plus',projected_path='unused',projected_sha256='sha'),
        'm':dict(kind='nominal_minus',projected_path='unused',projected_sha256='sha')})
    with pytest.raises(ValueError):public_nominal_view(manifest)
