import numpy as np
from hyra_exp3.links import build_graph, regression_rows

def test_frozen_graph_reads_no_true_labels_or_future_nominal_values():
    side=32
    yy,xx=np.meshgrid(np.arange(side),np.arange(side),indexing='ij')
    panel=dict(positions_m=np.column_stack((xx.ravel(),yy.ravel()))*10+5,
        elevation_m=-xx.ravel()/side*.3,nominal_sample_m=np.full((10,side*side),.3),label_truth_sample_m=np.zeros((10,side*side)))
    values=np.ones((16,16)).tolist()
    bank=dict(banks={'32':dict(train_gain=values,held_out_train_gain=values,
        lag_steps=np.zeros((16,16),int).tolist(),held_out_train_improving_families=(np.ones((16,16))*6).tolist())})
    a=build_graph(bank,panel,side)
    panel['label_truth_sample_m'][:]=999
    panel['nominal_sample_m'][1:]=999
    b=build_graph(bank,panel,side)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert (a['P_mask'].sum(1)<=8).all()
    for target,source in zip(*np.nonzero(a['P_mask'])):
        assert source%4 <= target%4

def test_link_regression_inputs_are_past_observations_only():
    obs=np.arange(30*16.).reshape(30,16)
    panel=dict(observed=obs,nominal=np.zeros_like(obs),labels=obs.copy(),global_mean=obs.mean(1))
    b,x,y=regression_rows(panel,2,3,2)
    assert np.array_equal(x,obs[np.arange(3,27)-2,3])
    assert np.array_equal(y,obs[np.arange(3,27)+3,2])
    assert np.array_equal(b[:,1],obs[np.arange(3,27),2])
