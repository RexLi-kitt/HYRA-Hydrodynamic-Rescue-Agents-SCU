import numpy as np
from hyra_exp3.model_features import full_raw_frame
from hyra_exp3.common_data import collect_patrol
from hyra_exp3.records import save_observations
from hyra_exp3.simulation import EpisodeConfig, run_episode
from hyra_exp3.windows import NominalPredictor

def test_all_raw_samples_are_present_and_delayed_samples_are_absent(tmp_path):
    side=32
    times=np.arange(61.)
    field=dict(times_s=times,depths_m=np.full((61,side*side),.2),elevation_m=np.zeros(side*side),positions_m=np.zeros((side*side,2)))
    panel,raw,_=collect_patrol(field,field,side,4,42,duration_s=60)
    path=tmp_path/'observations.npz'
    save_observations(path,raw)
    with np.load(path) as saved:
        obs={k:saved[k] for k in saved.files}
    features=full_raw_frame(panel,obs,30,10)
    assert features.shape==(1024,13,46)
    legal=(obs['sample_time_s']<=20)
    assert int(features[:,:,26:].sum())==int(legal.sum())
    for i in np.flatnonzero(legal):
        cell=obs['row'][i]*side+obs['col'][i]
        slot=int(12-(30-obs['sample_time_s'][i])/5)
        assert features[cell,slot,6+obs['drone'][i]]==np.float32(obs['depth_m'][i])

def test_live_engine_retains_full_overlapping_raw_history_for_all_trained_methods():
    class FullHistory(NominalPredictor):
        requires_full_raw_history=True
        def issue(self, now, observations):
            if now==20:
                assert len(observations)==20  # 4 co-located drones x 5 ticks.
            return super().issue(now,observations)
    times=np.arange(31.)
    depth=np.full((31,1,1),.2)
    c=EpisodeConfig(map_shape=(1,1),n_drones=4,n_targets=0,duration_s=30,sensor_radius_cells=0)
    run_episode(depth,c,times=times,predictor=FullHistory(times,depth),starts=[(0,0)]*4)
