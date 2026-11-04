"""Parent parameter sampler extracted unchanged from the original dataset module."""
import numpy as np

def parent_parameters(seed):
    rng=np.random.default_rng(seed)
    return dict(seed=int(seed), phase=float(rng.uniform(-np.pi,np.pi)),
                wavelength_cells=float(rng.uniform(30.,65.)),
                meander_cells=float(rng.uniform(12.,27.)),
                center_cells=float(rng.uniform(95.,145.)),
                valley_curvature=float(rng.uniform(.00002,.00006)),
                longitudinal_slope=float(rng.uniform(.0025,.005)),
                roughness_amplitude_m=float(rng.uniform(.05,.18)),
                peak_flow_m3_s=float(rng.uniform(8.,30.)),
                peak_time_s=int(rng.choice([300,450,600])),
                final_flow_m3_s=float(rng.uniform(.2,1.)))
