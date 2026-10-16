"""Independent ANUGA reference, nominal state and rolling forecast simulations."""


import numpy as np
from scipy.spatial import cKDTree
from ..artifacts import write_json


def domain_for(params, rate_factor=1.0, initial=None, start=0, rate_switch_s=None, switched_factor=None):
    import anuga
    domain = anuga.rectangular_cross_domain(10,10,len1=100.,len2=100.)
    domain.set_store(False)
    domain.set_flow_algorithm("DE0")
    def terrain(x,y):
        return -params["slope"]*x + params["ridge"]*np.exp(-((y-50)/12)**2)
    domain.set_quantity("elevation",terrain)
    domain.set_quantity("friction",.03)
    domain.set_quantity("stage",expression="elevation + 0.15")
    if initial is not None:
        for key, values in initial.items():
            domain.set_quantity(key,values,location="centroids")
    domain.set_starttime(float(start))
    def boundary(t):
        # Change the slope continuously: no artificial instantaneous water jump.
        elapsed=max(t-params["onset"],0)
        if rate_switch_s is None:
            integrated=rate_factor*elapsed
        else:
            before=max(min(t,rate_switch_s)-params["onset"],0)
            after=max(t-max(rate_switch_s,params["onset"]),0)
            integrated=rate_factor*before+switched_factor*after
        return [.15+params["rate"]*integrated,0.,0.]
    left = anuga.Time_boundary(domain,boundary)
    wall = anuga.Reflective_boundary(domain)
    domain.set_boundary(dict(left=left,right=wall,top=wall,bottom=wall))
    return domain


def simulate(domain, finaltime, xy, yieldstep=2):
    distance, indices = cKDTree(domain.get_centroid_coordinates()).query(xy)
    times, depths, states = [],[],[]
    for t in domain.evolve(yieldstep=yieldstep,finaltime=finaltime):
        times.append(float(t))
        depths.append(np.maximum(domain.quantities["stage"].centroid_values-
            domain.quantities["elevation"].centroid_values,0)[indices].copy())
        states.append({key:domain.quantities[key].centroid_values.copy()
                       for key in ("stage","xmomentum","ymomentum")})
    return np.asarray(times),np.asarray(depths),states,dict(indices=indices.tolist(),max_distance_m=float(distance.max()))


def generate_event(params, output_path, metadata):
    import anuga
    xy = np.array([(5+10*x,5+10*y) for y in range(10) for x in range(10)])
    reference = domain_for(params)
    times,truth,reference_states,mapping = simulate(reference,360,xy)
    nominal = domain_for(params,.75)
    ntimes,nominal_depth,nominal_states,_ = simulate(nominal,300,xy)
    releases = np.arange(0,301,10,dtype=float)
    forecasts,valid_times = [],[]
    for release in releases:
        index = int(np.flatnonzero(ntimes == release)[0])
        forecast = domain_for(params,.75,nominal_states[index],release)
        ft,fd,_,_ = simulate(forecast,release+60,xy)
        forecasts.append(fd)
        valid_times.append(ft)
    metadata.update(source="ANUGA",solver_version=anuga.__version__,
        units=dict(time="s",xy="m",depth="m"),params=params,mapping=mapping,
        forecasting="Independent nominal trajectory; fresh domain every10s; bias factor0.75; no truth/observations reset")
    from pathlib import Path
    raw_path = Path(output_path).with_suffix(".solver.npz")
    np.savez_compressed(raw_path,centroid_xy_m=reference.get_centroid_coordinates(),
        bed_m=reference.quantities["elevation"].centroid_values,time_s=times,nominal_time_s=ntimes,
        **{f"reference_{key}":np.array([s[key] for s in reference_states]) for key in reference_states[0]},
        **{f"nominal_{key}":np.array([s[key] for s in nominal_states]) for key in nominal_states[0]})
    metadata["raw_solver_file"] = raw_path.name
    export_event(dict(time_s=times,truth_depth_m=truth,xy_m=xy),
        dict(release_s=releases,forecast_valid_s=np.array(valid_times),forecast_depth_m=np.array(forecasts),
             nominal_time_s=ntimes,nominal_depth_m=nominal_depth),output_path,metadata)


def export_event(reference_source, forecast_source, output_path, metadata):
    from pathlib import Path
    p = Path(output_path)
    if p.exists():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**reference_source,**forecast_source)
    write_json(p.with_suffix(".json"),metadata)
