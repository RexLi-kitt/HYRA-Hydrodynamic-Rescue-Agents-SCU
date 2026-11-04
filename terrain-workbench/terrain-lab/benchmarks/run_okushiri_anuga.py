"""ANUGA reference adapter: original coordinates, SI units and unshifted times."""
import os
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,hashlib,importlib.metadata as md
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
import anuga
from anuga.operators.base_operator import Operator

class AcceptedStepRecorder(Operator):
    """Read-only observer called once after each accepted transport step."""
    def __init__(self, domain):
        super().__init__(domain, label='accepted_step_audit')
        self.dt=[]
    def __call__(self):
        self.dt.append(float(self.domain.timestep))


def run(data,out,spacing,end,cfl=.4):
    data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'timeseries.npz').exists():raise FileExistsError('Use a new output directory')
    source=np.loadtxt(data/'Benchmark_2_Bathymetry.asc',skiprows=6)
    wave=np.loadtxt(data/'Benchmark_2_input.txt',skiprows=1)
    obs=np.loadtxt(data/'output_ch5-7-9.txt',skiprows=1)
    if not (0<end<=22.5):raise ValueError('Input only supports 0--22.5 s')
    if not np.isclose(end/.05,round(end/.05)):raise ValueError('End must be an observation time')
    prov=json.loads((data/'provenance.json').read_text())
    for f in prov['files']:
        assert hashlib.sha256((data/f['file']).read_bytes()).hexdigest()==f['sha256']
    def bed(x,y):
        return map_coordinates(source,[source.shape[0]-1-np.asarray(y)/.014,np.asarray(x)/.014],order=1,mode='nearest')
    nx=int(np.ceil(5.448/spacing));ny=int(np.ceil(3.402/spacing))
    d=anuga.rectangular_cross_domain(nx,ny,len1=5.448,len2=3.402)
    d.set_flow_algorithm('DE0');d.set_cfl(cfl);d.set_store(False)
    recorder=AcceptedStepRecorder(d)
    d.set_quantity('elevation',bed);d.set_quantity('stage',0.);d.set_quantity('friction',.0025)
    def incoming(t):
        if t < -1e-10 or t>22.5+1e-10:raise ValueError('Boundary extrapolation forbidden')
        return np.array([np.interp(t,wave[:,0],wave[:,1]),0.,0.])
    b=anuga.Transmissive_n_momentum_zero_t_momentum_set_stage_boundary(d,incoming)
    wall=anuga.Reflective_boundary(d)
    d.set_boundary({tag:b if tag=='left' else wall for tag in d.get_boundary_tags()})
    gauges=np.array([[4.521,y] for y in [1.196,1.696,2.196]])
    config={'version':md.version('anuga'),'algorithm':'DE0','cfl':cfl,'domain_m':[5.448,3.402],
            'nx':nx,'ny':ny,'actual_rect_spacing':[5.448/nx,3.402/ny],'triangles':len(d),
            'manning':.0025,'initial_stage_m':0,'end_s':end,'source_commit':prov['commit'],
            'bed':'original-node bilinear field at triangular vertices; differs from HYRA cell-center bed',
            'sampling':'ANUGA quantity interpolation at fixed physical gauges',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'configuration.json').write_text(json.dumps(config,indent=2))
    ts=[];pred=[];vol=[];boundary=[];fractional=[];start=time.perf_counter()
    for t in d.evolve(yieldstep=.05,finaltime=end):
        ts.append(float(t));pred.append(d.quantities['stage'].get_values(interpolation_points=gauges).tolist())
        vol.append(float(d.get_water_volume()))
        boundary.append(float(d.get_boundary_flux_integral()))
        fractional.append(float(d.get_fractional_step_volume_integral()))
        h=d.quantities['stage'].centroid_values-d.quantities['elevation'].centroid_values
        assert np.isfinite(h).all() and h.min()>-1e-10
    ts=np.array(ts);pred=np.array(pred);count=len(ts)
    assert np.allclose(ts,obs[:count,0],atol=1e-10,rtol=0)
    assert np.isfinite(pred).all()
    dt=np.asarray(recorder.dt)
    assert len(dt)>0 and np.isfinite(dt).all() and (dt>0).all()
    assert abs(dt.sum()-end)<1e-12
    residual=np.asarray(vol)-vol[0]-(np.asarray(boundary)-boundary[0])-(np.asarray(fractional)-fractional[0])
    np.savez_compressed(out/'timestep_audit.npz',accepted_dt_s=dt)
    np.savez_compressed(out/'timeseries.npz',time_s=ts,predicted_m=pred,observed_m=obs[:count,1:]/100,volume_m3=vol,boundary_volume_m3=boundary,fractional_volume_m3=fractional,balance_residual_m3=residual)
    result={**config,'samples':count,'wall_s':time.perf_counter()-start,'min_depth_m':float(h.min()),'complete_physical_window':end==22.5,'steps':len(dt),'min_dt_s':float(dt.min()),'max_dt_s':float(dt.max()),'dt_sum_s':float(dt.sum()),'max_abs_balance_residual_m3':float(abs(residual).max()),'time_integrator':d.timestepping_method}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('out');p.add_argument('--spacing',type=float,default=.05);p.add_argument('--end',type=float,default=.1)
    p.add_argument('--cfl',type=float,default=.4)
    a=p.parse_args();run(a.data,a.out,a.spacing,a.end,a.cfl)
