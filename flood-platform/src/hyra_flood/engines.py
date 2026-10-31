"""Backends run in their own Python environments, never silently substituted."""
import hashlib,importlib.metadata,os,sys,time
from pathlib import Path
import numpy as np
from .contracts import load_case,output_times,new_output,write_json,sha

CAPABILITIES={
 'legacy_hyra':{'implemented':True,'mesh':'square cells','case_scope':'closed rectangular verification cases'},
 'anuga_cpu':{'implemented':True,'mesh':'4 triangles per case cell','case_scope':'flat-bed closed rectangular verification cases'},
 'lisflood_dg2_gpu':{'implemented':False,'blocker':'binary/version/license/build and controlled case adapters not yet verified'},
 'local_ns':{'implemented':False,'blocker':'requires physical justification, local geometry/BC, validation and conservative coupling'},
}

def legacy_transform(fields):
    # HYRA rows increase south; canonical rows increase north. Momentum is a vector.
    return {k:(-v[::-1].copy() if k=='qy_m2_s' else v[::-1].copy()) for k,v in fields.items()}

def _legacy(c,f,legacy_lab):
    if legacy_lab is None:raise ValueError('Explicit legacy lab path required')
    lab=Path(legacy_lab).resolve();sys.path.insert(0,str(lab));os.environ['HYRA_FLOOD_BACKEND']='numba'
    from flood_solver import FloodSolver
    for module in ('flood_solver','flood_numerics','flood_acceleration'):
        if module in sys.modules and Path(sys.modules[module].__file__).resolve()!=lab/(module+'.py'):
            raise RuntimeError('Conflicting cached legacy module; run each backend in an isolated process')
    v=legacy_transform(f);n=c['numerics'];g=c['grid']
    s=FloodSolver(v['bed_m'],v['depth_m'],g['dx_m'],boundary='closed',manning=v['manning'],
        flux_method=n['native_flux'],low_froude=0,cfl=n['cfl'],max_timestep=n['max_dt_s'],spatial_order=2,time_integrator='rk2')
    s.mx[:]=v['qx_m2_s'];s.mz[:]=v['qy_m2_s']
    frames=[];vol=[];ledger=[];dt=[]
    def observe(sol):
        # Native advance also reports its initial state: only step increments
        # are accepted steps, not repeated observations at output boundaries.
        if sol.steps>observe.steps:
            if sol.steps!=observe.steps+1:raise ValueError('Missed accepted step')
            dt.append(float(sol.time-observe.previous));observe.steps=sol.steps
        observe.previous=sol.time
    observe.previous=0.;observe.steps=0
    for t in output_times(c):
        s.advance(float(t),observer=observe)
        frames.append((s.h[::-1].copy(),s.mx[::-1].copy(),-s.mz[::-1].copy()))
        b=s.summary()['balance'];vol.append(b['current_m3']);ledger.append(b['residual_m3'])
    meta={'backend':'legacy_hyra','version':'frozen_by_source_hash','native_numerics':s.numerical_config(),
        'source_sha256':{name:sha(lab/name) for name in ('flood_solver.py','flood_numerics.py','flood_acceleration.py')},
        'projection':'exact row reversal and y-momentum sign change; no resampling',
        'native_cells':int(s.h.size),'numerical_adjustment_m3':s.balance['numerical_adjustment_m3']}
    return frames,np.array(vol),np.zeros(len(vol)),np.zeros(len(vol)),np.array(dt),meta

def _anuga(c,f):
    if np.ptp(f['bed_m'])>0:raise NotImplementedError('ANUGA adapter v0.1 requires flat bed; never replace terrain silently')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    import anuga
    from anuga.operators.base_operator import Operator
    g=c['grid'];ny,nx=g['shape'];dx=g['dx_m'];ox,oy=g['origin_xy_m'];n=c['numerics']
    d=anuga.rectangular_cross_domain(nx,ny,len1=nx*dx,len2=ny*dx,origin=(0.,0.))
    d.set_flow_algorithm(n['anuga_algorithm']);d.set_cfl(n['cfl']);d.set_store(False)
    d.set_evolve_max_timestep(n['max_dt_s'])
    # Keep mesh arithmetic local to avoid large-origin area cancellation.
    cent=d.get_centroid_coordinates(absolute=True);areas=d.get_areas()
    cent=cent+np.array([ox,oy])
    ix=np.floor((cent[:,0]-ox)/dx).astype(int);iy=np.floor((cent[:,1]-oy)/dx).astype(int)
    if (ix<0).any() or (ix>=nx).any() or (iy<0).any() or (iy>=ny).any():raise ValueError('Mesh outside physical domain')
    index=iy*nx+ix;cell_areas=np.bincount(index,weights=areas,minlength=ny*nx)
    np.testing.assert_allclose(cell_areas,dx*dx,rtol=1e-12,atol=1e-12)
    d.set_quantity('elevation',float(f['bed_m'][0,0]))
    for name,values in (('stage',f['bed_m']+f['depth_m']),('xmomentum',f['qx_m2_s']),('ymomentum',f['qy_m2_s']),('friction',f['manning'])):
        d.set_quantity(name,values[iy,ix],location='centroids')
    wall=anuga.Reflective_boundary(d);d.set_boundary({tag:wall for tag in d.get_boundary_tags()})
    class Steps(Operator):
        def __init__(self,domain):super().__init__(domain,label='accepted_dt');self.values=[]
        def __call__(self):self.values.append(float(self.domain.timestep))
    rec=Steps(d)
    def project(values):return (np.bincount(index,weights=values*areas,minlength=ny*nx)/cell_areas).reshape(ny,nx)
    frames=[];vol=[];boundary=[];sources=[];ts=[]
    for t in d.evolve(yieldstep=c['time']['output_interval_s'],finaltime=c['time']['end_s']):
        h=d.quantities['stage'].centroid_values-d.quantities['elevation'].centroid_values
        if not np.isfinite(h).all() or h.min() < -1e-10:raise ValueError('Native ANUGA depth invalid')
        frames.append(tuple(project(v) for v in (h,d.quantities['xmomentum'].centroid_values,d.quantities['ymomentum'].centroid_values)))
        vol.append(float(d.get_water_volume()));boundary.append(float(d.get_boundary_flux_integral()));sources.append(float(d.get_fractional_step_volume_integral()));ts.append(t)
    np.testing.assert_allclose(ts,output_times(c),atol=1e-10,rtol=0)
    meta={'backend':'anuga_cpu','version':importlib.metadata.version('anuga'),'algorithm':n['anuga_algorithm'],
        'mesh_coordinate_offset_xy_m':[ox,oy],'actual_integrator':d.timestepping_method,'cfl':n['cfl'],'native_cells':len(d),
        'projection':'area-weighted conserved depth and momenta from exactly covered 4-triangle cells; not point interpolation',
        'mesh_sha256':hashlib.sha256(cent.tobytes()+areas.tobytes()).hexdigest()}
    return frames,np.array(vol),np.array(boundary),np.array(sources),np.array(rec.values),meta

def run(case,backend,out,legacy_lab=None):
    if backend not in CAPABILITIES or not CAPABILITIES[backend]['implemented']:raise NotImplementedError(f'{backend} is not implemented; no fallback')
    c,f,fingerprint=load_case(case);started=time.perf_counter();folder=new_output(out)
    write_json(folder/'status.json',{'state':'running','case_sha256':fingerprint,'backend':backend})
    try:
        result=_legacy(c,f,legacy_lab) if backend=='legacy_hyra' else _anuga(c,f)
        frames,vol,boundary,sources,dt,meta=result;frames=np.array(frames)
        if frames.shape!=(len(output_times(c)),3,*c['grid']['shape']) or not np.isfinite(frames).all():raise ValueError('Invalid engine fields')
        if dt.size==0 or not np.isfinite(dt).all() or (dt<=0).any() or abs(dt.sum()-c['time']['end_s'])>1e-8:raise ValueError('Invalid accepted-step audit')
        np.testing.assert_allclose(frames[0,0],f['depth_m'],atol=1e-12,rtol=0)
        for i,key in ((1,'qx_m2_s'),(2,'qy_m2_s')):np.testing.assert_allclose(frames[0,i],f[key],atol=1e-12,rtol=0)
        projected=frames[:,0].sum(axis=(1,2))*c['grid']['dx_m']**2
        np.testing.assert_allclose(projected,vol,rtol=1e-12,atol=1e-10)
        residual=vol-vol[0]-(boundary-boundary[0])-(sources-sources[0])
        if abs(residual).max()>1e-10*max(abs(vol[0]),1.):raise ValueError('Closed verification mass gate failed')
        np.savez_compressed(folder/'fields.npz',time_s=output_times(c),bed_m=f['bed_m'],depth_m=frames[:,0],qx_m2_s=frames[:,1],qy_m2_s=frames[:,2],native_volume_m3=vol,net_boundary_volume_m3=boundary,net_source_volume_m3=sources,accepted_dt_s=dt)
        meta.update(schema='hyra-flood-run-v1',complete=True,case=c,case_sha256=fingerprint,fields_sha256=sha(folder/'fields.npz'),
            platform_source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
            wall_s=time.perf_counter()-started,accepted_steps=len(dt),min_dt_s=float(dt.min()),max_dt_s=float(dt.max()),
            max_abs_mass_residual_m3=float(abs(residual).max()),max_projection_volume_error_m3=float(abs(projected-vol).max()),
            validation_status='engineering verification only; not physical accuracy acceptance')
        write_json(folder/'run.json',meta);write_json(folder/'status.json',{'state':'complete'});return meta
    except Exception as e:
        write_json(folder/'status.json',{'state':'failed','error':str(e)});raise
