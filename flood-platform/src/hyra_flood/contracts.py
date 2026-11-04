"""Strict case and artifact contracts. SI, east/north, row zero at south."""
from pathlib import Path
import hashlib,json
import numpy as np

SCHEMA='hyra-flood-case-v1'
COORDINATES='local_x_east_y_north_row0_south'
UNITS={'length':'m','time':'s','velocity':'m/s'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def exact_keys(obj,keys,where):
    if not isinstance(obj,dict) or set(obj)!=set(keys):raise ValueError(f'{where}: missing or unknown keys')

def load_case(path):
    path=Path(path).resolve();c=read_json(path)
    exact_keys(c,('schema','id','evidence','coordinates','units','grid','arrays','boundaries','time','numerics','gauges'),'case')
    if c['schema']!=SCHEMA or c['coordinates']!=COORDINATES or c['units']!=UNITS:raise ValueError('Unsupported schema/coordinates/units')
    if not isinstance(c['id'],str) or not c['id']:raise ValueError('Missing case id')
    exact_keys(c['evidence'],('category','source','role'),'evidence')
    if c['evidence']['category'] not in ('analytic','laboratory','field','synthetic'):raise ValueError('Unknown evidence category')
    if c['evidence']['role'] not in ('verification','calibration','holdout'):raise ValueError('Unknown evidence role')
    if not isinstance(c['evidence']['source'],str) or not c['evidence']['source'].strip():raise ValueError('Evidence source required')
    g=c['grid'];exact_keys(g,('shape','dx_m','origin_xy_m'),'grid')
    if len(g['shape'])!=2 or any(type(n)!=int or n<2 for n in g['shape']):raise ValueError('Grid shape')
    if isinstance(g['dx_m'],bool) or not np.isfinite(g['dx_m']) or g['dx_m']<=0:raise ValueError('Grid spacing')
    if len(g['origin_xy_m'])!=2 or not np.isfinite(g['origin_xy_m']).all():raise ValueError('Grid origin')
    a=c['arrays'];exact_keys(a,('path','sha256'),'arrays')
    ap=(path.parent/a['path']).resolve()
    if path.parent not in ap.parents:raise ValueError('Array path must stay inside case directory')
    if sha(ap)!=a['sha256']:raise ValueError('Input array SHA256 mismatch')
    with np.load(ap,allow_pickle=False) as z:
        if set(z.files)!=set(('bed_m','depth_m','qx_m2_s','qy_m2_s','manning')):raise ValueError('Missing/unsupported fields (no silent geometry omission)')
        fields={k:z[k].astype(np.float64) for k in z.files}
    for name,v in fields.items():
        if v.shape!=tuple(g['shape']) or not np.isfinite(v).all():raise ValueError(f'Invalid array {name}')
    if (fields['depth_m']<0).any() or (fields['manning']<0).any():raise ValueError('Negative depth/roughness')
    dry=fields['depth_m']==0
    if np.any(fields['qx_m2_s'][dry]!=0) or np.any(fields['qy_m2_s'][dry]!=0):raise ValueError('Nonzero momentum in zero water')
    exact_keys(c['boundaries'],('west','east','south','north'),'boundaries')
    for edge,b in c['boundaries'].items():
        exact_keys(b,('kind',),edge)
        if b['kind']!='reflective':raise NotImplementedError('v0.1 accepts closed verification cases only; forcing requires new audited adapter')
    t=c['time'];exact_keys(t,('end_s','output_interval_s'),'time')
    if not np.isfinite(list(t.values())).all() or min(t.values())<=0:raise ValueError('Invalid time')
    if abs(t['end_s']/t['output_interval_s']-round(t['end_s']/t['output_interval_s']))>1e-10:raise ValueError('End must be an output time')
    n=c['numerics'];exact_keys(n,('cfl','max_dt_s','native_flux','anuga_algorithm'),'numerics')
    if not np.isfinite([n['cfl'],n['max_dt_s']]).all() or not 0<n['cfl']<=.45 or n['max_dt_s']<=0:raise ValueError('Invalid numerics')
    if n['native_flux'] not in ('rusanov','knp','hllc') or n['anuga_algorithm']!='DE1':raise ValueError('Unsupported explicitly selected algorithm')
    ids=[];oy,ox=g['origin_xy_m'][1],g['origin_xy_m'][0];ny,nx=g['shape'];dx=g['dx_m']
    for p in c['gauges']:
        exact_keys(p,('id','x_m','y_m'),'gauge');ids.append(p['id'])
        if not isinstance(p['id'],str) or not p['id'] or not np.isfinite([p['x_m'],p['y_m']]).all():raise ValueError('Invalid gauge')
        if not ox<=p['x_m']<ox+nx*dx or not oy<=p['y_m']<oy+ny*dx:raise ValueError('Gauge outside domain; never clamp')
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate gauges')
    return c,fields,sha(path)

def output_times(c):
    t=c['time'];return np.arange(round(t['end_s']/t['output_interval_s'])+1,dtype=float)*t['output_interval_s']

def new_output(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);return path

def load_run(folder):
    folder=Path(folder);m=read_json(folder/'run.json')
    if m.get('schema')!='hyra-flood-run-v1' or m.get('complete') is not True:raise ValueError('Incomplete/nonstandard run')
    if sha(folder/'fields.npz')!=m['fields_sha256']:raise ValueError('Result SHA256 mismatch')
    with np.load(folder/'fields.npz',allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
    c=m['case'];t=output_times(c);shape=(len(t),*c['grid']['shape'])
    if a['time_s'].shape!=t.shape or not np.allclose(a['time_s'],t,rtol=0,atol=1e-10):raise ValueError('Missing or shifted times')
    for k in ('depth_m','qx_m2_s','qy_m2_s'):
        if a[k].shape!=shape or not np.isfinite(a[k]).all():raise ValueError('Invalid full field')
    if (a['depth_m'] < -1e-10).any():raise ValueError('Negative depth')
    if a['bed_m'].shape!=shape[1:] or not np.isfinite(a['bed_m']).all():raise ValueError('Invalid bed')
    for k in ('native_volume_m3','net_boundary_volume_m3','net_source_volume_m3'):
        if a[k].shape!=t.shape or not np.isfinite(a[k]).all():raise ValueError('Invalid water ledger')
    projected=a['depth_m'].sum(axis=(1,2))*c['grid']['dx_m']**2
    if not np.allclose(projected,a['native_volume_m3'],rtol=1e-12,atol=1e-10):raise ValueError('Conservative projection audit failed')
    dt=a['accepted_dt_s']
    if dt.ndim!=1 or dt.size==0 or not np.isfinite(dt).all() or (dt<=0).any() or abs(dt.sum()-t[-1])>1e-8:raise ValueError('Invalid accepted time steps')
    return m,a
