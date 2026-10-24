"""Merewether physical-model adapter; published local prototype coordinates.

Use the consistent 2022 experiment bundle, not ANUGA's separate historic case.
No simulation output is used as experimental ground truth.
"""
from pathlib import Path
import csv
import hashlib
import json
import time
import urllib.request
import zipfile
import numpy as np
import raster_import

LAB=Path(__file__).resolve().parent
RECORD='https://zenodo.org/api/records/6914888'
PAPER='https://doi.org/10.1061/JHEND8.HYENG-13244'
FILES=('merewether-0p175m.dem','merewether-0p175m.par','merewether.bci','merewether.stage',
       'Location_of_velocity_sampling_points.csv')


def fetch(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    def get(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'HYRA-Merewether-validation'}),timeout=30)
    record=folder/'setup-record.json'
    if not record.exists():
        with get(RECORD) as response:record.write_bytes(response.read())
    meta=json.loads(record.read_text(encoding='utf-8'))
    item=next(f for f in meta['files'] if f['key']=='Setup_files_and_data.zip')
    package=folder/item['key']
    checksum=item['checksum'].split(':')[1]
    if not package.exists() or hashlib.md5(package.read_bytes()).hexdigest()!=checksum:
        started=time.monotonic();count=0;last=started;pending=package.with_suffix('.part')
        with get(item['links']['self']) as response,pending.open('wb') as output:
            while True:
                chunk=response.read(262144)
                if not chunk:break
                output.write(chunk);count+=len(chunk)
                if time.monotonic()-last>5:
                    print(json.dumps(dict(stage='download',bytes=count,total=item['size'])),flush=True);last=time.monotonic()
                if time.monotonic()-started>240:raise TimeoutError('Experiment download exceeded 240 seconds')
        if hashlib.md5(pending.read_bytes()).hexdigest()!=checksum:raise ValueError('Experiment package checksum mismatch')
        pending.replace(package)
    experiment=folder/'experiment';experiment.mkdir(exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        for filename in FILES:
            name=next(n for n in archive.namelist() if '/Test_4_' in n and n.endswith('/'+filename))
            target=experiment/filename
            target.write_bytes(archive.read(name))
    provenance=dict(record='https://zenodo.org/records/6914888',doi='10.5281/zenodo.6914888',
        license=meta['metadata'].get('license'),package_md5=checksum,paper=PAPER,
        files={f:hashlib.sha256((experiment/f).read_bytes()).hexdigest() for f in FILES},
        observation_kind='maximum velocity measurements from physical model, rescaled to prototype',
        missing='Four water-stage locations are supplied, but measured water-level values are not in this bundle')
    (folder/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return experiment,provenance


def observations(folder):
    with (Path(folder)/'Location_of_velocity_sampling_points.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    result=[]
    for row in rows:
        speed=float(row['Speed_m_s']);angle=float(row['Mod_Angle'])
        # Mod_Angle is the supplied mathematical angle in the rotated DEM frame.
        # Do not use Angle_TNth (bearing in the original geographic frame).
        x,y=float(row['x']),float(row['y']);theta=np.deg2rad(angle)
        if not np.isfinite([speed,angle,x,y]).all() or speed<0:raise ValueError('Invalid measured velocity')
        result.append(dict(id=row['ID'],x_m=x,y_m=y,speed_m_s=speed,angle_xy_deg=angle,
                           u_m_s=float(speed*np.cos(theta)),v_m_s=float(speed*np.sin(theta))))
    return result


def make_case(root,folder,factor=4,manning=.042,method='max',*,layered=False):
    folder=Path(folder);bed,valid,header=raster_import.read_ascii(folder/'merewether-0p175m.dem')
    layer_data=None;extra={}
    if layered:
        if not valid.all():raise ValueError('Layer extraction needs complete source surface')
        import obstacle_layers
        layer_data=obstacle_layers.merewether_layers(bed)
        coarse=obstacle_layers.aggregate(layer_data,factor)
        top,_,_=raster_import.coarsen(bed,valid,factor,'max')
        bed,solid,aggregation=coarse['ground'],coarse['solid'],coarse['audit']
        extra=dict(obstacle_top=top,obstacle_fraction=coarse['coverage'])
    else:bed,solid,aggregation=raster_import.coarsen(bed,valid,factor,method)
    dx=header['cellsize_m']*factor;nz,nx=bed.shape
    # Source QFIX is discharge per unit width, not total discharge.
    boundary=(folder/'merewether.bci').read_text().splitlines()
    s=next(line.split() for line in boundary if line.strip().startswith('S'))
    start,end,q_per_m=float(s[1]),float(s[2]),float(s[4])
    if s[3]!='QFIX' or not 0<=start<end<=nx*dx:raise ValueError('Unsupported experiment inflow')
    left=np.arange(nx)*dx;overlap=np.maximum(0,np.minimum(left+dx,end)-np.maximum(left,start))
    cells=[[nz-1,int(x)] for x in np.flatnonzero(overlap)]
    faces=[dict(edge='south',offset=int(x),fraction=float(np.clip(overlap[x]/dx,0,1))) for x in np.flatnonzero(overlap)]
    port=dict(id='merewether-inlet',kind='river',role='inlet',edge='south',label='实验南侧38米入流口',
              x=int(np.flatnonzero(overlap)[0]),z=nz-1,native_cells=cells,faces=faces,
              width_blocks=float((end-start)/dx),width_m=end-start,
              inward_tangent=dict(vector_xz=[0,-1],method='published_boundary_normal'))
    wx=np.zeros((nz,nx+1),bool);wy=np.zeros((nz+1,nx),bool)
    wx[:,0]=True;wx[:,-1]=True;wy[-1,:]=overlap==0
    scene=dict(mode='current',flood=dict(metres_per_block=1,roughness_mode='uniform',manning=manning,
        boundary='open',initial_water=False,baseflow=False,warmup_s=0,rain_mm_h=0,infiltration_mm_h=0,
        rain_duration_s=0,inflow_m3_s=0,duration_s=800,grid_factor=1,frames=81,
        flux_method='rusanov',spatial_order=2,time_integrator='rk2',low_froude=0,cfl=.4,
        max_timestep=1,max_steps=1000000),external_inflows=[dict(id=port['id'],kind='river',enabled=True,
        inlet_direction_mode='normal',inlet_velocity_m_s=0,hydrograph=[[0,q_per_m*(end-start)]])])
    meta=dict(name='Merewether / Morgan-Selwyn physical model',source='https://zenodo.org/records/6914888',
        paper=PAPER,coordinate_system='local rotated prototype x-east/y-north, metres; not a geographic EPSG',
        vertical_datum='published prototype DEM datum, unchanged',native=header,aggregation=aggregation,
        spatial_scale='prototype SI values already rescaled by data authors; do not rescale again',
        inflow=dict(edge='south',start_x_m=start,end_x_m=end,discharge_per_width_m2_s=q_per_m,
                    total_discharge_m3_s=q_per_m*(end-start)),
        boundaries='north free outflow; east/west and south outside inlet closed',
        roughness=dict(n=manning,units='s/m^(1/3)',reference_n=.042,tutorial_n=.02,
            calibrated_here=False,literature_calibration='0.042 fitted at water-level point 4 by source authors',
            reason='uniform road-base experimental surface; buildings and piers represented by supplied DEM, not extra vegetation drag'),
        sampling=dict(velocity='40 published maximum velocity vectors; not a time series',
            water='4 locations only; measured stage values unavailable in this package'),
        original_dem_sha256=hashlib.sha256((folder/'merewether-0p175m.dem').read_bytes()).hexdigest())
    if layered:meta['obstacle_layers']=dict(**layer_data['metadata'],geometry_audit=aggregation)
    variant='layered-v1' if layered else method
    name=f'merewether-f{factor}-{variant}-n{manning:g}'.replace('.','p')
    tid=raster_import.write(root,name,bed,dx,meta,solid=solid,wall_x=wx,wall_y=wy,scenario=scene,crossings=[port],**extra)
    scene['terrain_id']=tid
    target=Path(root)/tid
    (target.parent/'scenario.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf-8')
    (target.parent/'observations.json').write_text(json.dumps(observations(folder),indent=2),encoding='utf-8')
    if layered:np.savez_compressed(target.parent/'native-obstacle-layers.npz',**{k:v for k,v in layer_data.items() if isinstance(v,np.ndarray)})
    return tid,scene
