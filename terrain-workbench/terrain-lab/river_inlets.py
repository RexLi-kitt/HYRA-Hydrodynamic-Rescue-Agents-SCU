"""Native-grid river boundary crossings and explicit external volume sources.

Direction is a geometric prior, not a measured flow. Ambiguous crossings remain
unclassified and no extra discharge is enabled automatically.
"""
from collections import Counter,deque
from pathlib import Path
import hashlib,json,math,re,uuid
import numpy as np

SCHEMA='hyra-river-inlets-v2'

def water_types(types,bed,water,zones,terrain_ids,zone_ids,sea_level):
    """Recover sea cells overwritten by the generator's river influence.

    Only submerged, sea-connected river tags with explicit RiverZone.None
    qualify. Riverbed/banks/valley cells remain rivers even below sea level.
    """
    types=np.asarray(types);corrected=types.copy()
    ocean=np.isin(types,[int(k) for k,v in terrain_ids.items() if 'ocean' in v])
    river=np.isin(types,[int(k) for k,v in terrain_ids.items() if v=='river'])
    none=[int(k) for k,v in zone_ids.items() if v=='None']
    sea_code=next((int(k) for k,v in terrain_ids.items() if v=='ocean'),None)
    if zones is None or not none or sea_code is None:return corrected,np.zeros_like(ocean)
    eligible=river&np.isin(zones,none)&(bed<sea_level)&np.isfinite(water)&(water>bed)&(water<=sea_level+.01)
    # Propagate only from actual ocean cells; an isolated low river/lake is not sea.
    connected=ocean.copy();queue=deque(map(tuple,np.argwhere(ocean)))
    nz,nx=types.shape
    while queue:
        z,x=queue.popleft()
        for zz,xx in ((z-1,x),(z+1,x),(z,x-1),(z,x+1)):
            if 0<=zz<nz and 0<=xx<nx and eligible[zz,xx] and not connected[zz,xx]:
                connected[zz,xx]=True;queue.append((zz,xx))
    changed=connected&eligible;corrected[changed]=sea_code
    return corrected,changed

def detect(river,bed,water,origin=(0,0),step=1):
    river=np.asarray(river,bool);bed=np.asarray(bed,float);water=np.asarray(water,float)
    if river.ndim!=2 or min(river.shape)<2 or bed.shape!=river.shape or water.shape!=river.shape:
        raise ValueError('河道入口识别需要同尺寸二维数组')
    if not np.isfinite(bed).all() or not math.isfinite(step) or step<=0:raise ValueError('河道高程或间距无效')
    nz,nx=river.shape
    faces=([('north',x,0,x) for x in range(nx)] + [('east',z,z,nx-1) for z in range(nz)]
        + [('south',x,nz-1,x) for x in range(nx-1,-1,-1)] + [('west',z,z,0) for z in range(nz-1,-1,-1)])
    hits=[bool(river[z,x]) for _,_,z,x in faces]
    if not any(hits):return []
    # Traverse the perimeter once after a dry face; corners are a single crossing.
    start=next((i+1 for i,v in enumerate(hits) if not v),0)%len(faces)
    groups=[];run=[]
    for j in range(len(faces)):
        i=(start+j)%len(faces)
        if hits[i]:run.append(faces[i])
        elif run:groups.append(run);run=[]
    if run:groups.append(run)
    result=[]
    for group in groups:
        cells=sorted({(z,x) for _,_,z,x in group});edge=Counter(f[0] for f in group).most_common(1)[0][0]
        distance={'north':lambda z,x:z,'south':lambda z,x:nz-1-z,
                  'west':lambda z,x:x,'east':lambda z,x:nx-1-x}[edge]
        radius=min(8,max(2,min(nz,nx)//4));queue=deque((z,x,0) for z,x in cells);seen=set(cells);inside=[]
        while queue:
            z,x,d=queue.popleft()
            if distance(z,x)>=min(3,radius):inside.append((z,x))
            if d>=radius:continue
            for dz in (-1,0,1):
                for dx in (-1,0,1):
                    zz,xx=z+dz,x+dx
                    if 0<=zz<nz and 0<=xx<nx and (zz,xx) not in seen and river[zz,xx]:
                        seen.add((zz,xx));queue.append((zz,xx,d+1))
        stage=np.where(np.isfinite(water)&(water>bed+.01),water,bed)
        edge_stage=float(np.median([stage[z,x] for z,x in cells]))
        drop=0.;basis='insufficient_interior';role='uncertain'
        if inside:
            drop=edge_stage-float(np.median([stage[z,x] for z,x in inside]));basis='water_surface'
            if abs(drop)<=.01:
                drop=float(np.median([bed[z,x] for z,x in cells])-np.median([bed[z,x] for z,x in inside]));basis='bed_slope'
            if abs(drop)>.01:role='inlet' if drop>0 else 'outlet'
        central=group[len(group)//3:max(len(group)//3+1,2*len(group)//3)]
        _,_,z,x=min(central,key=lambda f:(bed[f[2],f[3]],f[2],f[3]))
        digest=hashlib.sha256(json.dumps(sorted(group),separators=(',',':')).encode()).hexdigest()[:10]
        result.append(dict(id='river-'+digest,kind='river',role=role,edge=edge,
            label=f'河道交点 {len(result)+1}',x=int(x),z=int(z),native_cells=[list(c) for c in cells],
            faces=[dict(edge=e,offset=int(o),z=int(zz),x=int(xx)) for e,o,zz,xx in group],
            width_blocks=len(group)*step,world_blocks=[origin[0]+x*step,origin[1]+z*step],
            inward_stage_drop_blocks=drop,direction_basis=basis,direction_inferred=True,
            enabled=False,hydrograph=[[0,50],[120,0]]))
    return result


def annotate(root,terrain):
    root=Path(root);fields=terrain['fields'];shape=terrain['shape']
    names=['terrain','ground_height' if terrain['full'] else 'height',
           'water_surface' if terrain['full'] else 'riverWaterLevel']
    if 'riverZone' in fields:names.append('riverZone')
    paths=[root/fields[n]['url'].lstrip('/') for n in names]
    signature=dict(schema=SCHEMA,shape=shape,origin=terrain['origin'],step=terrain['step'],
        full=terrain['full'],sea_level=terrain['sea_level'],zone_ids=fields.get('riverZone',{}).get('values',{}),
        files=[dict(name=n,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for n,p in zip(names,paths)])
    folder=(root/terrain['id']).resolve()
    if root.resolve() not in folder.parents:raise ValueError('入口标注目录不正确')
    target=folder/'river-inlets.json'
    if target.is_file():
        try:
            cached=json.loads(target.read_text(encoding='utf-8'))
            if cached.get('signature')==signature and (folder/'hydrology-terrain.i32').is_file():return cached
        except (ValueError,OSError):pass
    def read(n,p):return np.fromfile(p,dtype=fields[n]['dtype']).reshape(shape)
    arrays=[read(n,p) for n,p in zip(names,paths)];types,bed,water=arrays[:3]
    if not terrain['full']:bed=bed*terrain['scale'];water=water*terrain['scale']
    types,changed=water_types(types,bed,water,arrays[3] if len(arrays)>3 else None,
        terrain['terrain_ids'],fields.get('riverZone',{}).get('values',{}),terrain['sea_level'])
    corrected=folder/'hydrology-terrain.i32';native_tmp=corrected.with_suffix('.i32.'+uuid.uuid4().hex+'.tmp')
    types.astype('<i4').tofile(native_tmp);native_tmp.replace(corrected)
    river=np.isin(types,[int(k) for k,v in terrain['terrain_ids'].items() if v=='river'])
    points=detect(river,bed,water,terrain['origin'],terrain['step'])
    data=dict(schema=SCHEMA,signature=signature,river_cells=int(river.sum()),crossings=points,
        sea_reclassified_cells=int(changed.sum()),
        terrain_field=dict(url=f'/{terrain["id"]}/hydrology-terrain.i32',dtype='<i4',
            values=terrain['terrain_ids'],file='hydrology-terrain.i32'),
        direction_method='native river boundary intersection; local inward stage/bed gradient; uncalibrated prior',
        message='已标注河道边界交点，流向为地形推断，可手动确认' if points else
            '河流未穿过地图边界，可手动设置入水点' if river.any() else '当前地图没有河流，可手动设置入水点')
    tmp=target.with_suffix('.json.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(target)
    return data


def curve(points):
    if not isinstance(points,list) or not 1<=len(points)<=256:raise ValueError('来水过程需要1到256个时间流量点')
    result=[];prev=-1
    for p in points:
        if not isinstance(p,(list,tuple)) or len(p)!=2:raise ValueError('来水过程使用[秒,m³/s]')
        t,q=p
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in p):raise ValueError('来水时间与流量必须有限')
        if not 0<=t<=3600 or t<=prev or not 0<=q<=10000:raise ValueError('来水时间需递增，流量需在0到10000之间')
        result.append([float(t),float(q)]);prev=t
    return result


def configure(world,crossings,settings,native_shape):
    if not isinstance(settings,list) or len(settings)>32:raise ValueError('最多配置32个外源来水点')
    native_z,native_x=native_shape;factor=world.meta['coarsening_factor'];available={p['id']:p for p in crossings}
    entries=[];seen=set()
    for item in settings:
        if not isinstance(item,dict):raise ValueError('入水点需要是对象')
        sid=item.get('id')
        if not isinstance(sid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,48}',sid) or sid in seen:raise ValueError('入水点编号需要唯一')
        seen.add(sid);enabled=item.get('enabled',True)
        if not isinstance(enabled,bool):raise ValueError('入水点启用标记需要布尔值')
        kind=item.get('kind','manual')
        if kind=='river':
            if sid not in available:raise ValueError('河道入口已变化，请重新读取地图')
            p=available[sid]
            cells=p['native_cells'];geometry=dict(p)
            weights=Counter((z//factor,x//factor) for z,x in cells)
        elif kind=='manual':
            x,z=item.get('x'),item.get('z')
            if any(isinstance(v,bool) or not isinstance(v,int) for v in (x,z)) or not 0<=x<native_x or not 0<=z<native_z:
                raise ValueError('手动入水位置需要是原始网格范围内的整数')
            weights=Counter({(z//factor,x//factor):1});geometry=dict(id=sid,kind=kind,x=x,z=z,label=sid,role='forced_point')
        else:raise ValueError('请选择河道入口或手动入水点')
        points=curve(item.get('hydrograph',[[0,50],[120,0]]))
        if enabled and any(world.base.solid[z,x] for z,x in weights):raise ValueError('外源来水不能注入不透水建筑格')
        total=sum(weights.values());coarse=[dict(z=int(z),x=int(x),weight=w/total) for (z,x),w in sorted(weights.items())]
        entry=dict(geometry,enabled=enabled,hydrograph=points,cells=coarse,delivery='volume_source_at_inlet_cells')
        if kind=='river':
            velocity=item.get('inlet_velocity_m_s',0)
            if isinstance(velocity,bool) or not isinstance(velocity,(int,float)) or not math.isfinite(velocity) or not 0<=velocity<=30:
                raise ValueError('河口入流速度需要在0到30 m/s之间，0为自动')
            faces=p.get('faces')
            if not faces:
                faces=[]
                for z,x in cells:
                    for edge,hit,offset in [('west',x==0,z),('east',x==native_x-1,z),('north',z==0,x),('south',z==native_z-1,x)]:
                        if hit:faces.append(dict(edge=edge,offset=offset))
            counts=Counter((f['edge'],f['offset']//factor) for f in faces)
            if not counts:raise ValueError('河口必须与地图外边界相交')
            boundary_faces=[];nz,nx=world.base.shape
            for (edge,offset),count in sorted(counts.items()):
                z,x={'west':(offset,0),'east':(offset,nx-1),'north':(0,offset),'south':(nz-1,offset)}[edge]
                wall={'west':world.base.wall_x[z,0],'east':world.base.wall_x[z,nx],
                      'north':world.base.wall_y[0,x],'south':world.base.wall_y[nz,x]}[edge]
                if enabled and (world.base.solid[z,x] or wall):raise ValueError('河口边界被建筑或阻水边封闭，请先移除阻挡')
                boundary_faces.append(dict(edge=edge,offset=int(offset),z=int(z),x=int(x),
                    width_m=count*world.solver.dx/factor,weight=count/sum(counts.values())))
            entry.update(delivery='boundary_discharge_with_momentum',inlet_velocity_m_s=float(velocity),
                boundary_faces=boundary_faces,velocity_method='specified' if velocity else 'local_depth_with_critical_floor',
                zero_discharge='no_exchange_on_configured_inlet_fraction')
            face_cells=Counter()
            for face in boundary_faces:face_cells[(face['z'],face['x'])]+=face['weight']
            entry['cells']=[dict(z=z,x=x,weight=weight) for (z,x),weight in sorted(face_cells.items())]
        entries.append(entry)
    world.external_inflows=entries;world.river_crossings=crossings
    world.solver.set_external_inflows(entries)
