"""Shared-grid physical environment and hydraulic map state."""
from __future__ import annotations
import io
import json
import math
import zipfile
import numpy as np


def json_safe(value,array_encoder=None):
    if isinstance(value,np.ndarray): return array_encoder(value) if array_encoder else json_safe(value.tolist())
    if isinstance(value,dict): return {k:json_safe(v,array_encoder) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v,array_encoder) for v in value]
    if isinstance(value,(np.integer,np.bool_)): return value.item()
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def index(value,size):
    if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or not 0<=value<size:
        raise ValueError('网格位置需要是范围内的整数')
    return int(value)


class BaseMap:
    def __init__(self,bed,dx,config=None,manning=.035):
        config=config or {}; self.z=np.asarray(config.get('z',bed),float).copy(); self.shape=self.z.shape
        if self.shape!=np.asarray(bed).shape:raise ValueError('导入高程必须与计算网格同尺寸')
        if self.z.ndim!=2 or min(self.shape)<2 or not np.isfinite(self.z).all() or not math.isfinite(dx) or dx<=0:
            raise ValueError('地图高程或间距不正确')
        self.dx=float(dx); nz,nx=self.shape
        defaults=dict(n=np.full(self.shape,manning),solid=np.zeros(self.shape,bool),
            source_id=np.full(self.shape,-1,np.int32),
            drain_capacity=np.zeros(self.shape),wall_x=np.zeros((nz,nx+1),bool),
            wall_y=np.zeros((nz+1,nx),bool),terrain_class=np.zeros(self.shape,np.int32))
        for key,default in defaults.items():
            raw=config.get(key,default)
            numeric=np.asarray(raw,float)
            if numeric.shape!=default.shape or not np.isfinite(numeric).all(): raise ValueError(f'{key} 数组尺寸或数值不正确')
            if default.dtype==bool and not np.isin(numeric,[0,1]).all(): raise ValueError(f'{key} 需要布尔值')
            if key in ('source_id','terrain_class') and (numeric!=np.floor(numeric)).any(): raise ValueError(f'{key} 需要整数')
            setattr(self,key,numeric.astype(default.dtype))
        for p in config.get('patches',[]): self.patch(p)
        if (self.n<0).any() or (self.n>.5).any(): raise ValueError('每格糙率需要在 0 到 0.5 之间（含端点）')
        if (self.drain_capacity<0).any(): raise ValueError('排水能力不能为负')
        if (self.source_id<-1).any() or (self.solid & (self.source_id>=0)).any(): raise ValueError('水源编号无效或位于建筑内')
        self.hydrographs={}
        for key,points in config.get('hydrographs',{}).items():
            if not str(key).isdigit() or len(points)<1: raise ValueError('水源过程需要编号与时间流量点')
            prev=-1; clean=[]
            for t,q in points:
                if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (t,q)) or t<0 or t<=prev or not 0<=q<=10000:
                    raise ValueError('入水过程时间必须递增，流量必须非负且有限')
                clean.append([float(t),float(q)]);prev=t
            self.hydrographs[int(key)]=clean
        ids=set(int(v) for v in np.unique(self.source_id) if v>=0)
        if ids-set(self.hydrographs): raise ValueError('每个水源编号都需要对应的入水过程')
        self.hydrographs={k:v for k,v in self.hydrographs.items() if k in ids}

    def patch(self,p):
        names=dict(building='solid',roughness='n',drain='drain_capacity',
                   source='source_id',terrain='terrain_class',wall_x='wall_x',wall_y='wall_y')
        if not isinstance(p,dict) or p.get('kind') not in names: raise ValueError('未知地图编辑类型')
        a=getattr(self,names[p['kind']]);nz,nx=a.shape
        x=index(p.get('x'),nx);z=index(p.get('z'),nz)
        w=index(p.get('width',1)-1,nx-x)+1;h=index(p.get('height',1)-1,nz-z)+1
        value=p.get('value',1)
        if not isinstance(value,(int,float,bool)) or not math.isfinite(value): raise ValueError('地图编辑值无效')
        if a.dtype==bool and value not in (0,1): raise ValueError('建筑与边墙标记需要布尔值')
        if np.issubdtype(a.dtype,np.integer) and value!=int(value): raise ValueError('编号和类别需要整数')
        a[z:z+h,x:x+w]=value

    def configure(self,solver):
        solver.bed=self.z.copy()
        solver.manning=self.n.copy()
        solver.set_environment(solid=self.solid,wall_x=self.wall_x,wall_y=self.wall_y,drain_capacity=self.drain_capacity)
        entries=[]
        for sid,points in self.hydrographs.items():
            cells=[list(map(int,c)) for c in np.argwhere(self.source_id==sid)]
            entries.append(dict(id=sid,cells=cells,hydrograph=points))
        solver.set_source_schedule(entries)

    def snapshot(self):
        result={k:getattr(self,k) for k in ('z','n','solid','source_id','drain_capacity','wall_x','wall_y','terrain_class')}
        if hasattr(self,'static_solid'):result['static_solid']=self.static_solid
        if hasattr(self,'refinement_level'):result['refinement_level']=self.refinement_level
        return result


class MapWorld:
    def __init__(self,solver,meta,layers=None,structures=None,people=None):
        self.solver=solver;self.meta=meta;self.config=meta['config']
        layers=dict(layers or {})
        layers.setdefault('n',solver.manning)
        self.base=BaseMap(solver.bed,solver.dx,layers,self.config['manning'])
        from hydraulic_structures import Structures
        self.structures=Structures(structures or [],self.base.z,self.base.dx)
        self.base.static_solid=self.base.solid.copy()
        self.base.solid |= self.structures.buildings
        if np.any(self.base.solid & (self.base.source_id>=0)):
            raise ValueError('建筑与变化水源重叠，请移动建筑或水源')
        solver.structures=self.structures
        self.base.configure(solver)
        source=(self.config['source_z']//meta['coarsening_factor'],self.config['source_x']//meta['coarsening_factor'])
        if self.config['inflow_m3_s'] and self.base.solid[source]:
            raise ValueError('指定洪水来水不能放在建筑格内')
        import people as people_layer
        levees=[(e['axis'],e['z'],e['x']) for obj in self.structures.compiled
                if obj['kind']!='building' and self.structures.active(obj,0.)
                for e in obj['edges']]
        self.people=people_layer.normalize(people,shape=self.base.shape,dx=self.base.dx,
            solid=self.base.solid,bed=self.base.z,wall_x=self.base.wall_x,wall_y=self.base.wall_y,
            levee_edges=levees,terrain_class=self.base.terrain_class,terrain_ids=meta.get('terrain_ids'))
        self.generated_initial=solver.h.copy()
        self.external_inflows=[];self.river_crossings=[]

    def advance_to(self,end,cancelled=lambda:False,progress=None):
        if not isinstance(end,(int,float)) or not math.isfinite(end) or end<self.solver.time:
            raise ValueError('模拟时间需要有限且不能倒退')
        c=self.config
        self.solver.advance(end,rain=c['rain_mm_h']/3600000,rain_until=c['rain_duration_s'],
            infiltration=c['infiltration_mm_h']/3600000,inflow=c['inflow_m3_s'],
            source=(c['source_z']//self.meta['coarsening_factor'],c['source_x']//self.meta['coarsening_factor']),
            cancelled=cancelled,progress=progress)

    def grid(self):
        return dict(shape=list(self.base.shape),axis_order=['z','x'],
                    origin_blocks=self.meta.get('origin_blocks',[0,0]),dx_m=self.base.dx)

    def water(self,compact=False):
        u,v=self.solver.velocity()
        result=dict(h=self.solver.h,u=u,v=v,maximum_depth=self.solver.maximum,summary=self.solver.summary())
        if not compact:result.update(volume=self.solver.h*self.base.dx**2,qx=self.solver.qx,qy=self.solver.qy)
        return result

    def forcing(self):
        count=np.zeros(self.base.shape,np.int32)
        for entry in self.external_inflows:
            if entry['enabled']:
                for cell in entry['cells']:count[cell['z'],cell['x']]+=1
        return dict(external_inflows=self.external_inflows,river_crossings=self.river_crossings,
                    inlet_count=count,external_rate_m_s=self.solver.forcing_rate(),
                    boundary_inflows=self.solver.boundary_forcing()[1],
                    accumulated_m3=self.solver.external_volume_by_id)

    def snapshot(self,array_encoder=None,compact=False):
        result=dict(schema='hyra-multilayer-v2',time_s=self.solver.time,grid=self.grid(),
            provenance=self.meta.get('terrain_source'),config=self.config,scenario_metadata=self.meta,
            source_hydrographs=self.base.hydrographs,M_base=self.base.snapshot(),M_water=self.water(compact),M_forcing=self.forcing(),
            M_structures=dict(schema='hyra-structures-v1',coordinate_system='local metres from displayed grid outer corner',
                              objects=self.structures.compiled),
            M_people=self.people)
        if hasattr(self.solver,'mesh_snapshot'):
            result['M_mesh']=self.solver.mesh_snapshot(compact)
            result['M_mesh']['origin_m']=[v*self.config['metres_per_block']-self.base.dx/2 for v in self.meta.get('origin_blocks',[0,0])]
            result['M_mesh']['coordinate_definition']='local x_m/z_m from origin_m; parent grid origin is the first sample centre'
        return json_safe(result,array_encoder)

    def arrays(self):
        result={'M_base.'+k:v for k,v in self.base.snapshot().items()}
        result.update({'M_water.'+k:v for k,v in self.water().items() if isinstance(v,np.ndarray)})
        result.update({'M_water.'+k:v for k,v in dict(mx=self.solver.mx,mz=self.solver.mz,
            initial_depth=self.solver.initial,generated_initial_depth=self.generated_initial,
            baseflow_rate=self.solver.baseflow_rate).items()})
        result.update({'M_forcing.'+k:v for k,v in self.forcing().items() if isinstance(v,np.ndarray)})
        import people as people_layer
        result['M_people.count']=people_layer.cell_counts(self.people,self.base.shape)
        if hasattr(self.solver,'mesh_snapshot'):
            result.update({'M_mesh.'+k:v for k,v in self.solver.mesh_snapshot().items() if isinstance(v,np.ndarray)})
            result['M_mesh.origin_m']=np.asarray([v*self.config['metres_per_block']-self.base.dx/2 for v in self.meta.get('origin_blocks',[0,0])])
        return result

    def export_zip(self):
        data=io.BytesIO();np.savez_compressed(data,**self.arrays());out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('map_model.json',json.dumps(self.snapshot(),ensure_ascii=False,allow_nan=False))
            z.writestr('layers.npz',data.getvalue())
            z.writestr('data_dictionary.json',json.dumps(DATA_DICTIONARY,ensure_ascii=False,indent=2))
        return out.getvalue()


DATA_DICTIONARY=dict(schema='hyra-multilayer-v2',axis_order=['z','x'],
    M_structures=dict(schema='hyra-structures-v1',
        coordinate_system='local x_m/z_m from displayed grid outer corner, positive right/down; independent of grid_factor',
        objects='original metre geometry plus compiled cell_bounds/edges; building height is metadata, solid footprint excludes water',
        crest_mode='relative: max(adjacent ground)+crest_m; absolute: crest_m',
        temporary='enabled only during flood time [start_s,end_s); no temporary structures in baseflow warmup',
        hydraulic_model='ANUGA Villemonte overtopping blend, zero below crest; no seepage or structural failure model',
        grid_mapping='rectangular intersecting cells; polylines snapped to connected staircase interior faces'),
    M_people=dict(schema='hyra-people-v1',
        coordinate_system='local x_m/z_m from displayed grid outer corner, positive right/down; independent of grid_factor',
        mode='uniform, cluster, cluster_isolated or manual; the authored mode is metadata for reproduction',
        start='rescue entry point; automatic picks the lowest standable boundary cell',
        items='stable id, metre position, [z,x] cell, group (-1 isolated) and kind',
        counts='total people, reachable people and size of the connected walkable area',
        legality='rejects building cells, ocean cells, sealed starts and targets unreachable from the start',
        reachability='four-neighbour walk over cells that are not solid and faces without a sealed wall or an active levee; flood water is not a barrier'),
    M_mesh=dict(schema='hyra-static-leaf-v1',parent='flat base-grid index',child='-1 coarse; 0/1 upper left/right, 2/3 lower left/right',
        origin_m='outer-corner origin in metres; absolute centres are origin_m + (x_m,z_m)',
        h='actual leaf depth, m',u='actual leaf +X velocity, m/s',v='actual leaf +Z velocity, m/s',
        maximum_depth='actual leaf cumulative maximum depth, m',area_m2='actual cell area',
        x_m='local leaf centre +X from origin_m, m',z_m='local leaf centre +Z from origin_m, m',
        width_m='leaf square side length, m',bed='inherited parent ground elevation, m',
        mx='leaf depth times +X velocity, m2/s',mz='leaf depth times +Z velocity, m2/s',
        face_left='leaf index or -1 exterior',face_right='leaf index or -1 exterior',
        face_axis='1 +X, 0 +Z',face_length_m='shared physical subface length',face_mass_flux='last RK-weighted mass flux, m2/s'),
    M_base=dict(z='ground elevation, m',n='Manning coefficient, s/m^(1/3)',solid='impermeable cell, bool',
        static_solid='impermeable cells before authored objects; for editing preview, retains legacy overlaps',
        source_id='-1 none or hydrograph id',drain_capacity='per-cell capacity, m3/s',
        wall_x='[nz,nx+1] impermeable faces',wall_y='[nz+1,nx] impermeable faces',
        terrain_class='integer land-cover class',refinement_level='0 native parent, 1 split into four leaf cells'),
    M_water=dict(h='depth, m',u='cell +x velocity, m/s',v='cell +z velocity, m/s',
        maximum_depth='maximum depth since flood t=0, m',volume='per-cell volume, m3',
        qx='[nz,nx+1] RK-weighted transported face flux over last solver step, m2/s',
        qy='[nz+1,nx] RK-weighted transported face flux over last solver step, m2/s'),
    refined_projection='M_water h/mx/mz are leaf area averages; u/v from averaged momentum/depth; maximum_depth is area-averaged leaf maxima; qx/qy are face-length averages',
    M_forcing=dict(external_inflows='stable id, native position, normalized coarse cells, enabled flag and [s,m3/s] curve',
        river_crossings='native river/edge crossings; inferred inlet/outlet/uncertain; no discharge enabled automatically',
        inlet_count='count of enabled external points mapped to each cell',external_rate_m_s='equivalent supplied depth rate for cell sources and boundary inflows, m/s; boundary inflows are applied only through face fluxes',
        boundary_inflows='prescribed inward discharge with X/Z velocity and momentum_direction_xz by outer face; Q=width*reference_depth*inward_normal_velocity',
        accumulated_m3='per-point supplied volume since flood t=0; independent of daily baseflow'),
    source_hydrographs='source_id -> strictly increasing [simulation s,total m3/s], piecewise constant',
    frames=dict(fields=['depth','u','v','maximum_depth'],dtype='<f4',order='[field,z,x]',
        interpolation='linear interpolation for display only; exported frames are calculated states'),
    mesh_frames=dict(fields=['depth','u','v','maximum_depth'],dtype='<f4',order='[field,leaf]',
        geometry='M_mesh parent/child/area; fixed throughout a run'),
    solver='Native square-grid SWE, limited hydrostatic reconstruction, SSP-RK2/Euler, Rusanov/KNP, adaptive CFL; split source/sinks; M_mesh fields describe legacy refined archives only')
