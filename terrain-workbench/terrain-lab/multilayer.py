"""Shared-grid physical environment and hydraulic map state."""
from __future__ import annotations
import io
import json
import math
import zipfile
import numpy as np


def json_safe(value):
    if isinstance(value,np.ndarray): return json_safe(value.tolist())
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
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
        if (self.n<=0).any() or (self.n>.5).any(): raise ValueError('每格糙率需要在 0 到 0.5 之间')
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
        return {k:getattr(self,k) for k in ('z','n','solid','source_id','drain_capacity','wall_x','wall_y','terrain_class')}


class MapWorld:
    def __init__(self,solver,meta,layers=None):
        self.solver=solver;self.meta=meta;self.config=meta['config']
        self.base=BaseMap(solver.bed,solver.dx,layers,self.config['manning'])
        self.base.configure(solver)
        source=(self.config['source_z']//meta['coarsening_factor'],self.config['source_x']//meta['coarsening_factor'])
        if self.config['inflow_m3_s'] and self.base.solid[source]:
            raise ValueError('指定洪水来水不能放在建筑格内')
        self.generated_initial=solver.h.copy()
        self.external_inflows=[];self.river_crossings=[]

    def advance_to(self,end,cancelled=lambda:False):
        if not isinstance(end,(int,float)) or not math.isfinite(end) or end<self.solver.time:
            raise ValueError('模拟时间需要有限且不能倒退')
        c=self.config
        self.solver.advance(end,rain=c['rain_mm_h']/3600000,rain_until=c['rain_duration_s'],
            infiltration=c['infiltration_mm_h']/3600000,inflow=c['inflow_m3_s'],
            source=(c['source_z']//self.meta['coarsening_factor'],c['source_x']//self.meta['coarsening_factor']),
            cancelled=cancelled)

    def grid(self):
        return dict(shape=list(self.base.shape),axis_order=['z','x'],
                    origin_blocks=self.meta.get('origin_blocks',[0,0]),dx_m=self.base.dx)

    def water(self):
        u,v=self.solver.velocity()
        return dict(h=self.solver.h,u=u,v=v,maximum_depth=self.solver.maximum,
                    volume=self.solver.h*self.base.dx**2,qx=self.solver.qx,qy=self.solver.qy,
                    summary=self.solver.summary())

    def forcing(self):
        count=np.zeros(self.base.shape,np.int32)
        for entry in self.external_inflows:
            if entry['enabled']:
                for cell in entry['cells']:count[cell['z'],cell['x']]+=1
        return dict(external_inflows=self.external_inflows,river_crossings=self.river_crossings,
                    inlet_count=count,external_rate_m_s=self.solver.forcing_rate(),
                    boundary_inflows=self.solver.boundary_forcing()[1],
                    accumulated_m3=self.solver.external_volume_by_id)

    def snapshot(self):
        return json_safe(dict(schema='hyra-multilayer-v2',time_s=self.solver.time,grid=self.grid(),
            provenance=self.meta.get('terrain_source'),config=self.config,scenario_metadata=self.meta,
            source_hydrographs=self.base.hydrographs,M_base=self.base.snapshot(),M_water=self.water(),M_forcing=self.forcing()))

    def arrays(self):
        result={'M_base.'+k:v for k,v in self.base.snapshot().items()}
        result.update({'M_water.'+k:v for k,v in self.water().items() if isinstance(v,np.ndarray)})
        result.update({'M_water.'+k:v for k,v in dict(mx=self.solver.mx,mz=self.solver.mz,
            initial_depth=self.solver.initial,generated_initial_depth=self.generated_initial,
            baseflow_rate=self.solver.baseflow_rate).items()})
        result.update({'M_forcing.'+k:v for k,v in self.forcing().items() if isinstance(v,np.ndarray)})
        return result

    def export_zip(self):
        data=io.BytesIO();np.savez_compressed(data,**self.arrays());out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('map_model.json',json.dumps(self.snapshot(),ensure_ascii=False,allow_nan=False))
            z.writestr('layers.npz',data.getvalue())
            z.writestr('data_dictionary.json',json.dumps(DATA_DICTIONARY,ensure_ascii=False,indent=2))
        return out.getvalue()


DATA_DICTIONARY=dict(schema='hyra-multilayer-v2',axis_order=['z','x'],
    M_base=dict(z='ground elevation, m',n='Manning coefficient, s/m^(1/3)',solid='impermeable cell, bool',
        source_id='-1 none or hydrograph id',drain_capacity='per-cell capacity, m3/s',
        wall_x='[nz,nx+1] impermeable faces',wall_y='[nz+1,nx] impermeable faces',
        terrain_class='integer land-cover class'),
    M_water=dict(h='depth, m',u='cell +x velocity, m/s',v='cell +z velocity, m/s',
        maximum_depth='maximum depth since flood t=0, m',volume='per-cell volume, m3',
        qx='[nz,nx+1] last transported face flux, m2/s',qy='[nz+1,nx] last transported face flux, m2/s'),
    M_forcing=dict(external_inflows='stable id, native position, normalized coarse cells, enabled flag and [s,m3/s] curve',
        river_crossings='native river/edge crossings; inferred inlet/outlet/uncertain; no discharge enabled automatically',
        inlet_count='count of enabled external points mapped to each cell',external_rate_m_s='equivalent supplied depth rate for cell sources and boundary inflows, m/s; boundary inflows are applied only through face fluxes',
        boundary_inflows='instantaneous prescribed inward discharge and normal velocity by outer face; Q=width*reference_depth*velocity',
        accumulated_m3='per-point supplied volume since flood t=0; independent of daily baseflow'),
    source_hydrographs='source_id -> strictly increasing [simulation s,total m3/s], piecewise constant',
    frames=dict(fields=['depth','u','v','maximum_depth'],dtype='<f4',order='[field,z,x]',
        interpolation='linear interpolation for display only; exported frames are calculated states'),
    solver='full SWE finite volumes, adaptive CFL, hydrostatic reconstruction; no agent or observation system')
