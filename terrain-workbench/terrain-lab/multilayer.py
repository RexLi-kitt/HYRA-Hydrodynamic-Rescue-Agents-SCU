"""PPT map contract: fixed environment, hydraulic truth, per-agent knowledge.

Grid arrays use [z,x], with qy/wall_y pointing along +z. Simulator truth is
available to operator tools; policy_view deliberately excludes it.
"""
from __future__ import annotations
import io
import json
import math
import re
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
            flight_allowed=np.ones(self.shape,bool),source_id=np.full(self.shape,-1,np.int32),
            drain_capacity=np.zeros(self.shape),wall_x=np.zeros((nz,nx+1),bool),
            wall_y=np.zeros((nz+1,nx),bool),person_class=np.zeros(self.shape,np.int32),
            terrain_class=np.zeros(self.shape,np.int32))
        for key,default in defaults.items():
            raw=config.get(key,default)
            numeric=np.asarray(raw,float)
            if numeric.shape!=default.shape or not np.isfinite(numeric).all(): raise ValueError(f'{key} 数组尺寸或数值不正确')
            if default.dtype==bool and not np.isin(numeric,[0,1]).all(): raise ValueError(f'{key} 需要布尔值')
            if key in ('source_id','person_class','terrain_class') and (numeric!=np.floor(numeric)).any(): raise ValueError(f'{key} 需要整数')
            setattr(self,key,numeric.astype(default.dtype))
        for p in config.get('patches',[]): self.patch(p)
        if (self.n<=0).any() or (self.n>.5).any(): raise ValueError('每格糙率需要在 0 到 0.5 之间')
        if (self.drain_capacity<0).any(): raise ValueError('排水能力不能为负')
        if not np.isin(self.person_class,[0,1,2]).all(): raise ValueError('人员类别需要为 0、1 或 2')
        if (self.source_id<-1).any() or (self.solid & (self.source_id>=0)).any(): raise ValueError('水源编号无效或位于建筑内')
        if (self.solid & (self.person_class>0)).any(): raise ValueError('示例人员不能放在不透水建筑格内')
        self.signal_ref=None
        if 'signal_field' in config:
            a=np.asarray(config['signal_field'],float)
            if a.shape!=self.shape or not np.isfinite(a).all(): raise ValueError('信号场尺寸或数值不正确')
            self.signal_ref=a.copy()
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
        names=dict(building='solid',no_fly='flight_allowed',roughness='n',drain='drain_capacity',
                   source='source_id',person='person_class',wall_x='wall_x',wall_y='wall_y')
        if not isinstance(p,dict) or p.get('kind') not in names: raise ValueError('未知地图编辑类型')
        a=getattr(self,names[p['kind']]);nz,nx=a.shape
        x=index(p.get('x'),nx);z=index(p.get('z'),nz)
        w=index(p.get('width',1)-1,nx-x)+1;h=index(p.get('height',1)-1,nz-z)+1
        value=p.get('value',0 if p['kind']=='no_fly' else 1)
        if not isinstance(value,(int,float,bool)) or not math.isfinite(value): raise ValueError('地图编辑值无效')
        if a.dtype==bool and value not in (0,1): raise ValueError('建筑、边墙与禁飞标记需要布尔值')
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
        return {k:getattr(self,k) for k in ('z','n','solid','flight_allowed','source_id','drain_capacity','wall_x','wall_y')}


class Knowledge:
    def __init__(self,shape,legacy=None):
        self.shape=shape
        for name in ('water_last','u_last','v_last','t_water','terrain_field','t_terrain','signal_field','t_signal'):
            setattr(self,name,np.full(shape,np.nan))
        for name in ('water_valid','terrain_valid','signal_valid','search_clear','search_person','person_valid','explored_ever','solid_last'):
            setattr(self,name,np.zeros(shape,bool))
        self.person_last=np.zeros(shape,np.int32);self.t_person=np.full(shape,np.nan)
        self.legacy_channels=None
        prior=legacy if isinstance(legacy,dict) else {}
        if isinstance(legacy,dict):legacy=prior.get('legacy_channels')
        if legacy is not None:
            a=np.asarray(legacy,float)
            if a.ndim!=3 or a.shape[1:]!=shape or a.shape[0]<2 or not np.isfinite(a).all():
                raise ValueError('原探索通道需要至少两个同网格的有限数组')
            self.legacy_channels=a.copy()
            self.explored_ever |= (a[:2]!=0).any(axis=0)
        for field,valid,time in (('terrain_field','terrain_valid','t_terrain'),('signal_field','signal_valid','t_signal')):
            if field not in prior:continue
            a=np.asarray(prior[field],float);v=np.asarray(prior.get(valid,np.isfinite(a)),bool)
            if a.shape!=shape or v.shape!=shape or not np.isfinite(a[v]).all():raise ValueError('原地形或信号估计数组不正确')
            getattr(self,field)[v]=a[v];getattr(self,valid)[v]=True;getattr(self,time)[v]=0
        for field in ('search_clear','search_person','explored_ever'):
            if field in prior:
                a=np.asarray(prior[field])
                if a.shape!=shape or not np.isin(a,[0,1]).all():raise ValueError('探索记录尺寸或标记不正确')
                getattr(self,field)[:] |= a.astype(bool)
        self.explored_ever |= self.search_clear | self.search_person

    def merge(self,other,now):
        # Newest valid measurement wins; unknown never replaces known zero.
        if self.shape!=other.shape:raise ValueError('共享地图必须使用同一网格')
        for valid,time in (('water_valid','t_water'),('terrain_valid','t_terrain'),('signal_valid','t_signal'),('person_valid','t_person')):
            t=getattr(other,time)[getattr(other,valid)]
            if not np.isfinite(t).all() or (t<0).any() or (t>now+1e-9).any():raise ValueError('共享观测的采集时间无效或位于未来')
        count=0
        for valid,time,fields in (('water_valid','t_water',('water_last','u_last','v_last')),
            ('terrain_valid','t_terrain',('terrain_field','solid_last')),
            ('signal_valid','t_signal',('signal_field',)),('person_valid','t_person',('person_last',))):
            v=getattr(other,valid);t=getattr(other,time);own=getattr(self,time)
            take=v & np.isfinite(t) & (t<=now+1e-9) & (~getattr(self,valid) | (t>own+1e-9))
            for field in fields: getattr(self,field)[take]=getattr(other,field)[take]
            getattr(self,time)[take]=t[take];getattr(self,valid)[take]=True
            if valid=='water_valid':count=int(take.sum())
        self.search_clear|=other.search_clear;self.search_person|=other.search_person;self.explored_ever|=other.explored_ever
        return count

    def observe(self,base,solver,footprint,valid=True):
        if not valid:return 0
        now=solver.time
        terrain=footprint & (~self.terrain_valid | (now>=self.t_terrain))
        self.terrain_field[terrain]=base.z[terrain];self.solid_last[terrain]=base.solid[terrain]
        self.t_terrain[terrain]=now;self.terrain_valid[terrain]=True
        visible=footprint & ~base.solid
        take=visible & (~self.water_valid | (now>=self.t_water))
        u,v=solver.velocity()
        for field,a in (('water_last',solver.h),('u_last',u),('v_last',v)):
            getattr(self,field)[take]=a[take]
        self.t_water[take]=now;self.water_valid[take]=True
        self.person_last[visible]=base.person_class[visible];self.person_valid[visible]=True;self.t_person[visible]=now
        self.search_person[visible & (base.person_class>0)]=True
        self.search_clear[visible & (base.person_class==0)]=True
        self.explored_ever|=visible
        if base.signal_ref is not None:
            self.signal_field[visible]=base.signal_ref[visible];self.signal_valid[visible]=True;self.t_signal[visible]=now
        return int(take.sum())

    def age(self,now):return np.where(self.water_valid,now-self.t_water,np.nan)

    def snapshot(self,now):
        return dict(M_obs={k:getattr(self,k) for k in ('water_last','water_valid','u_last','v_last','t_water')},
            information_age=self.age(now),
            M_exp={k:getattr(self,k) for k in ('explored_ever','search_clear','search_person',
                'terrain_field','terrain_valid','signal_field','signal_valid','person_last','person_valid',
                't_person','t_terrain','t_signal','solid_last','legacy_channels')})

    def pool_water(self,factor,now):
        nz,nx=self.shape
        if not isinstance(factor,int) or factor<1 or nz%factor or nx%factor:raise ValueError('池化尺寸不正确')
        def total(a):return a.reshape(nz//factor,factor,nx//factor,factor).sum(axis=(1,3))
        count=total(self.water_valid)
        value=np.divide(total(np.where(self.water_valid,self.water_last,0)),count,
                        out=np.full(count.shape,np.nan),where=count>0)
        oldest=np.where(self.water_valid,self.t_water,np.inf).reshape(nz//factor,factor,nx//factor,factor).min(axis=(1,3))
        oldest[count==0]=np.nan
        return dict(water_last=value,water_valid=count>0,known_fraction=count/factor**2,
                    oldest_observed_at_s=oldest,information_age=now-oldest,factor=factor)


def line_cells(x0,z0,x1,z1):
    # Exact grid traversal, including both neighbors when a segment touches a
    # corner. Sparse samples can miss a very short crossing of a forbidden cell.
    x,z=x0,z0;dx=x1-x0;dz=z1-z0;cells={(z,x)}
    sx=1 if dx>0 else -1;sz=1 if dz>0 else -1
    tx=.5/abs(dx) if dx else math.inf;tz=.5/abs(dz) if dz else math.inf
    stepx=1/abs(dx) if dx else math.inf;stepz=1/abs(dz) if dz else math.inf
    while (x,z)!=(x1,z1):
        if abs(tx-tz)<1e-12:
            cells.add((z,x+sx));cells.add((z+sz,x));x+=sx;z+=sz;tx+=stepx;tz+=stepz
        elif tx<tz:x+=sx;tx+=stepx
        else:z+=sz;tz+=stepz
        cells.add((z,x))
    return cells


class MapWorld:
    def __init__(self,solver,meta,layers=None,agents=None,legacy=None):
        self.solver=solver;self.meta=meta;self.base=BaseMap(solver.bed,solver.dx,layers,meta['config']['manning'])
        self.base.configure(solver);self.team=Knowledge(self.base.shape,legacy)
        self.agents={};self.knowledge={};self.events=[]
        self.config=meta['config'];self.origin=meta.get('origin_blocks',[0,0])
        source=(self.config['source_z']//meta['coarsening_factor'],self.config['source_x']//meta['coarsening_factor'])
        if self.config['inflow_m3_s'] and self.base.solid[source]:raise ValueError('指定洪水来水不能放在建筑格内')
        for a in agents or [dict(id='uav-1',x=self.base.shape[1]//2,z=self.base.shape[0]//2,radius=4)]:self.add_agent(a)
        self.event('created',{})

    def event(self,action,data):
        if len(self.events)>=500:raise ValueError('本次场景已达500次操作，请导出后建立新场景')
        self.events.append(dict(time_s=self.solver.time,action=action,data=json_safe(data)))

    def add_agent(self,a):
        name=a.get('id')
        if not isinstance(name,str) or name=='team' or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',name) or name in self.agents:
            raise ValueError('无人机编号需要唯一，使用字母、数字或连字符')
        nz,nx=self.base.shape;x=index(a.get('x'),nx);z=index(a.get('z'),nz)
        radius=a.get('radius',4)
        if not isinstance(radius,(int,float)) or not math.isfinite(radius) or not 0<=radius<=128:raise ValueError('观测半径无效')
        if not self.base.flight_allowed[z,x]:raise ValueError('无人机不能位于禁飞区')
        if len(self.agents)>=8:raise ValueError('最多支持8架无人机')
        self.agents[name]=dict(id=name,x=x,z=z,radius=float(radius));self.knowledge[name]=Knowledge(self.base.shape)

    def advance(self,seconds):
        if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<seconds<=60:raise ValueError('每次推进需要 0 到 60 秒')
        if len(self.events)>=500:raise ValueError('操作记录已满，请导出后重新建立场景')
        c=self.config
        self.solver.advance(self.solver.time+seconds,rain=c['rain_mm_h']/3600000,
            rain_until=c['rain_duration_s'],infiltration=c['infiltration_mm_h']/3600000,
            inflow=c['inflow_m3_s'],source=(c['source_z']//self.meta['coarsening_factor'],c['source_x']//self.meta['coarsening_factor']))
        self.event('advance',dict(seconds=seconds))

    def move(self,name,x,z):
        if len(self.events)>=500:raise ValueError('操作记录已满，请导出后重新建立场景')
        a=self.agents[name];nz,nx=self.base.shape;x=index(x,nx);z=index(z,nz)
        for zz,xx in line_cells(a['x'],a['z'],x,z):
            if not 0<=zz<nz or not 0<=xx<nx or not self.base.flight_allowed[zz,xx]:raise ValueError('移动路径穿过禁飞区')
        a.update(x=x,z=z);self.event('move',dict(agent=name,x=x,z=z))

    def observe(self,name,valid=True):
        if not isinstance(valid,bool):raise ValueError('观测有效标记需要布尔值')
        if len(self.events)>=500:raise ValueError('操作记录已满，请导出后重新建立场景')
        a=self.agents[name];zz,xx=np.indices(self.base.shape)
        footprint=(xx-a['x'])**2+(zz-a['z'])**2<=a['radius']**2
        count=self.knowledge[name].observe(self.base,self.solver,footprint,valid)
        self.event('observe',dict(agent=name,valid=valid,updated=count));return count

    def share(self,name,target='team'):
        if len(self.events)>=500:raise ValueError('操作记录已满，请导出后重新建立场景')
        source=self.team if name=='team' else self.knowledge[name];dest=self.team if target=='team' else self.knowledge[target]
        count=dest.merge(source,self.solver.time);self.event('share',dict(agent=name,target=target,updated=count));return count

    def positions(self,knowledge):
        found=[dict(x=int(x),z=int(z),category=int(knowledge.person_last[z,x]),observed_at_s=float(knowledge.t_person[z,x]))
               for z,x in np.argwhere(knowledge.person_valid & (knowledge.person_last>0))]
        return dict(uavs=list(self.agents.values()),discovered_persons=found)

    def policy_view(self,name,pool=1):
        k=self.team if name=='team' else self.knowledge[name]
        # All numeric channels are finite; validity channels distinguish unknown
        # from a confirmed zero. No truth elevation, water or hidden people here.
        uavs=np.zeros(self.base.shape)
        for a in self.agents.values():uavs[a['z'],a['x']]=1
        channels=dict(water_depth=np.where(k.water_valid,k.water_last,0),water_valid=k.water_valid,
            water_u=np.where(k.water_valid,k.u_last,0),water_v=np.where(k.water_valid,k.v_last,0),
            water_age_s=np.where(k.water_valid,k.age(self.solver.time),0),
            terrain=np.where(k.terrain_valid,k.terrain_field,0),terrain_valid=k.terrain_valid,
            signal=np.where(k.signal_valid,k.signal_field,0),signal_valid=k.signal_valid,
            explored_ever=k.explored_ever,search_clear=k.search_clear,search_person=k.search_person,
            person_seen=np.where(k.person_valid,k.person_last,0),person_valid=k.person_valid,
            uav_positions=uavs,flight_allowed=self.base.flight_allowed)
        result=dict(schema='hyra-agent-view-v1',time_s=self.solver.time,grid=self.grid(),
                    channel_names=list(channels),channels=channels,M_pos=self.positions(k),
                    M_obs=k.snapshot(self.solver.time)['M_obs'],M_exp=k.snapshot(self.solver.time)['M_exp'])
        if pool!=1:result['pooled_water']=k.pool_water(pool,self.solver.time)
        return json_safe(result)

    def grid(self):return dict(shape=list(self.base.shape),axis_order=['z','x'],origin_blocks=self.origin,dx_m=self.base.dx)

    def snapshot(self):
        u,v=self.solver.velocity()
        return json_safe(dict(schema='hyra-multilayer-v1',time_s=self.solver.time,grid=self.grid(),
            provenance=self.meta.get('terrain_source'),config=self.config,scenario_metadata=self.meta,
            source_hydrographs=self.base.hydrographs,M_base=self.base.snapshot(),
            M_ref=dict(terrain_class=self.base.terrain_class,solid=self.base.solid,
                       person_class=self.base.person_class,signal_field=self.base.signal_ref),
            M_water=dict(h=self.solver.h,u=u,v=v,volume=self.solver.h*self.base.dx**2,
                         qx=self.solver.qx,qy=self.solver.qy,summary=self.solver.summary()),
            team=self.team.snapshot(self.solver.time),M_pos=self.positions(self.team),
            agents={name:dict(position=a,**self.knowledge[name].snapshot(self.solver.time),
                             M_pos=self.positions(self.knowledge[name])) for name,a in self.agents.items()},
            events=self.events))

    def export_zip(self):
        state=self.snapshot();arrays={}
        def collect(prefix,obj):
            if isinstance(obj,np.ndarray):arrays[prefix]=obj
            elif isinstance(obj,dict):
                for key,value in obj.items():collect(prefix+'.'+key,value)
        collect('M_base',self.base.snapshot());collect('M_ref',dict(terrain_class=self.base.terrain_class,
            solid=self.base.solid,person_class=self.base.person_class,signal_field=self.base.signal_ref))
        collect('M_water',dict(h=self.solver.h,mx=self.solver.mx,mz=self.solver.mz,qx=self.solver.qx,qy=self.solver.qy,
            initial_depth=self.solver.initial,maximum_depth=self.solver.maximum,baseflow_rate=self.solver.baseflow_rate))
        collect('team',self.team.snapshot(self.solver.time))
        for name,k in self.knowledge.items():collect('agents.'+name,k.snapshot(self.solver.time))
        data=io.BytesIO();np.savez_compressed(data,**arrays);out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('map_model.json',json.dumps(state,ensure_ascii=False,allow_nan=False))
            z.writestr('layers.npz',data.getvalue())
            z.writestr('data_dictionary.json',json.dumps(DATA_DICTIONARY,ensure_ascii=False,indent=2))
            z.writestr('events.json',json.dumps(self.events,ensure_ascii=False,allow_nan=False))
        return out.getvalue()


DATA_DICTIONARY=dict(schema='hyra-multilayer-v1',axis_order=['z','x'],
    M_base=dict(z='ground elevation, m',n='Manning coefficient, s/m^(1/3)',solid='impermeable cell, bool',
        flight_allowed='known regulatory flight permission, bool',source_id='-1 none or hydrograph id',
        drain_capacity='per-cell capacity, m3/s',wall_x='[nz,nx+1] impermeable faces',wall_y='[nz+1,nx] impermeable faces'),
    M_water=dict(h='depth, m',u='cell +x velocity, m/s',v='cell +z velocity, m/s',
        qx='[nz,nx+1] last transported face flux, m2/s',qy='[nz+1,nx] last transported face flux, m2/s'),
    M_obs=dict(water_last='last valid depth, null/NaN when unknown',water_valid='known dry is valid with zero depth',
        u_last='last valid +x velocity, m/s',v_last='last valid +z velocity, m/s',
        t_water='measurement time on simulation clock, s',information_age='current time minus measurement time'),
    M_ref=dict(person_class='0 none, 1 not yet rescued, 2 rescued',signal_field='optional configured truth; not fabricated'),
    M_exp=dict(explored_ever='ever validly searched',search_clear='ever searched without a person',
        search_person='ever searched with a person',terrain_field='observed elevation or imported estimate',
        signal_field='observed or imported signal estimate',legacy_channels='optional original channels retained verbatim'),
    M_pos=dict(uavs='team UAV positions',discovered_persons='only last validly observed person locations'),
    isolation='operator snapshot includes truth; policy_view contains observations only',
    source_hydrographs='source_id -> strictly increasing [simulation s,total m3/s], piecewise constant',
    observation_model='no noise or latency; circular footprint; solid cells hide water and people; no 3D occlusion',
    solver='existing full SWE finite volumes; the PPT local-inertial alternative is not substituted',
    legacy='no original training network or GMRF estimator exists in this repository; adapter exposes named channels')
