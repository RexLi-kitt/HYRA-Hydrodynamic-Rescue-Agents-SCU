"""Causal frozen-model WindowEstimate adapter; constructor accepts no truth."""
from __future__ import annotations
import math
import time
import numpy as np
import torch

from .common_data import sampled_field
from .links import build_graph
from .model_features import full_raw_frame
from .models import SharedForecastBackbone, regions_for_positions
from .windows import NominalPredictor,WindowEstimate,service_start_intervals


class LearnedPredictor(NominalPredictor):
    requires_full_raw_history=True

    def __init__(self,times,nominal_depths,positions_m,elevation_m,model,bank,method,device='cpu',**kwargs):
        super().__init__(times,nominal_depths,**kwargs)
        self.side=int(round(len(positions_m)**.5))
        if self.depths.shape[1:]!=(self.side,self.side):
            raise ValueError('Expected a square public task grid')
        self.positions=np.asarray(positions_m,np.float32)
        self.elevation=np.asarray(elevation_m,np.float32)
        self.method,self.device=method,device
        self.model=model.to(device).eval()
        self.public_field=dict(times_s=self.times,depths_m=self.depths.reshape(len(self.times),-1))
        self.sample_times=np.arange(0.,self.times[-1]+1e-8,5.)
        self.sample_nominal=sampled_field(self.public_field,self.sample_times)
        public=dict(positions_m=self.positions,elevation_m=self.elevation,nominal_sample_m=self.sample_nominal)
        self.graph=build_graph(bank,public,self.side)
        self.region_ids=torch.as_tensor(regions_for_positions(self.positions,self.side),dtype=torch.long,device=device)
        self.query_ids=torch.arange(self.side*self.side,device=device)
        self.cache={};self.prediction=None;self.issue_latencies_ms=[]
        self.queries=[];self.publications=[];self.published_windows={}
        self._pending_observations=()
        # Device/kernel initialization belongs to startup, outside plan deadlines.
        started=time.perf_counter()
        self._compute(0.,[])
        self.warmup_wall_s=time.perf_counter()-started
        self.prediction=None

    def _compute(self,now,observations):
        n=self.side*self.side
        # Sensor events can coincide with a fractional flight arrival a few
        # ulps below the nominal tick. Use the same tolerant tick convention
        # as physical execution, while publication times retain exact `now`.
        base=math.floor((now+1e-7)/5)*5
        feature_clock=base if abs(now-base)<1e-7 else now
        start=max(0.,base-60.)
        selected=self.sample_times[(self.sample_times>=start)&(self.sample_times<=base)]
        nominal_indexes=np.rint(selected/5).astype(int)
        count=np.zeros((len(selected),n),np.uint8)
        mean=np.zeros(count.shape,np.float32);std=np.zeros_like(mean)
        names=('sample_time_s','arrival_time_s','row','col','drone','depth_m')
        obs={name:np.empty(0,int if name in ('row','col','drone') else float) for name in names}
        if observations:
            obs=dict(sample_time_s=np.asarray([r['sample_time'] for r in observations]),
                arrival_time_s=np.asarray([r['arrival_time'] for r in observations]),
                row=np.asarray([r['cell'][0] for r in observations],int),col=np.asarray([r['cell'][1] for r in observations],int),
                drone=np.asarray([r['drone'] for r in observations],int),depth_m=np.asarray([r['depth_m'] for r in observations]))
            legal=(obs['sample_time_s']>=now-60.-1e-9)&(obs['arrival_time_s']<=now+1e-9)
            obs={k:v[legal] for k,v in obs.items()}
            ticks=np.rint((obs['sample_time_s']-start)/5).astype(int)
            flat=ticks*n+obs['row']*self.side+obs['col']
            size=len(selected)*n
            counts=np.bincount(flat,minlength=size).reshape(count.shape)
            total=np.bincount(flat,weights=obs['depth_m'],minlength=size).reshape(count.shape)
            squared=np.bincount(flat,weights=obs['depth_m']**2,minlength=size).reshape(count.shape)
            present=counts>0
            mean[present]=total[present]/counts[present]
            std[present]=np.sqrt(np.maximum(0.,squared[present]/counts[present]-(total[present]/counts[present])**2))
            count[:]=counts
        panel=dict(times_s=selected,mean_depth_m=mean,std_depth_m=std,count=count,
            nominal_sample_m=self.sample_nominal[nominal_indexes])
        history=full_raw_frame(panel,obs,feature_clock)
        offsets=np.arange(13)*5.
        valid=now+offsets<=min(now+self.horizon_s,self.times[-1])+1e-8
        future_times=np.minimum(now+offsets,self.times[-1])
        nominal=sampled_field(self.public_field,future_times).T.copy()
        nominal[:,~valid]=0.
        static=np.column_stack((self.positions/(self.side*10.),self.elevation,nominal[:,0]))
        with torch.inference_mode():
            predicted=self.model(torch.as_tensor(history,device=self.device),torch.as_tensor(static,dtype=torch.float32,device=self.device),
                self.region_ids,self.query_ids,torch.as_tensor(nominal,device=self.device),
                torch.as_tensor(np.broadcast_to(valid,nominal.shape).copy(),device=self.device),self.graph,self.method)
            self.prediction=predicted.cpu().numpy()
        if not np.isfinite(self.prediction).all():
            raise FloatingPointError('Nonfinite frozen-model prediction')
        self.supported=bool(count.sum())

    def issue(self,now,arrived_observations):
        if not self.times[0]<=now<=self.times[-1]:
            raise ValueError('No forecast beyond saved nominal support')
        super().issue(now,arrived_observations)
        self.valid_until=min(float(self.times[-1]),now+5.*math.floor((self.valid_until-now+1e-8)/5.))
        self._curve_count=int(round((self.valid_until-self.issued_at)/5))+1
        self._published_times=np.minimum(self.issued_at+np.arange(self._curve_count)*5.,self.valid_until)
        self._published_time_tuple=tuple(map(float,self._published_times))
        started=time.perf_counter()
        # Publish the clock and freeze the arrived input immediately. Compute
        # a curve only when the planner actually requests a location; a fully
        # covered map with no pending task needs no unused full-map forecast.
        self._pending_observations=tuple(dict(cell=tuple(r['cell']),sample_time=float(r['sample_time']),
            arrival_time=float(r['arrival_time']),depth_m=float(r['depth_m']),drone=int(r['drone'])) for r in arrived_observations)
        self.prediction=None
        self.supported=False
        self.cache={}
        self.publications.append((self.version,self.issued_at,self.valid_until))
        self.issue_latencies_ms.append(1000*(time.perf_counter()-started))

    def _ensure_prediction(self):
        if self.prediction is None:
            started=time.perf_counter()
            self._compute(self.issued_at,self._pending_observations)
            # Whole-safe/whole-unsafe curves need no per-query segment scan.
            # Promote extrema to float64 so float32(.8) is not incorrectly
            # accepted as <= the exact 0.8m service threshold.
            self._curve_max=self.prediction[:,:self._curve_count].max(1).astype(float)
            self._curve_min=self.prediction[:,:self._curve_count].min(1).astype(float)
            self._pending_observations=()
            self.issue_latencies_ms[-1]+=1000*(time.perf_counter()-started)

    def _estimate(self,cell,now):
        if self.issued_at is None or now<self.issued_at-1e-9:
            raise ValueError('Forecast not published')
        cell=tuple(map(int,cell))
        if not (0<=cell[0]<self.side and 0<=cell[1]<self.side):
            raise ValueError('Query outside public task grid')
        self._ensure_prediction()
        if cell not in self.cache:
            index=cell[0]*self.side+cell[1]
            depth=self.prediction[index,:self._curve_count]
            if self._curve_count<2 or self.valid_until-self.issued_at<self.service_duration_s-1e-9:
                intervals=()
            elif self._curve_max[index]<=self.threshold_m:
                intervals=((self.issued_at,self.valid_until-self.service_duration_s),)
            elif self._curve_min[index]>self.threshold_m:
                intervals=()
            else:
                intervals=service_start_intervals(self._published_times,depth,self.service_duration_s,self.threshold_m)
            self.cache[cell]=WindowEstimate(cell,self.issued_at,self.version,self._published_time_tuple,tuple(map(float,depth)),
                intervals,self.supported,self.valid_until,evidence=f'frozen_{self.method}_same_full_arrived_raw_history',
                service_duration_s=self.service_duration_s)
            self.published_windows[self.version,cell]=self.cache[cell]
        return self.cache[cell]

    def estimate(self,cell,now):
        result=self._estimate(cell,now)
        self.queries.append((float(now),int(cell[0]),int(cell[1]),self.version,0))
        return result

    def estimate_many(self,cells,now):
        results=[self._estimate(cell,now) for cell in cells]
        self.queries.extend((float(now),int(cell[0]),int(cell[1]),self.version,1) for cell in cells)
        return results


def load_frozen_model(checkpoint,device='cpu'):
    record=torch.load(checkpoint,map_location=device,weights_only=True)
    model=SharedForecastBackbone(hidden=record['hidden']).to(device)
    model.load_state_dict(record['state_dict'],strict=True)
    return model.eval(),record
