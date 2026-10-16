"""Separate observable inputs, nominal priors, and offline reference labels."""
from pathlib import Path
import hashlib
import json
import numpy as np

from .lead import SOURCES,TARGETS
from .upstream_data import ForecastTable,source_records
from .full_upstream import full_histories
from .nominal_response import public_nominal_view,interpolate_nodes
from .metric_tasks import metric_cases,metric_scores


class MetricFamilyData:
    def __init__(self,manifest_path,design):
        self.path=Path(manifest_path);self.manifest=json.loads(self.path.read_text(encoding='utf-8'))
        completion=json.loads((self.path.parent/'completion.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(self.path.read_bytes()).hexdigest()==completion['manifest_sha256']
        self.design=design;self.cache={};self.nominal=public_nominal_view(self.manifest);self._contexts={};self._cases={}
        versions=self.manifest['forecast_versions'];release=sorted(int(v) for v in versions)
        frames=[];depth=[]
        for r in release:
            spec=versions[str(r)];field=self.field(spec['field_key']);rows=spec['published_rows']
            frames.append(field['time_s'][:rows]);depth.append(field['linear'][:rows])
        self.forecasts=ForecastTable(np.array(release),np.array(frames),np.array(depth),tuple(range(100)))
        self.source_forecasts=ForecastTable(self.forecasts.release_s,self.forecasts.valid_s,
                                           self.forecasts.depth_m[:,:,SOURCES],tuple(SOURCES))

    def field(self,key):
        if key not in self.cache:
            info=self.manifest['fields'][key];p=Path(info['projected_path'])
            if hashlib.sha256(p.read_bytes()).hexdigest()!=info['projected_sha256']:raise ValueError('Point field changed')
            with np.load(p,allow_pickle=False) as z:self.cache[key]={k:z[k] for k in ('time_s','linear')}
        return self.cache[key]

    def reference(self,mesh,condition,anchor,center):
        spec=self.manifest['references'][str(mesh)];base=self.field(spec['base'])
        if condition=='E0':return base['time_s'],base['linear']
        if condition!='E1':raise ValueError('Unknown offline condition')
        pulse=self.field(spec['branches'][f'a{anchor}_y{center}']);prefix=base['time_s']<anchor
        return np.r_[base['time_s'][prefix],pulse['time_s']],np.vstack([base['linear'][prefix],pulse['linear']])

    def nominal_context(self,anchor):
        if anchor not in self._contexts:
            now=anchor+np.array(self.design['decision_offsets_s']);offsets=np.arange(0,51,2)
            release=[];curves=[]
            for t in now:
                r,y=self.forecasts.at_decision(t,offsets);release.append(r);curves.append(y)
            full=np.array(curves) # time,horizon,point
            mean=full[:,0].mean(axis=1);drift=(full[:,3].mean(axis=1)-mean)/6
            base=full[:,:,TARGETS].transpose(0,2,1)
            p=self.nominal['known_params'];static=np.array([p['slope']*1000,p['ridge']/.1,p['friction']/.03,p['onset']/30,p['rate_nominal']/.003])
            common=[]
            for i,t in enumerate(now):
                shared=np.r_[t/390,(t-release[i])/10,static,mean[i],drift[i]*100,
                             full[i,0,SOURCES],full[i,0,TARGETS]]
                rows=[]
                for j,target in enumerate(TARGETS):
                    onehot=np.eye(len(TARGETS))[j]
                    rows.append(np.r_[base[i,j],shared,(5+10*(target%10))/100,(5+10*(target//10))/100,onehot])
                common.append(rows)
            self._contexts[anchor]=dict(now_s=now,base=base,common=np.array(common),nominal_mean=mean,
                nominal_drift=drift,current_release_s=np.array(release),source_current=full[:,0,SOURCES])
        return self._contexts[anchor]

    def inputs(self,mesh,condition,anchor,center,phase_index,noise_std,delay,replicate):
        times,depth=self.reference(mesh,condition,anchor,center)
        phase=self.design['sample_phase_fractions'][phase_index]
        valid=np.arange(anchor-60+5*phase,anchor+30+1e-9,5.)
        meta=self.manifest['metadata'];seed=[*meta['seed_identity'],937,anchor,phase_index,replicate]
        epsilon=noise_std*np.random.default_rng(np.random.SeedSequence(seed)).normal(size=(len(valid),len(SOURCES)))
        observed=source_records(times,depth[:,SOURCES],SOURCES,self.source_forecasts,valid,delay,epsilon)
        context=self.nominal_context(anchor)
        history=full_histories(observed['records'],context['now_s'])
        return dict(context,history=history,observation_sha256=observed['observation_sha256'],
                    observation_references=observed['references'])

    def labels(self,mesh,condition,anchor,center):
        times,depth=self.reference(mesh,condition,anchor,center);context=self.nominal_context(anchor)
        query=(context['now_s'][:,None,None]+np.arange(0,51,2)[None,None,:])
        query=np.broadcast_to(query,(len(context['now_s']),len(TARGETS),26))
        points=np.broadcast_to(np.array(TARGETS)[None,:,None],query.shape)
        truth=interpolate_nodes(times,depth,query.ravel(),points.ravel()).reshape(query.shape)
        return dict(truth=truth,residual=truth-context['base'])

    def cases(self,anchor):
        if anchor not in self._cases:
            cases,coverage=metric_cases(100,anchor,self.design['origin_ids'],self.design['speeds_m_per_s'],self.design['decision_offsets_s'])
            i=int(np.flatnonzero(self.forecasts.release_s==anchor)[0]);times=self.forecasts.valid_s[i];depth=self.forecasts.depth_m[i]
            scores=metric_scores(times,depth,cases,anchor)
            critical=np.array([min(abs(.8-m) for s in row for m in s['maxima'])<=self.design['critical_band_m'] for row in scores])
            self._cases[anchor]=dict(cases=cases,coverage=coverage,critical=critical,
                row_times=np.repeat(self.nominal_context(anchor)['now_s'],len(TARGETS)),
                row_targets=np.tile(TARGETS,len(self.design['decision_offsets_s'])))
        return self._cases[anchor]


def schedules(design):
    for noise in design['noise_std_m']:
        reps=1 if noise==0 else design['noisy_replicates']
        for delay in design['delays_s']:
            for phase in range(len(design['sample_phase_fractions'])):
                for replicate in range(reps):
                    yield dict(phase_index=phase,noise_std=noise,delay=delay,replicate=replicate,weight=1/reps)
