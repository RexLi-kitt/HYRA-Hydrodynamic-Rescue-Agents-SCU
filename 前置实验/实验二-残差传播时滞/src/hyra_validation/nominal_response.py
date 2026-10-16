"""Known nominal-response prior. Actual event metadata and reference fields are excluded."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.special import log_ndtr,logsumexp

from .lead import SOURCES,TARGETS


def public_nominal_view(manifest):
    keys={manifest['nominal_base'],*manifest['nominal_sensitivities'].values()}
    for calibration in manifest['nominal_calibration'].values():
        keys.add(calibration['zero']);keys.update(calibration['probes'].values())
    keys.update(v['field_key'] for v in manifest['forecast_versions'].values())
    fields={}
    for key in keys:
        info=manifest['fields'][key]
        if not info['kind'].startswith('nominal'):raise ValueError('Reference field in public nominal view')
        if info.get('restart') and info['restart']['source_kind']!='same_mesh_independent_nominal_base':
            raise ValueError('Physical prior cannot use reference initial states')
        fields[key]={k:info[k] for k in ('kind','projected_path','projected_sha256')}
    meta=manifest['metadata'];p=meta['params'];factor=meta['nominal_specifications'][0]['factor']
    return dict(fields=fields,nominal_base=manifest['nominal_base'],
        nominal_sensitivities=manifest['nominal_sensitivities'],nominal_calibration=manifest['nominal_calibration'],
        forecast_versions=manifest['forecast_versions'],
        known_params={k:p[k] for k in ('slope','ridge','friction','onset')}|dict(rate_nominal=p['rate']*factor))


def interpolate_nodes(times,values,query,points,zero_before=False):
    times=np.asarray(times);query=np.asarray(query,float);points=np.asarray(points,int)
    if np.any(~np.isfinite(query)) or np.any(query>times[-1]+1e-9) or (not zero_before and np.any(query<times[0])):
        raise ValueError('Response query outside calibrated support')
    q=np.clip(query,times[0],times[-1]);upper=np.searchsorted(times,q,side='right')
    upper=np.clip(upper,1,len(times)-1);lower=upper-1
    w=(q-times[lower])/(times[upper]-times[lower])
    out=(1-w)*values[lower,points]+w*values[upper,points]
    return np.where(query<times[0],0.,out) if zero_before else out


def half_gaussian_posterior(design,y,noise_std,beta_std,amplitude_std):
    """Batched two-parameter Gaussian posterior, with nonnegative pulse amplitude."""
    x=np.asarray(design,float);y=np.asarray(y,float)
    if x.ndim!=3 or x.shape[2]!=2 or x.shape[1]!=len(y) or min(noise_std,beta_std,amplitude_std)<=0:
        raise ValueError('Invalid source inverse problem')
    var=noise_std**2;prior_var=np.array([beta_std**2,amplitude_std**2])
    precision=np.einsum('hni,hnj->hij',x,x)/var+np.diag(1/prior_var)
    covariance=np.linalg.inv(precision);q=np.einsum('hni,n->hi',x,y)/var
    mean=np.einsum('hij,hj->hi',covariance,q)
    sign,logdet=np.linalg.slogdet(precision)
    if np.any(sign<=0):raise ValueError('Nonpositive posterior precision')
    log_evidence=-.5*(len(y)*np.log(2*np.pi*var)+(y@y)/var-np.sum(q*mean,axis=1)+logdet+np.log(prior_var).sum())
    sd=np.sqrt(covariance[:,1,1]);z=mean[:,1]/sd;log_probability=log_ndtr(z)
    mills=np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_probability)
    amplitude=np.maximum(0.,mean[:,1]+sd*mills)
    adjusted=mean.copy();adjusted[:,0]+=covariance[:,0,1]/covariance[:,1,1]*(amplitude-mean[:,1]);adjusted[:,1]=amplitude
    log_evidence+=np.log(2.)+log_probability
    return dict(mean=adjusted,untruncated_mean=mean,covariance=covariance,log_evidence=log_evidence)


class NominalResponseLibrary:
    def __init__(self,view,relative_step=.02):
        self.view=view;self.cache={}
        def field(key):
            info=view['fields'][key]
            if not info['kind'].startswith('nominal'):raise ValueError('Non-nominal calibration')
            p=Path(info['projected_path'])
            if hashlib.sha256(p.read_bytes()).hexdigest()!=info['projected_sha256']:raise ValueError('Calibration checksum mismatch')
            with np.load(p,allow_pickle=False) as z:self.cache[key]={k:z[k] for k in ('time_s','linear')}
            return self.cache[key]
        self.field=field
        base=field(view['nominal_base']);plus=field(view['nominal_sensitivities']['plus']);minus=field(view['nominal_sensitivities']['minus'])
        np.testing.assert_array_equal(plus['time_s'],minus['time_s'])
        self.sensitivity_times=plus['time_s'];self.sensitivity=(plus['linear']-minus['linear'])/(2*relative_step)
        states=[];responses=[];self.centers=sorted({int(c) for item in view['nominal_calibration'].values() for c in item['probes']})
        relative_times=None
        for anchor,item in sorted(view['nominal_calibration'].items(),key=lambda p:int(p[0])):
            zero=field(item['zero']);states.append(float(zero['linear'][0].mean()));curves=[]
            for center in self.centers:
                probe=field(item['probes'][str(center)])
                np.testing.assert_array_equal(zero['time_s'],probe['time_s'])
                curves.append(probe['linear']-zero['linear'])
            responses.append(curves);relative_times=zero['time_s']-zero['time_s'][0]
        order=np.argsort(states);self.states=np.array(states)[order];self.responses=np.array(responses)[order]
        if np.any(np.diff(self.states)<=0):raise ValueError('Calibration states must be distinct')
        self.response_times=relative_times

    def mixed_response(self,center_index,nominal_state):
        hi=int(np.clip(np.searchsorted(self.states,nominal_state,side='right'),1,len(self.states)-1));lo=hi-1
        w=float(np.clip((nominal_state-self.states[lo])/(self.states[hi]-self.states[lo]),0,1))
        return (1-w)*self.responses[lo,center_index]+w*self.responses[hi,center_index]

    def source_problem(self,history_values,now,nominal_mean,nominal_drift,onset_ages=np.arange(0,41,2),offsets=np.arange(0,51,2)):
        h=np.asarray(history_values,float)
        if h.shape!=(6,12,6):raise ValueError('Physical model requires all common raw records')
        active=h[:,:,5].ravel()>0;ids=np.repeat(SOURCES,12)[active];age=h[:,:,3].ravel()[active]
        y=h[:,:,2].ravel()[active];source_times=now-age
        g=interpolate_nodes(self.sensitivity_times,self.sensitivity,source_times,ids)
        target_ids=np.repeat(TARGETS,len(offsets));future=np.tile(offsets,len(TARGETS))+now
        gy=interpolate_nodes(self.sensitivity_times,self.sensitivity,future,target_ids).reshape(len(TARGETS),len(offsets))
        designs=[np.stack([g,np.zeros_like(g)],axis=-1)];physical=[np.zeros_like(gy)];recent=[np.zeros_like(gy)];wrong=[np.zeros_like(gy)];spatial=[np.zeros_like(gy)]
        hypothesis_centers=[-1];hypothesis_ages=[0.]
        for ci in range(len(self.centers)):
            for event_age in onset_ages:
                response=self.mixed_response(ci,nominal_mean-nominal_drift*event_age)
                pulse=interpolate_nodes(self.response_times,response,event_age-age,ids,zero_before=True)
                designs.append(np.stack([g,pulse],axis=-1));query=np.tile(offsets,len(TARGETS))+event_age
                physical.append(interpolate_nodes(self.response_times,response,query,target_ids,True).reshape(gy.shape))
                first=self.response_times<=60;source_signal=np.max(abs(response[first][:,SOURCES]),axis=1)
                peak=float(source_signal.max());source_marker=0. if peak<.001 else float(self.response_times[first][np.flatnonzero(source_signal>=.1*peak)[0]])
                lags=[]
                for target in TARGETS:
                    signal=abs(response[first,target]);peak=float(signal.max())
                    target_marker=source_marker if peak<.001 else float(self.response_times[first][np.flatnonzero(signal>=.1*peak)[0]])
                    lags.append(max(target_marker-source_marker,0.))
                shifted=query+np.repeat(lags,len(offsets))
                recent.append(interpolate_nodes(self.response_times,response,shifted,target_ids,True).reshape(gy.shape))
                wrong.append(interpolate_nodes(self.response_times,response,query+20,target_ids,True).reshape(gy.shape))
                shuffled=np.array([r+50 if r//10<5 else r-50 for r in target_ids])
                spatial.append(interpolate_nodes(self.response_times,response,query,shuffled,True).reshape(gy.shape))
                hypothesis_centers.append(ci);hypothesis_ages.append(float(event_age))
        count=len(designs)-1;prior=np.r_[.5,np.full(count,.5/count)]
        return dict(design=np.array(designs),y=y,global_target=gy,physical=np.array(physical),recency=np.array(recent),
                    wrong_lag=np.array(wrong),wrong_spatial=np.array(spatial),log_prior=np.log(prior),
                    hypothesis_centers=np.array(hypothesis_centers),hypothesis_ages=np.array(hypothesis_ages))


def response_predictions(problem,noise_std=.02,beta_std=.02,amplitude_std=2.):
    posterior=half_gaussian_posterior(problem['design'],problem['y'],noise_std,beta_std,amplitude_std)
    log_weight=posterior['log_evidence']+problem['log_prior'];weight=np.exp(log_weight-logsumexp(log_weight))
    mean=posterior['mean'];global_component=problem['global_target']*float(weight@mean[:,0])
    out={mode:global_component+np.einsum('h,htq->tq',weight*mean[:,1],problem[mode])
         for mode in ('physical','recency','wrong_lag','wrong_spatial')}
    out.update(global_component=global_component,hypothesis_weights=weight,
               event_probability=float(1-weight[0]),posterior=posterior)
    return out
