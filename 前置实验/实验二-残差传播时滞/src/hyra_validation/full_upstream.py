"""All received 5-second source records; no legacy ten-second thinning."""
import numpy as np
from .hidden_target_inputs import causal_history,SOURCES


def full_histories(records,decision_times):
    histories=[causal_history(records,float(now),history_s=60,bin_s=5) for now in decision_times]
    values=np.stack([h['values'] for h in histories])
    return dict(values=values,record_ids=np.stack([h['record_ids'] for h in histories]),
                source_ids=np.array(SOURCES),decision_times=np.asarray(decision_times,float))


def ordinary_full(history,models,offsets=np.arange(0,51,2)):
    """RW/CV Kalman, persistence and EWMA on every actual source sample."""
    values=np.asarray(history['values'],float)
    if values.ndim!=4 or values.shape[1:]!=(6,12,6):raise ValueError('Expected all six by twelve raw records')
    ntime=len(values);raw=values.reshape(-1,12,6)[:,::-1]
    y=raw[:,:,2];time=-raw[:,:,3];mask=raw[:,:,5]>0;n=len(raw)
    out={'nominal':np.zeros((n,len(offsets)))};latest=np.zeros(n)
    for k in range(12):latest=np.where(mask[:,k],y[:,k],latest)
    out['persistence']=np.repeat(latest[:,None],len(offsets),axis=1)
    for half in models['ewma_half_life_s']:
        value=np.zeros(n);last=np.zeros(n);initialized=np.zeros(n,bool)
        for k in range(12):
            active=mask[:,k];first=active & ~initialized
            alpha=1-2**(-np.maximum(time[:,k]-last,0)/half)
            value=np.where(first,y[:,k],np.where(active,value+alpha*(y[:,k]-value),value))
            last=np.where(active,time[:,k],last);initialized|=active
        out[f'ewma_{half}']=np.repeat(value[:,None],len(offsets),axis=1)
    for kind,qs in [('rw',models['kf_rw_process_variance_rate_m2_per_s']),('cv',models['kf_cv_process_spectral_density_m2_per_s3'])]:
        for q in qs:
            for r in models['kf_measurement_std_m']:
                x=np.zeros((n,2));p=np.zeros((n,2,2));p[:,0,0]=models['kf_initial_bias_std_m']**2
                p[:,1,1]=models['kf_initial_bias_rate_std_m_per_s']**2 if kind=='cv' else 0
                last=np.zeros(n);initialized=np.zeros(n,bool)
                for k in range(12):
                    active=mask[:,k];dt=np.where(active & initialized,time[:,k]-last,0)
                    if np.any(dt<0):raise ValueError('History is not chronological')
                    f=np.broadcast_to(np.eye(2),(n,2,2)).copy()
                    if kind=='cv':f[:,0,1]=dt
                    x=np.einsum('nij,nj->ni',f,x);p=f@p@f.transpose(0,2,1)
                    if kind=='cv':
                        p[:,0,0]+=q*dt**3/3;p[:,0,1]+=q*dt**2/2;p[:,1,0]+=q*dt**2/2;p[:,1,1]+=q*dt
                    else:p[:,0,0]+=q*dt
                    gain=p[:,:,0]/(p[:,0,0]+r*r)[:,None]
                    proposed=x+gain*(y[:,k]-x[:,0])[:,None]
                    ikh=np.broadcast_to(np.eye(2),(n,2,2)).copy();ikh[:,:,0]-=gain
                    proposed_p=ikh@p@ikh.transpose(0,2,1)+r*r*gain[:,:,None]*gain[:,None,:]
                    x=np.where(active[:,None],proposed,x);p=np.where(active[:,None,None],proposed_p,p)
                    last=np.where(active,time[:,k],last);initialized|=active
                out[f'kf_{kind}_q{q:g}_r{r:g}']=x[:,0,None]+(np.asarray(offsets)[None,:]-last[:,None])*x[:,1,None]
    return {name:array.reshape(ntime,6,-1) for name,array in out.items()}
