"""Scores never delete inconvenient points or turn missing data into dry land."""
import numpy as np
from .contracts import load_run

def strict_speed_score(ids,observed,predicted,relative=.05,zero_absolute=.02):
    ids=list(ids);o=np.asarray(observed,dtype=float);p=np.asarray(predicted,dtype=float)
    if len(ids)!=len(set(ids)) or o.shape!=p.shape or o.ndim!=1 or len(ids)!=len(o) or not len(o):raise ValueError('All unique points required')
    if not np.isfinite(o).all() or not np.isfinite(p).all() or (o<0).any() or (p<0).any():raise ValueError('Missing/nonphysical speeds must not be scored as zero')
    if not np.isfinite([relative,zero_absolute]).all() or min(relative,zero_absolute)<0:raise ValueError('Invalid tolerance')
    err=p-o;limit=np.where(o>0,relative*o,zero_absolute);ok=abs(err)<=limit
    return {'point_count':len(ids),'passed':int(ok.sum()),'rmse_m_s':float(np.sqrt(np.mean(err**2))),'mae_m_s':float(abs(err).mean()),
        'criteria':{'relative_nonzero':relative,'absolute_zero_m_s':zero_absolute,'basis':'user research target, not universal observation uncertainty'},
        'points':[dict(id=k,observed_m_s=float(a),predicted_m_s=float(b),absolute_error_m_s=float(abs(e)),tolerance_m_s=float(l),passed=bool(q)) for k,a,b,e,l,q in zip(ids,o,p,err,limit,ok)]}

def inundation_score(truth,prediction):
    t=np.asarray(truth);p=np.asarray(prediction)
    if t.shape!=p.shape or not np.isin(t,[-1,0,1]).all() or not np.isin(p,[-1,0,1,2]).all():raise ValueError('Expected unknown=-1, dry=0, wet=1, predicted uncertain=2')
    known=t>=0;classified=known & np.isin(p,[0,1]);wet=t==1;dry=t==0
    tp=int((wet&(p==1)).sum());fn=int((wet&(p==0)).sum());fp=int((dry&(p==1)).sum());tn=int((dry&(p==0)).sum())
    unresolved=int((known&~classified).sum());uw=int((wet&~classified).sum())
    return dict(tp=tp,fn=fn,fp=fp,tn=tn,unknown_observation=int((~known).sum()),unresolved_known_truth=unresolved,
        observed_wet_unresolved=uw,classified_csi=tp/(tp+fn+fp) if tp+fn+fp else None,
        wet_recall_lower_bound=tp/int(wet.sum()) if wet.any() else None,
        note='CSI applies only to classified observed cells; unresolved wet cells reduce recall lower bound; no overall accuracy')

def compare_runs(first,second):
    ma,a=load_run(first);mb,b=load_run(second)
    if ma['case_sha256']!=mb['case_sha256']:raise ValueError('Different case/input contract: refuse direct comparison')
    result={'kind':'software comparison, not experimental truth','first':ma['backend'],'second':mb['backend'],'case_sha256':ma['case_sha256'],'differences':{}}
    for k in ('depth_m','qx_m2_s','qy_m2_s'):
        e=b[k]-a[k];result['differences'][k]={'rmse':float(np.sqrt(np.mean(e**2))),'max_abs':float(abs(e).max())}
    result['native_meshes']=[ma['native_cells'],mb['native_cells']];return result


def gauge_series(folder,velocity_min_depth_m):
    """Same canonical cell-average operator for all backends; no point-value claim."""
    if not np.isfinite(velocity_min_depth_m) or velocity_min_depth_m<=0:raise ValueError('Explicit positive velocity depth required')
    m,a=load_run(folder);c=m['case'];g=c['grid'];ox,oy=g['origin_xy_m'];ny,nx=g['shape'];dx=g['dx_m'];rows=[]
    for point in c['gauges']:
        ix=int(np.floor((point['x_m']-ox)/dx));iy=int(np.floor((point['y_m']-oy)/dx))
        if not 0<=ix<nx or not 0<=iy<ny:raise ValueError('Gauge outside canonical grid')
        h=a['depth_m'][:,iy,ix];wet=h>=velocity_min_depth_m
        u=np.divide(a['qx_m2_s'][:,iy,ix],h,out=np.full_like(h,np.nan),where=wet)
        v=np.divide(a['qy_m2_s'][:,iy,ix],h,out=np.full_like(h,np.nan),where=wet)
        rows.append(dict(id=point['id'],depth_m=h.tolist(),u_east_m_s=[float(x) if np.isfinite(x) else None for x in u],
                    v_north_m_s=[float(x) if np.isfinite(x) else None for x in v],velocity_valid=wet.tolist()))
    return dict(time_s=a['time_s'].tolist(),sampling='containing canonical cell average; no snapping or point interpolation',
                velocity_min_depth_m=velocity_min_depth_m,gauges=rows)
