"""Compare complete archives against all original observations, without rerunning CFD."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hyra_flood.archive import score_merewether


def compare(old,new,source,out):
    old,new,source,out=map(Path,(old,new,source,out))
    if out.exists():raise FileExistsError('Do not overwrite a comparison')
    a=score_merewether(old,source);b=score_merewether(new,source)
    ca=json.loads((old/'config.json').read_text(encoding='utf8'))
    cb=json.loads((new/'config.json').read_text(encoding='utf8'))
    for key in ('source','sampling','roughness'):
        if ca[key]!=cb[key]:raise ValueError('Changed '+key)
    ra=json.loads((old/'result.json').read_text(encoding='utf8'))
    rb=json.loads((new/'result.json').read_text(encoding='utf8'))
    rows=[]
    for x,y,p,q in zip(a['points'],b['points'],ra['points'],rb['points']):
        if x['id']!=y['id'] or x['observed_m_s']!=y['observed_m_s']:raise ValueError('Misaligned observations')
        delta=y['absolute_error_m_s']-x['absolute_error_m_s']
        rows.append({'id':x['id'],'observed_m_s':x['observed_m_s'],
          'old_m_s':x['predicted_m_s'],'new_m_s':y['predicted_m_s'],
          'old_abs_error_m_s':x['absolute_error_m_s'],'new_abs_error_m_s':y['absolute_error_m_s'],
          'error_change_m_s':delta,'status':'improved' if delta < -1e-9 else 'degraded' if delta>1e-9 else 'unchanged',
          'old_pass':x['passed'],'new_pass':y['passed'],'tolerance_m_s':x['tolerance_m_s'],
          'old_max_depth_m':p['max_depth_m'],'new_max_depth_m':q['max_depth_m']})
    nwin=sum(x['status']=='improved' for x in rows);nlose=sum(x['status']=='degraded' for x in rows)
    result={'baseline':str(old),'candidate':str(new),'evidence':a['evidence'],
      'kind':'grid/geometric-discretization experiment, not new-engine validation',
      'criteria':a['criteria'],'scores':{name:{k:v[k] for k in ('passed','point_count','rmse_m_s','mae_m_s','points_below_sampling_depth')} for name,v in [('baseline',a),('candidate',b)]},
      'counts':{'improved':nwin,'degraded':nlose,'unchanged':40-nwin-nlose},
      'new_passes':[x['id'] for x in rows if x['new_pass'] and not x['old_pass']],
      'lost_passes':[x['id'] for x in rows if x['old_pass'] and not x['new_pass']],
      'total_abs_error_reduction_m_s':sum(-x['error_change_m_s'] for x in rows if x['error_change_m_s']<0),
      'total_abs_error_increase_m_s':sum(x['error_change_m_s'] for x in rows if x['error_change_m_s']>0),
      'rmse_reduction_percent':100*(1-b['rmse_m_s']/a['rmse_m_s']),
      'mae_reduction_percent':100*(1-b['mae_m_s']/a['mae_m_s']),
      'critical_points':[x for x in rows if x['id'] in ('19-J','16-K','16-F','21-I')],
      'balance':{'baseline':ra['balance'],'candidate':rb['balance']},
      'duration_s':[ra['duration_s'],rb['duration_s']],'wall_s':[ra['wall_s'],rb['wall_s']],
      'raw_audits':{'baseline':a['raw_audit'],'candidate':b['raw_audit']},'points':rows,
      'limits':['The new ANUGA/LISFLOOD backend is not this candidate.',
        'Grid refinement and geometric rasterization change together; no convergence proof.',
        'Numerical sampling wetness h>=.01m is not a rescue safety threshold.',
        'No independent dry-area evidence or continuous measured velocity; not a field acceptance.'],
      'comparison_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.mkdir(parents=True)
    for filename,obj in [('comparison.json',result),('baseline-score.json',a),('candidate-score.json',b)]:
        (out/filename).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(40);fig,ax=plt.subplots(2,1,figsize=(16,8),sharex=True,layout='constrained')
    ax[0].plot(x,[r['observed_m_s'] for r in rows],'ko',label='Laboratory observation')
    ax[0].plot(x,[r['old_m_s'] for r in rows],'s-',label='HYRA 0.70 m',ms=3,alpha=.8)
    ax[0].plot(x,[r['new_m_s'] for r in rows],'^-',label='HYRA 0.35 m',ms=3,alpha=.8)
    ax[0].set_ylabel('Maximum speed (m/s)');ax[0].legend(ncols=3);ax[0].grid(alpha=.2)
    ax[1].bar(x,[r['error_change_m_s'] for r in rows],color=['#c3473f' if r['error_change_m_s']>0 else '#278a63' for r in rows])
    ax[1].axhline(0,color='black',lw=.7);ax[1].set_ylabel('New minus old absolute error (m/s)')
    ax[1].set_xticks(x,[r['id'] for r in rows],rotation=90);ax[1].grid(axis='y',alpha=.2)
    fig.suptitle('Merewether: all 40 original points, full 0-800 s; lower error is better')
    fig.savefig(out/'all40-comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('scores','counts','rmse_reduction_percent','mae_reduction_percent','new_passes','lost_passes','critical_points')},ensure_ascii=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('source');p.add_argument('output');v=p.parse_args()
    compare(v.baseline,v.candidate,v.source,v.output)
