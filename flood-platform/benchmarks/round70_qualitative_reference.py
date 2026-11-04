"""Full reference audit using the previously frozen round69 qualitative rules."""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np

BASE=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
REPO=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029')
sys.path.insert(0,str(REPO/'flood-platform/benchmarks'))
from compare_okushiri_qualitative import validate_series, classify_windows, peak_interval, order, compare, count_rows, STATIONS

def analyze():
    out=BASE/'round70'
    protocol=json.loads((BASE/'round69/protocol.json').read_text(encoding='utf-8'))
    raw=np.loadtxt(BASE/'round39/output_ch5-7-9.txt',skiprows=1)
    reference=raw[raw[:,0]<=22.5]
    with np.load(BASE/'round47/exact_full/timeseries.npz') as data:
        ht=data['time_s'];hp=data['predicted_m'];ho=data['observed_m']
    previous=json.loads((BASE/'round69/analysis/comparison.json').read_text(encoding='utf-8'))['cases'][0]
    original_rows=previous['phase_rows'];rules=protocol['phase_rules']
    results=[];curves=[]
    for name in ['anuga_cfl04','anuga_cfl02']:
        directory=out/name
        with np.load(directory/'timeseries.npz',allow_pickle=False) as data:
            t=data['time_s'];o=data['observed_m'];p=data['predicted_m']
            residual=data['balance_residual_m3']
        validate_series(t,p,o,reference)
        with np.load(directory/'timestep_audit.npz') as data:dt=data['accepted_dt_s']
        run=json.loads((directory/'result.json').read_text())
        assert len(t)==451 and np.isfinite(p).all()
        assert np.isfinite(dt).all() and (dt>0).all() and abs(dt.sum()-22.5)<1e-10
        assert abs(residual).max()<1e-10 and run['min_depth_m']>=-1e-10
        rows=classify_windows(t,o,p,rules['primary_window_s'],rules['resolution_m'],rules['edge_block_fraction'])
        sensitivity=[]
        for width in rules['sensitivity_windows_s']:
            rr=classify_windows(t,o,p,width,rules['resolution_m'],rules['edge_block_fraction'])
            sensitivity.append({'width_s':width,'counts':count_rows(rr),'rows':rr})
        obs_peaks={s:peak_interval(t,o[:,j],rules['resolution_m'],.05) for j,s in enumerate(STATIONS)}
        model_peaks={s:peak_interval(t,p[:,j],rules['resolution_m'],.05) for j,s in enumerate(STATIONS)}
        pairs=[]
        for first,second in protocol['peak_order']['all_pairs']:
            a=order(obs_peaks[first]['sample_support_s'],obs_peaks[second]['sample_support_s'])
            b=order(model_peaks[first]['sample_support_s'],model_peaks[second]['sample_support_s'])
            pairs.append({'pair':[first,second],'observed_order':a,'model_order':b,'comparison':compare(a,b)})
        transitions=[]
        for a,b in zip(original_rows,rows):
            assert (a['station'],a['start_s'],a['end_s'])==(b['station'],b['start_s'],b['end_s'])
            if a['comparison']!=b['comparison']:
                transitions.append({'station':a['station'],'start_s':a['start_s'],'end_s':a['end_s'],
                    'hyra':a['comparison'],'reference':b['comparison']})
        result={'case':name,'evidence':'software diagnostic with laboratory observations; no production upgrade',
            'run':run,'phase_counts':count_rows(rows),'phase_rows':rows,
            'all_opposite_rows':[r for r in rows if r['comparison']=='inconsistent'],
            'all_unknown_rows':[r for r in rows if r['comparison']=='unknown'],
            'peak_order_counts':count_rows(pairs),'peak_order_rows':pairs,
            'observed_peaks':obs_peaks,'model_peaks':model_peaks,'sensitivity':sensitivity,
            'all_transitions_vs_hyra':transitions,
            'all_rows_consistent':all(r['comparison']=='consistent' for r in rows+pairs),
            'internal_rmse_mm':(np.sqrt(np.mean((p-o)**2,axis=0))*1000).tolist()}
        results.append(result);curves.append((name,t,p))
    changes=[]
    for a,b in zip(results[0]['phase_rows'],results[1]['phase_rows']):
        if a['comparison']!=b['comparison']:
            changes.append({'station':a['station'],'start_s':a['start_s'],'end_s':a['end_s'],
                'cfl04':a['comparison'],'cfl02':b['comparison']})
    audit={'source_protocol_sha256':hashlib.sha256((BASE/'round69/protocol.json').read_bytes()).hexdigest(),
        'primary_hyra_unchanged':{'case':'round47/exact_full','counts':previous['phase_counts']},
        'references':results,'time_sensitivity_label_changes':changes,
        'actual_max_dt_ratio':results[1]['run']['max_dt_s']/results[0]['run']['max_dt_s'],
        'interpretation_limits':['Euler vs RK2 and triangular vertex bed vs square cell-centre bed differ',
            'No mesh convergence or NS necessity conclusion','No calibration or resampling of original observations',
            'All rows retained; no conditional removal of failed/unknown rows','No field-event rescue acceptance']}
    (out/'qualitative-reference-comparison.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 完整参考定性对照（软件诊断，不是生产升级）','',
        '|配置|测站|时窗s|观测|模拟|结论|','|---|---|---|---|---|---|']
    for case in results:
        for r in case['phase_rows']:
            lines.append(f"|{case['case']}|{r['station']}|{r['start_s']}–{r['end_s']}|{r['observed']['label']}|{r['model']['label']}|{r['comparison']}|")
    (out/'all-reference-rows.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    for j,ax in enumerate(axes):
        ax.plot(ht,ho[:,j]*1000,color='black',lw=1.3,label='Laboratory record')
        ax.plot(ht,hp[:,j]*1000,color='#1568a3',lw=1.1,label='HYRA exact 6 mm (unchanged)')
        for name,t,p in curves:
            ax.plot(t,p[:,j]*1000,lw=.9,ls='--' if name.endswith('02') else '-',
                color='#c56e25' if name.endswith('02') else '#2e8b57',label=name)
        ax.set(ylabel=STATIONS[j]+' stage (mm)',xlim=(0,22.5));ax.grid(alpha=.15)
        if j==1:ax.axvspan(21,22,alpha=.12,color='#a54b38')
    axes[0].legend(fontsize=8,ncol=2)
    axes[-1].set_xlabel('Original laboratory time (s); no offsets or shifts')
    fig.suptitle('Independent SWE implementations; matching observations remains unproven as a whole')
    fig.savefig(out/'reference-full-curves.png',dpi=140);plt.close(fig)
    print(json.dumps({'references':[{'case':r['case'],'phase_counts':r['phase_counts'],
        'peak_counts':r['peak_order_counts'],'tail':[x for x in r['phase_rows'] if x['station']=='ch7' and x['start_s']==21],
        'opposite':[(x['station'],x['start_s']) for x in r['all_opposite_rows']]} for r in results],
        'time_label_changes':changes,'actual_max_dt_ratio':audit['actual_max_dt_ratio']},ensure_ascii=False,indent=2))

if __name__=='__main__':analyze()
