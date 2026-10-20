"""Family x training-seed paired statistics, with layouts averaged first."""
import numpy as np

SEEDS=(101,202,303)


def paired_effect(rows,method,baseline,*,families=None,replicates=10000,seed=2026101803):
    selected=[r for r in rows if r['method'] in (method,baseline)]
    ids=sorted(set(r['family_id'] for r in selected)) if families is None else sorted(families)
    keyed={}
    for r in selected:
        key=r['family_id'],r['training_seed'],r['layout'],r['method']
        if key in keyed:
            raise ValueError('Duplicate comparison cell')
        keyed[key]=float(r['summary']['success_rate'])
    differences=np.zeros((len(ids),3))
    for i,family in enumerate(ids):
        for j,training_seed in enumerate(SEEDS):
            pairs=[]
            for layout in (0,1):
                try:
                    pairs.append(keyed[family,training_seed,layout,method]-keyed[family,training_seed,layout,baseline])
                except KeyError as error:
                    raise ValueError(f'Missing paired method/seed/layout cell: {error}') from error
            differences[i,j]=np.mean(pairs)
    if not len(ids):
        raise ValueError('No independent families')
    rng=np.random.default_rng(seed)
    boot=np.empty(replicates)
    for i in range(replicates):
        fi=rng.integers(0,len(ids),len(ids));si=rng.integers(0,3,3)
        boot[i]=differences[fi[:,None],si[None,:]].mean()*100
    interval=np.quantile(boot,[.025,.975]).tolist()
    point=float(differences.mean()*100)
    return dict(method=method,baseline=baseline,delta_percentage_points=point,ci95_percentage_points=interval,
        family_ids=ids,training_seeds=list(SEEDS),layouts_averaged_first=True,
        family_seed_differences_pp=(differences*100).tolist(),bootstrap_replicates=replicates,bootstrap_seed=seed,
        primary_positive=interval[0]>0,engineering_target_3pp_met=point>=3.,
        resampling='independent family and training-seed indices, paired across methods; layouts are not independent units')


def select_bstar(rows,fixed_metrics,*,expected_families=12):
    scores={}
    for method in ('H','E'):
        group=[r for r in rows if r['method']==method and r['side']==50 and r['delay_s']==0]
        families=sorted({r['family_id'] for r in group})
        if len(families)!=expected_families or len(group)!=expected_families*2*3:
            raise ValueError('B* needs the complete primary-scale zero-delay validation matrix')
        keys={(r['family_id'],r['training_seed'],r['layout']) for r in group}
        if len(keys)!=len(group) or any((f,s,l) not in keys for f in families for s in SEEDS for l in (0,1)):
            raise ValueError('Validation method/seed/layout pairing incomplete')
        predictive=[r for r in fixed_metrics if r['method']==method and r['delay_s']==0]
        if len(predictive)!=expected_families*2*3:
            raise ValueError('Missing fixed-route validation diagnostics for B* tie-break')
        scores[method]=dict(success_rate=float(np.mean([r['summary']['success_rate'] for r in group])),
            window_error_rate=float(np.mean([r['window_error_rate'] for r in predictive])),
            prediction_mse_m2=float(np.mean([r['mse_m2'] for r in predictive])))
    def rank(method):
        s=scores[method]
        return (-s['success_rate'],s['window_error_rate'],s['prediction_mse_m2'],0 if method=='H' else 1)
    chosen=min(('H','E'),key=rank)
    return dict(B_star=chosen,scores=scores,selection_split='validation_only',selected_once=True,
        rule='primary-scale zero-delay service rate, then fixed-route window error, then fixed-route depth MSE, then H',
        P_or_test_results_used=False)
