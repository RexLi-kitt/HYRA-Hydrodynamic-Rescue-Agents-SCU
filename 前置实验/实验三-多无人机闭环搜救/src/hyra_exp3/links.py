"""Training-only sparse residual links, validated on held-out TRAIN families."""
from pathlib import Path
import numpy as np

from .artifacts import sha256, write_json
from .diagnostics import load_field
from .models import regions_for_positions
from .protocol import load_protocol


def regional_panel(panel, side):
    regions = regions_for_positions(panel["positions_m"], side)
    t = len(panel["times_s"])
    observed = np.full((t,16), np.nan)
    nominal, labels = np.zeros((t,16)), np.zeros((t,16))
    for region in range(16):
        cells = regions == region
        counts = panel["count"][:,cells].astype(float)
        n = counts.sum(1)
        raw = ((panel["mean_depth_m"][:,cells] - panel["nominal_sample_m"][:,cells]) * counts).sum(1)
        observed[n>0,region] = raw[n>0] / n[n>0]
        nominal[:,region] = panel["nominal_sample_m"][:,cells].mean(1)
        labels[:,region] = (panel["label_truth_sample_m"][:,cells] - panel["nominal_sample_m"][:,cells]).mean(1)
        last, last_tick = np.nan, -1000
        for tick in range(t):
            if np.isfinite(observed[tick,region]):
                last,last_tick = observed[tick,region],tick
            elif tick-last_tick <= 12:
                observed[tick,region] = last
    available = np.isfinite(observed)
    global_mean = np.nansum(observed,axis=1) / available.sum(1).clip(1)
    return dict(observed=observed,nominal=nominal,labels=labels,global_mean=global_mean)


def regression_rows(panel, target, source, lag):
    observed = panel["observed"]
    ticks = np.arange(max(3,lag),len(observed)-3)
    own, before = observed[ticks,target],observed[ticks-3,target]
    x = observed[ticks-lag,source]
    y = panel["labels"][ticks+3,target]
    base = np.column_stack((np.ones(len(ticks)),own,(own-before)/15.,panel["global_mean"][ticks],
                            panel["nominal"][ticks+3,target]-panel["nominal"][ticks,target]))
    valid = np.isfinite(base).all(1) & np.isfinite(x) & np.isfinite(y)
    return base[valid], x[valid], y[valid]


def _fit_bank(records):
    fit = [r for r in records if not r["held_out"]]
    held = [r for r in records if r["held_out"]]
    shape=(16,16)
    train_gain, held_gain = np.zeros(shape), np.zeros(shape)
    lag_steps, improving_families = np.zeros(shape,int),np.zeros(shape,int)
    for target in range(16):
        for source in range(16):
            if target==source:
                continue
            best = None
            for lag in range(13):
                pieces=[regression_rows(r["panel"],target,source,lag) for r in fit]
                if not pieces:
                    continue
                b=np.concatenate([p[0] for p in pieces]);x=np.concatenate([p[1] for p in pieces]);y=np.concatenate([p[2] for p in pieces])
                if len(y)<100:
                    continue
                bx=np.column_stack((b,x))
                ridge=1e-4
                beta0=np.linalg.solve(b.T@b+ridge*np.eye(b.shape[1]),b.T@y)
                beta1=np.linalg.solve(bx.T@bx+ridge*np.eye(bx.shape[1]),bx.T@y)
                mse0=float(np.mean((y-b@beta0)**2));mse1=float(np.mean((y-bx@beta1)**2))
                improvement=(mse0-mse1)/max(mse0,1e-10)
                if best is None or improvement>best[0]:
                    best=(improvement,lag,beta0,beta1)
            if best is None:
                continue
            gain,lag,beta0,beta1=best
            gains_by_family={}
            totals=[]
            for record in held:
                b,x,y=regression_rows(record["panel"],target,source,lag)
                if len(y)<10:
                    continue
                e0=(y-b@beta0)**2;e1=(y-np.column_stack((b,x))@beta1)**2
                totals.append((e0,e1))
                gains_by_family.setdefault(record["family_id"],[]).append(float(np.sum(e0-e1)))
            if not totals:
                continue
            e0=np.concatenate([p[0] for p in totals]);e1=np.concatenate([p[1] for p in totals])
            train_gain[target,source]=gain
            held_gain[target,source]=float(np.mean(e0-e1)/max(np.mean(e0),1e-10))
            lag_steps[target,source]=lag
            improving_families[target,source]=sum(sum(v)>0 for v in gains_by_family.values())
    return dict(train_gain=train_gain.tolist(),held_out_train_gain=held_gain.tolist(),
                lag_steps=lag_steps.tolist(),held_out_train_improving_families=improving_families.tolist())


def fit_links(root, data_run, output):
    data_run,output=Path(data_run),Path(output)
    paths=sorted((data_run/'jobs').glob('*/case.json'))
    cases=[load_protocol(p) for p in paths]
    cases=[c for c in cases if c['family']['split']=='train']
    if len(cases)!=72 or len({c['family']['family_id'] for c in cases})!=24:
        raise ValueError('All 24 training families and three scales required for link fitting')
    banks={}
    for side in (32,50,70):
        records=[]
        for case in cases:
            if case['side']!=side:
                continue
            for p in case['panels']:
                if sha256(p['path'])!=p['sha256']:
                    raise AssertionError('Common panel changed')
                panel=load_field(p['path'])
                records.append(dict(panel=regional_panel(panel,side),family_id=case['family']['family_id'],
                    held_out=case['family']['index_within_split']<6))
        banks[str(side)]=_fit_bank(records)
        print(f'Fitted train-only link bank for {side}x{side}',flush=True)
    result=dict(schema='hyra3-training-only-regional-links-v1',banks=banks,regions_per_axis=4,
        fit_families=sorted({c['family']['family_id'] for c in cases if c['family']['index_within_split']>=6}),
        held_out_TRAIN_families=sorted({c['family']['family_id'] for c in cases if c['family']['index_within_split']<6}),
        validation_or_test_used=False,top_k=8,P_min_improving_held_out_train_families=4,
        lag_seconds=[5*i for i in range(13)],selection_target='15s future residual conditional on own trend global residual and nominal trend',
        public_graph_rule='training scores and lags, adjacent/upstream public terrain and initial independent nominal wetness',
        source_cases=[dict(path=str(p),sha256=sha256(p)) for p in paths if load_protocol(p)['family']['split']=='train'])
    write_json(output,result)
    return result


def build_graph(bank, public_panel, side):
    learned=bank['banks'][str(side)]
    train=np.asarray(learned['train_gain'])
    held=np.asarray(learned['held_out_train_gain'])
    improved=np.asarray(learned['held_out_train_improving_families'])
    lags=np.asarray(learned['lag_steps'],int)
    region_ids=regions_for_positions(public_panel['positions_m'],side)
    bed=np.asarray([public_panel['elevation_m'][region_ids==r].mean() for r in range(16)])
    initial=np.asarray([public_panel['nominal_sample_m'][0,region_ids==r].mean() for r in range(16)])
    rows,cols=np.divmod(np.arange(16),4)
    distance=np.maximum(np.abs(rows[:,None]-rows[None]),np.abs(cols[:,None]-cols[None]))
    physical=(distance<=2)&(cols[None]<=cols[:,None])&(bed[None]>=bed[:,None]-.02)&(initial[None]>0)&(initial[:,None]>0)
    np.fill_diagonal(physical,False)
    e_mask,p_mask=np.zeros((16,16),bool),np.zeros((16,16),bool)
    for target in range(16):
        candidates=[s for s in range(16) if s!=target and train[target,s]>0]
        candidates.sort(key=lambda s:(-train[target,s],s))
        e_mask[target,candidates[:8]]=True
        candidates=[s for s in range(16) if physical[target,s] and held[target,s]>0 and improved[target,s]>=4]
        candidates.sort(key=lambda s:(-held[target,s],s))
        p_mask[target,candidates[:8]]=True
    return dict(E_mask=e_mask,E_lag_steps=lags,E_score=np.clip(train,-1,1).astype(np.float32),
                P_mask=p_mask,P_lag_steps=lags,P_score=np.clip(held,-1,1).astype(np.float32))
