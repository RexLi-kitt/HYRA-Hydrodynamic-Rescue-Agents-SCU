"""Common samples and paired family-level uncertainty."""
import numpy as np


def score_event(reference, diagnostic_outputs, evaluation_config):
    y, p = np.asarray(reference), np.asarray(diagnostic_outputs)
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Must score identical finite samples")
    positive = y >= evaluation_config["underestimation_threshold_m"]
    flagged = p >= evaluation_config["underestimation_threshold_m"]
    return dict(mae_m=float(np.abs(y-p).mean()), samples=int(y.size),
                severe_underestimate_recall=float(flagged[positive].mean()) if positive.any() else None,
                false_positive_rate=float(flagged[~positive].mean()) if (~positive).any() else None)


def compare_paired_events(event_scores, comparison_config):
    config = comparison_config
    by_family = {}
    for row in event_scores:
        by_family.setdefault(row["event_family_id"], {}).setdefault(row["method"], []).append(row[config["metric"]])
    result, rng = [], np.random.default_rng(config.get("seed",0))
    baselines=["B0_raw","B1_distance","B2_uniform"]
    if any(r["method"]=="B_local" for r in event_scores):baselines.insert(1,"B_local")
    for baseline in baselines:
        delta = [np.mean(v["B2_structure"])-np.mean(v[baseline]) for v in by_family.values()
                 if "B2_structure" in v and baseline in v]
        if len(delta) != len(by_family):
            raise ValueError("Incomplete paired families")
        d = np.asarray(delta)
        boot = rng.choice(d,(2000,len(d)),replace=True).mean(axis=1)
        result.append(dict(baseline=baseline, metric=config["metric"],
                           difference_b2_minus_baseline=float(d.mean()),
                           event_bootstrap_interval_95=np.quantile(boot,[.025,.975]).tolist(),
                           independent_families=len(d),
                           uncertainty="event variation after averaging seeds; pilot interval"))
        if event_scores and all("training_seed" in row for row in event_scores):
            seeds=sorted({row["training_seed"] for row in event_scores})
            family_names=sorted(by_family)
            cells=np.zeros((len(seeds),len(family_names)))
            for si,seed in enumerate(seeds):
                for fi,family in enumerate(family_names):
                    values={method:[r[config["metric"]] for r in event_scores if r["training_seed"]==seed
                        and r["event_family_id"]==family and r["method"]==method] for method in ("B2_structure",baseline)}
                    if not all(values.values()):
                        raise ValueError("Missing seed/family pair")
                    cells[si,fi]=np.mean(values["B2_structure"])-np.mean(values[baseline])
            crossed=[]
            for _ in range(2000):
                seed_index=rng.integers(0,len(seeds),len(seeds))
                family_index=rng.integers(0,len(family_names),len(family_names))
                crossed.append(cells[np.ix_(seed_index,family_index)].mean())
            result[-1]["event_and_training_seed_interval_95"]=np.quantile(crossed,[.025,.975]).tolist()
            result[-1]["training_seeds"]=len(seeds)
            result[-1]["per_seed_paired_difference"]={str(seed):float(cells[i].mean()) for i,seed in enumerate(seeds)}
    return result
