from pathlib import Path
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'generated' / 'signal-routes-20261027-v2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources():
    paths = list((ROOT / 'signal_routes').glob('*.py'))
    paths += [ROOT / name for name in ('scale100_env.py', 'scale100_train.py',
              'scale100_aux_resilient.py', 'scale100_rules.py', 'scale100_development.py')]
    paths += list((ROOT / 'sur_rescue').glob('*.py'))
    paths += list((ROOT / 'hyra_learning').glob('*.py'))
    paths += [ROOT / 'hyra_learning/vendor/pymarl_qmix.py']
    paths += [ROOT.parent / 'terrain-workbench/terrain-lab/flood_solver.py']
    return {str(p.relative_to(ROOT.parent)): digest(p) for p in sorted(set(paths))}


def registration():
    return dict(schema='hyra-signal-routes-v2', date='2026-10-27',
        status='registered_not_results', tasks=dict(primary='E2', regression='E1_zero_source'),
        methods=['Braw', 'Bpost', 'R1', 'R2', 'R3', 'C1', 'C2', 'C3'],
        fixed=dict(shape=[20, 20], dx_m=5, n_agents=2, targets=5, step_s=5,
                   service_s=10, service_depth_limit_m=.08, service_speed_limit_m_s=.3,
                   observation_channels=21, actions=8, auxiliary_data_weight=0, auxiliary_physics_weight=0),
        split=dict(fit=list(range(7101, 7107)), development=list(range(7201, 7204)),
                   holdout=list(range(7301, 7309)), unit='parent terrain',
                   fit_mechanisms=['slow', 'rapid'], holdout_mechanisms=['slow', 'rapid', 'double']),
        hydrographs=dict(slow=[[0, 0], [300, 1], [600, .4], [900, 0]],
                        rapid=[[0, 0], [50, 1], [150, .4], [300, 0], [900, 0]],
                        double=[[0, 0], [50, 1], [150, 0], [450, .7], [600, 0], [900, 0]]),
        inference=dict(coupling=.35, exact_observations=True,
                       features=['depth_m','speed_m_s','wet_observed_surface_slope_m_m','record_age_s'],
                       feature_normalization='fit-only mean/std per available feature; clip standardized values to [-5,5]',
                       surface_elevation='public_bed_m + observed_depth_m',
                       covariance_diagnostics_only=True, hidden_targets_allowed=False,
                       water_permutation='joint h/u/v/age within arrived cells; same rule train/eval'),
        selection=dict(durations_s=[900, 600, 450], rule_best_rescue_range=[.2, .9],
                       require_service_interruptions=True, fallback_duration_s=900,
                       checkpoint_order=['rescue_fraction', 'first_coverage_auc',
                                         'current_correct_coverage_auc', 'earlier_checkpoint']),
        pilot=dict(seeds=[11], episodes=128),
        formal=dict(seeds=[11, 29, 47], episodes=1024, updates=1021, checkpoints=[256, 512, 1024], jobs=24),
        evaluation=dict(scenario_seeds=[11, 29, 47], primary='rescue_fraction',
                        independent_parents=8, cluster_bootstrap=10000,
                        primary_comparisons=['R1-Bpost', 'R2-Bpost', 'R3-Bpost'], holm_alpha=.05),
        interpretation=['static personnel; no advection claim',
                        'initial-depth-conditioned target placement may convey a location prior',
                        'synthetic diagnostics do not establish a real sensor mechanism',
                        'SAR and quadratic CRF equivalence must be disclosed',
                        'budget does not establish convergence'])
