"""Separate scalar audit of the finite-configuration offline action opportunity counts."""
import argparse
from pathlib import Path
import sys
import numpy as np

from audit_pulse_service_alignment import read, digest, load, distances, score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.artifacts import write_json


def visible(times, base, pulse, now, phase, delay, gained):
    if not gained: return False
    valid = np.arange(20+5*phase, 300+1e-9, 5)
    indices = []
    for b in range(6):
        accepted = [j for j, t in enumerate(valid) if t+delay <= now and b*10 <= now-t < (b+1)*10]
        if accepted: indices.append(accepted[-1])
    for source in (10, 20, 30, 60, 70, 80):
        if not any((source//10 < 5) == (target//10 < 5) for target in gained): continue
        for index in indices:
            change = np.interp(valid[index], times, pulse[:, source]-base[:, source])
            if abs(change) >= .001: return True
    return False


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    config = read(run/'config.json'); parent = Path(config['parent_run'])
    geometry = read(Path(config['source_family_parent'])/'geometry.json')['layouts']
    layouts = {}
    for layout in geometry:
        a, b = layout['targets']; origin = layout['origin']; blocked = set(layout['blocked_regions'])
        ds = distances(origin, blocked)
        layouts[layout['layout_index']] = [([a, b], [ds[a], distances(a, blocked)[b]]),
                                          ([b, a], [ds[b], distances(b, blocked)[a]])]
    registry = read(run/'field_registry.json'); cache = {}; total = 0; phase0 = 0; phase10 = 0; evaluations = 0
    def field(family, kind, anchor=0, center=0):
        key = f'{family}_n80_dt2_{kind}_a{anchor}_y{center}'
        if key not in cache:
            info = registry[key]; assert digest(info['projected_path']) == info['projected_sha256']
            cache[key] = load(info['projected_path'])
        return cache[key]
    for family in config['design']['families']:
        previous = read(parent/(family+'.json'))['robust_cases']
        current = read(run/(family+'.json'))['opportunities']
        assert len(previous) == len(current)
        base = field(family, 'truth_base')
        for prev, row in zip(previous, current):
            anchor, center, now = row['anchor_s'], row['center_y_m'], row['now_s']
            assert prev['anchor_s'] == anchor and prev['physical_center_y_m'] == center
            assert prev['layout'] == row['layout'] and prev['now_s'] == now
            pulse = field(family, 'truth_pulse', anchor, center)
            baseline = prev['frozen_order']; actions = layouts[row['layout']]
            survives = prev['all_variants_improve']
            flags = [np.array(prev['all_variants_visible_D0_by_phase']),
                     np.array(prev['all_variants_visible_D10_by_phase'])]
            for method in ('linear', 'nearest'):
                scores = [score(pulse['time_s'], pulse[method], order, ds, now, anchor+60) for order, ds in actions]
                evaluations += 2
                gap = scores[1-baseline][0]-scores[baseline][0]
                assert gap == row[f'n80_{method}_alternative_minus_baseline']
                survives = survives and gap > 0
                gained = set(s for s in scores[1-baseline][3] if s is not None)-set(s for s in scores[baseline][3] if s is not None)
                for delay_index, delay in enumerate((0, 10)):
                    for j, phase in enumerate((0, .25, .5, .75)):
                        flags[delay_index][j] &= gap > 0 and visible(pulse['time_s'], base[method], pulse[method], now, phase, delay, gained)
            assert bool(survives) == row['expanded_improves']
            for i, delay in enumerate((0, 10)):
                np.testing.assert_array_equal(flags[i] & survives, row[f'expanded_visible_D{delay}_by_phase'])
            total += survives; phase0 += int(flags[0].sum()); phase10 += int(flags[1].sum())
    summary = read(run/'summary.json')
    assert total == summary['expanded_robust_opportunities']
    assert phase0 == summary['expanded_visible_D0'] and phase10 == summary['expanded_visible_D10']
    write_json(run/'opportunity_audit.json', dict(passed=True, scalar_order_evaluations=evaluations,
        expanded_robust_opportunities=total, expanded_visible_D0=phase0, expanded_visible_D10=phase10,
        interpretation='Offline paired-reference necessary opportunity, not an online detector or policy efficacy result'))
    print('OPPORTUNITY_AUDIT_PASSED', total, phase0, phase10)


if __name__ == '__main__': main()
