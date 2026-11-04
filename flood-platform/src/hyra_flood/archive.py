"""Read-only cross-check of saved Merewether data; no solver imports."""
from pathlib import Path
import hashlib
import json
import numpy as np


def verify_saved_run(folder):
    folder = Path(folder)
    result = json.loads((folder / 'result.json').read_text(encoding='utf8'))
    series = json.loads((folder / 'series.json').read_text(encoding='utf8'))
    assert result['complete'] and result['duration_s'] == 800
    assert len(series) == 1600 and len(result['points']) == 40
    times = np.asarray([r['time_s'] for r in series])
    np.testing.assert_allclose(times, np.arange(.5, 800.5, .5), rtol=0, atol=1e-7)
    speeds = np.asarray([r['speeds_m_s'] for r in series])
    depths = np.asarray([r['depths_m'] for r in series])
    stages = np.asarray([r['stages_m'] for r in series])
    assert speeds.shape == depths.shape == (1600, 40)
    assert stages.shape == (1600, 4)
    assert np.isfinite(speeds).all() and np.isfinite(depths).all() and np.isfinite(stages).all()
    assert (speeds >= 0).all() and (depths >= 0).all()
    np.testing.assert_array_equal(speeds[depths < .01], 0.)
    maxima = speeds.max(axis=0)
    max_depths = depths.max(axis=0)
    ids = [p['id'] for p in result['points']]
    assert len(set(ids)) == 40
    for i, p in enumerate(result['points']):
        np.testing.assert_allclose(p['model_max_speed_m_s'], maxima[i], rtol=0, atol=1e-12)
        np.testing.assert_allclose(p['max_depth_m'], max_depths[i], rtol=0, atol=1e-12)
        np.testing.assert_allclose(p['speed_error_m_s'], maxima[i] - p['speed_m_s'], rtol=0, atol=1e-12)
        first_time = times[speeds[:, i].argmax()] if maxima[i] > 0 else 0.
        np.testing.assert_allclose(p['time_of_model_max_s'], first_time, rtol=0, atol=1e-7)
        u, v = p['model_u_at_max_m_s'], p['model_v_at_max_m_s']
        np.testing.assert_allclose(np.hypot(u, v), maxima[i], rtol=0, atol=1e-12)
        if maxima[i] >= .05 and p['speed_m_s'] >= .05:
            delta = (float(np.degrees(np.arctan2(v, u))) % 360 - p['angle_xy_deg'] + 180) % 360 - 180
            np.testing.assert_allclose(p['direction_error_deg'], delta, rtol=0, atol=1e-10)
        else:
            assert p['direction_error_deg'] is None
    with np.load(folder / 'final.npz', allow_pickle=False) as final:
        for name in ['depth', 'mx', 'mz', 'bed', 'n', 'maximum_depth']:
            assert list(final[name].shape) == result['shape'], name
            assert np.isfinite(final[name]).all(), name
        assert (final['depth'] >= 0).all()
        assert (final['maximum_depth'] >= final['depth'] - 1e-12).all()
        final_volume = float(final['depth'].sum() * result['dx_m']**2)
    return {'status': 'passed', 'samples': len(series), 'points': len(ids),
            'final_volume_from_array_m3': final_volume,
            'sha256': {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                       for name in ['result.json', 'series.json', 'final.npz']},
            'limits': ['Only stored samples and final fields checked; no certification of every internal substep.',
                       'Boundary/source flux ledger is not independently reconstructed here.',
                       'No new experimental-accuracy conclusion.']}



# Imported from round62 raw-series verifier; scoring is now a platform service.
def score_merewether(folder,source_directory):
    import csv
    from .contracts import read_json,sha
    from .evaluation import strict_speed_score
    folder=Path(folder);source=Path(source_directory)
    audit=verify_saved_run(folder);cfg=read_json(folder/'config.json');r=read_json(folder/'result.json')
    for name,h in cfg['source']['files'].items():
        if sha(source/name)!=h:raise ValueError('Original source file changed')
    with (source/'Location_of_velocity_sampling_points.csv').open(encoding='utf-8-sig',newline='') as stream:original=list(csv.DictReader(stream))
    if len(original)!=40:raise ValueError('All40 original points required')
    for row,p in zip(original,r['points']):
        for key,col in [('id','ID'),('x_m','x'),('y_m','y'),('speed_m_s','Speed_m_s'),('angle_xy_deg','Mod_Angle')]:
            value=row[col] if key=='id' else float(row[col])
            if p[key]!=value:raise ValueError('Observation IDs/order/values changed')
    series=read_json(folder/'series.json');pred=np.max([row['speeds_m_s'] for row in series],axis=0)
    result=strict_speed_score([p['id'] for p in r['points']],[p['speed_m_s'] for p in r['points']],pred)
    result.update(evidence='laboratory measurements rescaled to prototype; not field truth',source=cfg['source'],
        raw_audit=audit,duration_s=800.,points_below_sampling_depth=[p['id'] for p in r['points'] if p['max_depth_m']<.01],
        spatial_extent_acceptance='unavailable: these40 points cannot quantify dry-area false positives')
    return result
