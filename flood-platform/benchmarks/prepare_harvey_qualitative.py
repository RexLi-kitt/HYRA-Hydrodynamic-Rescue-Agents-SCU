"""Fixed-window observational labels for Harvey; never invent a model comparison."""
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import argparse
import hashlib
import json
from collections import Counter


def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def validate_archive(rows, official):
    if len(rows) != 769 or len(official['features']) != 769:
        raise ValueError('Incomplete original observation window')
    props = {stamp(f['properties']['time']): f['properties'] for f in official['features']}
    if len(props) != 769:
        raise ValueError('Duplicate official observation time')
    start = stamp('2017-08-25T00:00:00Z')
    values = []
    for index, row in enumerate(rows):
        t = stamp(row['observed_start_utc'])
        if t != start + timedelta(minutes=15*index) or t != stamp(row['observed_end_utc']):
            raise ValueError('Wrong cadence, ordering or time support')
        p = props[t]
        if row['unit'] != 'm' or row['source_unit'] != 'ft' or p['unit_of_measure'] != 'ft':
            raise ValueError('Wrong gauge-height units')
        if not row['valid'] or row['variable'] != 'gauge_height' or row['station_id'] != 'USGS-08075000':
            raise ValueError('Wrong station, variable or validity')
        if p['monitoring_location_id'] != row['station_id'] or p['parameter_code'] != '00065' or p['time_series_id'] != row['series_id']:
            raise ValueError('Official record identity mismatch')
        if p['approval_status'] != row['approval_status'] or p['qualifier'] != row['qualifier']:
            raise ValueError('Official quality fields changed')
        ft = Decimal(row['source_value'])
        if not ft.is_finite() or ft != Decimal(p['value']):
            raise ValueError('Official/archive value mismatch')
        if (ft*100) != (ft*100).to_integral_value():
            raise ValueError('Unexpected source granularity')
        if abs(float(ft*Decimal('0.3048'))-row['value']) > 1e-10:
            raise ValueError('Metre conversion inconsistent')
        values.append(ft)
    return values


def net_label(delta, rounding_band):
    if delta > rounding_band:
        return '净上涨'
    if delta < -rounding_band:
        return '净退水'
    return '记录分辨率下无法判定净变化'


def prepare(archive_path, official_path, protocol_path, output):
    archive_path, official_path, protocol_path, output = map(Path, (archive_path, official_path, protocol_path, output))
    if output.exists():
        raise FileExistsError('Preserve previous labels')
    rows = json.loads(archive_path.read_text(encoding='utf8'))
    official = json.loads(official_path.read_text(encoding='utf8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf8'))
    actual_hash = hashlib.sha256(official_path.read_bytes()).hexdigest()
    if actual_hash != protocol['raw_sha256']:
        raise ValueError('Official input differs from frozen protocol')
    values = validate_archive(rows, official)
    records = []
    for lo in range(0, 768, 24):
        hi = lo+24
        v = values[lo:hi+1]
        delta = v[-1]-v[0]
        diffs = [b-a for a,b in zip(v[:-1], v[1:])]
        positive = sum(d > 0 for d in diffs)
        negative = sum(d < 0 for d in diffs)
        record = {
            'object_id':'USGS-08075000', 'criterion_id':'Q2-net-change-6h',
            'observed_start_utc':rows[lo]['observed_start_utc'],
            'observed_end_utc':rows[hi]['observed_end_utc'],
            'observation_label':net_label(delta, Decimal('.01')),
            'sensitivity_label_002ft':net_label(delta, Decimal('.02')),
            'source_row_indices_inclusive':[lo,hi], 'sample_count':25,
            'source_record_ids':[rows[lo]['id'],rows[hi]['id']],
            'start_relative_gauge_height_m':rows[lo]['value'],
            'end_relative_gauge_height_m':rows[hi]['value'],
            'endpoint_change_ft':str(delta),
            'within_window_recorded_up_steps':positive,
            'within_window_recorded_down_steps':negative,
            'within_window_recorded_equal_steps':sum(d == 0 for d in diffs),
            'within_window_both_directions':positive > 0 and negative > 0,
            'simulation_label':None, 'simulation_evidence_locator':None,
            'conclusion':'无法判断', 'task_impact':'尚无同域模拟；不可推断积水深度、到达或搜索排序',
        }
        records.append(record)
    peak = max(values)
    peak_times = [row['observed_start_utc'] for row,v in zip(rows,values) if v==peak]
    result = {
        'case':'Harvey2017 Brays Bayou USGS08075000',
        'evidence':'field gauge archive; current approved historical revision',
        'scope':'observation-only labels, NOT HYRA validation success',
        'raw_source_hash':actual_hash,
        'normalized_archive_hash':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        'source_recheck_all769_match':True, 'interval_s':900,
        'classification_rule':'Fixed UTC6h endpoint difference; .01ft combined endpoint rounding band, not instrument uncertainty',
        'label_counts':dict(Counter(r['observation_label'] for r in records)),
        'rounding_band_sensitivity_changed_windows':sum(r['observation_label']!=r['sensitivity_label_002ft'] for r in records),
        'windows_with_recorded_internal_reversals':sum(r['within_window_both_directions'] for r in records),
        'peak_relative_gauge_height_ft':str(peak),'peak_record_times_utc':peak_times,
        'comparison_counts':{'一致':0,'部分一致':0,'不一致':0,'无法判断':len(records)},
        'data_gaps':['no corresponding complete HYRA event simulation','2017 gauge vertical datum and historical channel terrain unresolved','no matched independent flood extent/dry-area observations','no observed ground-depth, velocity, road condition or person distribution'],
        'records':records,
    }
    output.mkdir()
    (output/'labels.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    table = ['# Harvey真实测站定性过程标签', '', '证据：现场测站归档；水位相对水尺，不是地面水深。仅准备观测标签，尚无对应HYRA模拟。固定UTC六小时窗口，段内反向变化另列；不代表单调上涨或退水。', '', '|UTC起点|UTC终点|观测净变化|段内有正反变化|模拟|对照|', '|---|---|---|---|---|---|']
    for row in records:
        table.append('|'+ '|'.join([row['observed_start_utc'],row['observed_end_utc'],row['observation_label'],'是' if row['within_window_both_directions'] else '否','未建立','无法判断'])+'|')
    (output/'observation-labels.md').write_text('\n'.join(table)+'\n',encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=[stamp(row['observed_start_utc']) for row in rows]
    fig, ax = plt.subplots(figsize=(12,4),layout='constrained')
    ax.plot(t,[row['value'] for row in rows],color='#185f9d',lw=1.3)
    for k in range(0,769,24):
        ax.axvline(t[k],color='grey',lw=.4,alpha=.25)
    ax.set(title='Harvey2017: observed gauge stage, not ground water depth or a simulation',ylabel='Gauge-relative height (m)',xlabel='UTC (all769 original15-minute records)')
    ax.grid(alpha=.15)
    fig.savefig(output/'observed-process.png',dpi=140)
    plt.close(fig)
    print(json.dumps({k:result[k] for k in ('label_counts','rounding_band_sensitivity_changed_windows','windows_with_recorded_internal_reversals','peak_relative_gauge_height_ft','peak_record_times_utc','comparison_counts')},ensure_ascii=False))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('archive');p.add_argument('official');p.add_argument('protocol');p.add_argument('output')
    a=p.parse_args()
    prepare(a.archive,a.official,a.protocol,a.output)
