"""Observation contracts shared by ingestion, replay API and checks (stdlib only)."""
from datetime import datetime, timezone, timedelta
import math


def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamp requires an explicit timezone')
    return result.astimezone(timezone.utc)


def utc(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def normalize_usgs(features, site='USGS-08075000'):
    records, seen, locations = [], {}, {}
    for feature in features:
        p = feature['properties']
        if p['monitoring_location_id'] != site or p['parameter_code'] != '00065':
            raise ValueError('Unexpected site or variable')
        if p['unit_of_measure'] != 'ft':
            raise ValueError('Unexpected source unit')
        when = utc(instant(p['time']))
        key = (p['time_series_id'], when)
        if key in seen:
            if seen[key] != p['value']:
                raise ValueError('Conflicting duplicate observation')
            continue
        seen[key] = p['value']
        # Preserve missing records without turning them into a zero-valued reading.
        try:
            value = float(p['value'])
        except (ValueError, TypeError):
            value = None
        valid = value is not None and math.isfinite(value)
        qualifier = p.get('qualifier')
        records.append(dict(id=f'{site}:{p["time_series_id"]}:{when}', station_id=site,
            series_id=p['time_series_id'], variable='gauge_height', unit='m',
            value=value * .3048 if valid else None, valid=valid,
            source_value=p['value'], source_unit='ft', observed_start_utc=when,
            observed_end_utc=when, available_at_utc=None, availability_basis='unknown',
            time_kind='instant', support_type='point', origin_kind='station_record',
            approval_status=p.get('approval_status'), qualifier=qualifier,
            source_last_modified=p.get('last_modified')))
        locations[site] = feature['geometry']['coordinates']
    if not records:
        raise ValueError('No records returned; refusing an empty event')
    if len({r['series_id'] for r in records}) != 1:
        raise ValueError('Multiple sensor series need an explicit selection')
    records.sort(key=lambda r: r['observed_end_utc'])
    return records, locations


def replay(records, at, *, assumed_latency_s=60, max_age_s=1800):
    """Causal release; no interpolation, no assumption that last_modified is delivery."""
    now = instant(at)
    if not math.isfinite(assumed_latency_s) or not 0 <= assumed_latency_s <= 86400:
        raise ValueError('Invalid assumed latency')
    visible = []
    for record in records:
        if record['time_kind'] != 'instant':
            continue  # Peak/max products are held out from online replay.
        observed = instant(record['observed_end_utc'])
        known = record.get('available_at_utc')
        available = instant(known) if known else observed + timedelta(seconds=assumed_latency_s)
        if observed > now or available > now:
            continue
        visible.append(dict(record, effective_available_at_utc=utc(available),
            effective_availability_basis='reported' if known else 'assumed',
            age_s=(now - observed).total_seconds()))
    visible.sort(key=lambda r: (r['observed_end_utc'], r['id']))
    latest = {}
    for record in visible:
        # An explicit missing record interrupts an earlier valid reading.
        latest[record['station_id']] = dict(record,
            stale=not record['valid'] or record['age_s'] > max_age_s)
    return dict(at_utc=utc(now), records=visible, latest=list(latest.values()),
        policy=dict(assumed_latency_s=assumed_latency_s, max_age_s=max_age_s,
            archived_revision=True, note='历史归档版；送达延迟是假设，不能重现当年的初报修订过程'))


def grid_plan(width_m, cell_m):
    if not math.isfinite(width_m) or not math.isfinite(cell_m) or not 32 <= width_m <= 10000 or not .25 <= cell_m <= 100:
        raise ValueError('Invalid grid size')
    side = math.ceil(width_m / cell_m)
    return dict(side=side, cells=side * side, actual_width_m=side * cell_m,
        four_float32_fields_mib=side * side * 16 / 1024 ** 2)


def cell_at(east, north, grid):
    """North-west-inclusive, south-east-exclusive; north-up rows [z,x]."""
    col = math.floor((east - grid['west_m']) / grid['cell_m'])
    row = math.floor((grid['north_m'] - north) / grid['cell_m'])
    if 0 <= row < grid['rows'] and 0 <= col < grid['cols']:
        return row, col
    return None
