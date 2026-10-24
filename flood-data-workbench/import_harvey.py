"""Reproducible bounded USGS acquisition. No flood solver or invented water fields."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from dataset import instant, normalize_usgs, utc, cell_at

ROOT = Path(__file__).resolve().parent
SITE = 'USGS-08075000'
START, END = '2017-08-25T00:00:00Z', '2017-09-02T00:00:00Z'
API = 'https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items'
ELEVATION = 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer'


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def download(url, path, *, offline=False, limit=12_000_000):
    if not path.exists() and not offline:
        with urlopen(Request(url, headers={'User-Agent': 'HYRA-flood-data-workbench/1.0'}), timeout=30) as stream:
            body = stream.read(limit + 1)
        if len(body) > limit:
            raise ValueError('Acquisition size limit exceeded')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    body = path.read_bytes()
    return body, dict(url=url, raw_file=str(path.relative_to(ROOT)).replace('\\', '/'),
        sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))


def terrain(folder, raw, location, offline):
    import numpy as np
    import rasterio
    from pyproj import Transformer
    # Reuse the existing parser; do not duplicate grid-origin/row-order logic.
    sys.path.insert(0, str(ROOT.parent / 'terrain-workbench' / 'terrain-lab'))
    from raster_import import read_ascii
    east, north = Transformer.from_crs(4326, 32615, always_xy=True).transform(*location)
    west, south = (int(east // 10) - 100) * 10, (int(north // 10) - 100) * 10
    query = dict(f='image', bbox=f'{west},{south},{west+2000},{south+2000}', bboxSR=32615,
        imageSR=32615, size='200,200', format='tiff', pixelType='F32',
        interpolation='RSP_NearestNeighbor', renderingRule=json.dumps({'rasterFunction': 'None'}))
    url = ELEVATION + '/exportImage?' + urlencode(query)
    _, provenance = download(url, raw / 'overview.tif', offline=offline)
    metadata, service_provenance = download(ELEVATION + '?f=pjson', raw / 'elevation-service.json', offline=offline)
    service = json.loads(metadata)
    with rasterio.open(raw / 'overview.tif') as source:
        if source.crs.to_epsg() != 32615 or source.width != 200 or source.height != 200:
            raise ValueError('Unexpected raster CRS or dimensions')
        expected = rasterio.transform.from_origin(west, south + 2000, 10, 10)
        if not source.transform.almost_equals(expected):
            raise ValueError('Unexpected exported raster transform')
        a = source.read(1, masked=True)
        values = a.filled(-9999).astype(float)
    # Exercise the same ASCII ingress used by the existing map importer.
    asc = raw / 'overview.asc'
    with asc.open('w', encoding='utf-8') as stream:
        stream.write(f'ncols 200\nnrows 200\nxllcorner {west}\nyllcorner {south}\ncellsize 10\nNODATA_value -9999\n')
        np.savetxt(stream, values, fmt='%.9g')
    bed, valid, meta = read_ascii(asc)
    valid &= np.isfinite(bed) & (np.abs(bed) < 10000)
    if valid.mean() < .9:
        raise ValueError('Terrain export has insufficient valid coverage')
    values = np.where(valid, bed, np.nan).astype('<f4')
    values.tofile(folder / 'elevation.f32')
    valid.astype('u1').tofile(folder / 'elevation-valid.u8')
    grid = dict(crs='EPSG:32615', west_m=meta['xllcorner'],
        north_m=meta['yllcorner'] + 2000, cell_m=10, rows=200, cols=200,
        axis_order='north-up; [row=z, column=x]', station_east_m=east, station_north_m=north)
    row, col = cell_at(east, north, grid)
    return dict(grid=grid, station_cell=[row, col], unit='m', source_kind='bare_earth_dem_service_export',
        source_resolution_m=None, export_spacing_m=10, vertical_datum='not_verified',
        acquisition_date='not_verified_for_2017', event_compatible=False,
        service_description=service.get('serviceDescription'),
        warning='现代3DEP地形的10米概览采样；采集年份与垂直基准未逐幅核验，不是2017年地形或1米计算地形。',
        provenance=[provenance, service_provenance],
        fields=dict(elevation=dict(url='/data/harvey-2017/elevation.f32', dtype='<f4', shape=[200,200]),
            valid=dict(url='/data/harvey-2017/elevation-valid.u8', dtype='|u1', shape=[200,200])),
        valid_cells=int(valid.sum()), min_m=float(bed[valid].min()), max_m=float(bed[valid].max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', help='Rebuild using the exact cached downloads')
    parser.add_argument('--terrain', action='store_true', help='Also fetch a 2 km overview, not a hydraulic grid')
    args = parser.parse_args()
    raw, folder = ROOT / '.runtime' / 'harvey-2017', ROOT / 'data' / 'harvey-2017'
    folder.mkdir(parents=True, exist_ok=True)
    url = API + '?' + urlencode(dict(f='json', monitoring_location_id=SITE, parameter_code='00065',
        datetime=f'{START}/{END}', limit=10000))
    features, provenance = [], []
    for page in range(10):
        body, audit = download(url, raw / f'gauge-{page:02}.json', offline=args.offline)
        data = json.loads(body)
        features.extend(data['features'])
        provenance.append(audit)
        next_urls = [link['href'] for link in data.get('links', []) if link.get('rel') == 'next']
        if not next_urls:
            break
        url = next_urls[0]
        if urlparse(url).hostname != 'api.waterdata.usgs.gov':
            raise ValueError('Unexpected pagination host')
    else:
        raise ValueError('Pagination limit exceeded')
    records, locations = normalize_usgs(features)
    if any(not instant(START) <= instant(r['observed_end_utc']) <= instant(END) for r in records):
        raise ValueError('Unexpected observation outside requested window')
    save_json(folder / 'observations.json', records)
    differences = Counter((instant(b['observed_end_utc']) - instant(a['observed_end_utc'])).total_seconds()
        for a, b in zip(records, records[1:]))
    event = dict(schema='hyra-observed-event-v1', id='harvey-2017', title='Harvey 2017 · Brays Bayou',
        subtitle='Houston，USGS 08075000；真实水位点记录与独立地形概览',
        start_utc=START, end_utc=END, display_timezone='America/Chicago',
        station=dict(id=SITE, name='Brays Bayou at Houston, TX', coordinates=locations[SITE], crs='EPSG:4326',
            variable='gauge_height', datum='gauge_relative_unresolved', unit='m'),
        counts=dict(total=len(records), valid=sum(r['valid'] for r in records)),
        interval_histogram_s={str(k): v for k, v in sorted(differences.items())},
        replay_policy=dict(assumed_latency_s=60, max_age_s=1800),
        provenance=provenance, ingested_at_utc=utc(datetime.now(timezone.utc)),
        limitations=['归档审定版数据，不等于当年实时初报。', '真实送达时间未知；默认60秒延迟仅是可调整的工程假设。',
            '相对水尺的水位不是地面水深；尚未核验2017年的水尺基准变更。',
            '一个测站不提供全域淹没范围、水深或流速；最大淹没图尚未匹配此工作区。'],
        capabilities=dict(gauge_series=True, terrain_overview=False, time_varying_extent=False,
            measured_depth=False, measured_velocity=False, solver_ready=False),
        sources=[dict(title='USGS 连续监测接口', url=API),
            dict(title='Harvey 官方事件资料', url='https://www.usgs.gov/centers/oklahoma-texas-water-science-center/science/post-hurricane-harvey-assessment')])
    if args.terrain:
        event['terrain'] = terrain(folder, raw, locations[SITE], args.offline)
        event['capabilities']['terrain_overview'] = True
    event['artifacts'] = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in [folder / 'observations.json'] + ([folder / 'elevation.f32', folder / 'elevation-valid.u8'] if args.terrain else [])}
    save_json(folder / 'manifest.json', event)
    print(json.dumps(dict(event=event['id'], counts=event['counts'], intervals=event['interval_histogram_s'],
        terrain=event['capabilities']['terrain_overview']), ensure_ascii=False))


if __name__ == '__main__':
    main()
