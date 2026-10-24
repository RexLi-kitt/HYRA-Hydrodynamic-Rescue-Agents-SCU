import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import cell_at, grid_plan, instant, normalize_usgs, replay
from server import Handler, load_event


def observation(when='2017-08-25T00:00:00Z', **extra):
    return dict(id=when, station_id='station', variable='gauge_height', observed_end_utc=when,
        available_at_utc=None, time_kind='instant', valid=True, value=1., **extra)


class TimeContract(unittest.TestCase):
    def test_delivery_gate_at_exact_boundary(self):
        records=[observation()]
        self.assertEqual(replay(records, '2017-08-25T00:00:59Z')['records'], [])
        self.assertEqual(len(replay(records, '2017-08-25T00:01:00Z')['records']), 1)

    def test_observation_gate_even_if_bad_early_delivery(self):
        r=observation('2017-08-25T00:15:00Z')
        r['available_at_utc']='2017-08-25T00:00:00Z'
        self.assertEqual(replay([r], '2017-08-25T00:10:00Z')['records'], [])

    def test_reported_delivery_overrides_assumption(self):
        r=observation();r['available_at_utc']='2017-08-25T01:00:00Z'
        self.assertEqual(replay([r], '2017-08-25T00:59:00Z', assumed_latency_s=0)['records'], [])
        self.assertEqual(replay([r], '2017-08-25T01:00:00Z')['records'][0]['effective_availability_basis'], 'reported')

    def test_last_modified_is_not_historical_delivery(self):
        r=observation(source_last_modified='2025-01-01T00:00:00Z')
        self.assertEqual(len(replay([r], '2017-08-25T00:01:00Z')['records']), 1)

    def test_timezone_requires_aware_time(self):
        self.assertEqual(instant('2017-08-25T08:00:00+08:00'), instant('2017-08-25T00:00:00Z'))
        with self.assertRaises(ValueError):instant('2017-08-25T00:00:00')

    def test_missing_record_interrupts_old_valid_reading(self):
        missing=observation('2017-08-25T00:15:00Z');missing.update(valid=False,value=None)
        packet=replay([observation(),missing], '2017-08-25T00:16:00Z')
        self.assertFalse(packet['latest'][0]['valid'])
        self.assertIsNone(packet['latest'][0]['value'])
        self.assertTrue(packet['latest'][0]['stale'])

    def test_stale_after_30_minutes(self):
        self.assertTrue(replay([observation()], '2017-08-25T00:31:00Z')['latest'][0]['stale'])

    def test_event_max_never_becomes_instantaneous_observation(self):
        r=observation();r['time_kind']='event_max'
        self.assertEqual(replay([r], '2017-09-01T00:00:00Z')['records'], [])


class SpatialContract(unittest.TestCase):
    def test_256_one_metre_grid(self):
        p=grid_plan(256,1)
        self.assertEqual(p['cells'],65536)
        self.assertEqual(p['four_float32_fields_mib'],1)
        self.assertEqual(grid_plan(512,1)['cells'],4*p['cells'])

    def test_north_up_half_open_boundaries(self):
        g=dict(west_m=1000,north_m=2000,cell_m=10,rows=2,cols=3)
        self.assertEqual(cell_at(1005,1995,g),(0,0))
        self.assertEqual(cell_at(1025,1985,g),(1,2))
        self.assertEqual(cell_at(1000,2000,g),(0,0))
        self.assertIsNone(cell_at(1030,1990,g))
        self.assertIsNone(cell_at(1000,1980,g))
        self.assertIsNone(cell_at(999,1990,g))

    def test_rejects_nonfinite_or_unbounded_grid(self):
        for width,cell in [(float('nan'),1),(256,0),(1000000,1)]:
            with self.assertRaises(ValueError):grid_plan(width,cell)


class RealArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.event,cls.records=load_event()

    def test_real_period_and_native_cadence(self):
        self.assertEqual(len(self.records),769)
        self.assertEqual(self.records[0]['observed_end_utc'],'2017-08-25T00:00:00Z')
        self.assertEqual(self.records[-1]['observed_end_utc'],'2017-09-02T00:00:00Z')
        for a,b in zip(self.records,self.records[1:]):
            self.assertEqual((instant(b['observed_end_utc'])-instant(a['observed_end_utc'])).total_seconds(),900)

    def test_ft_conversion_without_depth_claim(self):
        for r in self.records:
            self.assertEqual(r['variable'],'gauge_height')
            self.assertAlmostEqual(r['value'],float(r['source_value'])*.3048)
            self.assertIsNone(r['available_at_utc'])
        peak=max(self.records,key=lambda r:r['value'])
        self.assertEqual(peak['source_value'],'45.67')

    def test_causal_real_packet_and_no_synthetic_timestamps(self):
        p=replay(self.records,'2017-08-28T00:00:00Z')
        self.assertEqual(len(p['records']),288)
        self.assertEqual(p['latest'][0]['observed_end_utc'],'2017-08-27T23:45:00Z')
        self.assertTrue(all(instant(r['effective_available_at_utc'])<=instant(p['at_utc']) for r in p['records']))
        self.assertNotIn('depth',p)

    def test_portable_artifact_hashes_and_lengths(self):
        for name,audit in self.event['artifacts'].items():
            body=(ROOT/'data'/'harvey-2017'/name).read_bytes()
            self.assertEqual(len(body),audit['bytes'])
            self.assertEqual(hashlib.sha256(body).hexdigest(),audit['sha256'])
        self.assertEqual((ROOT/'data'/'harvey-2017'/'elevation.f32').stat().st_size,200*200*4)

    def test_terrain_is_explicitly_overview_only(self):
        t=self.event['terrain'];g=t['grid']
        self.assertFalse(t['event_compatible'])
        self.assertFalse(self.event['capabilities']['solver_ready'])
        self.assertEqual(g['crs'],'EPSG:32615')
        self.assertEqual(cell_at(g['station_east_m'],g['station_north_m'],g),tuple(t['station_cell']))

    def test_normalization_retains_missing_and_rejects_wrong_unit(self):
        # Source-shaped input for edge cases not present in this intact real archive.
        f=dict(properties=dict(time_series_id='sensor',monitoring_location_id='USGS-08075000',
            parameter_code='00065',unit_of_measure='ft',time='2017-08-25T00:00:00Z',value=None),
            geometry=dict(coordinates=[-95.,29.]))
        records,_=normalize_usgs([f]);self.assertIsNone(records[0]['value']);self.assertFalse(records[0]['valid'])
        f['properties']['unit_of_measure']='m'
        with self.assertRaises(ValueError):normalize_usgs([f])


class ReadOnlyAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.worker=threading.Thread(target=cls.http.serve_forever,daemon=True);cls.worker.start()
        cls.base=f'http://127.0.0.1:{cls.http.server_port}'

    @classmethod
    def tearDownClass(cls):cls.http.shutdown();cls.http.server_close();cls.worker.join()

    def test_public_assets_and_manifest(self):
        for path in ['/','/app.mjs','/shared/map-arrays.mjs','/shared/style.css','/api/event','/data/harvey-2017/elevation.f32']:
            with urlopen(self.base+path,timeout=5) as response:self.assertEqual(response.status,200)

    def test_runtime_code_and_full_archive_not_exposed(self):
        for path in ['/server.py','/.runtime/harvey-2017/gauge-00.json','/data/harvey-2017/observations.json','/shared/server.py','/data/../server.py']:
            with self.assertRaises(HTTPError) as error:urlopen(self.base+path,timeout=5)
            self.assertEqual(error.exception.code,404)

    def test_invalid_request_and_real_causal_response(self):
        with self.assertRaises(HTTPError) as error:urlopen(self.base+'/api/observations?at=bad',timeout=5)
        self.assertEqual(error.exception.code,400)
        with urlopen(self.base+'/api/observations?at=2017-08-25T00:00:00Z',timeout=5) as response:
            self.assertEqual(json.load(response)['records'],[])


if __name__=='__main__':unittest.main()
