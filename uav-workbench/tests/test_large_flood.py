import json
import base64
import math
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from large_flood.input import FloodInput
from large_flood.simulate import advance_waypoints, run, segments_close


def fixture(path, missing=False, bad_time=False):
    frames=[]
    with zipfile.ZipFile(path,'w') as z:
        for i in range(5):
            name=f'{i}.bin'; frame=np.zeros((4,20,20),dtype='<f4')
            if i>=2: frame[0, :, :]=.1; frame[3,:,:]=.1
            if not (missing and i==2): z.writestr(name,frame.tobytes())
            frames.append(dict(index=i,time_s=(i*5 + (1 if bad_time and i==2 else 0)),url=name))
        p=dict(grid=dict(shape=[20,20],dx_m=5.,axis_order=['z','x']),frame_fields=['depth','u','v','maximum_depth'],frame_dtype='<f4',frames=frames,units=dict(time='s',depth='m',velocity='m/s',distance='m'))
        z.writestr('process.json',json.dumps(p))


class LargeFloodTests(unittest.TestCase):
    def test_speed_and_waypoint_distance(self):
        straight,_,dist=advance_waypoints((0,0),[(100,0)],0,5)
        diagonal,_,dist2=advance_waypoints((0,0),[(100,100)],0,5)
        self.assertAlmostEqual(np.linalg.norm(straight),5)
        self.assertAlmostEqual(np.linalg.norm(diagonal),5)
        self.assertEqual((dist,dist2),(5,5))
        end,_,travelled=advance_waypoints((0,0),[(2,0),(2,10)],0,5)
        np.testing.assert_allclose(end,[2,3]); self.assertEqual(travelled,5)

    def test_crossing_and_separate_movements(self):
        self.assertTrue(segments_close((0,0),(10,0),(10,0),(0,0),1))
        self.assertFalse(segments_close((0,0),(10,0),(0,6),(10,6),5))

    def test_actual_frames_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'f.zip';fixture(path)
            f=FloodInput(path,shape=(20,20),duration=20)
            self.assertEqual(f.frame(9)[1],5); self.assertEqual(f.frame(10)[1],10)
            with self.assertRaises(ValueError): f.frame(21)
            with self.assertRaises(ValueError): FloodInput(path,shape=(20,20),duration=25)
            with self.assertRaises(ValueError): FloodInput(path,shape=(21,20),duration=20)
            broken=Path(d)/'broken.zip';fixture(broken,missing=True)
            with self.assertRaises(KeyError): FloodInput(broken,shape=(20,20),duration=20)
            bad=Path(d)/'bad.zip';fixture(bad,bad_time=True)
            with self.assertRaises(ValueError): FloodInput(bad,shape=(20,20),duration=20)

    def test_run_records_causal_along_path(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'f.zip';fixture(path)
            result=run(path,4,8,25,duration=20,shape=(20,20))
            self.assertGreater(len(result['observations']),4*5)
            self.assertTrue(all(o['frame_time']<=o['t']+1e-6 for o in result['observations']))
            self.assertTrue(all(0<=x<=100 and 0<=z<=100 for row in result['trajectories'] for x,z in row['positions']))
            self.assertEqual(result['summary']['collision_stops'],0)
            self.assertGreater(result['summary']['coverage'],0)
            self.assertTrue(all('wet_bits' in o for o in result['observations']))
            late=next(o for o in result['observations'] if o['frame_time'] == 10.)
            depth=np.frombuffer(base64.b64decode(late['depth_f32_le']),dtype='<f4')
            self.assertTrue(np.allclose(depth,.1))

if __name__=='__main__': unittest.main()
