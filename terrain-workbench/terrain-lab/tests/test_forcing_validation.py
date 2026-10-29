import sys,unittest,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver

def manual():return dict(id='a',enabled=True,cells=[dict(z=1,x=1,weight=1.)],hydrograph=[[0,1],[1,0]])
def boundary():return dict(id='b',enabled=True,delivery='boundary_discharge_with_momentum',hydrograph=[[0,1],[1,0]],inlet_velocity_m_s=1.,boundary_faces=[dict(edge='west',offset=1,z=1,x=0,width_m=1.,weight=1.)])
class ExternalValidationTests(unittest.TestCase):
    def sim(self):return FloodSolver(np.zeros((4,4)),np.zeros((4,4)),1,boundary='closed')
    def test_solid_both_orders_and_no_partial_change(self):
        for maker in (manual,boundary):
            for first in (True,False):
                s=self.sim();e=maker();p=e.get('cells',e.get('boundary_faces'))[0]
                solid=np.zeros((4,4),bool);solid[p['z'],p['x']]=True
                if first:
                    s.set_environment(solid=solid)
                    with self.assertRaisesRegex(ValueError,'solid'):s.set_external_inflows([e])
                    self.assertEqual(s.external_schedule,[])
                else:
                    s.set_external_inflows([e])
                    with self.assertRaisesRegex(ValueError,'solid'):s.set_environment(solid=solid)
                    self.assertFalse(s.solid.any())
                    s.advance(1);self.assertAlmostEqual(s.h.sum(),1,places=10)
    def test_detached_forcing_and_retained_counters(self):
        s=self.sim();e=manual();s.set_external_inflows([e]);e['hydrograph'][0][1]=4;e['cells'][0]['weight']=3
        s.advance(.5);s.set_external_inflows([manual()]);s.advance(1)
        self.assertAlmostEqual(s.h.sum(),1,places=10)
        self.assertAlmostEqual(s.external_volume_by_id['a'],1,places=10)
        s.set_external_inflows([])
        self.assertAlmostEqual(s.external_volume_by_id['a'],1,places=10)
    def test_running_environment_rejected_preserves_budget_and_peak(self):
        s=self.sim();s.advance(1,rain=.01);before=copy.deepcopy(s.summary()['balance']);peak=s.maximum.copy()
        with self.assertRaisesRegex(ValueError,'new simulation'):s.set_environment(drain_capacity=np.zeros((4,4)))
        self.assertEqual(s.summary()['balance'],before);np.testing.assert_array_equal(s.maximum,peak)
    def test_invalid_manual_does_not_replace_old(self):
        invalid=[]
        for weight in (-1,0,float('nan'),.5):
            e=manual();e['cells'][0]['weight']=weight;invalid.append([e])
        e=manual();e['cells'][0]['x']=-1;invalid.append([e])
        e=manual();e['cells']*=2;invalid.append([e])
        e=manual();e['hydrograph']=[[0,1],[0,2]];invalid.append([e])
        invalid.append([manual(),manual()])
        for bad in invalid:
            s=self.sim();s.set_external_inflows([manual()]);old=copy.deepcopy(s.external_schedule)
            with self.assertRaises(ValueError):s.set_external_inflows(bad)
            self.assertEqual(s.external_schedule,old)
    def test_invalid_boundary_and_occupied_face(self):
        variants=[]
        for d in ([0,1],[-1,0],[2,0],[float('nan'),0]):
            e=boundary();e['boundary_faces'][0]['momentum_direction_xz']=d;variants.append([e])
        e=boundary();e['boundary_faces'][0]['offset']=2;variants.append([e])
        e=boundary();e['boundary_faces'][0]['width_m']=2;variants.append([e])
        e=boundary();e['id']='c';variants.append([boundary(),e])
        for entries in variants:
            with self.assertRaises(ValueError):self.sim().set_external_inflows(entries)
        for first in (True,False):
            s=self.sim();wall=np.zeros((4,5),bool);wall[1,0]=True
            if first:
                s.set_environment(wall_x=wall)
                with self.assertRaisesRegex(ValueError,'wall'):s.set_external_inflows([boundary()])
            else:
                s.set_external_inflows([boundary()])
                with self.assertRaisesRegex(ValueError,'wall'):s.set_environment(wall_x=wall)
                self.assertFalse(s.wall_x.any())
    def test_disabled_solid_source_and_weighted_source(self):
        s=self.sim();solid=np.zeros((4,4),bool);solid[1,1]=True;s.set_environment(solid=solid)
        e=manual();e['enabled']=False;s.set_external_inflows([e]);s.advance(1);self.assertEqual(s.h.sum(),0)
        s=self.sim();e=manual();e['cells']=[dict(z=1,x=1,weight=.3),dict(z=2,x=2,weight=.7)]
        s.set_external_inflows([e]);s.advance(1);self.assertAlmostEqual(s.h.sum(),1,places=10)

if __name__=='__main__':unittest.main()
