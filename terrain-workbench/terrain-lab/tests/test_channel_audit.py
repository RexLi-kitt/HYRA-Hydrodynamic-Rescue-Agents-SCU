import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hydraulic_structures import Structures
from channel_audit import audit_channels

def buildings(gap=.5,shift=0):
    center=8+shift
    return [dict(id='lower',kind='building',x_m=7.5,z_m=0,width_m=1,length_m=center-gap/2),
            dict(id='upper',kind='building',x_m=7.5,z_m=center+gap/2,width_m=1,length_m=16-center-gap/2)]

class ChannelAuditTests(unittest.TestCase):
    def test_production_gaps(self):
        for gap in (.5,1):
            for shift in (0,.125,.25,.375):
                for dx in (.5,.25,.125):
                    with self.subTest(gap=gap,shift=shift,dx=dx):
                        size=round(16/dx);s=Structures(buildings(gap,shift),np.zeros((size,size)),dx)
                        before=s.buildings.copy();a=audit_channels(s)
                        self.assertEqual(len(a['channels']),1)
                        c=a['channels'][0]
                        # Independent aperture measurement from the production raster.
                        cross=s.buildings[:,round(7.5/dx):round(8.5/dx)]
                        aperture=np.count_nonzero(~cross.any(axis=1))*dx
                        self.assertAlmostEqual(c['grid_min_width_m'],aperture)
                        self.assertEqual(c['connected'],aperture>0)
                        self.assertAlmostEqual(c['physical_width_m'],gap)
                        np.testing.assert_array_equal(before,s.buildings)
    def test_axes(self):
        items=buildings(1,.375)
        a=audit_channels(Structures(items,np.zeros((32,32)),.5))['channels'][0]
        for b in items:
            b['x_m'],b['z_m']=b['z_m'],b['x_m']
            b['width_m'],b['length_m']=b['length_m'],b['width_m']
        b=audit_channels(Structures(items,np.zeros((32,32)),.5))['channels'][0]
        self.assertEqual(a['grid_min_width_m'],b['grid_min_width_m'])
        self.assertEqual(a['status'],b['status'])
        self.assertNotEqual(a['flow_axis'],b['flow_axis'])
    def test_obstructed_geometry(self):
        items=buildings(1)+[dict(id='middle',kind='building',x_m=7.5,z_m=7.8,width_m=1,length_m=.4)]
        a=audit_channels(Structures(items,np.zeros((32,32)),.5))
        self.assertEqual(a['excluded_obstructed'],1)
        self.assertFalse(any(c['building_ids']==['lower','upper'] for c in a['channels']))
    def test_budget_not_safe(self):
        s=Structures(buildings(1),np.zeros((32,32)),.5)
        a=audit_channels(s,cell_budget=0)
        self.assertFalse(a['complete']);self.assertEqual(a['channels'][0]['status'],'not_evaluated')
        a=audit_channels(s,max_candidates=0)
        self.assertFalse(a['complete']);self.assertEqual(a['channels'],[])
    def test_snapshot_integration(self):
        from flood_solver import FloodSolver
        from multilayer import MapWorld
        import flood_run,json
        config=flood_run.settings(dict(baseflow=False,warmup_s=0,inflow_m3_s=0), (32,32))
        solver=FloodSolver(np.zeros((32,32)),np.zeros((32,32)),.5)
        world=MapWorld(solver,dict(config=config,coarsening_factor=1),structures=buildings())
        snapshot=world.snapshot()
        audit=json.loads(json.dumps(snapshot))['M_structures']['channel_audit']
        self.assertEqual(audit['channels'][0]['status'],'closed')
        self.assertEqual(world.snapshot(compact=True)['M_structures']['channel_audit'],audit)

    def test_no_buildings(self):
        a=audit_channels(Structures([],np.zeros((4,4)),1))
        self.assertTrue(a['complete']);self.assertEqual(a['channels'],[])
    def test_rasterized_third_building(self):
        # A third building outside the physical corridor can intrude after rounding.
        items=buildings(1)+[dict(id='outside',kind='building',x_m=8.51,z_m=7.5,width_m=.2,length_m=1)]
        a=audit_channels(Structures(items,np.zeros((32,32)),.5))
        self.assertEqual(a['channels'][0]['status'],'open')

if __name__=='__main__':unittest.main()
