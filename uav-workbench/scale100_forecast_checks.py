"""Checks that physical field error cannot hide behind a global volume sum."""
import json,unittest
from pathlib import Path
from scale100_forecast_audit import *

class ForecastChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);row=next(r for r in json.loads(DATA.read_text())['cases'] if r['split']=='fit');cls.case=FloodCase(row['path']);cls.i=80
    def test_actual_solver_pair_has_zero_data_and_residual(self):
        c=self.case;i=self.i;pred=torch.tensor(np.stack((c.probes['before'][i],c.probes['after'][i])))[None]
        s=score(pred,c.probes['before'][i],c.probes['after'][i],c.bed,float(c.probes['dt'][i]),c.probes['source'][i],np.ones((20,20),bool))
        self.assertEqual(s['data_normalized_mse'],0.);self.assertLess(s['local_physics_normalized_mse'],1e-25);self.assertLess(s['volume_residual_abs_mean_m3'],1e-10)
    def test_global_volume_does_not_hide_local_error(self):
        c=self.case;i=self.i;pred=torch.tensor(np.stack((c.probes['before'][i],c.probes['after'][i])))[None]
        wet=np.flatnonzero(pred[0,1,0].numpy().ravel()>.002);a,b=map(int,wet[:2]);pred[0,1,0].flatten()[a]+=.001;pred[0,1,0].flatten()[b]-=.001
        s=score(pred,c.probes['before'][i],c.probes['after'][i],c.bed,float(c.probes['dt'][i]),c.probes['source'][i],np.ones((20,20),bool))
        self.assertLess(s['volume_residual_abs_mean_m3'],1e-10);self.assertGreater(s['local_physics_normalized_mse'],0.);self.assertGreater(s['microstep_h_mae_m'],0.)
    def test_known_unknown_errors_are_separate(self):
        before=np.zeros((3,20,20));after=before.copy();after[0]=.1;known=np.zeros((20,20),bool);known[:10]=True
        pred=torch.zeros(1,2,3,20,20,dtype=torch.float64);pred[0,1,0,:10]=.1
        s=score(pred,before,after,np.zeros((20,20)),.5,np.zeros((20,20)),known)
        self.assertEqual(s['known_h_mae_m'],0.);self.assertAlmostEqual(s['unknown_h_mae_m'],.1);self.assertAlmostEqual(s['microstep_h_mae_m'],.05)

if __name__=='__main__':unittest.main(verbosity=2)
