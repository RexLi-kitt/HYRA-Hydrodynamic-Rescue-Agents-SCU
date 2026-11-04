import json,unittest
from pathlib import Path
import numpy as np
import torch
from flood1km.learners import Learner,ALGORITHMS
from train_with_service import rollout
class IntegrationTests(unittest.TestCase):
    def test_all_six_actual_actions_and_next_masks(self):
        torch.set_num_threads(2)
        root=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs\uav6-pilot-20261101')
        item=json.loads((root/'data/dataset.json').read_text(encoding='utf-8'))['cases'][0]
        water=np.load(item['path'],mmap_mode='r')
        for a in ALGORITHMS:
            with self.subTest(algorithm=a):
                torch.manual_seed(1001);m=Learner(a,1468);rows=[]
                result=rollout(m,water,920000,rows.append)
                self.assertEqual(len(rows),result['ST']);self.assertGreater(result['forced_agent_steps'],0)
                for row,nxt in zip(rows[:-1],rows[1:]):
                    np.testing.assert_array_equal(row['next_mask'],nxt['mask'])
                    np.testing.assert_array_equal(row['next_action'],nxt['action'])
                    np.testing.assert_array_equal(row['next_obs'],nxt['obs'])
                before=[p.detach().clone() for p in m.parameters()]
                if a in ('hasac','maac'):m.update_off_policy(rows[:64])
                else:m.update_on_policy(rows)
                m.check_finite();self.assertTrue(any(not torch.equal(x,p) for x,p in zip(before,m.parameters())))
                print('verified',a,len(rows),'forced',result['forced_agent_steps'],flush=True)
if __name__=='__main__':unittest.main()
