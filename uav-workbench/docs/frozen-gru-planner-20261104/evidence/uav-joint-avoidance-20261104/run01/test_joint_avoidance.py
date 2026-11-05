import unittest
import numpy as np
from joint_avoidance import resolve_joint,swept_distances,MOVES

class Safety(unittest.TestCase):
    def test_safe_actions_exact(self):
        p=np.array([[5.,5.],[995.,5.],[5.,995.],[995.,995.]])
        ends=np.clip(p[:,None,:]+MOVES[None,:,:]*10,0,1000);wanted=np.array([3,2,3,2])
        a,changed=resolve_joint(p,ends,wanted,np.zeros((4,9)),5.)
        np.testing.assert_array_equal(a,wanted);self.assertFalse(changed)

    def test_headon_resolves_without_collision_or_dead_stop(self):
        p=np.array([[490.,500.],[510.,500.],[5.,5.],[995.,995.]])
        ends=np.clip(p[:,None,:]+MOVES[None,:,:]*10,0,1000);wanted=np.array([3,2,8,8]);scores=np.zeros((4,9))
        scores[0]=(ends[0]-p[0])[:,0];scores[1]=-(ends[1]-p[1])[:,0];scores[2:,8]=10000
        a,changed=resolve_joint(p,ends,wanted,scores,5.)
        self.assertTrue(changed);self.assertGreaterEqual(swept_distances(p,ends[np.arange(4),a]).min(),5.-1e-8)
        self.assertGreater(scores[0,a[0]]+scores[1,a[1]],0)
        np.testing.assert_array_equal(a[2:],[8,8])

    def test_original_stuck_pair_can_separate(self):
        p=np.array([[27.954786916,533.635773802],[900.,900.],[27.812613440,528.560966701],[900.,100.]])
        ends=np.clip(p[:,None,:]+MOVES[None,:,:]*10,0,1000);wanted=np.array([1,8,0,8]);scores=np.zeros((4,9))
        scores[0]=-(ends[0]-p[0])[:,1];scores[2]=(ends[2]-p[2])[:,1];scores[[1,3],8]=10000
        a,changed=resolve_joint(p,ends,wanted,scores,5.)
        self.assertTrue(changed);self.assertGreaterEqual(swept_distances(p,ends[np.arange(4),a]).min(),5.-1e-8)
        self.assertTrue((a[[0,2]]!=8).any())

if __name__=='__main__':unittest.main()
