"""Run using the isolated RivGraph interpreter; test original-mask preservation."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import rivgraph
except ImportError:
    rivgraph=None


@unittest.skipIf(rivgraph is None,'Use the isolated RivGraph interpreter')
class RivGraphWorkerTests(unittest.TestCase):
    def analyze(self,mask):
        from river_network_worker import analyze
        return analyze(mask,np.zeros(mask.shape),np.ones(mask.shape),1)

    def test_disconnected_channels_are_not_discarded(self):
        mask=np.zeros((30,30),bool);mask[4:7,2:28]=True;mask[20:23,2:28]=True
        result=self.analyze(mask)
        self.assertEqual(result['components_four'],2)
        self.assertEqual({link['component'] for link in result['graph']['links']},{1,2})

    def test_diagonal_single_cells_remain_separate(self):
        result=self.analyze(np.eye(5,dtype=bool))
        self.assertEqual(result['components_four'],5)
        self.assertEqual(result['components_eight'],1)
        self.assertEqual(result['links'],0)

    def test_closed_loop_is_retained(self):
        mask=np.zeros((30,30),bool)
        mask[5:25,5:25]=True;mask[8:22,8:22]=False
        result=self.analyze(mask)
        self.assertEqual(result['components_four'],1)
        self.assertEqual(len(result['graph']['unresolved_components']),1)
        self.assertGreater(len(result['graph']['unresolved_components'][0]['cells']),0)
        self.assertEqual(result['status'],'review')


if __name__=='__main__':unittest.main()
