import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from run_okushiri_hyra import sample_bed
class Coordinates(unittest.TestCase):
    def test_explicit_nodes(self):
        x=np.arange(9)*.014;y=np.arange(8)*.014
        xx,yy=np.meshgrid(x,y[::-1]);z=2*xx-3*yy+.7
        np.testing.assert_allclose(sample_bed(z,xx,yy,.014,1,'xya_node'),z,rtol=0,atol=1e-14)
    def test_legacy_centers(self):
        x=(np.arange(9)+.5)*.014;y=(np.arange(8)+.5)*.014
        xx,yy=np.meshgrid(x,y[::-1]);z=2*xx-3*yy+.7
        np.testing.assert_allclose(sample_bed(z,xx,yy,.014),z,rtol=0,atol=1e-14)
if __name__=='__main__':unittest.main()
