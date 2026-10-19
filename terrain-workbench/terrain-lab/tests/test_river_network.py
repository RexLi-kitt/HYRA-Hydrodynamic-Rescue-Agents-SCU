"""Effective-grid invariants: a source graph does not prove preservation."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from river_network import compare_grids, components


class RiverGridTests(unittest.TestCase):
    def test_diagonal_touch_is_not_a_shared_face(self):
        _, sizes=components(np.eye(4,dtype=bool))
        self.assertEqual(sizes,[1,1,1,1])

    def test_straight_channel_is_preserved(self):
        native=np.zeros((8,8),bool);native[2:4,:]=True
        coarse=np.zeros((4,4),bool);coarse[1,:]=True
        self.assertEqual(compare_grids(native,coarse,2)['warnings'],[])

    def test_coarsening_can_join_separate_channels(self):
        native=np.zeros((8,8),bool);native[0,:]=True;native[2,:]=True
        coarse=np.zeros((4,4),bool);coarse[:2,:]=True
        report=compare_grids(native,coarse,2)
        self.assertEqual(report['native_components'],2)
        self.assertEqual(report['merged_effective_components'],1)

    def test_lost_middle_cell_splits_channel(self):
        native=np.zeros((8,8),bool);native[2:4,:]=True
        coarse=np.zeros((4,4),bool);coarse[1,:]=True;coarse[1,2]=False
        report=compare_grids(native,coarse,2)
        self.assertEqual(report['lost_native_channel_cells'],4)
        self.assertEqual(report['split_native_components'],1)

    def test_wall_splits_same_grid_without_losing_channel_cells(self):
        mask=np.ones((4,4),bool)
        wx=np.zeros((4,5),bool);wx[:,2]=True
        report=compare_grids(mask,mask,1,wall_x=wx)
        self.assertEqual(report['lost_native_channel_cells'],0)
        self.assertEqual(report['split_native_components'],1)

    def test_empty_map(self):
        report=compare_grids(np.zeros((8,8),bool),np.zeros((4,4),bool),2)
        self.assertEqual(report['native_components'],0)
        self.assertEqual(report['warnings'],[])


if __name__=='__main__':unittest.main()
