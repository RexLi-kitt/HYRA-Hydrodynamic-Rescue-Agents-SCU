"""Person legality, rescue start and reachability checks on the hydraulic grid."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import people
import map_sessions


def grid(n=8, dx=1.0):
    shape = (n, n)
    return dict(shape=shape, dx=dx, solid=np.zeros(shape, bool), bed=np.zeros(shape),
                wall_x=np.zeros((n, n + 1), bool), wall_y=np.zeros((n + 1, n), bool))


def person(x, z, group=0, kind='trapped', pid=None):
    item = dict(x_m=x, z_m=z, group=group, kind=kind)
    if pid:
        item['id'] = pid
    return item


def build(g, items, start=None, **kw):
    return people.normalize(dict(mode='manual', seed=1, start=start, items=items), shape=g['shape'], dx=g['dx'],
                            solid=g['solid'], bed=g['bed'], wall_x=g['wall_x'], wall_y=g['wall_y'], **kw)


class PeopleGridTests(unittest.TestCase):
    def test_legal_people_compile_to_cells(self):
        g = grid()
        result = build(g, [person(1.5, 1.5), person(6.5, 6.5, group=-1)], start=dict(x_m=.5, z_m=.5))
        self.assertEqual([item['cell'] for item in result['items']], [[1, 1], [6, 6]])
        self.assertEqual([item['group'] for item in result['items']], [0, -1])
        self.assertEqual(result['start']['cell'], [0, 0])
        self.assertEqual(result['counts'], dict(total=2, reachable=2, connected_cells=64))

    def test_automatic_start_is_on_the_boundary(self):
        g = grid()
        result = build(g, [person(3.5, 3.5)])
        z, x = result['start']['cell']
        self.assertTrue(z in (0, g['shape'][0] - 1) or x in (0, g['shape'][1] - 1))
        self.assertTrue(result['start']['automatic'])

    def test_person_inside_building_is_rejected(self):
        g = grid()
        g['solid'][2, 2] = True
        with self.assertRaisesRegex(ValueError, '人员 1 位于建筑'):
            build(g, [person(2.5, 2.5)], start=dict(x_m=.5, z_m=.5))

    def test_start_inside_building_is_rejected(self):
        g = grid()
        g['solid'][0, 0] = True
        with self.assertRaisesRegex(ValueError, '救援起点位于建筑'):
            build(g, [], start=dict(x_m=.5, z_m=.5))

    def test_sealed_start_is_rejected(self):
        g = grid()
        g['wall_x'][0, 1] = True
        g['wall_y'][1, 0] = True
        with self.assertRaisesRegex(ValueError, '救援起点被完全封闭'):
            build(g, [], start=dict(x_m=.5, z_m=.5))

    def test_person_behind_a_sealed_wall_is_unreachable(self):
        g = grid()
        g['wall_x'][:, 4] = True
        with self.assertRaisesRegex(ValueError, '人员 1 与救援起点不连通'):
            build(g, [person(6.5, .5)], start=dict(x_m=.5, z_m=.5))
        g['wall_x'][:, 4] = False
        self.assertEqual(len(build(g, [person(6.5, .5)], start=dict(x_m=.5, z_m=.5))['items']), 1)

    def test_levee_faces_block_reachability(self):
        g = grid()
        edges = [(1, z, 4) for z in range(8)]
        with self.assertRaisesRegex(ValueError, '人员 2 与救援起点不连通'):
            build(g, [person(2.5, 2.5), person(6.5, 2.5)], start=dict(x_m=.5, z_m=2.5), levee_edges=edges)

    def test_out_of_bounds_and_ocean_are_rejected(self):
        g = grid()
        with self.assertRaisesRegex(ValueError, '超出地图范围'):
            build(g, [person(9.5, .5)], start=dict(x_m=.5, z_m=.5))
        classes = np.zeros(g['shape'], np.int32)
        classes[3, 3] = 4
        with self.assertRaisesRegex(ValueError, '人员 1 位于海域'):
            build(g, [person(3.5, 3.5)], start=dict(x_m=.5, z_m=.5),
                  terrain_class=classes, terrain_ids={'4': 'ocean'})

    def test_duplicate_ids_and_bad_group_are_rejected(self):
        g = grid()
        with self.assertRaisesRegex(ValueError, '编号需要唯一'):
            build(g, [person(1.5, 1.5, pid='p'), person(2.5, 2.5, pid='p')], start=dict(x_m=.5, z_m=.5))
        with self.assertRaisesRegex(ValueError, '分组'):
            build(g, [person(1.5, 1.5, group=100)], start=dict(x_m=.5, z_m=.5))

    def test_cell_counts_array(self):
        g = grid()
        result = build(g, [person(1.5, 1.5), person(1.6, 1.2)], start=dict(x_m=.5, z_m=.5))
        counts = people.cell_counts(result, g['shape'])
        self.assertEqual(int(counts.sum()), 2)
        self.assertEqual(int(counts[1, 1]), 2)

    def test_empty_people_layer_is_valid(self):
        g = grid()
        result = build(g, [])
        self.assertEqual(result['counts']['total'], 0)
        self.assertEqual(result['items'], [])
        self.assertIsNone(result['start'])


class DemoSceneIntegrationTests(unittest.TestCase):
    def test_demo_scene_keeps_people_in_snapshot_and_arrays(self):
        root = Path(tempfile.mkdtemp())
        data = dict(mode='demo', people=dict(mode='cluster_isolated', seed=7,
            parameters=dict(count=3, clusters=1, radius_m=10, isolated=1),
            items=[person(12.5, 12.5, group=0), person(17.5, 12.5, group=0), person(147.5, 147.5, group=-1)],
            start=dict(x_m=2.5, z_m=2.5)))
        world = map_sessions.create(root, {}, data)
        snapshot = world.snapshot()
        layer = snapshot['M_people']
        self.assertEqual(layer['schema'], 'hyra-people-v1')
        self.assertEqual(layer['counts']['total'], 3)
        self.assertEqual(layer['counts']['reachable'], 3)
        self.assertEqual(len(layer['items']), 3)
        self.assertIn('M_people.count', world.arrays())
        self.assertEqual(int(world.arrays()['M_people.count'].sum()), 3)

    def test_demo_scene_rejects_a_person_inside_a_building(self):
        root = Path(tempfile.mkdtemp())
        data = dict(mode='demo', people=dict(mode='manual',
            items=[person(72.5, 57.5)], start=dict(x_m=2.5, z_m=2.5)))
        with self.assertRaisesRegex(ValueError, '位于建筑'):
            map_sessions.create(root, {}, data)


if __name__ == '__main__':
    unittest.main()
