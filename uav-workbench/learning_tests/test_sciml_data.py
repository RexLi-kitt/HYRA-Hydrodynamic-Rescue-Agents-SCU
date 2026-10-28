from hashlib import sha256
import json
from pathlib import Path
import unittest

from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
from hyra_learning.sciml_data import cases, terrain, hydrographs


class NewStudyDataTests(unittest.TestCase):
    def test_parents_and_forcing_templates_do_not_cross_splits(self):
        design = json.loads((Path(__file__).resolve().parents[1]/'learning_configs/sciml_round1_design_20261025.json').read_text(encoding='utf-8'))
        entries = cases(design)
        self.assertEqual([sum(c['split'] == s for c in entries) for s in ('fit','selection','holdout')], [9,3,9])
        self.assertEqual(len({c['id'] for c in entries}), 21)
        parents, forcing = {}, {}
        by_parent = {}
        for split in ('fit','selection','holdout'):
            parents[split] = {c['parent_terrain'] for c in entries if c['split'] == split}
            forcing[split] = {c['forcing_family'] for c in entries if c['split'] == split}
        for a,b in [('fit','selection'),('fit','holdout'),('selection','holdout')]:
            self.assertFalse(parents[a] & parents[b])
            self.assertFalse(forcing[a] & forcing[b])
        hydro_hashes = set()
        for case in entries:
            bed = terrain(case['family'],case['parent_seed'])
            self.assertEqual(bed.shape,(32,32))
            self.assertTrue(np.isfinite(bed).all())
            fingerprint = sha256(bed.tobytes()).hexdigest()
            if case['family'] in by_parent:
                self.assertEqual(by_parent[case['family']],fingerprint)
            by_parent[case['family']] = fingerprint
            graphs = hydrographs(case)
            hydro_hashes.add(sha256(json.dumps(graphs,sort_keys=True).encode()).hexdigest())
            for points in graphs.values():
                self.assertEqual(points[0],[0,0] if isinstance(points[0],list) else (0,0))
                self.assertTrue(all(a[0] < b[0] for a,b in zip(points,points[1:])))
                self.assertTrue(all(t % 10 == 0 and q >= 0 for t,q in points))
        self.assertEqual(len(set(by_parent.values())),7)
        self.assertEqual(len(hydro_hashes),21)


if __name__ == '__main__':
    unittest.main()
