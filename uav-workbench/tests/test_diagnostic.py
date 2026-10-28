import unittest
import numpy as np
from hyra_uav.core import Memory
from hyra_uav.diagnostic import GROUPS, feature_groups, fit_ridge, node_features, ranking, rewire, replay_samples


class DiagnosticTests(unittest.TestCase):
    def test_changes_only_from_two_actual_observations_and_retained_interval(self):
        m=Memory((2,2),4)
        m.accept(dict(cells=[0],labels=[0],observed_s=0,arrived_s=0),0)
        self.assertEqual(node_features(m,20)[0,18],0)
        m.accept(dict(cells=[0],labels=[1],observed_s=10,arrived_s=12),12)
        f=node_features(m,20)
        self.assertEqual(f[0,15],1)
        self.assertAlmostEqual(float(f[0,16]),10/600)
        self.assertAlmostEqual(float(f[0,17]),10/600)
        self.assertEqual(f[1].sum(),0)
        with self.assertRaises(ValueError):node_features(m,11)

    def test_same_dimension_common_raw_evidence_and_ablation(self):
        n=4;pool=np.array([[1,2],[0,3],[0,3],[1,2]])
        graph=dict(pool=pool,distance=np.ones((n,2)),strength=np.full((n,2),.5),B3=np.array([[1,0]]*n),A2=np.array([[0,1]]*n))
        m=Memory((2,2),4)
        m.accept(dict(cells=[0,1],labels=[0,0],observed_s=0,arrived_s=0),0)
        m.accept(dict(cells=[1],labels=[1],observed_s=10,arrived_s=10),10)
        features=feature_groups(m,20,graph,np.zeros((2,2)))
        self.assertEqual(len({v.shape for v in features.values()}),1)
        # Common raw evidence is identical before the appended aggregation.
        for group in ("B2","B3","A2"):
            np.testing.assert_array_equal(features["B1"][:,:63],features[group][:,:63])
        self.assertTrue(np.any(features["A1"]!=features["B3"]))
        self.assertEqual(set(features),set(GROUPS))

    def test_wrong_graph_preserves_degrees_distance_rings_and_common_pool(self):
        n=20;distances=np.ones((n,n));np.fill_diagonal(distances,np.inf)
        pool=np.array([[j for j in range(n) if j!=i] for i in range(n)])
        graph=np.zeros(pool.shape,bool)
        for i in range(n):graph[i,np.isin(pool[i],[(i+k)%n for k in range(1,6)])]=True
        wrong,swaps=rewire(pool,graph,distances,attempts=2000)
        self.assertGreater(swaps,0)
        np.testing.assert_array_equal(graph.sum(1),wrong.sum(1))
        np.testing.assert_array_equal(np.bincount(pool[graph],minlength=n),np.bincount(pool[wrong],minlength=n))

    def test_ties_rounding_no_positives_and_empty_candidates(self):
        y=np.array([1,0,0]);cells=np.array([9,3,5]);scores=np.ones(3)
        result=ranking(y,scores,cells,.1)
        self.assertEqual(result["k"],1);self.assertEqual(result["precision"],0)
        self.assertIsNone(ranking(np.zeros(3),scores,cells,.1)["recall"])
        self.assertIsNone(ranking(np.array([]),np.array([]),np.array([]),.1)["precision"])

    def test_no_refreshed_or_unknown_candidates_and_no_endpoint_weight(self):
        shape=(2,2);pool=np.array([[1],[0],[3],[2]])
        graph=dict(pool=pool,distance=np.ones((4,1)),strength=np.ones((4,1)),B3=np.ones((4,1)),A2=np.ones((4,1)))
        replay=dict(grid=dict(shape=shape),config=dict(duration_s=20),
                    observations=[dict(cells=[0,1],labels=[0,0],observed_s=0,arrived_s=0),dict(cells=[1],labels=[1],observed_s=10,arrived_s=10)],
                    snapshots=[dict(time_s=t) for t in (0,10,20)],truth=[[0,0,0,0],[1,1,0,0],[0,1,0,0]])
        batches=list(replay_samples(replay,graph,np.zeros(shape)))
        self.assertEqual(len(batches),1)
        self.assertEqual(batches[0][1].tolist(),[0]);self.assertEqual(batches[0][-1].tolist(),[1])
        # Different hidden labels change training labels, never inputs.
        replay["truth"][1]=[0,0,1,1]
        changed=list(replay_samples(replay,graph,np.zeros(shape)))[0]
        for group in GROUPS:np.testing.assert_array_equal(batches[0][3][group],changed[3][group])

    def test_scaling_and_model_fit_use_only_supplied_fit_data(self):
        x=np.array([[0,1],[1,1],[2,1]],float);y=np.array([0,0,1])
        model=fit_ridge(x,y)
        np.testing.assert_array_equal(model["mean"],[1,1]);self.assertEqual(model["scale"][1],1)
        self.assertTrue(np.isfinite(model["weight"]).all())


if __name__=="__main__":unittest.main()
