from pathlib import Path
import tempfile
import unittest
import numpy as np
from hyra_uav.core import Memory
from hyra_uav.strong_features import representation,batches,latest_only,model_inputs,cache
from hyra_uav.relations import weighted_relations
from test_core import TEST_TEMP,bundle


class RelationTests(unittest.TestCase):
    def test_strict_A1_cannot_reconstruct_any_earlier_observed_changes(self):
        pool=np.array([[1,2],[0,3],[0,3],[1,2]])
        graph=dict(pool=pool,distance=np.ones((4,2)),terrain_factor=np.ones((4,2)),
                   B2_weight=np.ones((4,2)),B3_weight=np.array([[1.,.1]]*4),A2_weight=np.array([[.1,1.]]*4))
        first,second=Memory((2,2)),Memory((2,2))
        for memory,old_label in ((first,0),(second,1)):
            memory.accept(dict(cells=[0,1,2,3],labels=[old_label]*4,observed_s=0,arrived_s=0),0)
            memory.accept(dict(cells=[0,1,2,3],labels=[1]*4,observed_s=10,arrived_s=10),10)
        args=(20,graph,np.zeros((2,2)),np.array([0]))
        raw_a,tail_a=representation(first,*args);raw_b,_=representation(second,*args)
        self.assertFalse(np.array_equal(raw_a,raw_b))
        strict_a,strict_tail_a=representation(latest_only(first),*args)
        strict_b,strict_tail_b=representation(latest_only(second),*args)
        np.testing.assert_array_equal(strict_a,strict_b)
        np.testing.assert_array_equal(strict_tail_a['A1'],strict_tail_b['A1'])
        self.assertTrue(np.all(strict_a[:,15:19]==0))
        self.assertEqual(strict_a[0,12],1)  # Latest label and age survive.
        self.assertAlmostEqual(strict_a[0,13],10/600)
        data=dict(common=raw_a,common_A1=strict_a,**tail_a);data['A1']=strict_tail_a['A1']
        np.testing.assert_array_equal(model_inputs(data,'B3')[:,:raw_a.shape[1]],raw_a)
        np.testing.assert_array_equal(model_inputs(data,'A1')[:,:raw_a.shape[1]],strict_a)
        replay=dict(grid=dict(shape=[2,2]),config=dict(duration_s=20),
                    observations=[dict(cells=list(range(4)),labels=[v]*4,observed_s=t,arrived_s=t) for t,v in ((0,0),(5,1))],
                    snapshots=[dict(time_s=t) for t in (0,5,10,20)],truth=[[1]*4]*4)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            path=Path(folder)/'strict.npz'
            stats=cache(replay,graph,np.zeros((2,2)),path,strict_change_ablation=True)
            self.assertTrue(stats['strict_change_ablation'])
            with np.load(path) as data:
                self.assertTrue(np.all(data['common_A1'][:,3:12]==0))
                self.assertTrue(np.any(data['common'][:,3:12]!=0))
                self.assertTrue(np.all(data['A1'][:,15:]==0))
                self.assertFalse(np.array_equal(model_inputs(data,'A1'),model_inputs(data,'B3')))

    def test_weight_ablation_exact_support_degree_and_distance_mass(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            process=bundle(Path(temp)/"p.zip",np.zeros((3,9,9)))
            # A purely flat terrain contains no topographic relation to destroy.
            flat=weighted_relations(process,pool_size=32,terrain_cost_scale_m=2.5)
            self.assertTrue(flat["audit"]["support_exact"])
            self.assertTrue(flat["audit"]["distance_weighted_mass_exact"])
            self.assertAlmostEqual(flat["audit"]["mean_operator_total_variation"],0)
            process.bed=np.indices((9,9))[1]*.03
            graph=weighted_relations(process,pool_size=32,terrain_cost_scale_m=2.5)
            self.assertTrue(graph["audit"]["support_exact"])
            self.assertTrue(graph["audit"]["out_strength_exact"])
            self.assertGreater(graph["audit"]["mean_operator_total_variation"],0)
            self.assertTrue(graph["audit"]["distance_weighted_mass_exact"])

    def test_all_models_share_raw_history_and_A1_only_removes_propagated_changes(self):
        n=4;pool=np.array([[1,2],[0,3],[0,3],[1,2]])
        graph=dict(pool=pool,distance=np.ones((4,2)),terrain_factor=np.ones((4,2)),
                   B2_weight=np.ones((4,2)),B3_weight=np.array([[1.,.1]]*4),A2_weight=np.array([[.1,1.]]*4))
        memory=Memory((2,2));memory.accept(dict(cells=[0,1,2],labels=[0,0,0],observed_s=0,arrived_s=0),0)
        memory.accept(dict(cells=[1],labels=[1],observed_s=10,arrived_s=10),10)
        raw,tail=representation(memory,20,graph,np.zeros((2,2)),np.array([0]))
        self.assertEqual(raw.shape,(1,63))
        self.assertTrue(np.all(tail["A1"][:,15:19]==0))
        np.testing.assert_array_equal(tail["A1"][:,:15],tail["B3"][:,:15])
        self.assertGreater(tail["B3"][0,15],tail["A2"][0,15])
        replay=dict(grid=dict(shape=[2,2]),config=dict(duration_s=20),
                    observations=[dict(cells=[0],labels=[0],observed_s=0,arrived_s=0)],
                    snapshots=[dict(time_s=t) for t in (0,10,20)],truth=[[0]*4,[1]*4,[0]*4])
        before=list(batches(replay,graph,np.zeros((2,2))))[0]
        replay["truth"][1]=[0]*4
        after=list(batches(replay,graph,np.zeros((2,2))))[0]
        np.testing.assert_array_equal(before["common"],after["common"])
        self.assertNotEqual(before["y"].tolist(),after["y"].tolist())


if __name__=="__main__":unittest.main()
