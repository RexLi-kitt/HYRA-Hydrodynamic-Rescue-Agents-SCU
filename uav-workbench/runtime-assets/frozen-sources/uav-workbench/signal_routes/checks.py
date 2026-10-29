"""Independent numerical, permission, environment and resume acceptance tests."""
import unittest
from dataclasses import replace
import numpy as np
from scale100_env import ScaleEnv,config
from .belief import SignalBeliefEstimator,Parameters
from .environment import RouteEnv


class FakeCase:
    shape=(20,20); dx=5.; digest='synthetic-engineering-check'
    process={'grid':{'shape':[20,20],'dx_m':5,'axis_order':['z','x']}}
    bed=np.zeros(shape)
    def index(self,t):
        if t<0 or t>900 or t%5: raise ValueError('Actual 5s frames only')
        return int(t//5)
    def values(self,t):
        self.index(t); a=np.zeros((4,*self.shape)); a[0]=.02+float(t)/10000; a[1]=.03; return a
    def labels(self,t): return self.values(t)[0]>=.05


class BeliefChecks(unittest.TestCase):
    def setUp(self):
        self.shape=(3,4); self.rng=np.random.default_rng(7)
        self.s=self.rng.normal(size=self.shape); self.h=self.rng.normal(size=(3,*self.shape))*.03
        self.mask=np.zeros(self.shape,bool); self.mask[0,:]=True
        self.stamps=np.where(self.mask,0.,-1.)

    def make(self,mode,p=None):
        return SignalBeliefEstimator(mode,p).reset(self.shape).update(self.s,self.h,self.mask,self.stamps,5.)

    def test_spd_and_exact_dense_conditional(self):
        for mode in ('Bpost','R1','R2','R3','C1','C2','C3'):
            model=self.make(mode,Parameters(alpha=.7,kappa=2,beta=(.1,.1,0,0),strength=2,cross=.6))
            q,m=model.canonical(); dense=q.toarray()
            np.testing.assert_allclose(dense,dense.T,atol=1e-12)
            np.linalg.cholesky(dense)
            joint=len(m)==24; known=np.r_[self.mask.ravel(),self.mask.ravel()] if joint else self.mask.ravel()
            observed=(self.s.ravel()-.1)/.2
            values=np.r_[np.ones(12),observed] if joint else observed
            ref=m.copy(); ref[known]=values[known]
            ref[~known]+=np.linalg.solve(dense[np.ix_(~known,~known)],-dense[np.ix_(~known,known)]@(values[known]-m[known]))
            np.testing.assert_allclose(model.posterior()['mean'].ravel(),ref[-12:]*.2+.1,atol=1e-10)

    def test_zero_effect_recovers_shared_baseline(self):
        baseline=self.make('Bpost').posterior(True)
        for mode in ('R1','R2','R3','C1','C2','C3'):
            out=self.make(mode).posterior(True)
            np.testing.assert_allclose(out['mean'],baseline['mean'],atol=1e-12)
            np.testing.assert_allclose(out['variance'],baseline['variance'],atol=1e-12)

    def test_sar_residual_energy_equals_symmetric_precision(self):
        model=self.make('R1',Parameters(alpha=.8,kappa=2)); q,_=model.canonical()
        b=.35*((1-.8)*model.w0+.8*model.weights()); a=np.eye(12)-b.toarray()
        x=self.rng.normal(size=12)
        self.assertAlmostEqual(float(x@q@x),float((a@x)@(a@x)))
        self.assertGreater(np.max(np.abs(b.toarray()-b.toarray().T)),1e-5)

    def test_crf_joint_energy_identity_and_known_structure_equivalence(self):
        model=self.make('R3',Parameters(strength=2,cross=.6,kappa=1)); q,_=model.canonical()
        z=self.rng.normal(size=12); x=self.rng.normal(size=12); w=model.weights().toarray()
        energy=z@model.q0@z+x@model.q0@x+2*np.sum((x-.6*w@z)**2)
        self.assertAlmostEqual(float(np.r_[z,x]@q@np.r_[z,x]),float(energy))
        # If z is fixed, the CRF is exactly a Gaussian with a shifted signal mean.
        conditional_q=model.q0.toarray()+2*np.eye(12)
        conditional_mean=np.linalg.solve(conditional_q,2*.6*w@z)
        residual=q.toarray()[12:,:12]@z+conditional_q@conditional_mean
        np.testing.assert_allclose(residual,0,atol=1e-12)

    def test_unknown_storage_and_hidden_future_cannot_change_output(self):
        for mode in ('R1','R2','R3','C1','C2','C3'):
            p=Parameters(alpha=.7,kappa=2,beta=(.2,0,.1,0),strength=1,cross=.5)
            a=self.make(mode,p).posterior()['mean']
            signal=self.s.copy(); water=self.h.copy(); stamps=self.stamps.copy()
            signal[~self.mask]=1e9; water[:,~self.mask]=np.nan; stamps[~self.mask]=1e9
            b=SignalBeliefEstimator(mode,p).reset(self.shape).update(signal,water,self.mask,stamps,5.).posterior()['mean']
            np.testing.assert_array_equal(a,b)

    def test_future_arrival_rejected(self):
        stamps=self.stamps.copy(); stamps[self.mask]=6.
        with self.assertRaises(ValueError):
            SignalBeliefEstimator().reset(self.shape).update(self.s,self.h,self.mask,stamps,5.)

    def test_records_not_mutated_and_known_zero_is_exact(self):
        self.s[0,0]=0.; before=self.s.copy(); model=self.make('R3',Parameters(strength=2,cross=.6))
        out=model.posterior(True)
        np.testing.assert_array_equal(self.s,before)
        np.testing.assert_allclose(out['mean'][self.mask],self.s[self.mask],atol=1e-14)
        np.testing.assert_array_equal(out['variance'][self.mask],0.)

    def test_missing_dry_and_all_observed(self):
        for mode in ('Bpost','R1','R2','R3'):
            for mask in (np.zeros(self.shape,bool),np.ones(self.shape,bool)):
                model=SignalBeliefEstimator(mode,Parameters(alpha=1,kappa=4,strength=10,cross=2)).reset(self.shape)
                model.update(self.s,np.zeros_like(self.h),mask,np.where(mask,0.,-1.),0.)
                self.assertTrue(np.isfinite(model.posterior(True)['variance']).all())

    def test_permutation_is_joint_deterministic_and_mask_preserving(self):
        a=self.make('C1',Parameters(alpha=1,kappa=2)); b=self.make('C1',Parameters(alpha=1,kappa=2))
        np.testing.assert_array_equal(a.water,b.water)
        np.testing.assert_array_equal(a.valid,self.mask.ravel())
        original=sorted(map(tuple,self.h.reshape(3,-1)[:,self.mask.ravel()].T))
        permuted=sorted(map(tuple,a.water[:,self.mask.ravel()].T))
        self.assertEqual(original,permuted)

    def test_covariance_and_nll_match_independent_dense_inverse(self):
        model=self.make('R1',Parameters(alpha=.5,kappa=2)); q,_=model.canonical(); ids=np.flatnonzero(~self.mask.ravel())
        ref=np.linalg.inv(q.toarray()[np.ix_(ids,ids)])*.2**2
        np.testing.assert_allclose(model.covariance(ids),ref,atol=1e-12)
        score=model.score(self.s)
        self.assertTrue(np.isfinite(score['nll']))

    def test_surface_slope_uses_public_bed_and_physical_edge_length(self):
        bed=np.tile(np.arange(4)*.1,(3,1)); water=np.zeros((3,3,4)); water[0]=.05
        model=SignalBeliefEstimator('R2').reset(self.shape,dict(bed=bed,dx_m=5.))
        model.update(self.s,water,np.ones(self.shape,bool),np.zeros(self.shape),0.)
        raw,available=model.raw_features()
        self.assertTrue(available[:,2].all())
        self.assertGreater(float(raw[:,2].min()),0.)
        # Doubling grid spacing must halve the slope; depth differences alone cannot pass.
        other=SignalBeliefEstimator('R2').reset(self.shape,dict(bed=bed,dx_m=10.))
        other.update(self.s,water,np.ones(self.shape,bool),np.zeros(self.shape),0.)
        np.testing.assert_allclose(other.raw_features()[0][:,2],raw[:,2]/2,atol=1e-14)

    def test_level_free_surface_has_zero_slope_despite_varying_depth(self):
        bed=np.tile(np.arange(4)*.1,(3,1)); water=np.zeros((3,3,4)); water[0]=.5-bed
        model=SignalBeliefEstimator('R2').reset(self.shape,dict(bed=bed,dx_m=5.))
        model.update(self.s,water,np.ones(self.shape,bool),np.zeros(self.shape),0.)
        np.testing.assert_allclose(model.raw_features()[0][:,2],0.,atol=1e-14)

    def test_fit_normalization_and_missing_feature_gating(self):
        p=Parameters(feature_mean=(.1,.2,.3,10.),feature_scale=(.2,.4,.6,20.))
        model=self.make('R2',p); raw,available=model.raw_features()
        expected=np.where(available,np.clip((raw-np.array(p.feature_mean))/np.array(p.feature_scale),-5,5),0.)
        np.testing.assert_array_equal(model.features(),expected)
        np.testing.assert_array_equal(model.features()[~self.mask.ravel()],0.)
        with self.assertRaises(ValueError): Parameters(feature_scale=(1,1,0,1)).validate()

    def test_dry_or_unobserved_surface_edges_do_not_supply_slope(self):
        model=SignalBeliefEstimator('R2').reset(self.shape,dict(bed=np.ones(self.shape)))
        model.update(self.s,np.zeros_like(self.h),self.mask,self.stamps,5.)
        self.assertFalse(model.raw_features()[1][:,2].any())
        with self.assertRaises(ValueError): SignalBeliefEstimator().reset(self.shape,dict(bed=np.full(self.shape,np.nan)))


class EnvironmentChecks(unittest.TestCase):
    def test_generator_records_rewards_and_only_two_channels_change(self):
        case=FakeCase(); cfg=replace(config('E2','B3',11),duration_s=20.)
        base=ScaleEnv(case,cfg); base.reset()
        models=[RouteEnv(case,cfg,m,Parameters(alpha=.5,kappa=1,beta=(.1,.1,0,0),strength=.5,cross=.4))
                for m in ('Braw','Bpost','R1','R2','R3','C1','C2','C3')]
        for env in models: env.reset()
        for actions in ([1,1],[1,1],[3,2],[1,1]):
            view,reward,done=base.step(actions)
            for env in models:
                v,r,d=env.step(actions)
                self.assertEqual(reward,r); self.assertEqual(done,d)
                self.assertEqual(base.observations,env.observations)
                np.testing.assert_array_equal(base.fields._physical_signal,env.fields._physical_signal)
                kept=[i for i in range(21) if i not in (3,7)]
                np.testing.assert_array_equal(view.images[:,kept],v.images[:,kept])
                np.testing.assert_array_equal(base.signal,env.signal)

    def test_e1_is_exact_zero_and_identical_to_existing_view(self):
        case=FakeCase(); cfg=replace(config('E1','B3',11),duration_s=10.)
        base=ScaleEnv(case,cfg); reference=base.reset()
        for mode in ('Braw','Bpost','R1','R2','R3','C1','C2','C3'):
            env=RouteEnv(case,cfg,mode,Parameters(strength=2,cross=.8)); view=env.reset()
            np.testing.assert_array_equal(view.images,reference.images)
            np.testing.assert_array_equal(view.images[:,[3,7]],0.)

    def test_unobserved_target_mutation_does_not_change_estimator_view(self):
        env=RouteEnv(FakeCase(),replace(config('E2','B3',11),duration_s=10.),'R3',Parameters(strength=1,cross=.5))
        env.reset(); a=env.view().images.copy()
        hidden=[p for p,s in env.target_status.items() if s=='undiscovered']
        for p in hidden: del env.target_status[p]
        env.fields._physical_signal[:]=12345.
        env._belief_key=None
        np.testing.assert_array_equal(a,env.view().images)


if __name__=='__main__': unittest.main(verbosity=2)
