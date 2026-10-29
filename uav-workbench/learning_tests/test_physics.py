from pathlib import Path
import os
import sys
import unittest

from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch

os.environ['HYRA_FLOOD_BACKEND'] = 'numpy'
TERRAIN = Path(__file__).resolve().parents[2]/'terrain-workbench/terrain-lab'
sys.path.insert(0, str(TERRAIN))
from flood_solver import FloodSolver
from hyra_learning.physics import SWEOperator


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.operator = SWEOperator()

    def compare(self, bed, state, dt=.1, source=None, drains=None, solid=None, wx=None, wy=None, infiltration=0., dtype=torch.float64):
        reference = FloodSolver(bed, state[0], 5, boundary='closed')
        reference.set_environment(solid=solid, wall_x=wx, wall_y=wy, drain_capacity=drains)
        reference.mx, reference.mz = state[1].copy(), state[2].copy()
        if source is not None:
            reference.baseflow_rate = source.copy()
        reference.step(dt, infiltration=infiltration)
        expected = np.stack((reference.h, reference.mx, reference.mz))
        tensor = lambda a: None if a is None else torch.as_tensor(a, dtype=dtype)
        result = self.operator.step(tensor(state)[None], torch.as_tensor(bed, dtype=torch.float64), dt,
                                    source=tensor(source), drains=tensor(drains),
                                    solid=None if solid is None else torch.as_tensor(solid),
                                    wall_x=None if wx is None else torch.as_tensor(wx),
                                    wall_y=None if wy is None else torch.as_tensor(wy), infiltration=infiltration)
        np.testing.assert_allclose(result[0].detach().numpy(), expected, rtol=0,
                                   atol=1e-10 if dtype == torch.float64 else 2e-5)
        return result

    def test_lake_dry_source_front_and_sinks_match_reference(self):
        z, x = np.indices((7, 9), dtype=float)
        bed = .015*x+.02*np.sin(z)
        h = .4-bed
        state = np.stack((h, np.zeros_like(h), np.zeros_like(h)))
        result = self.compare(bed, state)
        np.testing.assert_allclose(result[0].numpy(), state, atol=1e-12)
        source = np.zeros_like(h)
        source[3, 2] = .4
        dry = state*0
        self.compare(bed, dry, source=source)
        front = state.copy()
        front[0, :, 5:] = 0
        front[1, :, :5] = .02
        self.compare(bed, front, source=source, drains=np.full_like(h, .04), infiltration=.00001)
        self.compare(bed, front, source=source, drains=np.full_like(h, 20.), infiltration=.2)
        self.compare(bed+10000, front, dtype=torch.float32)

    def test_reflective_walls_solids_drag_and_batch(self):
        rng = np.random.default_rng(14)
        bed = rng.uniform(0, .03, (7, 9))
        h = rng.uniform(.05, .2, bed.shape)
        state = np.stack((h, h*.2, h*-.1))
        solid = np.zeros_like(h, dtype=bool)
        solid[2:4, 4] = True
        state[:, solid] = 0
        wx, wy = np.zeros((7, 10), dtype=bool), np.zeros((8, 9), dtype=bool)
        wx[:, 5], wy[3, :3] = True, True
        expected = self.compare(bed, state, wx=wx, wy=wy, solid=solid)
        data = torch.as_tensor(np.stack((state, state)), dtype=torch.float64)
        result = self.operator.step(data, torch.as_tensor(bed), .1,
                                    solid=torch.as_tensor(solid), wall_x=torch.as_tensor(wx), wall_y=torch.as_tensor(wy))
        torch.testing.assert_close(result, expected.expand_as(result), atol=1e-12, rtol=0)
        self.compare(bed, state, wx=wx, wy=wy, solid=solid, dtype=torch.float32)

    def test_gradients_finite_in_dry_cells_and_finite_difference_in_wet_cells(self):
        dry = torch.zeros((1, 3, 5, 6), dtype=torch.float64, requires_grad=True)
        bed = torch.zeros((5, 6), dtype=torch.float64)
        result = self.operator.step(dry, bed, .1)
        result.sum().backward()
        self.assertTrue(torch.isfinite(dry.grad).all())
        generator = torch.Generator().manual_seed(17)
        state = torch.rand((1, 3, 5, 6), generator=generator, dtype=torch.float64)*.01
        state[:, 0] += .1
        state.requires_grad_()
        weights = torch.randn(state.shape, generator=generator, dtype=torch.float64)
        loss = (self.operator.step(state, bed, .03)*weights).sum()
        gradient = torch.autograd.grad(loss, state)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        plus, minus = state.detach().clone(), state.detach().clone()
        plus[0, 1, 2, 3] += 1e-7
        minus[0, 1, 2, 3] -= 1e-7
        numeric = ((self.operator.step(plus, bed, .03)-self.operator.step(minus, bed, .03))*weights).sum()/2e-7
        torch.testing.assert_close(numeric, gradient[0, 1, 2, 3], rtol=1e-4, atol=1e-5)
        if torch.cuda.is_available():
            cuda_state = state.detach().cuda().requires_grad_()
            cuda_result = self.operator.step(cuda_state, bed.cuda(), .03)
            torch.testing.assert_close(cuda_result.cpu(), self.operator.step(state.detach(), bed, .03), rtol=1e-10, atol=1e-12)
            cuda_result.square().sum().backward()
            self.assertTrue(torch.isfinite(cuda_state.grad).all())

    def test_unsupported_numerics_rejected(self):
        for options in [dict(boundary='open'), dict(flux_method='knp'), dict(spatial_order=1), dict(time_integrator='euler')]:
            with self.assertRaises(ValueError):
                SWEOperator(**options)

    def test_float32_drying_and_registered_prediction_bounds_have_finite_gradients(self):
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            h=torch.tensor([0.,1e-12,1e-8,1e-6,.001,.1],device=device)[None,None,:].expand(1,5,-1).clone()
            state=torch.stack((h,h*2,h*-.5),dim=1).requires_grad_()
            output=self.operator.step(state,torch.zeros_like(h),.01,drains=torch.full_like(h,.04))
            output.square().sum().backward()
            self.assertTrue(torch.isfinite(state.grad).all())
            z,x=torch.meshgrid(torch.arange(5,device=device),torch.arange(6,device=device),indexing='ij')
            h=torch.where((z+x)%2==0,5.,.0001)[None]
            u=torch.where(x%2==0,10.,-10.)[None]
            v=torch.where(z%2==0,10.,-10.)[None]
            state=torch.stack((h,h*u,h*v),dim=1).requires_grad_()
            source=torch.zeros_like(h);source[:,2,3]=2.
            output=self.operator.step(state,((x+z)*.01).double(),.01,source=source)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(output).all() and torch.isfinite(state.grad).all())
            self.assertTrue((output[:,0]>=0).all())


if __name__ == '__main__':
    unittest.main()
