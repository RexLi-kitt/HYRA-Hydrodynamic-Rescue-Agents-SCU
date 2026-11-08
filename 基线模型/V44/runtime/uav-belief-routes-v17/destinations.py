"""Causal destination selection; the only fit object is a bank of training-map templates."""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "uav-rescue-gate-v16"))
from audit16 import material, runtime, read, save, sha, s
from service14 import Switcher, DIRECTIONS, candidate_footprints
from bound_search import Cohorts

SIDE = 25
CELL = 1000 // SIDE
axis = np.arange(SIDE) * CELL + CELL / 2
CENTRES = np.stack(np.meshgrid(axis, axis), axis=-1)
FULL = np.stack(np.meshgrid(np.arange(100)*10+5, np.arange(100)*10+5), axis=-1)
BANK = OUT / "origin-templates.npz"


def average_tiles(grid):
    return grid.reshape(SIDE, 4, SIDE, 4).mean(axis=(1, 3))


def blur(grid):
    padded = np.pad(grid, 1, mode="constant")
    return sum(padded[i:i+SIDE, j:j+SIDE] * a * b
               for i, a in enumerate((.2, .6, .2))
               for j, b in enumerate((.2, .6, .2)))


def source_cases():
    for directory, ordinary, pressure in (
        ("uav-priority-confirm-v10", 97151001, 97161001),
        ("uav-short-deadline-learning-v11", 98751001, 98761001),
    ):
        yield str(ROOT / directory), ordinary, pressure


def build_bank():
    templates, origins = [], []
    for directory, ordinary, _ in source_cases():
        for i in range(16):
            key = (directory, ordinary+i, "ordinary", i)
            field, pop = material(key, 0)
            water = field[0]
            speed = np.hypot(water[1], water[2])
            observed = np.stack((average_tiles(water[0]), average_tiles(speed)))
            idx = (pop["points"][0] // CELL).astype(int)
            counts = np.zeros((SIDE, SIDE))
            np.add.at(counts, (idx[:, 1], idx[:, 0]), 1)
            for shift in (-3, -2, -1, 0, 1, 2, 3):
                sig = np.roll(observed, shift, axis=-1)
                den = np.roll(blur(counts), shift, axis=-1)
                if shift:
                    edge = slice(None, shift) if shift > 0 else slice(shift, None)
                    sig[..., edge] = 0
                    den[:, edge] = 0
                templates.append(sig)
                origins.append(den / max(den.sum(), 1e-12))
    np.savez_compressed(BANK, signature=np.asarray(templates, dtype="f4"),
                        origin=np.asarray(origins, dtype="f4"))


class DestinationSearch(Switcher):
    def __init__(self, use_templates):
        super().__init__("persist")
        self.fresh_search = True
        self.cohorts = Cohorts(16)
        self.cohorts.personal_window_seconds = 720
        self.use_templates = use_templates
        if use_templates:
            with np.load(BANK) as bank:
                self.signature = bank["signature"]
                self.origin = bank["origin"]
        self.visits = np.zeros((SIDE, SIDE))
        self.new_contacts = set()
        self.first_water = np.zeros((2, 100, 100))
        self.latched = np.zeros((100, 100), bool)
        self.goals = {}
        self.weighted_map = np.ones((SIDE, SIDE))
        self.free_steps = 0

    def act(self, o):
        self.cohorts.observe(o)
        if self.wet is not True:
            fresh = (o.last >= 0) & ~self.latched
            self.first_water[0, fresh] = o.measured[0, fresh]
            self.first_water[1, fresh] = np.hypot(o.measured[1, fresh], o.measured[2, fresh])
            self.latched |= fresh
            for j in np.flatnonzero(o.visible):
                if j not in self.new_contacts:
                    self.new_contacts.add(j)
                    x, y = (o.seen_xy[j] / CELL).astype(int).clip(0, SIDE-1)
                    self.visits[y, x] += 1
        return super().act(o)

    def ordinary_mass(self, o):
        if o.t % 20 == 0:
            fraction = average_tiles(self.latched)
            hyd = np.stack([average_tiles(a) for a in self.first_water]) / np.maximum(fraction, 1/16)
            wet = blur(np.minimum(1, (hyd[0] + hyd[1]) / .3) * (fraction > 0))
            contact = blur(self.visits)
            if self.use_templates:
                residual = np.abs(self.signature - hyd[None]) / np.array([.25,.2])[None,:,None,None]
                mismatch = (np.minimum(residual, 3) * fraction).sum(axis=(1,2,3))
                mismatch /= max(1, (fraction * (hyd[0] + hyd[1] > .02)).sum())
                contact_cost = (self.visits * np.log(.0004 + self.origin)).sum(axis=(1,2))
                logits = -5*mismatch + 4*contact_cost/(1+self.visits.sum())
                weight = np.exp(logits-logits.max())
                self.weighted_map = np.average(self.origin, axis=0, weights=weight)
            else:
                self.weighted_map = np.ones((SIDE,SIDE)) / (SIDE*SIDE)
            self.weighted_map = self.weighted_map / max(self.weighted_map.max(), 1e-12)
            self.weighted_map += .25 * wet + .4 * contact/max(.5, contact.max())
        explored = (o.last >= 0)
        recovery = np.where(explored, .04 + .30 * np.clip((o.t-o.last)/480,0,1), 1.)
        return self.weighted_map.repeat(4,0).repeat(4,1)*recovery

    def navigate(self, o, ends, length, assigned, targets, mass):
        scores = np.zeros((4,8))
        priority = average_tiles(mass)
        local_ids, local_mask = candidate_footprints(ends,DIRECTIONS)
        footprint = (mass.ravel()[local_ids]*local_mask).sum(axis=-1)
        reserved = [targets[i].copy() for i in np.flatnonzero(assigned>=0)]
        for i in np.flatnonzero(assigned>=0):
            self.goals.pop(i,None)
        for i in np.flatnonzero(assigned<0):
            self.free_steps += 1
            pos = o.pos[i]
            goal = self.goals.get(i)
            if goal is None or o.t >= goal[1] or np.linalg.norm(goal[0]-pos) < 46:
                distance = np.linalg.norm(CENTRES-pos, axis=-1)
                value = (priority + .35*blur(priority)) / (1 + distance/115)
                for previous in reserved:
                    value *= .16 + .84*np.minimum(1, np.linalg.norm(CENTRES-previous,axis=-1)/150)
                # Reaching a cell means observing a new sector, not waiting at its centre.
                value[distance<35] *= .20
                index = np.unravel_index(value.argmax(),value.shape)
                goal = CENTRES[index], o.t+40
                self.goals[i]=goal
            reserved.append(goal[0])
            approach = (np.linalg.norm(pos-goal[0])-np.linalg.norm(ends[i]-goal[0],axis=-1))/50
            local = footprint[i]/max(.1, footprint[i].max())
            scores[i] = approach + .85*local
            scores[i, length[i]<5] = -1e5
        return scores

    def search(self,o,ends,length,assigned,targets):
        # Before identifying the water regime, keep the shared planner for all arms.
        if self.wet is None:
            return super().search(o,ends,length,assigned,targets)
        return self.navigate(o,ends,length,assigned,targets,self.ordinary_mass(o))

    def pressure_search(self,o,ends,length,assigned,targets):
        mass = self.cohorts.mass(o.t,1.4)
        mass += .03*np.clip((o.t-o.last)/500,0,1)
        return self.navigate(o,ends,length,assigned,targets,mass).argmax(axis=1)
