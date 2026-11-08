"""Observation-only dispatch between three joint free-search plans."""
import numpy as np
from continuous44 import Continuous, DIRECTIONS, expected_ends
from windows import features

NAMES = ("base", "stripes", "flex")
RNG = np.random.default_rng(4411)
# Identical hypothetical origins for all options, independent of evaluation seed.
ORIGINS = RNG.uniform(0, 1000, (1536, 2))
COHORT = np.arange(len(ORIGINS)) % 12
CELLS = (ORIGINS / 10).astype(int)


class Fleet(Continuous):
    def __init__(self, selector, weights=None):
        super().__init__()
        self.selector = selector
        self.fitted = weights
        self.period = -1
        self.choice = 0
        self.decisions = []
        self.lanes = None

    def layout(self, o):
        axis = self.belief.axis
        k = int(np.argmax(np.abs(axis)))
        sign = 1 if axis[k] >= 0 else -1
        layouts = {}
        for variant in (1, 2):
            flex = int(np.argmax(sign * o.pos[:, k])) if variant == 2 else -1
            inside = [i for i in np.argsort(o.pos[:, 1-k]) if i != flex]
            layout = {}
            for rank, i in enumerate(inside):
                lo = rank * 1000 / len(inside)
                width = 1000 / len(inside)
                # Begin in the nearest of six transverse lanes; progress on each turn.
                lane = int(np.clip((o.pos[i, 1-k] - lo) * 6 / width, 0, 5))
                layout[int(i)] = [k, lo, width, lane, (rank + self.period) % 2, False]
            if flex >= 0:
                layout[flex] = [1-k, 975. if sign > 0 else 0., 25., 2,
                                int(o.pos[flex, 1-k] < 500), True]
            layouts[variant] = layout
        return layouts

    @staticmethod
    def sweep(pos, ends, lengths, layout, free):
        actions = np.zeros(4, int)
        for i in free:
            k, lo, width, lane, forward, edge = layout[i]
            def destination(lane, forward):
                point = np.empty(2)
                point[k] = 980 if forward else 20
                point[1-k] = lo + (lane + .5) * width / 6
                return point
            goal = destination(lane, forward)
            if np.linalg.norm(pos[i] - goal) < 48:
                forward = 1-forward
                lane = lane if edge else (lane+1) % 6
                layout[i][3:5] = [lane, forward]
                goal = destination(lane, forward)
            cost = np.linalg.norm(ends[i]-goal, axis=1)
            actions[i] = np.argmin(np.where(lengths[i] > 1, cost, np.inf))
        return actions

    def forecast(self, o, assigned, base, layouts):
        free = np.flatnonzero(assigned < 0)
        b = self.belief
        fall = b.fall[COHORT]
        keep = ((fall + b.personal_window_seconds >= o.t+10) &
                (b.unseen[COHORT, CELLS[:,1], CELLS[:,0]] > 0))
        orig = ORIGINS[keep]
        shift = b.shift[COHORT[keep]]
        drifting = np.isfinite(fall[keep])
        remain = fall[keep] + b.personal_window_seconds - o.t
        # Constant local flow for this short forecast. Trapping on cardinal
        # flow pressure fields uses the first edge reached, including corners.
        packet = []
        for step in range(1, 7):
            travel = shift + drifting[:, None] * b.field[1:] * (5*step)
            lim = np.where(travel >= 0, 1000-orig, -orig)
            frac = np.full_like(travel, np.inf)
            np.divide(lim, travel, where=travel != 0, out=frac)
            packet.append(orig + travel * np.minimum(1., frac.min(axis=1))[:, None])
        scores = []
        for variant in range(3):
            plans = {i: list(row) for i,row in layouts.get(variant, {}).items()}
            pos = o.pos.copy()
            captured = np.zeros(len(orig), bool)
            for t, people in enumerate(packet):
                ends, lengths = expected_ends(pos)
                if variant:
                    acts = self.sweep(pos, ends, lengths, plans, free)
                else:
                    acts = base.copy()
                    for i in free:
                        tail = self.plans_search.get(i, (0,None,[]))[2]
                        if t and len(tail):
                            acts[i] = tail[min(t-1, len(tail)-1)]
                pos[free] = ends[free, acts[free]]
                for i in free:
                    delta = people-pos[i]
                    captured |= ((delta*delta).sum(1) <= 400) & ((delta @ DIRECTIONS[acts[i]]) >= 0) & (remain >= (t+1)*5+10)
            scores.append(int(captured.sum()))
        return scores

    def pressure_search(self, o, ends, length, assigned, targets):
        period = (o.t-60)//120
        boundary = period >= 0 and period != self.period
        if boundary or self.choice == 0:
            base = super().pressure_search(o, ends, length, assigned, targets)
        if boundary:
            self.period = period
            layouts = self.layout(o)
            scores = self.forecast(o, assigned, base, layouts)
            x = features(o) + [float((assigned < 0).mean())] + [s/20 for s in scores]
            if self.selector in NAMES:
                pick = NAMES.index(self.selector)
            elif self.selector == "rule":
                pick = int(np.argmax(scores))
            else:
                w = self.fitted
                z = (np.asarray(x)-w["mean"])/w["scale"]
                pick = int(np.argmax(w["intercept"] + z @ w["coef"]))
            self.decisions.append(dict(t=int(o.t), x=x, forecast=scores, choice=pick))
            self.choice = pick
            self.lanes = layouts.get(pick)
            if pick:
                self.plans_search.clear()
        if self.choice == 0:
            return base
        return self.sweep(o.pos, ends, length, self.lanes, np.flatnonzero(assigned < 0))
