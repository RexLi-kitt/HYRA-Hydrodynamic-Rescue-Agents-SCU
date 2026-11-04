"""A causal, optional action constraint based only on observed failed motion.

This is a disclosed controller addition, not a change to collision physics or
a claim that the neural policy improved. Service masks always take precedence.
"""
import numpy as np


class RetryGuard:
    def __init__(self, threshold=3, cooldown=5):
        self.threshold = threshold
        self.cooldown = cooldown
        self.previous = None
        self.actions = None
        self.free = None
        self.streak = np.zeros(4, dtype=int)
        self.banned = np.zeros((4, 8), dtype=int)
        self.constrained_steps = 0

    def mask(self, obs, available, phases):
        obs = np.asarray(obs)
        if obs.shape != (4, 1468):
            raise ValueError('Frozen public observation schema required')
        pos = obs[0, 1326:1346].reshape(4, 5)[:, :2].astype(float) * 1000
        self.banned = np.maximum(self.banned - 1, 0)
        out = np.array(available, dtype=bool, copy=True)
        now_free = np.array([p == 'learned' for p in phases])
        for i in range(4):
            if not now_free[i]:
                self.streak[i] = 0
                self.banned[i] = 0
                continue
            if self.previous is not None:
                moved = np.linalg.norm(pos[i] - self.previous[i]) >= .001
                failed = self.free[i] and self.actions[i] < 8 and not moved
                self.streak[i] = self.streak[i] + 1 if failed else 0
                if moved:
                    self.banned[i] = 0
                if failed and self.streak[i] >= self.threshold:
                    self.banned[i, self.actions[i]] = self.cooldown
            out[i, :8] &= self.banned[i] == 0
            # After repeated failed movement, use another legal direction if
            # possible. A previously deliberate hover alone never activates it.
            if self.banned[i].any() and out[i, :8].any():
                out[i, 8] = False
            if not out[i].any():
                out[i] = available[i]
        self.previous = pos.copy()
        self.free = now_free
        changed = np.any(out != available, axis=1)
        self.constrained_steps += int(changed.sum())
        return out, dict(constrained_agents=int(changed.sum()),
                         streak=self.streak.tolist(), banned=self.banned.tolist())

    def remember(self, actions):
        self.actions = np.asarray(actions, dtype=int).copy()
