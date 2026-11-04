"""Read-only accepted-step audit for observers also called at output boundaries."""
import math


class AcceptedStepAudit:
    def __init__(self, solver, values):
        self.values = values
        self.last_steps = int(solver.steps)
        if len(values) != self.last_steps:
            raise ValueError('Accepted-step history length differs from solver steps')
        if any(not math.isfinite(dt) or dt <= 0 for dt in values):
            raise ValueError('Invalid accepted timestep in restart history')
        if abs(math.fsum(values)-solver.time) > 1e-10*max(1., abs(solver.time)):
            raise ValueError('Accepted-step history duration differs from solver time')

    def __call__(self, solver):
        steps = int(solver.steps)
        # advance() observes the initial state as well as each accepted step.
        if steps == self.last_steps:
            return
        if steps != self.last_steps+1:
            raise ValueError('Missed or reversed accepted step in observer')
        dt = float(solver.last_timestep['dt_s'])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid accepted timestep')
        self.values.append(dt)
        self.last_steps = steps
