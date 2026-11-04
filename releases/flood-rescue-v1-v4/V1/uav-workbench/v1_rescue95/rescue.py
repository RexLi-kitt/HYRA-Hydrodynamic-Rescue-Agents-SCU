"""Serve observed people; return -1 for aircraft available to patrol."""
import math
import numpy as np
from flood_dqn.env import EPS
from rescue_assignment_dqn.controller import navigation


class Rescue:
    def __init__(self):
        self.previous = {}
        self.retry = {}
        self.best = {}
        self.seen_at = {}
        self.last_pos = {}

    def actions(self, env):
        t = env.step_no
        actions = np.full(4, -1, dtype=int)
        pending = np.flatnonzero(env.found & ~env.rescued)
        goals = env.targets[pending]
        assignments = {}
        pairs = []
        for j in pending:
            cell = env.cells[j]
            goal = env.targets[j]
            stamp = int(env.stamps[cell])
            h, u, v = env.water[:,cell]
            good = h <= np.float32(.08) and np.hypot(u,v) <= np.float32(.3)
            self.seen_at.setdefault(j,t)
            if env.owner[j] >= 0:
                assignments[int(env.owner[j])] = j
                continue
            for i,p in enumerate(env.p):
                distance = np.linalg.norm(p-goal)
                key = (i,j)
                if self.previous.get(i) == j:
                    if np.array_equal(self.last_pos[i],p) and distance<=10+EPS:
                        # A submitted hover produced no progress: wait for a new good measurement.
                        self.retry[key] = (t+6, t)
                    best,since = self.best[key]
                    if distance < best-1:
                        self.best[key] = (distance,t)
                    elif distance>10+EPS and t-since>=3:
                        self.retry[key] = (t+4,t)
                due,when = self.retry.get(key,(-1,-1))
                if t<due and not (good and stamp>when):continue
                # Stale unsafe targets can be retried locally while searching. Fresh unsafe
                # readings never justify dispatch and cannot prolong an active hover.
                if not good and (t-stamp<=6 or distance>60):continue
                eta = max(0,math.ceil((distance-10-EPS)/50))
                if eta+2>300-t:continue
                pairs.append((eta, distance-5*(self.previous.get(i)==j),self.seen_at[j],i,j))
        assigned_goals=set(assignments.values())
        for _,_,_,i,j in sorted(pairs):
            if i not in assignments and j not in assigned_goals:
                assignments[i]=j
                assigned_goals.add(j)
        for i,j in assignments.items():
            distance=np.linalg.norm(env.p[i]-env.targets[j])
            if self.previous.get(i)!=j:self.best[i,j]=(distance,t)
            actions[i]=8 if distance<=10+EPS else navigation(env.p[i],env.targets[j],goals)
            self.last_pos[i]=env.p[i].copy()
        self.previous=assignments
        return actions
