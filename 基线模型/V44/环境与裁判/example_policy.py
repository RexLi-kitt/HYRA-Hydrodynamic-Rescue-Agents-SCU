"""A minimal observation-only integration example. Replace make_policy with your own."""
import numpy as np


class ExamplePolicy:
    def act(self,o):
        # The four aircraft travel towards the centre along the initial diagonals.
        # For a currently visible target already within 1 m, attempt the 10 s drop.
        actions=np.array([6,4,7,5]);targets=np.full(4,-1)
        used=set()
        for i in range(4):
            for j in np.flatnonzero(o.visible):
                if j not in used and np.linalg.norm(o.pos[i]-o.seen_xy[j])<=1:
                    actions[i]=8;targets[i]=j;used.add(int(j));break
        return actions,targets


def make_policy():
    return ExamplePolicy()
