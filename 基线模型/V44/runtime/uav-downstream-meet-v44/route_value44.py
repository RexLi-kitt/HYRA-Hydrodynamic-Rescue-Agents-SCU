"""Choose between two available V44 searches using a fitted paired delivery difference."""
from pathlib import Path
import json
import numpy as np
from continuous44 import Continuous

WEIGHTS = Path(__file__).with_name("route-value-weights.json")
SLOTS = (180, 510, 840)
FEATURES = ("bias", "score_gap", "along_flow", "new_ground", "team_separation",
            "edge_clearance", "time", "observed_speed")


class RouteValue(Continuous):
    def __init__(self, mode="learned", target=None):
        super().__init__()
        self.route_trial_mode = mode
        self.fork_slot = target
        self.decisions = {}
        if mode == "learned":
            weights = json.loads(WEIGHTS.read_text(encoding="utf-8"))
            if weights["features"] != list(FEATURES):
                raise ValueError("Incompatible V44 route value weights")
            self.route_weights = np.asarray(weights["weights"])

    def select_route(self,o,i,routes,poses,scores,best):
        slot = next((s for s in SLOTS if s <= o.t < s+60 and s not in self.decisions), None)
        if slot is None:
            return routes[best].tolist()
        other = np.flatnonzero(routes[:,0] != routes[best,0])
        if not len(other):
            return routes[best].tolist()
        alt = int(other[np.argmax(scores[other])])
        # Use only the executable commitment, not the uncommitted tail of a beam.
        paths = poses[[best,alt], :self.commit]
        end = paths[:,-1]
        cell = np.clip((paths/10).astype(int),0,99)
        fresh = (o.last[cell[:,:,1],cell[:,:,0]] < 0).mean(axis=1)
        others = np.delete(o.pos,i,axis=0)
        separation = np.linalg.norm(paths[:,:,None]-others,axis=-1).min(axis=2).mean(axis=1)
        edges = np.minimum(end,1000-end).min(axis=1)
        x = np.array([1., (scores[alt]-scores[best])/max(1.,abs(scores[best])),
                      (end[1]-end[0])@self.belief.axis/150, fresh[1]-fresh[0],
                      (separation[1]-separation[0])/150, (edges[1]-edges[0])/150,
                      o.t/900, np.linalg.norm(self.belief.field[1:])])
        take = ((self.route_trial_mode=="fork" and self.fork_slot==slot) or self.route_trial_mode=="always" or
                (self.route_trial_mode=="learned" and self.route_weights@x > 0))
        self.decisions[slot] = dict(t=int(o.t), drone=int(i), x=x.tolist(), take=bool(take),
                                    first=[int(routes[best,0]),int(routes[alt,0])])
        return routes[alt if take else best].tolist()
