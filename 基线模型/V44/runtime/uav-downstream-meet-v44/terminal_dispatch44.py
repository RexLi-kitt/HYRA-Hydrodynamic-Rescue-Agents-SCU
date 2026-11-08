"""Evaluate one real reassignment using separate end-of-mission and timing predictions."""
import numpy as np
from destinations import DestinationSearch
from early_delivery44 import QuickDispatch, state

STARTS=(60,180,300,480,720)
END=960


class BaselineCapture(DestinationSearch):
    """Keep the incumbent assignment just before QuickDispatch computes its alternative."""
    def assign(self,o):
        jobs,goals=super().assign(o)
        if self.arm:
            self.incumbent=(jobs.copy(),goals.copy(),self.since.copy())
        return jobs,goals


def proposal_features(o,prior,old,new,old_goals,new_goals):
    changed=old!=new
    before=set(old[old>=0])
    after=set(new[new>=0])
    def describe(jobs,goals):
        ids=np.flatnonzero(changed&(jobs>=0))
        if not len(ids):
            return [0.]*4
        js=jobs[ids]
        return [float(np.linalg.norm(o.pos[ids]-goals[ids],axis=1).sum()/200),
                float(np.minimum(o.t-o.seen_at[js],240).sum()/240),
                float(np.linalg.norm(o.seen_v[js],axis=1).sum()),
                float(o.visible[js].sum())]
    return np.r_[state(o), changed.sum(), len(before-after),len(after-before),
                 np.sum(changed&(prior==old)&(old>=0)),
                 np.subtract(describe(old,old_goals),describe(new,new_goals))]


class TerminalDispatch(QuickDispatch,BaselineCapture):
    def __init__(self,fit=None,force_at=None):
        super().__init__(None)
        self.terminal_fit=fit
        self.force_at=force_at
        self.tested=set()
        self.accepted=None
        self.events=[]

    def act(self,o):
        # The selector operates on actual changes, not on clock-based macro-actions.
        return DestinationSearch.act(self,o)

    def assign(self,o):
        slot=int(np.searchsorted(STARTS,o.t,side="right")-1)
        eligible=(0<=slot and o.t<END and slot not in self.tested and self.accepted is None)
        self.arm=2 if eligible else 0
        prior=self.track.copy()
        previous_changes=self.dispatch_changes
        new,goals=super().assign(o)
        if not eligible:
            return new,goals
        old,old_goals,old_since=self.incumbent
        if np.array_equal(old,new):
            return new,goals
        self.tested.add(slot)
        x=proposal_features(o,prior,old,new,old_goals,goals)
        pred=None
        take=(self.force_at==o.t)
        if self.terminal_fit is not None:
            w=self.terminal_fit
            z=np.clip((x-w["mean"])/w["scale"],-4.,4.)
            pred=w["bias"]+z@w["coef"]
            take=bool(pred[0]>=w["min_gain"] and (pred[0]>=1 or pred[1]>0))
        self.events.append(dict(t=int(o.t),slot=slot,x=x.tolist(),base=old.tolist(),proposed=new.tolist(),
                                prediction=None if pred is None else pred.tolist(),taken=take))
        if take:
            self.accepted=int(o.t)
            return new,goals
        self.track=old.copy()
        self.since=old_since
        self.dispatch_changes=previous_changes
        return old,old_goals


def build_terminal(fit=None,force_at=None):
    from model import load_model
    agent=load_model()
    agent.sparse=TerminalDispatch(fit,force_at)
    agent.sparse.cohorts.personal_window_seconds=720
    return agent
