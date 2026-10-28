from dataclasses import replace
import hashlib
import numpy as np
from scale100_env import ScaleEnv
from sur_rescue.environment import pooled, frozen
from .belief import SignalBeliefEstimator, Parameters


class RouteEnv(ScaleEnv):
    """Changes only the two actor signal channels, never raw records or truth."""
    def __init__(self, source, config, method='Bpost', parameters=None, direction='forward'):
        self.route_method = method
        self.estimator = None if method == 'Braw' else SignalBeliefEstimator(method, parameters, direction)
        super().__init__(source, config)

    def reset(self):
        self._belief_key = None; self.belief_audit = []
        if self.estimator: self.estimator.reset(self.shape,dict(bed=self.source.bed,dx_m=self.dx))
        return super().reset()

    def view(self):
        raw = super().view()
        if self.estimator is None or self.config.task == 'E1': return raw
        key = (self.time_s,len(self.observations))
        if key != self._belief_key:
            self.estimator.update(self.signal,self.water,self.explored,self.observed_s,self.time_s,self.obstacles)
            post = self.estimator.posterior(); self.signal_belief = post['mean']
            self.belief_audit.append(dict(time_s=self.time_s, **post['diagnostics']))
            self._belief_key = key
        images = raw.images.copy(); pad = np.pad(self.signal_belief,2)
        global_signal = pooled(self.signal_belief[None])[0]
        for i,(r,c) in enumerate(self.positions):
            images[i,3] = pad[r:r+5,c:c+5]; images[i,7] = global_signal
        return replace(raw, images=frozen(images))

    def metrics(self):
        out = super().metrics()
        out['rescue_fraction'] = out['rescued']/out['targets'] if out['targets'] else 0.
        out['first_coverage_auc'] = float(np.mean([np.mean(s['explored']) for s in self.snapshots[:-1]]))
        out['first_discovery_s'] = next((e['time_s'] for e in self.events if e['type']=='discovered'),None)
        out['first_rescue_s'] = next((e['time_s'] for e in self.events if e['type']=='rescued'),None)
        out['unrescued'] = out['targets']-out['rescued']
        out['service_interruptions'] = sum(e['type']=='service_interrupted' for e in self.events)
        out['inference_max_s'] = max((r['seconds'] for r in self.belief_audit),default=0.)
        out['inference_max_residual'] = max((r['residual_inf'] for r in self.belief_audit),default=0.)
        return out
