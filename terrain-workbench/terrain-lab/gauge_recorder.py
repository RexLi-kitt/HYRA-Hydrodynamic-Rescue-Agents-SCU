"""Per-accepted-step gauge records for one solver run, without extra output steps."""
from gauge_sampling import sample_gauge
from gauge_statistics import summarize_gauge_window

class GaugeRecorder:
    def __init__(self, solver, points, *, origin_center_m, run_id):
        if not isinstance(run_id,str) or not run_id:raise ValueError('run_id is required')
        if not points or any(not isinstance(k,str) or not k for k in points):raise ValueError('Named gauge points required')
        self.solver=solver;self.run_id=run_id;self.origin=tuple(origin_center_m)
        self.points={name:tuple(position) for name,position in points.items()}
        for position in self.points.values():sample_gauge(solver,*position,origin_center_m=self.origin)
        self.records={name:[] for name in self.points};self.last=None
    def __call__(self, solver):
        if solver is not self.solver:raise ValueError('Recorder belongs to a different solver')
        state=(float(solver.time),solver.steps)
        if self.last is not None:
            if state[0]<self.last[0] or state[1]<self.last[1]:raise ValueError('Scenario reset requires a new recorder')
            if state==self.last:return
            if state[0]<=self.last[0]:raise ValueError('Gauge time must increase')
        rows={name:dict(sample_gauge(solver,*position,origin_center_m=self.origin),run_id=self.run_id,gauge_id=name)
              for name,position in self.points.items()}
        for name,row in rows.items():self.records[name].append(row)
        self.last=state
    def summarize(self, name, *, start_s, end_s):
        return dict(summarize_gauge_window(self.records[name],start_s=start_s,end_s=end_s),run_id=self.run_id,gauge_id=name)
