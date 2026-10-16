"""Causal offline innovation score; never called by the controller."""
import numpy as np


def score_innovations(records, history_length=3, max_age_s=60, scale_floor_m=.02):
    histories={};seen=set();output=[]
    for record in sorted(records,key=lambda r:(r["arrival_time_s"],r["valid_time_s"],r["region_id"])):
        region=record["region_id"];key=(region,record["valid_time_s"])
        if record.get("residual_m") is None or key in seen:continue
        seen.add(key)
        previous=[r for r in histories.get(region,[]) if r["valid_time_s"]<record["valid_time_s"]
            and 0<=record["arrival_time_s"]-r["valid_time_s"]<=max_age_s][-history_length:]
        if len(previous)==history_length:
            values=np.array([r["residual_m"] for r in previous])
            innovation=record["residual_m"]-values.mean()
            variance=max(float(values.var()),scale_floor_m**2)
            output.append(dict(region=region,time_s=record["valid_time_s"],arrival_s=record["arrival_time_s"],
                signed_residual_m=record["residual_m"],absolute_residual_m=abs(record["residual_m"]),
                innovation_m=float(innovation),innovation_score=float(innovation**2/variance),
                history_times=[r["valid_time_s"] for r in previous]))
        histories.setdefault(region,[]).append(record)
    return output
