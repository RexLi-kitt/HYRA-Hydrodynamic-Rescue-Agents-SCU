"""Metric flight-time task geometry and complete service scoring.

All pairs and origins are fixed before reading water levels. Straight-line
flight is a declared task abstraction; the water threshold constrains service.
"""
from itertools import combinations
import numpy as np

from .lead import TARGETS
from .scaled_hydraulics import sampling_xy
from .numerical_stability import interval_maximum


def task_geometry(length_m, origins=(44,45,54,55), speeds=(5.,10.)):
    xy=sampling_xy(length_m);eligible=[];excluded=[]
    for a,b in combinations(sorted(TARGETS),2):
        between=float(np.linalg.norm(xy[a]-xy[b]))
        for origin in origins:
            for speed in speeds:
                if not np.isfinite(speed) or speed<=0:raise ValueError('Invalid flight speed')
                first_a=float(np.linalg.norm(xy[origin]-xy[a])/speed)
                first_b=float(np.linalg.norm(xy[origin]-xy[b])/speed)
                travel=between/speed
                item=dict(targets=[a,b],origin=origin,speed_m_per_s=speed,
                    cross_half=bool((a//10<5)!=(b//10<5)),
                    orders=[([a,b],[first_a,travel]),([b,a],[first_b,travel])],
                    worst_duration_s=max(first_a,first_b)+travel+24.)
                (eligible if item['worst_duration_s']<=48.+1e-12 else excluded).append(item)
    return eligible,excluded


def metric_cases(length_m,anchor_s,origins=(44,45,54,55),speeds=(5.,10.),offsets=(0,6,12,18,24,30)):
    eligible,excluded=task_geometry(length_m,origins,speeds)
    cases=[dict(case,layout=i,now_s=anchor_s+offset,decision_offset_s=offset)
           for i,case in enumerate(eligible) for offset in offsets if offset+case['worst_duration_s']<=60.+1e-12]
    return cases,dict(candidate_layouts=len(eligible)+len(excluded),eligible_layouts=len(eligible),
        excluded_layouts=len(excluded),available_cases=len(cases),
        excluded_cases=(len(eligible)+len(excluded))*len(offsets)-len(cases),
        deadline_excluded_cases=len(eligible)*len(offsets)-len(cases))


def metric_order(times,depth,order,travel_s,now_s,deadline_s):
    if len(order)!=2 or len(set(order))!=2 or len(travel_s)!=2 or np.any(np.asarray(travel_s)<0):
        raise ValueError('Invalid two-target action')
    clock=now_s;starts=[];maxima=[];succeeded=[]
    for target,travel in zip(order,travel_s):
        begin=clock+travel
        if begin+12>deadline_s+1e-10:raise ValueError('Action exceeds deadline')
        maximum=interval_maximum(times,depth[:,target],begin,begin+12)
        early=interval_maximum(times,depth[:,target],begin,begin+6)
        starts.append(float(begin));maxima.append(float(maximum));succeeded.append(target if maximum<=.8 else None)
        clock=begin+(6 if early>.8 else 12)
    return dict(count=sum(r is not None for r in succeeded),margin=min(.8-m for m in maxima),
                starts=starts,maxima=maxima,succeeded=succeeded)


def metric_scores(times,depth,cases,anchor_s):
    return [[metric_order(times,depth,*order,c['now_s'],anchor_s+60) for order in c['orders']] for c in cases]


def choose_metric(scores,cases):
    chosen=[]
    for row,case in zip(scores,cases):
        a,b=row;winner=0
        if b['count']>a['count']:winner=1
        elif b['count']==a['count']:
            if b['margin']>a['margin']+1e-12:winner=1
            elif abs(b['margin']-a['margin'])<=1e-12 and case['orders'][1][0]<case['orders'][0][0]:winner=1
        chosen.append(winner)
    return np.array(chosen,dtype=int)
