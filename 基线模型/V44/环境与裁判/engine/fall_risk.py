"""Instantaneous flow-induced first-fall hazard and cumulative survival; SI inputs."""
import numpy as np


def response(depth_m, speed_m_s):
    x, y = np.asarray(speed_m_s), np.asarray(depth_m)
    z = x*x*(1+.25*y/(1+x+y))
    return np.where(y>0,z/(1+z),0.)


def probability(seconds, depth_m, speed_m_s, coefficient):
    if seconds<0 or coefficient<0 or np.any(np.asarray(speed_m_s)<0) or np.any(np.asarray(depth_m)<0):
        raise ValueError('Nonnegative SI exposure required')
    return -np.expm1(-seconds*coefficient*response(depth_m,speed_m_s))
