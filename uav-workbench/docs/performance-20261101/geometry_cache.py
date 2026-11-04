"""Bounded exact-input memoization of public, deterministic geometry only."""
from functools import lru_cache
import numpy as np


def array_key(value):
    a = np.asarray(value)
    if a.shape != (2,) or a.dtype.kind != 'f' or not np.isfinite(a).all():
        return None
    return a.dtype.str, a.tobytes()


def install():
    import flood1km.env as env
    import observed_planner as planner
    import coverage_controller as coverage
    baseline = (env.min_separation, planner.min_separation, coverage.min_separation, planner.view)

    @lru_cache(maxsize=65536)
    def separation_cached(keys, duration, speed):
        arrays = [np.frombuffer(raw, dtype=dtype) for dtype, raw in keys]
        return baseline[0](*arrays, duration=float.fromhex(duration), speed=float.fromhex(speed))

    def separation(a0, a1, b0, b1, duration=5., speed=10.):
        keys = tuple(array_key(v) for v in (a0, a1, b0, b1))
        if None in keys or not np.isfinite(duration) or not np.isfinite(speed):
            return baseline[0](a0, a1, b0, b1, duration, speed)
        return separation_cached(keys, float(duration).hex(), float(speed).hex())

    @lru_cache(maxsize=512)
    def view_cached(p, heading):
        result = baseline[3](np.frombuffer(p[1], dtype=p[0]), np.frombuffer(heading[1], dtype=heading[0]))
        result.flags.writeable = False
        return result

    def view(p, heading):
        keys = (array_key(p), array_key(heading))
        if None in keys:
            return baseline[3](p, heading)
        return view_cached(*keys)

    def enable():
        env.min_separation = planner.min_separation = coverage.min_separation = separation
        planner.view = view

    def disable():
        env.min_separation, planner.min_separation, coverage.min_separation, planner.view = baseline

    def clear():
        separation_cached.cache_clear(); view_cached.cache_clear()

    def stats():
        return dict(separation=separation_cached.cache_info()._asdict(), view=view_cached.cache_info()._asdict())

    return dict(enable=enable, disable=disable, clear=clear, stats=stats)
