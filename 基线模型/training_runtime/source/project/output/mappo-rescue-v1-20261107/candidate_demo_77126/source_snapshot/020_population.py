"""Portable imports for the active collision, sensing and delivery implementation."""
import numpy as np
START=np.array([[5,5],[995,5],[5,995],[995,995]],dtype=float)
EPS=1e-7
COUNT=64
DURATION=2400
RULES={"personal_window_after_fall_s":720}
MOVES=np.array([[0,1],[0,-1],[-1,0],[1,0],[-1,1],[-1,-1],[1,1],[1,-1],[0,0]],float)
MOVES[:8]/=np.linalg.norm(MOVES[:8],axis=1,keepdims=True)

def sector(points, position, heading, radius):
    d = np.asarray(points) - position
    return (np.sum(d*d, axis=-1) <= radius*radius+1e-9) & (d @ heading >= -1e-9)

def segment_circle(p, d, center, radius):
    """First intersection in [0,1], including an initially occupied service disk."""
    offset = p-center
    if offset@offset <= radius*radius+1e-9: return 0.
    aa = d@d
    if aa < 1e-15: return None
    bb = 2*(offset@d); cc = offset@offset-radius*radius
    disc = bb*bb-4*aa*cc
    if disc < -1e-9: return None
    x = (-bb-np.sqrt(max(0.,disc)))/(2*aa)
    return float(np.clip(x,0,1)) if -1e-9 <= x <= 1+1e-9 else None
