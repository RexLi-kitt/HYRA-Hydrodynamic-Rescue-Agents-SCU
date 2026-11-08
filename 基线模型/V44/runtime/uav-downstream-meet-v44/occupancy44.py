"""Predict where currently unseen people can still receive a delivery in 30–60 seconds."""
import numpy as np
from continuous44 import Continuous

SIDE = 20
yy, xx = np.mgrid[:SIDE,:SIDE]
TILES = np.column_stack((xx.ravel(), yy.ravel())) * 50 + 25
line = np.arange(25,1000,50)
EDGE = np.concatenate([np.column_stack((np.full(20,x),line)) for x in (0,1000)] +
                      [np.column_stack((line,np.full(20,y))) for y in (0,1000)])
XY = np.vstack((TILES,EDGE))
EDGE_TILE = (EDGE.clip(0,999)//50).astype(int)
EDGE_CELL = (np.repeat(EDGE,5,axis=0) + np.vstack((
    np.column_stack((np.zeros(200),np.tile(np.arange(-20,21,10),40))),
    np.column_stack((np.tile(np.arange(-20,21,10),40),np.zeros(200)))))).clip(0,999)//10
EDGE_CELL = (100*EDGE_CELL[:,1]+EDGE_CELL[:,0]).astype(int)
IS_EDGE = np.arange(480)>=400


def regions(xy):
    ix,iy = (xy.clip(0,999.99)//50).astype(int).T
    result = 20*iy+ix
    # Match the retained first contact; apply x convention at exact corners.
    for side,hit,offset in ((2,xy[:,1]<=1e-7,ix),(3,xy[:,1]>=1000-1e-7,ix),
                            (0,xy[:,0]<=1e-7,iy),(1,xy[:,0]>=1000-1e-7,iy)):
        result[hit] = 400+20*side+offset[hit]
    return result


def pool(grid):
    interior = grid.reshape(20,5,20,5).mean((1,3)).ravel()
    edge = grid.ravel()[EDGE_CELL].reshape(80,5).mean(1)
    return np.r_[interior,edge]


def predictors(o, belief):
    age = o.t-o.last
    visited = pool(o.last>=0)
    fresh = pool((o.last>=0)&(age<=90))
    axis = belief.axis
    along = (XY-500)@axis/500
    time = o.t/2400
    elapsed = np.log1p(o.t/240)
    edge = IS_EDGE.astype(float)
    # Normal to each specific edge, so adjacent sides have distinct flux.
    normal = np.r_[np.zeros(400),np.repeat([-axis[0],axis[0],-axis[1],axis[1]],20)]
    contacts = np.zeros(480)
    known = o.seen_at >= 0
    if known.any():
        xy = o.seen_xy[known]
        drop = np.exp(-(o.t-o.seen_at[known])/180.)
        count = np.bincount(regions(xy),weights=drop,minlength=480)
        image = np.pad(count[:400].reshape(20,20),1)
        contacts[:400] = sum(image[i:i+20,j:j+20] for i in range(3) for j in range(3)).ravel()
        for k in range(4):
            sl = slice(400+k*20,420+k*20)
            contacts[sl] = np.convolve(count[sl],[.5,1,.5],mode="same")
    speed = np.linalg.norm(belief.field[1:])
    drift = np.mean(np.linalg.norm(belief.shift,axis=1))/500
    downstream = np.maximum(0,normal)
    return np.column_stack(np.broadcast_arrays(1.,time,elapsed,known.mean(),speed,drift,along,
        along*drift,edge,normal,edge*elapsed,downstream*elapsed,downstream*drift,
        np.abs(along),visited,fresh,visited*edge,fresh*edge,np.log1p(contacts)))


def heat_from_counts(counts):
    grid = np.repeat(np.repeat(counts[:400].reshape(20,20)/25,5,0),5,1).ravel()
    np.add.at(grid,EDGE_CELL,np.repeat(counts[400:]/5,5))
    return grid


class Occupancy(Continuous):
    def __init__(self, weights):
        super().__init__()
        self.occupancy_weights = np.asarray(weights)
        self.forecast_time = -1
        self.forecasts = 0

    def fields(self,o):
        fixed,mobile = super().fields(o)
        if o.t < 60:
            return fixed,mobile
        if self.forecast_time < 0 or o.t-self.forecast_time >= 30:
            design = predictors(o,self.belief)
            self.counts = np.exp(np.clip(design@self.occupancy_weights,-16,4))
            self.heat = heat_from_counts(self.counts)
            self.forecast_time = o.t
            self.forecasts += 1
        # One native unit corresponds to the initial uniform density per cell.
        # Moving search retains the inherited continuous displacement correction.
        addition = .5*self.heat*10000/len(o.seen_at)
        return fixed+addition,mobile
