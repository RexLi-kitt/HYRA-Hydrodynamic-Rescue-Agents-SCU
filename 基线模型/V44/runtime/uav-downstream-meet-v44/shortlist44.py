"""Exact V44 beam generation, returning its final candidates and original scores."""
from beam16 import expected_ends, DIRECTIONS, np

def shortlist(self,o,i,fields,reserved):
    horizon=self.horizon;axis=self.belief.axis;velocity=self.belief.field[1:]
    positions=o.pos[i,None].copy();values=np.zeros(1);tracks=np.empty((1,0),int)
    seen=[np.zeros((1,10000),bool),np.zeros((1,10000),bool)]
    for step in range(horizon):
        ends,length=expected_ends(positions)
        # Model the displacement of targets after the current observation.
        g0,id0,m0=self.candidates(ends,DIRECTIONS,fields[0],seen[0],False,np.zeros(2),axis)
        g1,id1,m1=self.candidates(ends,DIRECTIONS,fields[1],seen[1],True,(step+1)*5*velocity,axis)
        gain=g0+g1
        scale=max(.1,float(gain.max()))
        gain+=self.cross*scale*(1-np.abs(DIRECTIONS@axis))
        for reserved_path in reserved:
            # Reserve nearby search territory as well as the other aircraft's precise path.
            centre=reserved_path[min(step,len(reserved_path)-1)]
            dist=np.linalg.norm(ends-centre,axis=-1)
            gain-=.9*scale*np.maximum(0,1-dist/150)
        scores=values[:,None]+(.97**step)*gain
        scores[length<39.9]=-1.e10
        sorted_indices=np.argsort(-scores.ravel(),kind="stable")
        parent,actions=np.unravel_index(sorted_indices,scores.shape)
        good=scores[parent,actions]>-1.e9
        parent,actions=parent[good],actions[good]
        if not len(parent):break
        # Retain competing ways into the same area but avoid flooding the beam with loops.
        ends_sorted=ends[parent,actions]
        keys=np.c_[np.rint(ends_sorted/8).astype(int),actions]
        _,unique=np.unique(keys,axis=0,return_index=True)
        keep=np.sort(unique)[:self.width];parent=parent[keep];actions=actions[keep]
        for b,(ids,mask) in enumerate(((id0,m0),(id1,m1))):
            seen[b]=seen[b][parent].copy()
            jj,cc=np.nonzero(mask[parent,actions])
            seen[b][jj,ids[parent[jj],actions[jj],cc]]=True
        positions=ends[parent,actions];values=scores[parent,actions]
        tracks=np.column_stack((tracks[parent],actions))
    if not tracks.shape[1]:return np.array([[int(np.argmax(expected_ends(o.pos[i])[1]))]]),np.zeros(1)
    return tracks,values
