"""Exact cached window geometry, sharing repeated windows across metric cases."""
import numpy as np


class CompiledMetricCases:
    def __init__(self,cases,row_times,row_targets,curve_size=26):
        lookup={(float(t),int(r)):i for i,(t,r) in enumerate(zip(row_times,row_targets))}
        windows=[];identities={};first6=[];common=[]
        def window(row,begin,duration):
            key=(row,float(begin),duration)
            if key not in identities:identities[key]=len(windows);windows.append(key)
            return identities[key]
        for case in cases:
            early=[];full=[]
            for order,travel in case['orders']:
                a=lookup[float(case['now_s']),order[0]];b=lookup[float(case['now_s']),order[1]]
                start=travel[0];early.append(window(a,start,6))
                full.append([window(a,start,12),window(b,start+6+travel[1],12),window(b,start+12+travel[1],12)])
            first6.append(early);common.append(full)
        self.first6=np.array(first6);self.common=np.array(common);self.window_count=len(windows)
        self.prefer1=np.array([c['orders'][1][0]<c['orders'][0][0] for c in cases])
        row,begin,duration=np.array(windows).T;row=row.astype(int);end=begin+duration
        if np.any(begin<0) or np.any(end>2*(curve_size-1)+1e-9):raise ValueError('Window outside forecast')
        def endpoint(t):
            lo=np.floor(t/2).astype(int);hi=np.minimum(lo+1,curve_size-1)
            return row*curve_size+lo,row*curve_size+hi,t/2-lo
        self.left=endpoint(begin);self.right=endpoint(end)
        node=np.floor(begin/2).astype(int)[:,None]+np.arange(8)
        self.valid=(node*2>=begin[:,None]) & (node*2<=end[:,None])
        self.nodes=row[:,None]*curve_size+np.clip(node,0,curve_size-1)
        self.curve_size=curve_size

    def scores(self,curves):
        curves=np.asarray(curves)
        if curves.ndim==2:curves=curves[None]
        if curves.shape[-1]!=self.curve_size:raise ValueError('Unexpected curve size')
        flat=curves.reshape(len(curves),-1)
        def endpoint(spec):
            lo,hi,w=spec;return flat[:,lo]+(flat[:,hi]-flat[:,lo])*w
        internal=np.max(np.where(self.valid,flat[:,self.nodes],-np.inf),axis=-1)
        values=np.maximum(np.maximum(endpoint(self.left),endpoint(self.right)),internal)
        common=values[:,self.common];abort=values[:,self.first6]>.8
        second=np.where(abort,common[:,:,:,1],common[:,:,:,2]);first=common[:,:,:,0]
        count=(first<=.8).astype(np.int8)+(second<=.8).astype(np.int8);margin=np.minimum(.8-first,.8-second)
        choose=(count[:,:,1]>count[:,:,0]) | ((count[:,:,1]==count[:,:,0]) &
            ((margin[:,:,1]>margin[:,:,0]+1e-12) | ((abs(margin[:,:,1]-margin[:,:,0])<=1e-12)&self.prefer1)))
        return dict(count=count,margin=margin,common_maxima=common,chosen=choose.astype(np.int8))
