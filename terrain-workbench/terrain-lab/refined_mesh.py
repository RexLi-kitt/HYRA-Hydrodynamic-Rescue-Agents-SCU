"""Static two-level square leaf mesh; one shared flux per (possibly split) face."""
import numpy as np


def inland_mask(classes,names,solid=None,buffer=1):
    classes=np.asarray(classes)
    targets=[int(k) for k,v in names.items() if v in ('river','lake','wetland')]
    ocean=np.isin(classes,[int(k) for k,v in names.items() if 'ocean' in v])
    selected=np.isin(classes,targets)&~ocean
    mask=selected.copy()
    for _ in range(buffer):
        p=np.pad(mask,1)
        mask=np.logical_or.reduce([p[z:z+classes.shape[0],x:x+classes.shape[1]] for z in range(3) for x in range(3)])&~ocean
    if solid is not None:mask&=~np.asarray(solid,bool)
    return selected,mask


class LeafMesh:
    def __init__(self,mask,dx,boundary='closed'):
        self.mask=np.asarray(mask,bool).copy();self.shape=nz,nx=self.mask.shape;self.dx=dx
        counts=np.where(self.mask.ravel(),4,1)
        self.start=(np.cumsum(counts)-counts).astype(np.int32)
        self.parent=np.repeat(np.arange(nx*nz,dtype=np.int32),counts)
        self.count=len(self.parent)
        self.child=np.full(self.count,-1,np.int8)
        refined=self.start[self.mask.ravel()]
        self.child[(refined[:,None]+np.arange(4)).ravel()]=np.tile(np.arange(4,dtype=np.int8),len(refined))
        fine=self.child>=0
        self.width=np.where(fine,.5,1.)*dx;self.area=self.width**2
        self.x=(self.parent%nx+np.where(fine,.25+.5*(self.child%2),.5))*dx
        self.z=(self.parent//nx+np.where(fine,.25+.5*(self.child//2),.5))*dx
        columns=[]
        def add(left,right,axis,x,z,length,edge,offset,baseface,periodic=False):
            n=len(left)
            columns.append((left,right,np.full(n,axis,np.int8),x,z,np.broadcast_to(length,(n,)),
                np.full(n,edge,np.int8),offset,baseface,np.full(n,periodic,bool)))
        def between(pl,pr,axis,periodic=False):
            split=self.mask.ravel()[pl]|self.mask.ravel()[pr]
            repeats=np.where(split,2,1);j=np.repeat(np.arange(len(pl)),repeats)
            sub=np.arange(len(j))-(np.cumsum(repeats)-repeats)[j]
            a,b=pl[j],pr[j];s=split[j]
            lf=np.where(self.mask.ravel()[a],(1+2*sub if axis==1 else 2+sub),0)
            rf=np.where(self.mask.ravel()[b],(2*sub if axis==1 else sub),0)
            x=(a%nx+(1 if axis==1 else np.where(s,.25+.5*sub,.5)))*dx
            z=(a//nx+(np.where(s,.25+.5*sub,.5) if axis==1 else 1))*dx
            face=(a//nx*(nx+1)+a%nx+1) if axis==1 else ((a//nx+1)*nx+a%nx)
            add(self.start[a]+lf,self.start[b]+rf,axis,x,z,np.where(s,.5,1)*dx,-1,np.full(len(j),-1),face,periodic)
        ids=np.arange(nz*nx).reshape(nz,nx)
        between(ids[:,:-1].ravel(),ids[:,1:].ravel(),1)
        between(ids[:-1].ravel(),ids[1:].ravel(),0)
        # Four child/child faces inside each subdivided parent.
        p=np.flatnonzero(self.mask.ravel());start=self.start[p]
        for axis in (1,0):
            for sub in (0,1):
                left=start+(2*sub if axis==1 else sub);right=left+(1 if axis==1 else 2)
                x=(p%nx+(.5 if axis==1 else .25+.5*sub))*dx
                z=(p//nx+(.25+.5*sub if axis==1 else .5))*dx
                add(left,right,axis,x,z,dx/2,-1,np.full(len(p),-1),np.full(len(p),-1))
        if boundary=='periodic':
            between(ids[:,-1],ids[:,0],1,True);between(ids[-1],ids[0],0,True)
        else:
            for edge,parents in enumerate((ids[:,0],ids[:,-1],ids[0],ids[-1])):
                axis=1 if edge<2 else 0;split=self.mask.ravel()[parents]
                repeats=np.where(split,2,1);j=np.repeat(np.arange(len(parents)),repeats)
                sub=np.arange(len(j))-(np.cumsum(repeats)-repeats)[j]
                p=parents[j];s=split[j]
                child={0:2*sub,1:1+2*sub,2:sub,3:2+sub}[edge]
                cell=self.start[p]+np.where(s,child,0);missing=np.full(len(cell),-1)
                left,right=(missing,cell) if edge in (0,2) else (cell,missing)
                x=(np.full(len(p),0 if edge==0 else nx) if axis==1 else p%nx+np.where(s,.25+.5*sub,.5))*dx
                z=(p//nx+np.where(s,.25+.5*sub,.5) if axis==1 else np.full(len(p),0 if edge==2 else nz))*dx
                baseface=p//nx*(nx+1)+(0 if edge==0 else nx) if axis==1 else (0 if edge==2 else nz)*nx+p%nx
                add(left,right,axis,x,z,np.where(s,.5,1)*dx,edge,j,baseface)
        (self.left,self.right,self.axis,self.fx,self.fz,self.length,self.edge,
         self.offset,self.baseface,self.periodic)=[np.concatenate(c) for c in zip(*columns)]
        for name in ('left','right','offset','baseface'):setattr(self,name,getattr(self,name).astype(np.int32))
        self.lv=self.left>=0;self.rv=self.right>=0
        self.l=np.where(self.lv,self.left,self.right);self.r=np.where(self.rv,self.right,self.left)
        self.ox_l=self.fx-self.x[self.l];self.oz_l=self.fz-self.z[self.l]
        self.ox_r=self.fx-self.x[self.r];self.oz_r=self.fz-self.z[self.r]
        self.ox_r[self.periodic&(self.axis==1)]-=nx*dx
        self.oz_r[self.periodic&(self.axis==0)]-=nz*dx
        self.internal=self.lv&self.rv

    def sum_left(self,values):
        return np.bincount(self.left[self.lv],weights=values[self.lv],minlength=self.count)

    def sum_right(self,values):
        return np.bincount(self.right[self.rv],weights=values[self.rv],minlength=self.count)

    def divergence(self,values):
        return self.sum_left(values*self.length)-self.sum_right(values*self.length)

    def project(self,values):
        return (np.bincount(self.parent,weights=values*self.area,minlength=np.prod(self.shape))/self.dx**2).reshape(self.shape)

    def face_projection(self,flux,axis):
        selected=(self.axis==axis)&(self.baseface>=0)
        nz,nx=self.shape;shape=(nz,nx+1) if axis==1 else (nz+1,nx)
        result=np.bincount(self.baseface[selected],weights=flux[selected]*self.length[selected]/self.dx,minlength=np.prod(shape)).reshape(shape)
        if self.periodic.any():
            if axis==1:result[:,0]=result[:,-1]
            else:result[0]=result[-1]
        return result
