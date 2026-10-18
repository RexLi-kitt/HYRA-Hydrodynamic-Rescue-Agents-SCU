// Leaf geometry is static; the displayed state is loaded from real mesh frames.
export function meshGeometry(mesh,shape){
 if(!mesh)return null;
 const n=mesh.cell_count,[rows,cols]=shape;
 if(!Number.isSafeInteger(n)||n<1||mesh.parent.length!==n||mesh.child.length!==n)throw Error('加密网格几何不完整');
 const start=new Int32Array(rows*cols).fill(-1),fine=[];
 for(let i=0;i<n;i++){
  const parent=mesh.parent[i],child=mesh.child[i];
  if(!Number.isInteger(parent)||parent<0||parent>=rows*cols||![-1,0,1,2,3].includes(child))throw Error('加密网格编号不正确');
  if(start[parent]<0)start[parent]=i;
  if(child>=0)fine.push(i);
 }
 for(let p=0;p<start.length;p++){
  const s=start[p];if(s<0)throw Error('加密网格遗漏地块');
  if(mesh.child[s]>=0)for(let k=0;k<4;k++)if(mesh.parent[s+k]!==p||mesh.child[s+k]!==k)throw Error('计算子格不完整');
 }
 return {n,start,fine,weights:Float64Array.from(mesh.child,c=>c<0?4:1)};
}
export function meshState(mesh){
 const n=mesh.cell_count,out=new Float32Array(4*n);
 for(const [k,key] of ['h','u','v','maximum_depth'].entries()){
  if(mesh[key].length!==n)throw Error('加密水情不完整');out.set(mesh[key],k*n);
 }
 return out;
}
export function meshPixel(parent,child,cols){
 return (2*Math.floor(parent/cols)+(child>>1))*2*cols+2*(parent%cols)+(child&1);
}
export function selectedLeaf(geometry,mesh,parent,child){
 if(!geometry||child===undefined)return -1;
 const index=geometry.start[parent];return mesh.child[index]>=0?index+child:-1;
}
export function exportMesh(mesh,state){
 if(!mesh)return undefined;
 const n=mesh.cell_count,result={schema:mesh.schema,cell_count:n,parent:Array.from(mesh.parent),child:Array.from(mesh.child)};
 if(mesh.origin_m)result.origin_m=Array.from(mesh.origin_m);
 for(const [k,key] of ['h','u','v','maximum_depth'].entries())result[key]=Array.from(state.subarray(k*n,(k+1)*n));
 return result;
}
