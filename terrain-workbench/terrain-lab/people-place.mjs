// Person placement and reachability on the hydraulic grid.
// Shared by the editor preview; the server re-checks the same rules on apply.
import {staticSolids,snappedEdges,isActive} from './structure-editor.mjs';
import {oceanMask} from './map-display.mjs';

export const GROUP_COLORS=['#e4572e','#f2a20c','#7b5cd6','#1f9e8f','#d64550'];
export const MAX_PEOPLE=200;

export function createRng(seed){
 let a=Math.floor(Number(seed)||0)>>>0;
 if(!a)a=0x9e3779b9;
 return()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};
}

// Fixed environment only: buildings, sealed walls and levees active at t=0.
export function buildModel(scene,structures=[]){
 const [nz,nx]=scene.grid.shape,dx=scene.grid.dx_m,b=scene.M_base;
 const base=scene.M_base.static_solid||staticSolids(scene);
 const solid=Array.from({length:nz},(_,z)=>Uint8Array.from(base[z],v=>v?1:0));
 const blockedX=Array.from({length:nz},(_,z)=>Uint8Array.from(b.wall_x[z],v=>v?1:0));
 const blockedY=Array.from({length:nz+1},(_,z)=>Uint8Array.from(b.wall_y[z],v=>v?1:0));
 for(const o of structures){
  if(o.kind==='building'){
   const x0=Math.max(0,Math.min(nx,Math.floor(o.x_m/dx+1e-10))),z0=Math.max(0,Math.min(nz,Math.floor(o.z_m/dx+1e-10)));
   const x1=Math.max(0,Math.min(nx,Math.ceil((o.x_m+o.width_m)/dx-1e-10))),z1=Math.max(0,Math.min(nz,Math.ceil((o.z_m+o.length_m)/dx-1e-10)));
   for(let z=z0;z<z1;z++)solid[z].fill(1,x0,x1);
  }else if(isActive(o,0)){
   for(const [axis,z,x] of snappedEdges(o.points_m,scene.grid.shape,dx)){if(axis===1)blockedX[z][x]=1;else blockedY[z][x]=1;}
  }
 }
 const ids=scene.scenario_metadata?.terrain_ids||scene.terrain_ids||{};
 const flat=oceanMask(b.terrain_class,ids);
 const ocean=Array.from({length:nz},(_,z)=>flat.subarray(z*nx,(z+1)*nx));
 return {shape:[nz,nx],dx,solid,blockedX,blockedY,ocean,bed:b.z};
}

export function closedCell(model,z,x){return Boolean(model.solid[z][x]||model.ocean[z][x]);}

export function openFaces(model,z,x){
 const [nz,nx]=model.shape;let count=0;
 if(x>0&&!closedCell(model,z,x-1)&&!model.blockedX[z][x])count++;
 if(x+1<nx&&!closedCell(model,z,x+1)&&!model.blockedX[z][x+1])count++;
 if(z>0&&!closedCell(model,z-1,x)&&!model.blockedY[z][x])count++;
 if(z+1<nz&&!closedCell(model,z+1,x)&&!model.blockedY[z+1][x])count++;
 return count;
}

export function reachableFrom(model,start){
 const [nz,nx]=model.shape,seen=Array.from({length:nz},()=>new Uint8Array(nx));
 if(!start||closedCell(model,start.z,start.x))return seen;
 seen[start.z][start.x]=1;const stack=[[start.z,start.x]];
 while(stack.length){
  const [z,x]=stack.pop();
  const visit=(zz,xx,open)=>{if(zz<0||zz>=nz||xx<0||xx>=nx||!open||seen[zz][xx]||closedCell(model,zz,xx))return;seen[zz][xx]=1;stack.push([zz,xx]);};
  visit(z,x-1,x>0&&!model.blockedX[z][x]);
  visit(z,x+1,x+1<nx&&!model.blockedX[z][x+1]);
  visit(z-1,x,z>0&&!model.blockedY[z][x]);
  visit(z+1,x,z+1<nz&&!model.blockedY[z+1][x]);
 }
 return seen;
}

export function defaultStart(model){
 const [nz,nx]=model.shape,dx=model.dx;let best=null;
 const consider=(z,x)=>{if(closedCell(model,z,x)||openFaces(model,z,x)===0)return;const bed=Number(model.bed[z][x]);if(!best||bed<best.bed-1e-9)best={z,x,bed};};
 for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)if(z===0||z===nz-1||x===0||x===nx-1)consider(z,x);
 if(!best)for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)consider(z,x);
 return best?{z:best.z,x:best.x,x_m:(best.x+.5)*dx,z_m:(best.z+.5)*dx,automatic:true}:null;
}

export function legalIndices(model,reachable){
 const [nz,nx]=model.shape,out=[];
 for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)if(reachable[z][x]&&!closedCell(model,z,x))out.push(z*nx+x);
 return out;
}

const shuffle=(values,rng)=>{for(let i=values.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[values[i],values[j]]=[values[j],values[i]];}return values;};

function spreadPick(indices,count,minDistance,nx,dx,rng){
 const indexSet=new Set(indices),chosen=[];
 const distance=(a,b)=>Math.hypot((a%nx-b%nx)*dx,(((a/nx)|0)-((b/nx)|0))*dx);
 const pool=shuffle([...indices],rng);
 for(let pass=0;pass<6&&chosen.length<count;pass++){
  const limit=minDistance*Math.pow(.65,pass);
  for(const i of pool){if(chosen.length>=count)break;if(!indexSet.has(i))continue;if(chosen.some(c=>distance(c,i)<limit-1e-9))continue;chosen.push(i);}
 }
 for(const i of pool){if(chosen.length>=count)break;if(!chosen.includes(i))chosen.push(i);}
 return chosen;
}

function nearestCenter(centers,nx,dx,i){
 const x=i%nx,z=(i/nx)|0;let best=Infinity;
 for(const c of centers)best=Math.min(best,Math.hypot((c%nx-x)*dx,(((c/nx)|0)-z)*dx));
 return best;
}

function clusterLayout(model,reachable,{count,clusters,radius,seed}){
 const [nz,nx]=model.shape,dx=model.dx,rng=createRng(seed),legal=legalIndices(model,reachable);
 if(!legal.length)return {error:'没有可放置人员的合法区域；请检查建筑、墙或起点位置。'};
 const centers=spreadPick(legal,Math.min(clusters,Math.max(1,Math.floor(count))),Math.max(3*radius,8*dx),nx,dx,rng);
 if(!centers.length)return {error:'无法选择聚簇中心，请减小聚簇半径或移动起点。'};
 const cells=new Set(),items=[];
 for(let i=0;i<count;i++){
  const cluster=i%centers.length,center=centers[cluster],cx=center%nx,cz=(center/nx)|0;
  let radiusNow=radius;
  let options=legal.filter(k=>Math.hypot((k%nx-cx)*dx,((k/nx|0)-cz)*dx)<=radiusNow&&!cells.has(k));
  for(let grow=0;!options.length&&grow<4;grow++){radiusNow*=1.6;options=legal.filter(k=>Math.hypot((k%nx-cx)*dx,((k/nx|0)-cz)*dx)<=radiusNow&&!cells.has(k));}
  if(!options.length)return {error:'聚簇半径内没有足够的合法格，请增大半径或减少人数。'};
  shuffle(options,rng);const pick=options[0];cells.add(pick);
  items.push({cell:[(pick/nx)|0,pick%nx],group:cluster});
 }
 return {items,centers};
}

export function placePeople(model,reachable,{mode='uniform',count=6,clusters=1,radius=10,isolated=2,seed=1}={}){
 const [nz,nx]=model.shape,dx=model.dx;
 if(mode==='manual')return {items:[]};
 if(mode==='uniform'){
  const legal=legalIndices(model,reachable);
  if(!legal.length)return {error:'没有可放置人员的合法区域；请检查建筑、墙或起点位置。'};
  const rng=createRng(seed),area=legal.length*dx*dx,spacing=Math.max(1.5*dx,.75*Math.sqrt(area/Math.max(1,count)));
  const picked=spreadPick(legal,Math.min(count,legal.length),spacing,nx,dx,rng);
  return {items:picked.map(i=>({cell:[(i/nx)|0,i%nx],group:0}))};
 }
 const groups=mode==='cluster_isolated'?Math.max(1,Math.min(count,count-isolated)):count;
 const layout=clusterLayout(model,reachable,{count:groups,clusters,radius,seed});
 if(layout.error)return layout;
 if(mode==='cluster')return {items:layout.items};
 const rng=createRng(seed^0x5f3759df),blocked=new Set(layout.items.map(p=>p.cell[0]*nx+p.cell[1])),legal=legalIndices(model,reachable);
 const isolation=Math.max(3*radius,10*dx),wanted=Math.min(isolated,Math.max(0,count-groups));
 if(!wanted)return {items:layout.items};
 const far=legal.filter(i=>!blocked.has(i)&&nearestCenter(layout.centers,nx,dx,i)>=isolation);
 let pool=far.length>=wanted?far:legal.filter(i=>!blocked.has(i)&&nearestCenter(layout.centers,nx,dx,i)>=isolation*.6);
 if(pool.length<wanted)pool=legal.filter(i=>!blocked.has(i));
 shuffle(pool,rng);
 const chosen=[];
 for(const i of pool){if(chosen.length>=wanted)break;if(chosen.some(c=>Math.hypot((c%nx-i%nx)*dx,((c/nx|0)-(i/nx|0))*dx)<2*radius))continue;chosen.push(i);}
 for(const i of pool){if(chosen.length>=wanted)break;if(!chosen.includes(i))chosen.push(i);}
 return {items:[...layout.items,...chosen.map(i=>({cell:[(i/nx)|0,i%nx],group:-1}))]};
}

export function validate(model,start,items){
 const errors=[],unreachable=[];
 if(!start){errors.push('尚未设置救援起点。');return {ok:false,errors,unreachable};}
 const cell={z:start.z,x:start.x};
 if(cell.z<0||cell.z>=model.shape[0]||cell.x<0||cell.x>=model.shape[1]){errors.push('救援起点超出地图范围。');return {ok:false,errors,unreachable};}
 if(model.solid[cell.z][cell.x])errors.push('救援起点位于建筑或封闭占地内。');
 else if(model.ocean[cell.z][cell.x])errors.push('救援起点位于海域，请移到陆地。');
 else if(openFaces(model,cell.z,cell.x)===0)errors.push('救援起点被完全封闭，四条边均不可通行。');
 const reach=errors.length?null:reachableFrom(model,cell);
 items.forEach((item,i)=>{
  const [z,x]=item.cell;
  if(z<0||z>=model.shape[0]||x<0||x>=model.shape[1]){errors.push(`人员 ${i+1} 超出地图范围。`);return;}
  if(model.solid[z][x])errors.push(`人员 ${i+1} 位于建筑或封闭占地内。`);
  else if(model.ocean[z][x])errors.push(`人员 ${i+1} 位于海域，请移到陆地。`);
  else if(reach&&!reach[z][x])unreachable.push(i+1);
 });
 if(!errors.length&&unreachable.length){
  const listed=unreachable.slice(0,8).join('、'),more=unreachable.length>8?'等':'';
  errors.push(`人员 ${listed}${more} 与救援起点不连通，请移动人员或改选起点。`);
 }
 return {ok:errors.length===0,errors,unreachable,connected:reach?reach.reduce((n,row)=>n+row.reduce((m,v)=>m+v,0),0):0};
}

export function cellFromMetres(x_m,z_m,model){
 const [nz,nx]=model.shape,dx=model.dx;
 return {z:Math.max(0,Math.min(nz-1,Math.floor(z_m/dx))),x:Math.max(0,Math.min(nx-1,Math.floor(x_m/dx)))};
}

export function personAt(model,cell,group=0,kind='trapped'){
 return {cell:[cell.z,cell.x],group,kind};
}
