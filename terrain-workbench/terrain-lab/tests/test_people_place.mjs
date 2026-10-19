import assert from 'node:assert/strict';
import {buildModel,reachableFrom,defaultStart,legalIndices,placePeople,validate,cellFromMetres,openFaces} from '../people-place.mjs';

function grid(nz,nx,fill=0){return Array.from({length:nz},()=>Array.from({length:nx},()=>fill));}
function scene(nz=20,nx=20,dx=2){
 return {grid:{shape:[nz,nx],dx_m:dx},
  M_base:{z:grid(nz,nx,10),solid:grid(nz,nx,0),static_solid:grid(nz,nx,0),
   wall_x:Array.from({length:nz},()=>Array.from({length:nx+1},()=>0)),
   wall_y:Array.from({length:nz+1},()=>Array.from({length:nx},()=>0)),
   terrain_class:grid(nz,nx,0)},
  scenario_metadata:{terrain_ids:{'0':'land','4':'ocean'}}};
}
const cellOf=(model,item)=>[Math.floor(item.z_m/model.dx),Math.floor(item.x_m/model.dx)];

// Model: draft buildings and levees are included; ocean is blocked.
const s=scene();
const model=buildModel(s,[]);
assert.deepEqual(model.shape,[20,20]);
assert.equal(model.dx,2);
const start=defaultStart(model);
assert.ok(start&&start.z>=0&&start.x>=0);
const reach=reachableFrom(model,start);
assert.equal(legalIndices(model,reach).length,400,'flat open map is fully reachable');
console.log('default start on the boundary and full-grid reachability passed.');

// Uniform placement: deterministic, legal, distinct and spread out.
const a=placePeople(model,reach,{mode:'uniform',count:6,seed:42});
const b=placePeople(model,reach,{mode:'uniform',count:6,seed:42});
assert.equal(a.items.length,6);
assert.deepEqual(a.items,b.items,'same seed reproduces the layout');
const keys=new Set(a.items.map(p=>p.cell.join(',')));
assert.equal(keys.size,6,'one person per selected cell');
for(const p of a.items){
 assert.ok(p.cell[0]>=0&&p.cell[0]<20&&p.cell[1]>=0&&p.cell[1]<20);
 assert.equal(model.solid[p.cell[0]][p.cell[1]],0);
 assert.ok(reach[p.cell[0]][p.cell[1]],'person must be reachable from the start');
}
let minimum=Infinity;
for(let i=0;i<6;i++)for(let j=i+1;j<6;j++)minimum=Math.min(minimum,Math.hypot((a.items[i].cell[1]-a.items[j].cell[1])*2,(a.items[i].cell[0]-a.items[j].cell[0])*2));
assert.ok(minimum>2,'uniform placement keeps a visible separation');
console.log('uniform placement is deterministic, legal and spread out (min gap '+minimum.toFixed(1)+' m).');

// Cluster: everyone stays near one centre.
const c=placePeople(model,reach,{mode:'cluster',count:5,clusters:1,radius:6,seed:9});
assert.equal(c.items.length,5);
let maxRadius=0;
const cx=c.items.reduce((n,p)=>n+p.cell[1],0)/5,cz=c.items.reduce((n,p)=>n+p.cell[0],0)/5;
for(const p of c.items)maxRadius=Math.max(maxRadius,Math.hypot((p.cell[1]-cx)*2,(p.cell[0]-cz)*2));
assert.ok(maxRadius<=12,'cluster members stay inside the radius envelope');
assert.ok(c.items.every(p=>p.group===0));
console.log('cluster placement keeps all members within '+maxRadius.toFixed(1)+' m of the group centre.');

// Cluster + isolated: most stay together, the isolated ones stand apart.
const wide=scene(40,40);
const wideModel=buildModel(wide,[]);
const wideReach=reachableFrom(wideModel,defaultStart(wideModel));
const m=placePeople(wideModel,wideReach,{mode:'cluster_isolated',count:6,clusters:1,radius:5,isolated:2,seed:3});
assert.equal(m.items.length,6);
const clustered=m.items.filter(p=>p.group>=0),isolated=m.items.filter(p=>p.group<0);
assert.equal(clustered.length,4);
assert.equal(isolated.length,2);
const gx=clustered.reduce((n,p)=>n+p.cell[1],0)/clustered.length,gz=clustered.reduce((n,p)=>n+p.cell[0],0)/clustered.length;
for(const p of isolated){
 const distance=Math.hypot((p.cell[1]-gx)*2,(p.cell[0]-gz)*2);
 assert.ok(distance>=10,'isolated person stays away from the cluster: '+distance.toFixed(1)+' m');
}
console.log('cluster + isolated placement separates the isolated people by at least 10 m.');

// A sealed wall splits the map: the far person must be reported as unreachable.
const walled=scene();
for(let z=0;z<20;z++)walled.M_base.wall_x[z][10]=1;
const wm=buildModel(walled,[]);
const ws={z:10,x:2},wr=reachableFrom(wm,ws);
assert.equal(wr[10][18],0,'wall blocks the eastern half');
const blocked=validate(wm,ws,[{cell:[10,2]},{cell:[10,18]}]);
assert.equal(blocked.ok,false);
assert.deepEqual(blocked.unreachable,[2]);
assert.match(blocked.errors.join(' '),/人员 2 与救援起点不连通/);
console.log('unreachable targets are detected across a sealed wall.');

// A start boxed in by wall faces is refused.
const boxed=scene();
boxed.M_base.wall_x[0][1]=1;boxed.M_base.wall_y[1][0]=1;
const bm=buildModel(boxed,[]);
assert.equal(openFaces(bm,0,0),0);
const bad=validate(bm,{z:0,x:0},[]);
assert.equal(bad.ok,false);
assert.match(bad.errors.join(' '),/完全封闭/);
console.log('sealed start detection passed.');

// Buildings from the draft structure editor become illegal cells immediately.
const built=scene();
const withBuilding=buildModel(built,[{kind:'building',x_m:8,z_m:8,width_m:8,length_m:8,height_m:6}]);
for(const [z,x] of [[4,4],[5,6],[7,7]])assert.equal(withBuilding.solid[z][x],1,'draft building blocks its cells');
assert.equal(withBuilding.solid[3][3],0);
console.log('draft buildings are folded into the legality model.');

// Metre to cell conversion clamps to the grid.
assert.deepEqual(cellFromMetres(0,0,model),{z:0,x:0});
assert.deepEqual(cellFromMetres(999,999,model),{z:19,x:19});
console.log('All people placement checks passed.');
