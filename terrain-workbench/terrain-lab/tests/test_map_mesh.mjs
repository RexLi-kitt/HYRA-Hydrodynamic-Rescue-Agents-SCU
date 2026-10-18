import assert from 'node:assert/strict';
import {meshGeometry,meshState,meshPixel,selectedLeaf,exportMesh} from '../map-mesh.mjs';
import {decodeRows} from '../map-arrays.mjs';
import {speedStats} from '../map-display.mjs';
const mesh={schema:'hyra-static-leaf-v1',cell_count:7,parent:[0,1,1,1,1,2,3],child:[-1,0,1,2,3,-1,-1],
 h:[1,1,2,3,4,1,1],u:[0,1,2,3,4,0,0],v:[0,0,0,0,0,0,0],maximum_depth:[1,1,2,3,4,1,1]};
const g=meshGeometry(mesh,[2,2]),state=meshState(mesh);
assert.deepEqual([...g.start],[0,1,5,6]);assert.deepEqual(g.fine,[1,2,3,4]);
assert.equal(selectedLeaf(g,mesh,1,2),3);assert.equal(selectedLeaf(g,mesh,0,2),-1);
assert.deepEqual([0,1,2,3].map(c=>meshPixel(1,c,2)),[2,3,6,7]);
assert.deepEqual(exportMesh(mesh,state),mesh);
const childBuffer=new Int8Array([-1,0,1,2,3]).buffer;
assert.deepEqual([...decodeRows(childBuffer,{dtype:'|i1',shape:[5]})],[-1,0,1,2,3]);
assert.throws(()=>meshGeometry({...mesh,child:[-1,0,1,3,3,-1,-1]},[2,2]),/子格/);
// Equal-area statistics: splitting one slow cell must not change the P95.
const n=7,data=new Float32Array(n*4);data.fill(1,0,n);data.set([10,1,1,1,1,2,3],n);
const stats=speedStats(data,n,[],g.weights);
const expanded=[10,10,10,10,1,1,1,1,2,2,2,2,3,3,3,3].sort((a,b)=>a-b);
assert.equal(stats.p95,expanded[Math.ceil(expanded.length*.95)-1]);
for(let size=1;size<100;size++){
 const values=new Float32Array(size*4),weights=Float64Array.from({length:size},(_,i)=>i%3===0?4:1),reference=[];
 values.fill(1,0,size);
 for(let i=0;i<size;i++){values[size+i]=(i*17)%23+1;for(let k=0;k<weights[i];k++)reference.push(values[size+i]);}
 reference.sort((a,b)=>a-b);
 assert.equal(speedStats(values,size,[],weights).p95,reference[Math.ceil(reference.length*.95)-1]);
}
console.log('Actual leaf geometry, child picking, fine pixels, binary indices, export and area-weighted P95 passed.');
