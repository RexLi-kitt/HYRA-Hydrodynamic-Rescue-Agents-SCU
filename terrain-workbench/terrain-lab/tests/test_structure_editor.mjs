import assert from 'node:assert/strict';
import {snappedEdges,rectangle,isActive,staticSolids,StructureEditor} from '../structure-editor.mjs';
assert.deepEqual(rectangle([12,20],[4,4],[16,16],4),{x_m:4,z_m:4,width_m:8,length_m:16});
assert.deepEqual(rectangle([64,64],[64,64],[16,16],4),{x_m:60,z_m:60,width_m:4,length_m:4});
const path=snappedEdges([[2,1],[6,5]],[8,12],1);
assert.equal(path.length,8);
assert.deepEqual(new Set(path.map(e=>e.join())),new Set(snappedEdges([[6,5],[2,1]],[8,12],1).map(e=>e.join())));
assert.deepEqual(snappedEdges([[6,0],[6,8]],[8,12],1),Array.from({length:8},(_,i)=>[1,i,6]));
const obj={kind:'temporary_levee',start_s:2,end_s:5};
assert.equal(isActive(obj,1),false);assert.equal(isActive(obj,2),true);assert.equal(isActive(obj,5),false);
console.log('Metre rectangles, snapped barrier paths and activation timing passed.');

// A deleted object's footprint must not be baked into the editing background.
const scene={grid:{shape:[3,4],dx_m:1},M_base:{solid:[[0,0,0,1],[0,1,1,0],[0,0,0,0]]},
 M_structures:{objects:[{id:'b',kind:'building',x_m:1,z_m:1,width_m:2,length_m:1,height_m:6,cell_bounds:[1,1,3,2]}]}};
assert.deepEqual(staticSolids(scene).map(r=>Array.from(r)),[[0,0,0,1],[0,0,0,0],[0,0,0,0]]);
assert.equal(scene.M_base.solid[1][1],1,'preview must not mutate the applied hydraulic layer');
const solids=[[0,0,0,1],[0,1,0,0],[0,0,0,0]];
assert.equal(staticSolids({...scene,M_base:{...scene.M_base,static_solid:solids}}),solids,'retain overlapping legacy solids');
const fills=[],ctx={save(){},restore(){},setLineDash(){},fillRect(...r){fills.push(r);},strokeRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){}};
const editor={getScene:()=>scene,objects:[],points:[],start:null,hover:null,selected:null};
StructureEditor.prototype.draw.call(editor,ctx,640,0);
assert.equal(fills.length,0,'deleted draft building must disappear immediately');
StructureEditor.prototype.draw.call(editor,ctx,640,0,scene.M_structures.objects);
assert.equal(fills.length,1,'calculated views retain the actual applied building');
console.log('Deleted-building preview clears while applied results and legacy overlaps are preserved.');
