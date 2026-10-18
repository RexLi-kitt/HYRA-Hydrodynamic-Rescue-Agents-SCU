import assert from 'node:assert/strict';
import {decodeRows,hydrateScene} from '../map-arrays.mjs';
const values=new Float64Array([1,2,3,4,5,6]);
const rows=decodeRows(values.buffer,{dtype:'<f8',shape:[2,3]});
assert.deepEqual(rows.map(r=>[...r]),[[1,2,3],[4,5,6]]);
assert.equal(rows[1].buffer,values.buffer); // Rows are views, not full-grid copies.
assert.throws(()=>decodeRows(values.buffer,{dtype:'<f4',shape:[2,3]}),/不完整/);
assert.throws(()=>decodeRows(values.buffer,{dtype:'bad',shape:[2,3]}),/格式/);
const scene={M_base:{z:{array_url:'/generated/test/z.bin',dtype:'<f8',shape:[2,3]},solid:[[0,0,0],[0,0,0]]}};
await hydrateScene(scene,async url=>{assert.equal(url,'/generated/test/z.bin');return values.buffer;});
assert.deepEqual(scene.M_base.z.map(r=>[...r]),[[1,2,3],[4,5,6]]);
assert.deepEqual(scene.M_base.solid,[[0,0,0],[0,0,0]]);
await assert.rejects(hydrateScene({M_base:{z:{array_url:'https://example.com/x'}}}),/路径/);
console.log('Binary grid transport, exact values, zero-copy rows and legacy JSON scenes passed.');
