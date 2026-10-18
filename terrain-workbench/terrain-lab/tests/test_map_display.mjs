import assert from 'node:assert/strict';
import {visibleCrossings,inflowVectors,bearingLabel,inflowDirection,oceanMask,speedStats,speedColor,speedPalette} from '../map-display.mjs';

// Keep configured minor crossings even when the wider candidates are filtered.
const ports=Array.from({length:16},(_,i)=>({id:`river-${i}`,kind:'river',role:i%3===0?'uncertain':'inlet',x:i%8*32,z:i<8?0:255,width_blocks:80-i}));
const before=JSON.stringify(ports),keep=new Set(['river-15']);
const main=visibleCrossings(ports,'main',keep);
assert.ok(main.length<=6);assert.ok(main.some(p=>p.id==='river-15'));
for(let i=0;i<main.length;i++)for(let j=i+1;j<main.length;j++)assert.ok(Math.hypot((main[i].x-main[j].x)/256,(main[i].z-main[j].z)/256)>=.12);
assert.equal(visibleCrossings(ports,'all').length,16);
assert.deepEqual(visibleCrossings(ports,'configured',keep).map(p=>p.id),['river-15']);
assert.equal(JSON.stringify(ports),before);
const many=new Set(ports.slice(0,9).map(p=>p.id));assert.equal(visibleCrossings(ports,'main',many).length,9);
assert.ok(visibleCrossings(ports,'inlet',new Set(['river-0'])).some(p=>p.id==='river-0'));
for(const [edge,axis] of [['west','+X'],['east','−X'],['north','+Z'],['south','−Z']])assert.ok(inflowDirection({edge}).includes(axis));
assert.match(inflowDirection({faces:[{edge:'west'},{edge:'north'}]}),/\+X.*\+Z/);
assert.match(inflowDirection({kind:'manual'}),/无指定入流方向/);
assert.deepEqual(inflowVectors({edge:'west',inward_tangent:{vector_xz:[.8,.6]}}),[[.8,.6]]);
assert.match(bearingLabel([.8,.6]),/东南.*37°/);
assert.match(inflowDirection({edge:'north',inlet_direction_mode:'angle',inlet_angle_deg:45}),/自定义.*45°/);
assert.deepEqual(inflowVectors({edge:'west',inlet_direction_mode:'normal',inward_tangent:{vector_xz:[.8,.6]}}),[[1,0]]);

// A single high-speed cell must not flatten all ordinary speeds; dry cells are excluded.
const n=22,data=new Float32Array(n*4);data.fill(1,0,n);data.fill(1,n,2*n);data[n+20]=100;data[21]=0;data[n+21]=1000;
assert.deepEqual(speedStats(data,n),{peak:100,p95:1,sampleCount:21,oceanFallback:false});
data.fill(0);assert.deepEqual(speedStats(data,n),{peak:0,p95:0,sampleCount:0,oceanFallback:false});
// Even >95% slow ocean cannot set the inland scale. Preserve the all-water peak.
const count=1000,coastal=new Float32Array(count*4),sea=new Uint8Array(count);
coastal.fill(1,0,count);coastal.fill(.001,count,2*count);sea.fill(1);sea.fill(0,990);
coastal.fill(2,count+990,2*count);coastal[count]=20;
const coastalBefore=coastal.slice();
assert.equal(speedStats(coastal,count).p95,coastal[count+1]);
assert.deepEqual(speedStats(coastal,count,sea),{peak:20,p95:2,sampleCount:10,oceanFallback:false});
assert.deepEqual(coastal,coastalBefore);
// Dry and motionless inland cells do not prevent the ocean-only fallback.
coastal.fill(0,990,count);
assert.deepEqual(speedStats(coastal,count,sea),{peak:20,p95:coastal[count+1],sampleCount:990,oceanFallback:true});
coastal.fill(1,990,count);coastal.fill(0,count+990,2*count);
assert.equal(speedStats(coastal,count,sea).oceanFallback,true);
// Never infer sea from elevation: coasts, rivers, lakes and flooded land stay in.
assert.deepEqual([...oceanMask([[1,2,3],[4,5,6]],{1:'deep_ocean',2:'ocean',3:'coast',4:'river',5:'lake',6:'flats'})],[1,1,0,0,0,0]);
assert.deepEqual([...oceanMask([[1,2]],{})],[0,0]);
// Exact selection agrees with a sorted reference for ties, ordering and mixed sea.
for(const count of [1,19,20,21,1001])for(const order of ['up','down','ties']){
 const d=new Float32Array(count*4),mask=new Uint8Array(count);d.fill(1,0,count);
 for(let i=0;i<count;i++){d[count+i]=order==='ties'?i%3+1:order==='up'?i+1:count-i;mask[i]=Number(i%5===0);}
 const eligible=Array.from(d.subarray(count,2*count)).filter((v,i)=>!mask[i]);
 const expected=(eligible.length?eligible:Array.from(d.subarray(count,2*count))).sort((a,b)=>a-b);
 assert.equal(speedStats(d,count,mask).p95,expected[Math.ceil(expected.length*.95)-1]);
}
assert.deepEqual(speedColor(0,5),speedPalette[0]);assert.deepEqual(speedColor(100,5),speedPalette.at(-1));
assert.deepEqual(speedColor(1.25,5,true),speedColor(2.5,5,false));
assert.notDeepEqual(speedColor(1,5),speedColor(1,2));
console.log('Crossing visibility, preserved sources, edge directions, inland P95, ocean fallback and color mapping passed.');
