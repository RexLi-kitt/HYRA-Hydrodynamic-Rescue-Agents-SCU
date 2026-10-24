import assert from 'node:assert/strict';
import {auditWarnings,auditSummary,drawChannelAudit} from '../channel-audit.mjs';
assert.match(auditSummary(null),/重新应用/);
const a={complete:false,excluded_obstructed:0,channels:[
 {status:'closed',bounds_m:[1,2,1.5,4]},
 {status:'narrowed',bounds_m:[2,3,3,4]},
 {status:'open',bounds_m:[3,4,4,5]},
 {status:'not_evaluated',bounds_m:[4,5,5,6]}]};
assert.equal(auditWarnings(a).length,3);
assert.match(auditSummary(a),/不完整/);
let calls=[];
const ctx={save(){},restore(){},setLineDash(){},strokeRect(...args){calls.push(args)}};
drawChannelAudit(ctx,a,{shape:[16,32],dx_m:.5},640);
assert.equal(calls.length,3);assert.deepEqual(calls[0],[40,160,20,160]);
drawChannelAudit(ctx,null,{shape:[16,32],dx_m:.5},640);
assert.equal(calls.length,3);
console.log('Channel audit display checks passed');
