// Display choices never change the physical crossings or simulation values.
export const inwardVectors={west:[1,0],east:[-1,0],north:[0,1],south:[0,-1]};
const directionNames={west:'西边界 → 向右（+X）',east:'东边界 ← 向左（−X）',north:'北边界 ↓ 向下（+Z）',south:'南边界 ↑ 向上（−Z）'};
export function inflowVectors(p){
 const faces=p.boundary_faces||p.faces||[{edge:p.edge}];
 const result=[];
 for(const face of faces){
  const normal=inwardVectors[face.edge];if(!normal)continue;
  let vector=normal;
  if(p.inlet_direction_mode==='angle'){
   const angle=Number(p.inlet_angle_deg||0)*Math.PI/180;vector=[Math.cos(angle),Math.sin(angle)];
  }else if(p.inlet_direction_mode==='normal')vector=normal;
  else if(face.momentum_direction_xz)vector=face.momentum_direction_xz;
  else if(p.inward_tangent?.vector_xz)vector=p.inward_tangent.vector_xz;
  const length=Math.hypot(...vector);vector=vector.map(v=>v/length);
  const inward=vector[0]*normal[0]+vector[1]*normal[1];
  if(inward<.15&&p.inlet_direction_mode!=='angle')vector=normal;
  if(!result.some(v=>Math.hypot(v[0]-vector[0],v[1]-vector[1])<1e-6))result.push(vector);
 }
 return result;
}
export function bearingLabel(vector){
 const angle=(Math.atan2(vector[1],vector[0])*180/Math.PI+360)%360;
 const names=['东 →','东南 ↘','南 ↓','西南 ↙','西 ←','西北 ↖','北 ↑','东北 ↗'];
 return `${names[Math.round(angle/45)%8]} · ${angle.toFixed(0)}°`;
}
export function inflowDirection(p){
 if(p.kind==='manual')return '内部体积补给，无指定入流方向；流向由水面坡度计算';
 if(p.inlet_direction_mode||p.inward_tangent||p.boundary_faces?.some(f=>f.momentum_direction_xz)){
  const mode=p.inlet_direction_mode==='angle'?'自定义':p.inlet_direction_mode==='normal'?'边界法向':'沿河道';
  return `${mode}：${inflowVectors(p).map(bearingLabel).join('；')}`;
 }
 const edges=[...new Set((p.boundary_faces||p.faces||[{edge:p.edge}]).map(f=>f.edge))];
 return edges.map(e=>directionNames[e]||'边界方向未知').join('；');
}
export function portCode(p,enabled=false){return `${p.kind==='manual'?'点':enabled?'来':p.role==='inlet'?'入':p.role==='outlet'?'出':'?'}${String(p.label||p.id).match(/\d+$/)?.[0]||''}`;}
export function visibleCrossings(points,mode,keepIds=new Set(),shape=[256,256]){
 const keep=points.filter(p=>keepIds.has(p.id));
 if(mode==='configured')return keep;
 if(mode==='all')return points;
 if(mode==='inlet')return points.filter(p=>p.role==='inlet'||keepIds.has(p.id));
 const ranked=[...points].sort((a,b)=>Number(a.role==='uncertain')-Number(b.role==='uncertain')||(b.width_blocks||0)-(a.width_blocks||0)||a.id.localeCompare(b.id));
 const chosen=[...keep];
 for(const p of ranked){
  if(chosen.length>=Math.max(6,keep.length))break;
  if(chosen.some(q=>q.id===p.id||Math.hypot((p.x-q.x)/shape[1],(p.z-q.z)/shape[0])<.12))continue;
  chosen.push(p);
 }
 const ids=new Set(chosen.map(p=>p.id));return points.filter(p=>ids.has(p.id));
}
export function oceanMask(classes,terrainIds={}){
 const codes=new Set(Object.entries(terrainIds).filter(([,name])=>String(name).toLowerCase().includes('ocean')).map(([code])=>Number(code)));
 const mask=new Uint8Array(classes.length*(classes[0]?.length||0));let i=0;
 for(const row of classes)for(const code of row)mask[i++]=Number(codes.has(code));
 return mask;
}
function selectRank(values,k){
 let lo=0,hi=values.length-1;
 while(lo<hi){
  const pivot=values[(lo+hi)>>>1];let lower=lo,i=lo,upper=hi;
  // Three-way partition also handles large areas of identical velocities.
  while(i<=upper){
   if(values[i]<pivot){const v=values[lower];values[lower++]=values[i];values[i++]=v;}
   else if(values[i]>pivot){const v=values[upper];values[upper--]=values[i];values[i]=v;}
   else i++;
  }
  if(k<lower)hi=lower-1;else if(k>upper)lo=upper+1;else return values[k];
 }
 return values[k];
}
function weightedRank(values,weights,target){
 let lo=0,hi=values.length-1;
 const swap=(a,b)=>{let v=values[a];values[a]=values[b];values[b]=v;v=weights[a];weights[a]=weights[b];weights[b]=v;};
 while(lo<hi){
  const pivot=values[(lo+hi)>>>1];let lower=lo,i=lo,upper=hi;
  while(i<=upper){if(values[i]<pivot)swap(lower++,i++);else if(values[i]>pivot)swap(i,upper--);else i++;}
  let below=0,equal=0;for(let j=lo;j<lower;j++)below+=weights[j];for(let j=lower;j<=upper;j++)equal+=weights[j];
  if(target<=below)hi=lower-1;else if(target<=below+equal)return pivot;else{target-=below+equal;lo=upper+1;}
 }
 return values[lo];
}
export function speedStats(data,n,sea=[],areas=null){
 const values=new Float64Array(n),weights=areas?new Float64Array(n):null;let inland=0,ocean=0,peak=0,inlandArea=0,oceanArea=0;
 for(let i=0;i<n;i++)if(data[i]>=.01){const v=Math.hypot(data[n+i],data[2*n+i]);if(Number.isFinite(v)&&v>0){
  peak=Math.max(peak,v);const area=areas?areas[i]:1;if(!Number.isFinite(area)||area<=0)continue;
  const index=sea[i]?n-++ocean:inland++;values[index]=v;if(weights)weights[index]=area;
  if(sea[i])oceanArea+=area;else inlandArea+=area;
 }}
 // Let rivers and flooded land set the contrast, regardless of ocean area.
 // Ocean-only scenes still get an adaptive scale. The peak always covers all water.
 const speeds=inland?values.subarray(0,inland):values.subarray(n-ocean);
 const sampleWeights=weights?(inland?weights.subarray(0,inland):weights.subarray(n-ocean)):null;
 const p95=!speeds.length?0:sampleWeights?weightedRank(speeds,sampleWeights,.95*(inland?inlandArea:oceanArea)):selectRank(speeds,Math.ceil(.95*speeds.length)-1);
 return {peak,p95,sampleCount:speeds.length,oceanFallback:!inland&&ocean>0};
}
export const speedPalette=[[36,59,139],[22,139,192],[45,199,180],[244,222,79],[235,72,47]];
export function speedColor(value,limit,enhanced=false){
 let t=limit>0?Math.min(1,Math.max(0,value/limit)):0;if(enhanced)t=Math.sqrt(t);
 const k=t*(speedPalette.length-1),i=Math.min(speedPalette.length-2,Math.floor(k));
 return speedPalette[i].map((v,c)=>v+(speedPalette[i+1][c]-v)*(k-i));
}
