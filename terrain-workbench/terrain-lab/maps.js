import {discharge,windowCurve,portLabel,validateCurve} from './inflow-controls.mjs';
import {framePair,blendFrames,playbackTime} from './maps-playback.mjs';
const $=id=>document.getElementById(id),ids=['baseMap','depthMap','speedMap','maximumMap'],S=640;
const fmt=(v,n=2)=>Number(v).toLocaleString('zh-CN',{maximumFractionDigits:n});
let terrain,scene,process={state:'ready',frames:[]},patches=[],selected={x:0,z:0},imported=null;
let externalDraft=[],pickInflow=false,inflowDirty=false;
let busy=false,playing=false,time=0,display,baseImage,terrainMin=0,terrainMax=1,epoch=0,pollTimer,seekVersion=0,seeking=false;
const cache=new Map(),pending=new Map();let lastDraw=0;
const fields={mapResolution:'resolution',mapUnit:'metres_per_block',mapDuration:'duration_s',
 mapManning:'manning',mapRain:'rain_mm_h',mapRainUntil:'rain_duration_s',mapInfiltration:'infiltration_mm_h',
 mapBaseflowRate:'baseflow_m3_s',mapWarmup:'warmup_s',sourceX:'source_x',sourceZ:'source_z'};
function status(text,error=false){$('mapStatus').textContent=text;$('mapStatus').classList.toggle('error',error);}
async function json(url,options){const r=await fetch(url,options),j=await r.json();if(!r.ok){const error=Error(j.error||'读取失败');error.process_id=j.process_id;error.status=r.status;throw error;}return j;}
function post(url,data={}){return json(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});}
function active(){return ['queued','running'].includes(process.state);}
function controls(){
 for(const id of ['createMap','runMap','importMapConfig','addRiverInflow','addManualInflow','pickInflow'])$(id).disabled=busy||active()||!terrain;
 $('runMap').textContent=scene&&process.state!=='ready'?'重新模拟洪水':'开始洪水模拟';
 $('addRiverInflow').disabled=busy||active()||!$('riverInlet').value;
 for(const el of $('externalInflowList').querySelectorAll('button,input'))el.disabled=busy||active();
 $('cancelMap').disabled=busy||!active()||process.cancel_requested;
 $('playMap').textContent=playing?'暂停播放':time>=available()&&available()>0?'重播':'播放';
 $('playMap').disabled=!scene||(process.frames.length<2&&!playing);
 $('rewindMap').disabled=!scene;$('mapTimeline').disabled=!scene||process.frames.length<2;
 $('exportFrame').disabled=!scene||seeking;$('exportPng').disabled=!scene||seeking;
 $('downloadMap').hidden=!process.archive_url;
 if(process.archive_url)$('downloadMap').href=process.archive_url;
}
function config(){
 const mode=$('mapSource').value,flood={frames:imported?.flood?.frames??121};
 for(const [id,name] of Object.entries(fields))flood[name]=Number($(id).value);
 flood.inflow_m3_s=0; // The point editor now uses explicit, separately accounted external sources.
 flood.boundary=$('mapBoundary').value;flood.initial_water=$('mapInitialWater').checked;
 flood.baseflow=mode==='current'&&$('mapBaseflow').checked;
 if(mode==='demo'){
  flood.warmup_s=0;flood.resolution=32;
  flood.source_x=Math.min(31,flood.source_x);flood.source_z=Math.min(31,flood.source_z);
 }
 const layers={...(imported?.layers||{}),patches:[...(imported?.layers?.patches||[]),...patches]};
 const hydro=JSON.parse($('hydrographs').value);
 if(Object.keys(hydro).length)layers.hydrographs=hydro;else if(mode==='current')layers.hydrographs={};
 return {...(imported||{}),mode,terrain_id:terrain.id,flood,layers,external_inflows:externalDraft.map(p=>({...p}))};
}
function draft(){$('draftInfo').textContent=`草稿：${patches.length} 项编辑；应用或重新模拟后生效。`;}
async function create(start=false){
 if(busy||active())return;
 busy=true;controls();status('正在准备地图与洪水场景…');
 try{
  const c=config(),next=await post('/api/maps',c);inflowDirty=false;installScene(next);
  status('地图已应用。可开始整段洪水模拟，或编辑固定环境。');
  if(start){await post(`/api/maps/${scene.id}/run`);process.state='queued';remember();playing=true;status('洪水开始计算，过程就绪后自动连续播放。');poll(epoch);}
 }catch(e){let recover=e.process_id;if(!recover&&e.status===409)try{recover=localStorage.getItem('hyra-map-process');}catch{}if(recover){try{await restore(recover);status(e.message);}catch(recovery){status(recovery.message,true);}}else status(e.message,true);}finally{busy=false;controls();}
}
function remember(){try{localStorage.setItem('hyra-map-process',scene.id);}catch{}history.replaceState(null,'',`#${scene.id}`);}
function installScene(next){
 clearTimeout(pollTimer);epoch++;cache.clear();pending.clear();scene=next;process=next.process;time=0;playing=false;seeking=false;seekVersion++;selected={x:0,z:0};
 const w=scene.M_water;display=new Float32Array([...w.h.flat(),...w.u.flat(),...w.v.flat(),...w.maximum_depth.flat()]);
 terrainMin=Infinity;terrainMax=-Infinity;for(const row of scene.M_base.z)for(const h of row){terrainMin=Math.min(terrainMin,h);terrainMax=Math.max(terrainMax,h);}
 inletControls();makeBase();$('mapTimeline').max=scene.config.duration_s;$('durationLabel').textContent=`${fmt(scene.config.duration_s)} 秒`;
 $('worldInfo').textContent=`${scene.provenance==='synthetic_multilayer_demo'?'建筑与水源示例':'当前生成地图'} · ${scene.grid.shape[1]} × ${scene.grid.shape[0]} · 间距 ${fmt(scene.grid.dx_m)} 米 · 全部图层同步`;
 $('computeStatus').textContent='地图已建立，初始水体已显示。';$('mapProgress').value=0;render();
}
async function restore(id){
 const next=await json(`/api/maps/${id}`);installScene(next);remember();
 const mode=scene.provenance==='synthetic_multilayer_demo'?'demo':'current';
 externalDraft=normalizeInflows(next.scenario_config?.external_inflows||next.M_forcing?.external_inflows||[]);inflowDirty=false;inletControls();
 imported=scene.scenario_config||{mode,terrain_id:terrain.id,layers:{...scene.M_base,hydrographs:scene.source_hydrographs},flood:scene.config};
 patches=[];draft();$('mapSource').value=mode;
 for(const [key,name] of Object.entries(fields))$(key).value=scene.config[name];
 $('mapBoundary').value=scene.config.boundary;$('mapInitialWater').checked=scene.config.initial_water;$('mapBaseflow').checked=scene.config.baseflow;
 if(scene.config.inflow_m3_s>0&&!externalDraft.some(p=>p.id==='legacy-point'))externalDraft.push({id:'legacy-point',kind:'manual',x:scene.config.source_x,z:scene.config.source_z,enabled:true,hydrograph:[[0,scene.config.inflow_m3_s]]});
 inletControls();makeBase();$('hydrographs').value=JSON.stringify(imported.layers?.hydrographs||{});
 await poll(epoch);status(active()?'已恢复正在运行的洪水过程，可以继续查看或取消。':'已恢复洪水过程，可播放与导出。');controls();
 if(process.frames.length){const pair=framePair(process.frames,0);await Promise.all([requestFrame(pair.lo),requestFrame(pair.hi)]);renderTime(0);}
}
async function poll(token){
 try{
  const id=scene.id,p=await json(`/api/maps/${id}/status`);if(token!==epoch)return;process=p;
  $('computeStatus').textContent=p.message;$('mapProgress').value=p.progress;
  $('availableTime').textContent=`已计算 ${fmt(available(),1)} / ${fmt(scene.config.duration_s)} 秒`;
  if(p.initial_conditions){const w=p.initial_conditions;
   $('scenarioInfo').textContent=`边界：${{ocean:'海平面双向交换',open:'自由出流',closed:'封闭'}[scene.config.boundary]} · 日常基流 ${fmt(scene.scenario_metadata.river_baseflow?.actual_m3_s||0)} m³/s\n${w.duration_s?`基流预热 ${fmt(w.duration_s)} 秒，${w.converged?'达到稳定阈值':'未达到稳定阈值'}`:'未进行基流预热'}。地图图层与洪水过程均使用同一网格。`;
  }
  if(p.frames.length){const pair=framePair(p.frames,time);requestFrame(pair.lo);requestFrame(pair.hi);}
  if(p.state==='done')status('完整洪水过程已保存。可播放、暂停或拖动时间轴查看任意时刻。');
  if(p.state==='cancelled')status('计算已取消，已完成的时段仍可播放和导出。');
  if(p.state==='failed'){playing=false;status(p.message,true);}
  if(!active()&&p.frames.length<2)playing=false;
  controls();if(active())pollTimer=setTimeout(()=>poll(token),700);
 }catch(e){if(token!==epoch)return;playing=false;status(`读取计算进度失败：${e.message}`,true);controls();}
}
function available(){return process.frames.at(-1)?.time_s||0;}
function requestFrame(index){
 if(cache.has(index))return Promise.resolve(cache.get(index));if(pending.has(index))return pending.get(index);
 const f=process.frames[index];if(!f)return Promise.resolve(null);const token=epoch;
 const task=(async()=>{
  const r=await fetch(f.url);if(!r.ok)throw Error('洪水过程帧读取失败');
  const buffer=await r.arrayBuffer(),expected=scene.grid.shape[0]*scene.grid.shape[1]*4;
  if(buffer.byteLength!==expected*4)throw Error('洪水过程帧不完整');
  const data=new Float32Array(buffer);if(data.some(v=>!Number.isFinite(v)))throw Error('过程帧包含无效数值');
  if(token!==epoch)return null;
  cache.set(index,data);while(cache.size>8){const victim=[...cache.keys()].find(k=>k!==index);cache.delete(victim);}
  return data;
 })().catch(e=>{if(token===epoch){playing=false;status(e.message,true);controls();}return null;}).finally(()=>{if(token===epoch)pending.delete(index);});
 pending.set(index,task);return task;
}
function renderTime(target){
 const pair=framePair(process.frames,target);if(!pair)return false;
 const a=cache.get(pair.lo),b=cache.get(pair.hi);if(!a||!b){requestFrame(pair.lo);requestFrame(pair.hi);return false;}
 blendFrames(a,b,pair.alpha,display);time=pair.time;render(pair);
 requestFrame(Math.min(pair.hi+1,process.frames.length-1));return true;
}
function crossings(){return $('mapSource').value==='demo'?[]:(scene?.M_forcing?.river_crossings||terrain?.river_inlets?.crossings||[]);}
function inletCurve(){return validateCurve($('customInflowCurve').checked?JSON.parse($('inflowCurve').value):windowCurve(Number($('inflowStart').value),Number($('inflowEnd').value),Number($('mapInflow').value)));}
function normalizeInflows(values){if(!Array.isArray(values)||values.length>32)throw Error('外源点列表最多32项');return values.map(p=>({...p,kind:p.kind??'manual',enabled:p.enabled??true,hydrograph:validateCurve(p.hydrograph??[[0,50],[120,0]])}));}
function inletControls(){
 const points=crossings(),old=$('riverInlet').value;$('riverInlet').replaceChildren();
 if(!points.length){const o=document.createElement('option');o.value='';o.textContent='没有河道边界交点，可手动选点';$('riverInlet').append(o);}
 else for(const p of points){const o=document.createElement('option');o.value=p.id;o.textContent=portLabel(p);$('riverInlet').append(o);}
 if(points.some(p=>p.id===old))$('riverInlet').value=old;else if(points.some(p=>p.role==='inlet'))$('riverInlet').value=points.find(p=>p.role==='inlet').id;
 $('riverInletInfo').textContent=`识别到 ${points.filter(p=>p.role==='inlet').length} 个入口、${points.filter(p=>p.role==='outlet').length} 个出口、${points.filter(p=>p.role==='uncertain').length} 个待确认交点。${points.length?'方向为地形推断，选择河口后设置上游来水。':'可在地图上自行标注入水点。'}`;
 $('externalInflowList').replaceChildren(...externalDraft.map(p=>{
  const row=document.createElement('div');row.className='inlet-row';const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=p.enabled;check.setAttribute('aria-label',`启用入水点 ${p.id}`);
  check.onchange=()=>{p.enabled=check.checked;forcingEdited();};label.append(check,document.createTextNode(p.kind==='river'?portLabel(points.find(v=>v.id===p.id)||p):`手动入水点 ${p.id} · (${p.x},${p.z})`));
  const desc=document.createElement('small');desc.textContent=p.hydrograph.map(([t,q])=>`${fmt(t)}s: ${fmt(q)}m³/s`).join(' → ');
  const edit=document.createElement('button');edit.textContent='修改流量';edit.setAttribute('aria-label',`修改入水点 ${p.id}`);edit.onclick=()=>{try{p.hydrograph=inletCurve();forcingEdited();}catch(e){status(e.message,true);}};
  const del=document.createElement('button');del.textContent='移除';del.setAttribute('aria-label',`移除入水点 ${p.id}`);del.onclick=()=>{externalDraft=externalDraft.filter(v=>v.id!==p.id);forcingEdited();};
  row.append(label,desc,edit,del);return row;
 }));controls();
}
function forcingEdited(){inflowDirty=true;inletControls();makeBase();if(scene)render(framePair(process.frames,time));status('外源来水草稿已修改，应用地图或重新模拟后生效。');}
function addExternal(p){
 if(active()||busy)return;
 try{const existing=externalDraft.find(e=>e.id===p.id);if(!existing&&externalDraft.length>=32)throw Error('最多32个外源来水点');const e={...p,enabled:true,hydrograph:inletCurve()};if(existing)Object.assign(existing,e);else externalDraft.push(e);forcingEdited();}catch(e){status(e.message,true);}
}
function externalMarkers(ctx){
 const factor=scene.scenario_metadata.coarsening_factor||1,[nz,nx]=scene.grid.shape,sx=S/nx,sz=S/nz;
 const points=crossings(),configured=new Map(externalDraft.map(p=>[p.id,p]));
 const all=[...points,...externalDraft.filter(p=>p.kind==='manual')];
 for(const p of all){const entry=configured.get(p.id),enabled=entry?.enabled,px=(Math.floor(p.x/factor)+.5)*sx,pz=(Math.floor(p.z/factor)+.5)*sz;
  const x=Math.max(9,Math.min(S-9,px)),z=Math.max(9,Math.min(S-9,pz));ctx.save();ctx.strokeStyle=enabled?'#8c3bb2':p.role==='outlet'?'#777e87':'#8c5ca0';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=3;
  ctx.beginPath();ctx.arc(x,z,7,0,Math.PI*2);ctx.stroke();
  if(enabled){ctx.beginPath();if(p.kind==='manual'){ctx.moveTo(x-4,z);ctx.lineTo(x+4,z);ctx.moveTo(x,z-4);ctx.lineTo(x,z+4);}else{const [dx,dz]=({west:[1,0],east:[-1,0],north:[0,1],south:[0,-1]})[p.edge]||[1,0];ctx.moveTo(x-dx*5-dz*3,z-dz*5+dx*3);ctx.lineTo(x+dx*5,z+dz*5);ctx.lineTo(x-dx*5+dz*3,z-dz*5-dx*3);}ctx.stroke();}
  if(enabled){ctx.font='bold 17px sans-serif';ctx.textAlign=x>S/2?'right':'left';const text=`${inflowDirty?'草稿 ':''}${fmt(discharge(entry.hydrograph,time))} m³/s`;const tx=x+(x>S/2?-13:13),tz=Math.max(22,Math.min(S-8,z));ctx.strokeStyle='#fff';ctx.lineWidth=4;ctx.strokeText(text,tx,tz);ctx.fillText(text,tx,tz);}
  ctx.restore();
 }
}
$('riverInlet').onchange=controls;
$('addRiverInflow').onclick=()=>{const p=crossings().find(p=>p.id===$('riverInlet').value);if(p)addExternal({id:p.id,kind:'river'});};
$('addManualInflow').onclick=()=>{let n=1;while(externalDraft.some(p=>p.id===`manual-${n}`))n++;addExternal({id:`manual-${n}`,kind:'manual',x:Number($('sourceX').value),z:Number($('sourceZ').value)});};
$('pickInflow').onclick=()=>{pickInflow=!pickInflow;$('pickInflow').setAttribute('aria-pressed',String(pickInflow));status(pickInflow?'点击任意地图选择强制来水位置，然后添加入水点。':'已退出入水选点。');};
function rgbHeight(z){const t=(z-terrainMin)/Math.max(.01,terrainMax-terrainMin);return [112+116*t,152+74*t,117+74*t];}
function rgbDepth(h){const t=Math.min(1,Math.max(0,h)/Math.max(.1,Number($('depthScale').value)||3));return [155-134*t,217-156*t,234-110*t];}
function rgbSpeed(v){const t=Math.min(1,v/2);return t<.5?[151+166*t,210+18*t,189-154*t]:[234,219-290*(t-.5),112-88*(t-.5)];}
function image(colors){const [nz,nx]=scene.grid.shape,cv=document.createElement('canvas');cv.width=nx;cv.height=nz;
 const img=new ImageData(nx,nz);for(let i=0;i<nx*nz;i++)img.data.set([...colors(i).map(Math.round),255],i*4);cv.getContext('2d').putImageData(img,0,0);return cv;}
function markers(ctx,sources=true){
 const [nz,nx]=scene.grid.shape,sx=S/nx,sz=S/nz,b=scene.M_base;
 ctx.fillStyle='#414e4e';for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)if(b.solid[z][x])ctx.fillRect(x*sx,z*sz,sx,sz);
 ctx.strokeStyle='#182d2c';ctx.lineWidth=2;ctx.beginPath();
 for(let z=0;z<nz;z++)for(let x=0;x<=nx;x++)if(b.wall_x[z][x]){ctx.moveTo(x*sx,z*sz);ctx.lineTo(x*sx,(z+1)*sz);}
 for(let z=0;z<=nz;z++)for(let x=0;x<nx;x++)if(b.wall_y[z][x]){ctx.moveTo(x*sx,z*sz);ctx.lineTo((x+1)*sx,z*sz);}ctx.stroke();
 if(sources)for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)if(b.source_id[z][x]>=0||b.drain_capacity[z][x]>0){ctx.fillStyle=b.source_id[z][x]>=0?'#12ae75':'#10b5d1';ctx.fillRect((x+.22)*sx,(z+.22)*sz,.56*sx,.56*sz);}
 if(scene.config.inflow_m3_s>0){const f=scene.scenario_metadata.coarsening_factor||1,x=scene.config.source_x/f,z=scene.config.source_z/f;ctx.strokeStyle='#e47929';ctx.lineWidth=3;ctx.strokeRect(Math.floor(x)*sx,Math.floor(z)*sz,sx,sz);}
}
function makeBase(){
 if(!scene)return;const [nz,nx]=scene.grid.shape,b=scene.M_base,mode=$('baseLayer').value;
 const colors={terrain:[[119,168,130],[179,183,120],[173,151,120],[116,151,184],[159,174,126],[153,144,167]]};
 const img=image(i=>{const z=Math.floor(i/nx),x=i%nx;
  if(mode==='terrain')return colors.terrain[Math.abs(b.terrain_class[z][x])%6];
  if(mode==='roughness'){const n=Math.min(1,b.n[z][x]/.15);return [223-136*n,230-87*n,186-84*n];}
  if(mode==='structures'||mode==='sources')return [217,229,219];return rgbHeight(b.z[z][x]);});
 baseImage=document.createElement('canvas');baseImage.width=baseImage.height=S;const ctx=baseImage.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(img,0,0,S,S);
 if(mode==='elevation'){
  const sx=S/nx,sz=S/nz,interval=Math.max(.25,(terrainMax-terrainMin)/12);ctx.strokeStyle='#315c4944';ctx.lineWidth=.8;ctx.beginPath();
  for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){const band=Math.floor(b.z[z][x]/interval);
   if(x+1<nx&&Math.floor(b.z[z][x+1]/interval)!==band){ctx.moveTo((x+1)*sx,z*sz);ctx.lineTo((x+1)*sx,(z+1)*sz);}
   if(z+1<nz&&Math.floor(b.z[z+1][x]/interval)!==band){ctx.moveTo(x*sx,(z+1)*sz);ctx.lineTo((x+1)*sx,(z+1)*sz);}
  }ctx.stroke();
 }
 markers(ctx);
 const captions={elevation:`高程 ${fmt(terrainMin)}—${fmt(terrainMax)} m · 等高线与地物`,terrain:'独立地表类型图层 · 查点显示类别',roughness:'逐格 Manning 糙率 · 深色表示较大阻力',structures:'建筑与公共边阻水墙',sources:'绿色内源 · 紫色外源 · 青色排水口'};
 $('baseCaption').textContent=captions[mode];
}
function arrows(ctx){
 if(!$('flowArrows').checked)return;
 const [nz,nx]=scene.grid.shape,N=nx*nz,sx=S/nx,sz=S/nz,stride=Math.max(1,Math.ceil(Math.max(nx,nz)/Number($('flowDensity').value)));
 const labelBoxes=[];ctx.font='17px sans-serif';ctx.lineWidth=2;ctx.textAlign='center';
 for(let z=0;z<nz;z+=stride)for(let x=0;x<nx;x+=stride){
  // Choose a real fastest wet cell in each tile; narrow rivers survive sampling.
  let best=-1,speed=0;for(let zz=z;zz<Math.min(nz,z+stride);zz++)for(let xx=x;xx<Math.min(nx,x+stride);xx++){const i=zz*nx+xx,v=Math.hypot(display[N+i],display[2*N+i]);if(display[i]>=.01&&v>speed){best=i;speed=v;}}
  if(best<0||speed<.002)continue;
  const px=(best%nx+.5)*sx,pz=(Math.floor(best/nx)+.5)*sz,u=display[N+best]/speed,v=display[2*N+best]/speed,len=Math.min(35,8+speed*18),dx=u*len,dz=v*len;
  ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(px-dx/2,pz-dz/2);ctx.lineTo(px+dx/2,pz+dz/2);ctx.lineTo(px+dx/2-u*7-v*4,pz+dz/2-v*7+u*4);ctx.moveTo(px+dx/2,pz+dz/2);ctx.lineTo(px+dx/2-u*7+v*4,pz+dz/2-v*7-u*4);ctx.stroke();
  if($('flowLabels').checked){const label=fmt(speed,speed<.1?3:2),w=ctx.measureText(label).width;const ly=Math.max(20,Math.min(S-5,pz+25));const box=[px-w/2-3,ly-16,px+w/2+3,ly+4];if(labelBoxes.some(b=>box[0]<b[2]+3&&box[2]>b[0]-3&&box[1]<b[3]+3&&box[3]>b[1]-3))continue;labelBoxes.push(box);ctx.fillStyle='#143e49b8';ctx.fillRect(box[0],box[1],w+6,20);ctx.fillStyle='#fff';ctx.fillText(label,px,ly);}
 }
}
function draw(id,kind){
 const ctx=$(id).getContext('2d'),[nz,nx]=scene.grid.shape,N=nz*nx,b=scene.M_base;ctx.imageSmoothingEnabled=false;
 if(kind==='base')ctx.drawImage(baseImage,0,0);
 else{
  const img=image(i=>{const h=display[i],z=Math.floor(i/nx),x=i%nx;
   if(kind==='maximum'&&display[3*N+i]>=.01)return rgbDepth(display[3*N+i]);
   if(kind==='speed'&&h>=.01)return rgbSpeed(Math.hypot(display[N+i],display[2*N+i]));
   if(kind==='depth'&&h>=.01)return rgbDepth(h);return rgbHeight(b.z[z][x]);});
  ctx.drawImage(img,0,0,S,S);markers(ctx,false);if(kind==='depth')arrows(ctx);
 }
 externalMarkers(ctx);
 const sx=S/nx,sz=S/nz;ctx.strokeStyle='#f9bd38';ctx.lineWidth=2;ctx.strokeRect(selected.x*sx,selected.z*sz,sx,sz);
 if(kind==='base'){ctx.strokeStyle='#da982b';ctx.setLineDash([4,3]);for(const p of patches)ctx.strokeRect(p.x*sx,p.z*sz,(p.width||1)*sx,(p.height||1)*sz);ctx.setLineDash([]);}
}
function inspect(){
 const [nz,nx]=scene.grid.shape,N=nz*nx,x=selected.x,z=selected.z,i=z*nx+x,b=scene.M_base,h=display[i],u=display[N+i],v=display[2*N+i];
 const types=terrain.terrain_ids||{},label=scene.provenance==='synthetic_multilayer_demo'?b.terrain_class[z][x]:types[b.terrain_class[z][x]]||b.terrain_class[z][x];
 const incoming=(scene.M_forcing?.external_inflows||[]).filter(p=>p.enabled&&p.cells.some(c=>c.x===x&&c.z===z)).map(p=>`${p.id} ${fmt(discharge(p.hydrograph,time))} m³/s`).join('；');
 $('mapCellDetail').textContent=`${fmt(time,1)} 秒 · 格 (${x}, ${z}) · 地面 ${fmt(b.z[z][x])} m · 地表 ${label} · 糙率 ${fmt(b.n[z][x],3)} · 建筑 ${b.solid[z][x]?'是':'否'}\n水深 ${fmt(h,3)} m · 水面 ${fmt(b.z[z][x]+h,3)} m · 流速 ${fmt(Math.hypot(u,v),3)} m/s · X / Z 速度 ${fmt(u,3)} / ${fmt(v,3)} m/s\n累计最大水深 ${fmt(display[3*N+i],3)} m · 本格储水 ${fmt(h*scene.grid.dx_m**2)} m³ · 排水能力 ${fmt(b.drain_capacity[z][x])} m³/s · 水源编号 ${b.source_id[z][x]}${incoming?'\n外源来水：'+incoming:''}`;
}
function render(pair){
 if(!scene)return;pair??=framePair(process.frames,time);draw('baseMap','base');draw('depthMap','depth');draw('speedMap','speed');draw('maximumMap','maximum');inspect();
 $('worldTime').textContent=`${fmt(time,1)} 秒`;$('mapTimeline').value=time;
 const limit=Math.max(.1,Number($('depthScale').value)||3);$('depthLimit').textContent=`≥ ${fmt(limit)} m`;$('maximumLimit').textContent=`≥ ${fmt(limit)} m`;
 $('depthCaption').textContent=pair&&pair.alpha>1e-6&&pair.alpha<1-1e-6?'帧间平滑显示 · 流向与 m/s':'计算水情 · 流向与 m/s';
 const exact=pair?process.frames[pair.lo]:scene.M_water.summary,b=exact.balance;
 $('mapBalance').textContent=`最近已计算帧 ${fmt(exact.time_s,1)} 秒 · 水量收支误差 ${Number(b.residual_m3).toExponential(2)} m³\n初始 ${fmt(b.initial_m3)} + 降雨 ${fmt(b.rainfall_m3)} + 指定来水 ${fmt(b.inflow_m3)} + 日常基流 ${fmt(b.baseflow_m3)} + 变化水源 ${fmt(b.scheduled_inflow_m3)} + 外源来水 ${fmt(b.external_inflow_m3||0)} + 外海流入 ${fmt(b.sea_inflow_m3)} − 外海流出 ${fmt(b.sea_outflow_m3)} − 自由出流 ${fmt(b.outflow_m3)} − 入渗 ${fmt(b.infiltration_m3)} − 排水 ${fmt(b.drainage_m3)} = 当前 ${fmt(b.current_m3)} m³`;
}
function animate(now){
 if(playing&&scene&&now-lastDraw>=33){const elapsed=(now-lastDraw)/1000;lastDraw=now;
  const next=playbackTime(time,elapsed,Number($('playSpeed').value),available());
  if(renderTime(next)&&time>=available()-1e-6&&!active()){playing=false;controls();}
 }
 requestAnimationFrame(animate);
}
$('createMap').onclick=()=>create(false);$('runMap').onclick=()=>create(true);
$('cancelMap').onclick=async()=>{try{await post(`/api/maps/${scene.id}/cancel`);process.cancel_requested=true;controls();}catch(e){status(e.message,true);}};
function seekTo(target,resume=false){
 playing=false;const token=++seekVersion,pair=framePair(process.frames,target);
 if(!pair){controls();return;}seeking=true;controls();
 Promise.all([requestFrame(pair.lo),requestFrame(pair.hi)]).then(()=>{
  if(token!==seekVersion)return;
  if(renderTime(target)){status(`已定位 ${fmt(time,1)} 秒；各地图层同步。`);playing=resume;}
  seeking=false;controls();
 });
}
$('playMap').onclick=()=>{if(!playing&&time>=available()-1e-6){seekTo(0,true);return;}seekVersion++;seeking=false;playing=!playing;controls();};
$('rewindMap').onclick=()=>seekTo(0);
$('mapTimeline').oninput=()=>seekTo(Math.min(Number($('mapTimeline').value),available()));
$('baseLayer').onchange=()=>{makeBase();render();};
for(const id of ['flowArrows','flowLabels','flowDensity','depthScale'])$(id).onchange=()=>render(framePair(process.frames,time));
for(const id of ids){
 $(id).onclick=e=>{if(!scene||busy)return;const r=$(id).getBoundingClientRect(),[nz,nx]=scene.grid.shape;
  selected={x:Math.max(0,Math.min(nx-1,Math.floor((e.clientX-r.left)/r.width*nx))),z:Math.max(0,Math.min(nz-1,Math.floor((e.clientY-r.top)/r.height*nz)))};
  if(pickInflow){const f=scene.scenario_metadata.coarsening_factor||1;$('sourceX').value=selected.x*f;$('sourceZ').value=selected.z*f;pickInflow=false;$('pickInflow').setAttribute('aria-pressed','false');status('入水位置已选定，点击添加手动强制来水点。');}
 else
  if($('editTool').value!=='inspect'){
   if(active()){status('固定环境需在计算结束或取消后编辑。',true);return;}
   patches.push({kind:$('editTool').value,...selected,width:Number($('patchWidth').value),height:Number($('patchHeight').value),value:Number($('patchValue').value)});draft();status('已加入地图编辑草稿，应用或重新模拟后生效。');
  }render(framePair(process.frames,time));
 };
 $(id).onkeydown=e=>{const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];if(!scene||!d)return;e.preventDefault();selected.x=Math.max(0,Math.min(scene.grid.shape[1]-1,selected.x+d[0]));selected.z=Math.max(0,Math.min(scene.grid.shape[0]-1,selected.z+d[1]));render(framePair(process.frames,time));};
}
$('editTool').onchange=()=>{$('patchValue').value=({roughness:.06,drain:1,source:0,terrain:0})[$('editTool').value]??1;};
$('clearDraft').onclick=()=>{patches=[];draft();render();};
$('mapSource').onchange=()=>{imported=null;patches=[];externalDraft=[];inflowDirty=true;inletControls();draft();$('hydrographs').value='{}';$('mapBoundary').value=$('mapSource').value==='demo'?'closed':'ocean';status('地图来源已更改，应用或开始模拟后生效。');};
function download(blob,name){const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function displayedForcing(){
 const [nz,nx]=scene.grid.shape,rate=Array.from({length:nz},()=>Array(nx).fill(0)),entries=scene.M_forcing?.external_inflows||[];
 const byId={};for(const p of entries){const q=p.enabled?discharge(p.hydrograph,time):0;byId[p.id]=q;for(const cell of p.cells)rate[cell.z][cell.x]+=q*cell.weight/scene.grid.dx_m**2;}
 const pair=framePair(process.frames,time),actual=pair?process.frames[pair.lo]:scene.M_water.summary;
 return {external_inflows:entries,river_crossings:scene.M_forcing?.river_crossings||[],
  inlet_count:scene.M_forcing?.inlet_count||Array.from({length:nz},()=>Array(nx).fill(0)),
  external_rate_m_s:rate,instantaneous_discharge_by_id_m3_s:byId,
  accumulated_m3:actual.external_inflow_by_id_m3||{},accounted_frame_time_s:actual.time_s};
}
function saveJson(data,name){download(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),name);}
$('saveMapConfig').onclick=()=>{try{saveJson(config(),'hyra-map-flood-scenario.json');}catch(e){status(e.message,true);}};
$('exportFrame').onclick=()=>{if(!scene)return;const [nz,nx]=scene.grid.shape,N=nz*nx,pair=framePair(process.frames,time),layers={};for(const [k,j] of ['depth','u','v','maximum_depth'].entries())layers[j]=Array.from({length:nz},(_,z)=>Array.from(display.slice(k*N+z*nx,k*N+(z+1)*nx)));saveJson({schema:'hyra-map-display-v1',time_s:time,grid:scene.grid,display_interpolated:Boolean(pair&&pair.alpha>1e-6&&pair.alpha<1-1e-6),source_frame_times_s:pair?[process.frames[pair.lo].time_s,process.frames[pair.hi].time_s]:[0],M_forcing:displayedForcing(),layers},'hyra-displayed-water.json');};
$('exportPng').onclick=()=>$('depthMap').toBlob(blob=>download(blob,'hyra-map-water-depth.png'));
$('importMapConfig').onchange=async()=>{try{const file=$('importMapConfig').files[0];if(!file)return;if(file.size>8*1024*1024)throw Error('场景配置文件过大');const c=JSON.parse(await file.text());if(!c||!['current','demo'].includes(c.mode)||c.agents||c.legacy_exploration)throw Error('请选择纯地图与洪水场景配置');imported=c;patches=[];externalDraft=normalizeInflows(c.external_inflows||[]);inflowDirty=true;draft();$('mapSource').value=c.mode;for(const [id,name] of Object.entries(fields))if(c.flood?.[name]!==undefined)$(id).value=c.flood[name];for(const [id,name] of [['mapBaseflow','baseflow'],['mapInitialWater','initial_water']])if(c.flood?.[name]!==undefined)$(id).checked=c.flood[name];if(c.flood?.boundary)$('mapBoundary').value=c.flood.boundary;if(c.flood?.inflow_m3_s>0&&!externalDraft.some(p=>p.id==='legacy-point'))externalDraft.push({id:'legacy-point',kind:'manual',x:c.flood.source_x,z:c.flood.source_z,enabled:true,hydrograph:[[0,c.flood.inflow_m3_s]]});$('hydrographs').value=JSON.stringify(c.layers?.hydrographs||{});inletControls();status('地图场景配置已载入，应用或开始模拟后生效。');}catch(e){status(e.message,true);}};
json('/api/current').then(async t=>{
 terrain=t;$('terrainInfo').textContent=`当前种子 ${t.config.seed} · ${t.shape[1]} × ${t.shape[0]} 原始网格。`;
 for(const [id,size] of [['sourceX',t.shape[1]],['sourceZ',t.shape[0]]]){$(id).max=size-1;$(id).value=Math.min(size-1,Number($(id).value));}
 controls();let saved=location.hash.slice(1);if(!saved)try{saved=localStorage.getItem('hyra-map-process')||'';}catch{}
 if(/^maps-[a-f0-9]{12}$/.test(saved)){busy=true;controls();try{await restore(saved);return;}catch{try{localStorage.removeItem('hyra-map-process');}catch{}history.replaceState(null,'',location.pathname);}finally{busy=false;controls();}}
 await create(false);
}).catch(e=>status(e.message,true));
requestAnimationFrame(animate);
