import {auditWarnings,auditSummary,drawChannelAudit} from './channel-audit.mjs';
import {discharge,windowCurve,portLabel,validateCurve} from './inflow-controls.mjs';
import {framePair,blendFrames,playbackTime} from './maps-playback.mjs';
import {StructureEditor,staticSolids} from './structure-editor.mjs?v=2';
import {PeopleEditor} from './people-editor.mjs?v=1';
import {hydrateScene} from './map-arrays.mjs';
import {meshGeometry,meshState,meshPixel,selectedLeaf,exportMesh} from './map-mesh.mjs';
import {inwardVectors,inflowVectors,bearingLabel,inflowDirection,portCode,visibleCrossings,oceanMask,speedStats,speedColor} from './map-display.mjs?v=5';
const $=id=>document.getElementById(id),ids=['baseMap','depthMap','speedMap','maximumMap'],S=640;
const fmt=(v,n=2)=>Number(v).toLocaleString('zh-CN',{maximumFractionDigits:n});
let terrain,scene,process={state:'ready',frames:[]},patches=[],selected={x:0,z:0},imported=null;
let externalDraft=[],pickInflow=false,inflowDirty=false;
const selectedRiverIds=new Set();
let busy=false,playing=false,time=0,display,baseImage,terrainMin=0,terrainMax=1,epoch=0,pollTimer,seekVersion=0,seeking=false;
let speedScale=0,speedOceanMask=[],focusedPort=null,markerHits=[];
let meshDisplay=null,geometry=null,meshOcean=[];
let structureEditor,structureDragClick=false;
let peopleEditor;
const meshCache=new Map();
const cache=new Map(),pending=new Map();let lastDraw=0;
const fields={mapUnit:'metres_per_block',mapGridFactor:'grid_factor',mapDuration:'duration_s',
 mapManning:'manning',mapRain:'rain_mm_h',mapRainUntil:'rain_duration_s',mapInfiltration:'infiltration_mm_h',
 mapBaseflowRate:'baseflow_m3_s',mapWarmup:'warmup_s',sourceX:'source_x',sourceZ:'source_z'};
const numericalFields={mapFlux:['flux_method','rusanov'],mapSpatialOrder:['spatial_order',2],
 mapIntegrator:['time_integrator','rk2'],mapLowFroude:['low_froude',0],mapCfl:['cfl',.4],
 mapLimiter:['limiter_theta',1.5],mapVelocityDepth:['velocity_depth',1e-6],
 mapMaxTimestep:['max_timestep',1],mapMaxSteps:['max_steps',500000]};
function loadNumerics(flood={}){for(const [id,[key,fallback]] of Object.entries(numericalFields))$(id).value=flood[key]??fallback;$('mapRoughnessMode').value=flood.roughness_mode??'uniform';}
function readNumerics(){
 const result={};
 for(const [id,[key,fallback]] of Object.entries(numericalFields)){
  const el=$(id);if(!el.checkValidity()){
   for(let parent=el.parentElement;parent;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;
   el.reportValidity();throw Error('请检查水动力数值参数的范围。');
  }
  if(!el.value)throw Error('请选择有效的水动力计算方法。');
  result[key]=typeof fallback==='number'?Number(el.value):el.value;
 }return result;
}
function status(text,error=false){$('mapStatus').textContent=text;$('mapStatus').classList.toggle('error',error);}
async function json(url,options){const r=await fetch(url,options),j=await r.json();if(!r.ok){const error=Error(j.error||'读取失败');error.process_id=j.process_id;error.status=r.status;throw error;}return j;}
function post(url,data={}){return json(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});}
function active(){return ['queued','running'].includes(process.state);}
function layeredObstacles(){return scene?.scenario_metadata?.imported?.obstacle_layers||terrain?.imported?.obstacle_layers;}
function controls(){
 for(const el of document.querySelectorAll('.settings-panel input,.settings-panel select,.settings-panel textarea'))el.disabled=busy||active();
 $('mapGridFactor').disabled=busy||active()||$('mapSource').value==='demo'||!!layeredObstacles();
 for(const el of $('numericsPanel').querySelectorAll('input,select,button'))el.disabled=busy||active();
 for(const id of ['createMap','runMap','importMapConfig','addRiverInflow','addManualInflow','pickInflow'])$(id).disabled=busy||active()||!terrain;
 $('runMap').textContent=scene&&process.state!=='ready'?'重新模拟洪水':'开始洪水模拟';
 $('addRiverInflow').disabled=busy||active()||!selectedRiverIds.size;
 $('addRiverInflow').hidden=!selectedRiverIds.size;
 $('saveInflowSettings').disabled=busy||active()||!externalDraft.some(p=>p.id===focusedPort);
 for(const el of $('riverCandidates').querySelectorAll('input'))el.disabled=busy||active();
 for(const el of $('externalInflowList').querySelectorAll('button,input'))el.disabled=busy||active();
 peopleEditor?.fields();
 $('cancelMap').disabled=busy||!active()||process.cancel_requested;
 $('cancelMap').hidden=!active();
 $('playMap').textContent=playing?'暂停播放':time>=available()&&available()>0?'重播':'播放';
 $('playMap').disabled=!scene||(process.frames.length<2&&!playing);
 $('rewindMap').disabled=!scene;$('mapTimeline').disabled=!scene||process.frames.length<2;
 $('exportFrame').disabled=!scene||seeking;$('exportPng').disabled=!scene||seeking;
 $('downloadMap').hidden=!process.archive_url;
 if(process.archive_url)$('downloadMap').href=process.archive_url;
}
function config(){
 const mode=$('mapSource').value,flood={...(imported?.flood||{}),frames:imported?.flood?.frames??121};
 for(const [id,name] of Object.entries(fields))flood[name]=Number($(id).value);
 Object.assign(flood,readNumerics());
 flood.refine_inland_water=false;flood.refinement_buffer=0;
 flood.inflow_m3_s=0; // The point editor now uses explicit, separately accounted external sources.
 flood.boundary=$('mapBoundary').value;flood.initial_water=$('mapInitialWater').checked;
 flood.roughness_mode=$('mapRoughnessMode').value;
 flood.baseflow=mode==='current'&&$('mapBaseflow').checked;
 if(mode==='demo'){
  flood.grid_factor=1;
  flood.warmup_s=0;
  flood.source_x=Math.min(31,flood.source_x);flood.source_z=Math.min(31,flood.source_z);
 }
 const layers={...(imported?.layers||{}),grid_factor:imported?.layers?.grid_factor??1,
  patches:[...(imported?.layers?.patches||[]),...patches]};
 const hydro=JSON.parse($('hydrographs').value);
 if(Object.keys(hydro).length)layers.hydrographs=hydro;else if(mode==='current')layers.hydrographs={};
 return {...(imported||{}),mode,terrain_id:terrain.id,flood,layers,structures:structureEditor.config(),people:peopleEditor.config(),external_inflows:externalDraft.map(p=>({...p}))};
}
function draft(){$('draftInfo').textContent=`草稿：${patches.length} 项编辑；应用或重新模拟后生效。`;}
function pendingSettings(dirty=true){$('channelAuditStale').hidden=!dirty;$('draftBadge').textContent=dirty?'有未应用修改':'已应用';$('draftBadge').classList.toggle('is-dirty',dirty);if(scene&&baseImage)render();}
function showView(view){
 document.querySelector('.map-grid').dataset.view=view;
 for(const button of document.querySelectorAll('[data-view]'))if(button.tagName==='BUTTON')button.setAttribute('aria-selected',String(button.dataset.view===view));
 for(const figure of document.querySelectorAll('figure[data-map]'))figure.hidden=view!=='compare'&&figure.dataset.map!==view;
}
function showPanel(panel){
 for(const button of document.querySelectorAll('[data-panel]'))button.setAttribute('aria-selected',String(button.dataset.panel===panel));
 for(const section of document.querySelectorAll('.settings-panel'))section.hidden=section.id!==`panel-${panel}`;
 document.querySelector('.settings-scroll').scrollTop=0;
 if(panel==='inflows'||panel==='environment'||panel==='people')showView('base');
}
async function create(start=false){
 if(busy||active())return;
 try{
  peopleEditor.assertValid();
  const c=config();busy=true;controls();status('正在准备地图与洪水场景…');
  const next=await hydrateScene(await post('/api/maps',c));inflowDirty=false;installScene(next);pendingSettings(false);remember();
  status('地图已应用。可开始整段洪水模拟，或编辑固定环境。');
  if(start){await post(`/api/maps/${scene.id}/run`);process.state='queued';remember();playing=true;status('洪水开始计算，过程就绪后自动连续播放。');poll(epoch);}
 }catch(e){let recover=e.process_id;if(!recover&&e.status===409)try{recover=localStorage.getItem('hyra-map-process');}catch{}if(recover){try{await restore(recover);status(e.message);}catch(recovery){status(recovery.message,true);}}else status(e.message,true);}finally{busy=false;controls();}
}
function remember(){try{localStorage.setItem('hyra-map-process',scene.id);}catch{}history.replaceState(null,'',`#${scene.id}`);}
function installScene(next){
 clearTimeout(pollTimer);epoch++;cache.clear();pending.clear();scene=next;showChannelAudit();process=next.process;time=0;playing=false;seeking=false;seekVersion++;selected={x:0,z:0};
 structureEditor.load(next.M_structures?.objects||next.scenario_config?.structures||[]);
 peopleEditor.load(next.M_people||next.scenario_config?.people||null);
 meshCache.clear();geometry=meshGeometry(scene.M_mesh,scene.grid.shape);meshDisplay=geometry?meshState(scene.M_mesh):null;
 const w=scene.M_water,[rows,cols]=scene.grid.shape,N=rows*cols;display=new Float32Array(4*N);
 for(const [k,name] of ['h','u','v','maximum_depth'].entries())for(let z=0;z<rows;z++)display.set(w[name][z],k*N+z*cols);
 const factor=scene.scenario_metadata.coarsening_factor||1;
 $('nativeGridInfo').textContent=`计算网格 ${cols} × ${rows} 格 · 每格 ${fmt(scene.grid.dx_m)} 米 · ${factor>1?`每 ${factor} × ${factor} 个原始格合并为一个计算格`:'沿用原始地形网格'}`;
 speedOceanMask=oceanMask(scene.M_base.terrain_class,scene.scenario_metadata.terrain_ids||(scene.provenance==='synthetic_multilayer_demo'?{}:terrain.terrain_ids));
 meshOcean=geometry?Uint8Array.from(scene.M_mesh.parent,p=>speedOceanMask[p]):[];
 const refinement=scene.scenario_metadata.refinement;
 $('refinementInfo').textContent=refinement?.enabled?'正在查看历史加密过程；重新应用或模拟将使用统一格宽。':`统一格宽 ${fmt(scene.grid.dx_m)} 米 · ${fmt(N,0)} 个计算格。${factor>1?'高程和水深按面积平均，窄河道细节会变粗。':''}`;
 const networkCheck=scene.scenario_metadata.river_network_grid_check;
 if(networkCheck)$('refinementInfo').textContent+=` 河网复查：${networkCheck.status==='not_applicable'?'导入实验采用指定边界，无自然河网分类':networkCheck.warnings.length?networkCheck.warnings.join('；'):'未发现河道分类分区变化；仍需验证水位与流量'}。`;
 if(layeredObstacles())$('refinementInfo').textContent+=' 地面与阻水占地分层；障碍轮廓由原高程推断。本图格宽固定，更换格宽需重新生成障碍占地。';
 const n=scene.config;$('numericsApplied').textContent=`本场景：${n.flux_method==='knp'?'KNP':'Rusanov'} · ${n.spatial_order??2} 阶空间 · ${(n.time_integrator||'rk2').toUpperCase()} · 低弗劳德修正 ${n.low_froude??0} · CFL ${n.cfl??.4}`;
 const rough=scene.scenario_metadata.roughness;
 $('roughnessInfo').textContent=rough?`${rough.base_source==='explicit_n_layer'?'显式导入糙率':rough.mode==='landcover'?'按地表分类（'+(rough.source_kind==='actual_surface'?'完整地表':'快速预览近似')+'）':'统一糙率'} · n ${fmt(rough.effective_min_n??n.manning,3)}—${fmt(rough.effective_max_n??n.manning,3)} · 手动编辑优先。${rough.mode==='landcover'?'分类值为未率定初值。':''}`:'示例或历史糙率图层；手动编辑优先。';
 terrainMin=Infinity;terrainMax=-Infinity;for(const row of scene.M_base.z)for(const h of row){terrainMin=Math.min(terrainMin,h);terrainMax=Math.max(terrainMax,h);}
 inletControls();makeBase();$('mapTimeline').max=scene.config.duration_s;$('durationLabel').textContent=`${fmt(scene.config.duration_s)} 秒`;
 $('worldInfo').textContent=`${scene.provenance==='synthetic_multilayer_demo'?'建筑与水源示例':scene.provenance==='imported_experiment_raster'?'导入实验地形':'当前生成地图'} · ${cols} × ${rows} 计算格 · ${geometry?`最细 ${fmt(scene.grid.dx_m/2)} 米`:`格宽 ${fmt(scene.grid.dx_m)} 米`} · ${layeredObstacles()?'地面与障碍分层':'全部图层同步'}`;
 $('computeStatus').textContent='地图已建立，初始水体已显示。';$('mapProgress').value=0;render();
}
async function restore(id){
 let next;try{next=await json(`/api/maps/${id}`);}catch(e){if(e.status!==404||!/^(?:maps|merewether)-[a-z0-9-]+$/.test(id))throw e;next=await json(`/generated/${id}/maps/replay.json`);}
 await hydrateScene(next);installScene(next);remember();
 const mode=scene.provenance==='synthetic_multilayer_demo'?'demo':'current';
 externalDraft=normalizeInflows(next.scenario_config?.external_inflows||next.M_forcing?.external_inflows||[]);inflowDirty=false;inletControls();
 focusedPort=externalDraft[0]?.id??null;
 imported=scene.scenario_config||{mode,terrain_id:terrain.id,layers:{...scene.M_base,grid_factor:scene.config.grid_factor??1,hydrographs:scene.source_hydrographs},flood:scene.config};
 patches=[];draft();$('mapSource').value=mode;
 for(const [key,name] of Object.entries(fields))$(key).value=scene.config[name]??(name==='grid_factor'?1:'');
 loadNumerics(scene.config);
 $('mapBoundary').value=scene.config.boundary;$('mapInitialWater').checked=scene.config.initial_water;$('mapBaseflow').checked=scene.config.baseflow;
 if(scene.config.inflow_m3_s>0&&!externalDraft.some(p=>p.id==='legacy-point'))externalDraft.push({id:'legacy-point',kind:'manual',x:scene.config.source_x,z:scene.config.source_z,enabled:true,hydrograph:[[0,scene.config.inflow_m3_s]]});
 inletControls();restoreInflowEditor();makeBase();$('hydrographs').value=JSON.stringify(imported.layers?.hydrographs||{});
 await poll(epoch);status(active()?'已恢复正在运行的洪水过程，可以继续查看或取消。':process.state==='ready'?'场景已就绪，点击“开始洪水模拟”计算。':'已恢复洪水过程，可播放与导出。');controls();
 if(process.frames.length){const pair=framePair(process.frames,0);await Promise.all([requestFrame(pair.lo),requestFrame(pair.hi)]);renderTime(0);}
}
async function poll(token){
 try{
  const id=scene.id,p=await json(scene.archive_status_url||`/api/maps/${id}/status`);if(token!==epoch)return;process=p;
  if(updateSpeedScale())draw('speedMap','speed');
  const eta=active()&&Number.isFinite(p.estimated_remaining_s)&&p.wall_elapsed_s>=5?` · 预计剩余 ${Math.ceil(p.estimated_remaining_s/60)} 分钟`:'';
  $('computeStatus').textContent=p.message+eta;$('mapProgress').value=p.progress;
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
 const expected=scene.grid.shape[0]*scene.grid.shape[1]*4,meshCount=geometry?.n;
 const task=(async()=>{
  const read=async url=>{const r=await fetch(url);if(!r.ok)throw Error('洪水过程帧读取失败');return r.arrayBuffer();};
  if(meshCount&&!f.mesh_url)throw Error('加密计算帧缺少真实子格水情');
  const [buffer,meshBuffer]=await Promise.all([read(f.url),meshCount?read(f.mesh_url):null]);
  if(token!==epoch)return null;
  if(buffer.byteLength!==expected*4)throw Error('洪水过程帧不完整');
  const data=new Float32Array(buffer);if(data.some(v=>!Number.isFinite(v)))throw Error('过程帧包含无效数值');
  if(meshCount){
   if(meshBuffer.byteLength!==meshCount*4*4)throw Error('加密计算帧不完整');
   const leaf=new Float32Array(meshBuffer);if(leaf.some(v=>!Number.isFinite(v)))throw Error('加密计算帧包含无效数值');meshCache.set(index,leaf);
  }
  const capacity=Math.max(2,Math.min(8,Math.floor(128*1024**2/(buffer.byteLength+(meshBuffer?.byteLength||0)))));
  cache.set(index,data);while(cache.size>capacity){const victim=[...cache.keys()].find(k=>k!==index);cache.delete(victim);meshCache.delete(victim);}
  return data;
 })().catch(e=>{if(token===epoch){playing=false;status(e.message,true);controls();}return null;}).finally(()=>{if(token===epoch)pending.delete(index);});
 pending.set(index,task);return task;
}
function renderTime(target){
 const pair=framePair(process.frames,target);if(!pair)return false;
 const a=cache.get(pair.lo),b=cache.get(pair.hi);if(!a||!b){requestFrame(pair.lo);requestFrame(pair.hi);return false;}
 if(geometry){const ma=meshCache.get(pair.lo),mb=meshCache.get(pair.hi);if(!ma||!mb)return false;blendFrames(ma,mb,pair.alpha,meshDisplay);}
 blendFrames(a,b,pair.alpha,display);time=pair.time;render(pair);
 requestFrame(Math.min(pair.hi+1,process.frames.length-1));return true;
}
function crossings(){return $('mapSource').value==='demo'?[]:
 (scene?.scenario_config?.terrain_id===terrain?.id?terrain?.river_inlets?.crossings:scene?.M_forcing?.river_crossings)||terrain?.river_inlets?.crossings||[];}
function inletCurve(){return validateCurve($('customInflowCurve').checked?JSON.parse($('inflowCurve').value):windowCurve(Number($('inflowStart').value),Number($('inflowEnd').value),Number($('mapInflow').value)));}
function inletVelocity(){const v=Number($('inflowVelocity').value);if(!Number.isFinite(v)||v<0||v>30)throw Error('入流速度须为0—30 m/s，0为自动');return v;}
function inletDirectionSettings(){
 const mode=$('inflowDirectionMode').value,angle=Number($('inflowAngle').value);
 if(!Number.isFinite(angle))throw Error('请输入有效方位角');
 return {inlet_direction_mode:mode,inlet_angle_deg:(angle%360+360)%360};
}
function checkDirection(point,settings){
 if(settings.inlet_direction_mode!=='angle')return;
 const vector=inflowVectors({...point,...settings})[0];
 for(const edge of new Set((point.faces||point.boundary_faces||[{edge:point.edge}]).map(f=>f.edge))){
  const n=inwardVectors[edge];if(n&&vector[0]*n[0]+vector[1]*n[1]<.15)throw Error('此方位没有充分指向图内，请转动方向或选择“沿河道”。');
 }
}
function restoreInflowEditor(p=externalDraft[0]){
 if(!p)return;const curve=p.hydrograph,simple=curve.length===2&&curve[1][1]===0;
 $('mapInflow').value=curve[0][1];$('inflowStart').value=curve[0][0];
 if(simple)$('inflowEnd').value=curve[1][0];$('customInflowCurve').checked=!simple;
 $('inflowCurve').value=JSON.stringify(curve);$('inflowVelocity').value=p.inlet_velocity_m_s??0;
 $('inflowDirectionMode').value=p.inlet_direction_mode??'auto';$('inflowAngle').value=p.inlet_angle_deg??0;
 $('inflowAngleRange').value=$('inflowAngle').value;directionPreview();
}
function saveInflowEditor(p){
 const geometry=crossings().find(v=>v.id===p.id)||p,curve=inletCurve(),velocity=inletVelocity(),direction=inletDirectionSettings();
 if(p.kind==='river')checkDirection(geometry,direction);
 p.hydrograph=curve;if(p.kind==='river')Object.assign(p,{inlet_velocity_m_s:velocity},direction);
 forcingEdited();
}
function normalizeInflows(values){if(!Array.isArray(values)||values.length>32)throw Error('外源点列表最多32项');return values.map(p=>({...p,kind:p.kind??'manual',enabled:p.enabled??true,hydrograph:validateCurve(p.hydrograph??[[0,50],[120,0]])}));}
function shownCrossings(){return visibleCrossings(crossings(),$('crossingDisplay').value,new Set([...externalDraft.filter(p=>p.kind==='river').map(p=>p.id),...selectedRiverIds,...(focusedPort?[focusedPort]:[])]),terrain?.shape);}
function directionPreview(){
 const points=crossings().filter(p=>selectedRiverIds.has(p.id)||p.id===focusedPort);
 const manual=externalDraft.find(p=>p.id===focusedPort&&p.kind==='manual');if(manual)points.push(manual);
 const setting=inletDirectionSettings();$('inflowAngleControl').hidden=setting.inlet_direction_mode!=='angle';
 const resolved=points.map(p=>({...p,...setting}));
 $('inflowDirection').textContent=resolved.length?resolved.map(p=>`${portCode(p,externalDraft.some(e=>e.id===p.id&&e.enabled))}：${inflowDirection(p)}`).join('\n'):'先选择或定位一个河口，方向将自动贴合附近河道。';
 const vector=resolved.length?inflowVectors(resolved[0])[0]:setting.inlet_direction_mode==='angle'?[Math.cos(setting.inlet_angle_deg*Math.PI/180),Math.sin(setting.inlet_angle_deg*Math.PI/180)]:null;
 $('inflowBearingArrow').style.transform=`rotate(${vector?Math.atan2(vector[1],vector[0])*180/Math.PI:0}deg)`;
 $('inflowBearingReadout').textContent=vector?bearingLabel(vector):'选择河口';
}
function focusPort(p){
 focusedPort=p.id;
 showPanel('inflows');
 if(scene){const f=scene.scenario_metadata.coarsening_factor||1;selected={x:Math.floor(p.x/f),z:Math.floor(p.z/f)};}
 inletControls();if(scene)render(framePair(process.frames,time));
 status(`${portLabel(p)}；配置来水方向：${inflowDirection(p)}`);
 const pane=document.querySelector('.settings-scroll'),card=document.querySelector('.direction-card');
 pane.scrollTop+=card.getBoundingClientRect().top-pane.getBoundingClientRect().top-12;
}
function inletControls(){
 const points=crossings();for(const id of selectedRiverIds)if(!points.some(p=>p.id===id))selectedRiverIds.delete(id);
 const shown=shownCrossings();
 $('riverCandidates').replaceChildren(...shown.map(p=>{
  const row=document.createElement('div');row.className='candidate-row'+(p.id===focusedPort?' is-focused':'');
  const label=document.createElement('label');label.className='check';const check=document.createElement('input');check.type='checkbox';check.checked=selectedRiverIds.has(p.id);
  check.setAttribute('aria-label',`选择河口 ${p.id}`);check.onchange=()=>{if(check.checked)selectedRiverIds.add(p.id);else selectedRiverIds.delete(p.id);directionPreview();controls();if(scene)render(framePair(process.frames,time));};
  const enabled=externalDraft.some(v=>v.id===p.id&&v.enabled);
  const text=document.createElement('span');text.className=enabled?'port-active':p.role==='inlet'?'port-in':p.role==='outlet'?'port-out':'';text.textContent=`${portCode(p,enabled)} · ${portLabel(p)}${externalDraft.some(v=>v.id===p.id)?' · 已添加':''}`;
  label.append(check,text);const locate=document.createElement('button');locate.textContent='定位';locate.setAttribute('aria-label',`定位${p.label||p.id}`);locate.onclick=()=>focusPort(p);row.append(label,locate);return row;
 }));
 if(!shown.length)$('riverCandidates').textContent=points.length?'此筛选下没有河口，可切换为主要河口或全部。':'没有河道边界交点，可手动选点';
 $('riverInletInfo').textContent=`共 ${points.length} 处交点：${points.filter(p=>p.role==='inlet').length} 入、${points.filter(p=>p.role==='outlet').length} 出、${points.filter(p=>p.role==='uncertain').length} 待确认。显示 ${shown.length} 处，隐藏 ${points.length-shown.length} 处。主要河口按宽度、方向可信度和间距筛选；完整交点保留，可切换全部。`;
 $('crossingMapInfo').textContent=`固定地图显示 ${shown.length}/${points.length} 处候选；其他地图仅显示已配置／所选。点击标记或左侧“定位”查看方向。`;
 $('externalInflowCount').textContent=`已配置 ${externalDraft.length} 个外源来水点，启用 ${externalDraft.filter(p=>p.enabled).length} 个${inflowDirty?'（草稿，需应用或重新模拟）':''}。最多32个。`;
 $('externalInflowList').replaceChildren(...externalDraft.map(p=>{
  const geometry=points.find(v=>v.id===p.id)||p;
  const row=document.createElement('div');row.className='inlet-row'+(p.id===focusedPort?' is-focused':'');const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=p.enabled;check.setAttribute('aria-label',`启用入水点 ${p.id}`);
  check.onchange=()=>{p.enabled=check.checked;forcingEdited();};label.append(check,document.createTextNode(`${portCode(geometry,p.enabled)} · `+(p.kind==='river'?portLabel(geometry):`手动入水点 ${p.id} · (${p.x},${p.z})`)));
  const desc=document.createElement('small');desc.textContent=p.hydrograph.map(([t,q])=>`${fmt(t)}s: ${fmt(q)}m³/s`).join(' → ')+(p.kind==='river'?` · 速率 ${p.inlet_velocity_m_s?fmt(p.inlet_velocity_m_s)+' m/s':'自动'}`:' · 体积补给')+`\n${inflowDirection({...geometry,...p})}`;
  const locate=document.createElement('button');locate.textContent='定位／载入';locate.setAttribute('aria-label',`定位并载入入水点 ${p.id}`);locate.onclick=()=>{restoreInflowEditor(p);focusPort(geometry);};
  const edit=document.createElement('button');edit.textContent='保存当前设置';edit.setAttribute('aria-label',`修改入水点 ${p.id}`);edit.onclick=()=>{try{saveInflowEditor(p);}catch(e){status(e.message,true);}};
  const del=document.createElement('button');del.textContent='移除';del.setAttribute('aria-label',`移除入水点 ${p.id}`);del.onclick=()=>{externalDraft=externalDraft.filter(v=>v.id!==p.id);forcingEdited();};
  row.append(label,desc,locate,edit,del);return row;
 }));$('saveInflowSettings').disabled=busy||active()||!externalDraft.some(p=>p.id===focusedPort);directionPreview();controls();
}
function forcingEdited(){inflowDirty=true;pendingSettings();inletControls();makeBase();if(scene)render(framePair(process.frames,time));status('外源来水草稿已修改，应用地图或重新模拟后生效。');}
function addExternal(p){
 if(active()||busy)return;
 try{const existing=externalDraft.find(e=>e.id===p.id);if(!existing&&externalDraft.length>=32)throw Error('最多32个外源来水点');const e={...p,enabled:true,hydrograph:inletCurve()};if(existing)Object.assign(existing,e);else externalDraft.push(e);forcingEdited();}catch(e){status(e.message,true);}
}
function markerPosition(p){
 const factor=scene.scenario_metadata.coarsening_factor||1,[nz,nx]=scene.grid.shape;
 const x=Math.max(26,Math.min(S-26,(p.x/factor+.5)*S/nx)),z=Math.max(26,Math.min(S-26,(p.z/factor+.5)*S/nz));
 return [p.edge==='west'?44:p.edge==='east'?S-44:x,p.edge==='north'?44:p.edge==='south'?S-44:z];
}
function externalMarkers(ctx,kind){
 const factor=scene.scenario_metadata.coarsening_factor||1,[nz,nx]=scene.grid.shape,sx=S/nx,sz=S/nz;
 const points=crossings(),configured=new Map(externalDraft.map(p=>[p.id,p]));
 const all=[...(kind==='base'?shownCrossings():points.filter(p=>configured.has(p.id)||selectedRiverIds.has(p.id)||p.id===focusedPort)),...externalDraft.filter(p=>p.kind==='manual')];
 const positions=new Map(all.map(p=>[p.id,markerPosition(p)]));
 // Separate nearby badges along the edge while callout lines retain true positions.
 for(const edge of ['north','south','west','east']){
  const axis=edge==='north'||edge==='south'?0:1,g=all.filter(p=>p.edge===edge).sort((a,b)=>positions.get(a.id)[axis]-positions.get(b.id)[axis]);
  const gap=axis===0?52:38;
  for(let i=1;i<g.length;i++){const pos=positions.get(g[i].id);pos[axis]=Math.max(pos[axis],positions.get(g[i-1].id)[axis]+gap);}
  if(g.length&&positions.get(g.at(-1).id)[axis]>S-26){positions.get(g.at(-1).id)[axis]=S-26;for(let i=g.length-2;i>=0;i--)positions.get(g[i].id)[axis]=Math.min(positions.get(g[i].id)[axis],positions.get(g[i+1].id)[axis]-gap);}
 }
 if(kind==='base')markerHits=all.map(p=>({p,position:positions.get(p.id)}));
 for(const p of all){const entry=configured.get(p.id),enabled=entry?.enabled,focus=p.id===focusedPort||selectedRiverIds.has(p.id),[x,z]=positions.get(p.id);
  const color=enabled?'#7622b2':p.role==='inlet'?'#075bba':p.role==='outlet'?'#b2480b':'#526171';
  const ax=Math.max(3,Math.min(S-3,(Math.floor(p.x/factor)+.5)*sx)),az=Math.max(3,Math.min(S-3,(Math.floor(p.z/factor)+.5)*sz));
  ctx.save();ctx.strokeStyle='#fff';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(ax,az);ctx.lineTo(x,z);ctx.stroke();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
  ctx.fillStyle=color;ctx.beginPath();ctx.arc(ax,az,4,0,Math.PI*2);ctx.fill();
  ctx.fillStyle=focus?'#ffd451':'#fff';ctx.fillRect(x-24,z-16,48,32);ctx.fillStyle=color;ctx.fillRect(x-21,z-13,42,26);
  ctx.font='bold 16px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#fff';ctx.fillText(portCode(p,enabled),x,z);
  const vectors=inflowVectors({...p,...entry});
  if(p.kind!=='manual')for(const vec of vectors){if(!enabled&&p.role==='uncertain')continue;const sign=!enabled&&p.role==='outlet'?-1:1,[dx,dz]=vec.map(v=>v*sign),ex=x+dx*38,ez=z+dz*38;
   ctx.setLineDash(enabled?[]:[4,3]);ctx.beginPath();ctx.moveTo(x+dx*17,z+dz*17);ctx.lineTo(ex,ez);ctx.lineTo(ex-dx*8-dz*5,ez-dz*8+dx*5);ctx.moveTo(ex,ez);ctx.lineTo(ex-dx*8+dz*5,ez-dz*8-dx*5);ctx.strokeStyle='#fff';ctx.lineWidth=6;ctx.stroke();ctx.strokeStyle=color;ctx.lineWidth=3;ctx.stroke();ctx.setLineDash([]);
  }
  if(focus&&entry){const q=discharge(entry.hydrograph,time),text=`${inflowDirty?'草稿 ':''}${fmt(q)} m³/s`,tx=Math.max(8,Math.min(S-160,x+30)),tz=Math.max(50,Math.min(S-45,z+36));
   const state=boundaryInfo(entry.id),actual=!inflowDirty&&q>0&&state&&Math.abs(state.discharge_m3_s-q)<1e-6;
   const sub=p.kind==='manual'?'体积补给':actual?`边界入速 ${boundarySpeed(state)} m/s`:`设定入速 ${entry.inlet_velocity_m_s?fmt(entry.inlet_velocity_m_s)+' m/s':'自动'}`;
   ctx.font='bold 15px sans-serif';ctx.textAlign='left';ctx.fillStyle='#fff';ctx.fillRect(tx-4,tz-12,156,45);ctx.fillStyle=color;ctx.fillText(text,tx,tz);ctx.font='13px sans-serif';ctx.fillText(sub,tx,tz+20);
  }
  ctx.restore();
 }
}
$('addRiverInflow').onclick=()=>{
 if(active()||busy)return;
 try{
  const points=crossings().filter(p=>selectedRiverIds.has(p.id)),curve=inletCurve(),velocity=inletVelocity(),direction=inletDirectionSettings();
  if(externalDraft.length+points.filter(p=>!externalDraft.some(e=>e.id===p.id)).length>32)throw Error('最多32个外源来水点');
  for(const p of points)checkDirection(p,direction);
  for(const p of points){const entry={id:p.id,kind:'river',enabled:true,inlet_velocity_m_s:velocity,...direction,hydrograph:curve.map(v=>[...v])},existing=externalDraft.find(e=>e.id===p.id);if(existing)Object.assign(existing,entry);else externalDraft.push(entry);}
  selectedRiverIds.clear();forcingEdited();
 }catch(e){status(e.message,true);}
};
$('addManualInflow').onclick=()=>{let n=1;while(externalDraft.some(p=>p.id===`manual-${n}`))n++;addExternal({id:`manual-${n}`,kind:'manual',x:Number($('sourceX').value),z:Number($('sourceZ').value)});};
$('pickInflow').onclick=()=>{pickInflow=!pickInflow;$('pickInflow').setAttribute('aria-pressed',String(pickInflow));status(pickInflow?'点击任意地图选择强制来水位置，然后添加入水点。':'已退出入水选点。');};
function rgbHeight(z){const t=(z-terrainMin)/Math.max(.01,terrainMax-terrainMin);return [112+116*t,152+74*t,117+74*t];}
function rgbDepth(h){const t=Math.min(1,Math.max(0,h)/Math.max(.1,Number($('depthScale').value)||3));return [155-134*t,217-156*t,234-110*t];}
function updateSpeedScale(){
 let peak=0;for(const f of [scene.M_water.summary,...process.frames]){const v=Number(f.max_speed_m_s);if(Number.isFinite(v)&&v>peak)peak=v;}
 const mode=$('speedScaleMode').value,stats=geometry?speedStats(meshDisplay,geometry.n,meshOcean,geometry.weights):speedStats(display,scene.grid.shape[0]*scene.grid.shape[1],speedOceanMask);
 const limit=mode==='manual'?Number($('speedManualLimit').value):mode==='global'?peak:stats.p95;
 const next=Number.isFinite(limit)&&limit>0?limit:0,changed=next!==speedScale;speedScale=next;
 const label=v=>fmt(v,v>0&&v<.01?6:2),names={current:stats.sampleCount?`${stats.oceanFallback?'仅海洋有流动，使用海洋':'非海洋流动区域'} P95（${stats.sampleCount} 格）`:'当前无流动水域',global:'已计算帧峰值，统一色标',manual:'手动固定上限'};
 $('speedCaption').textContent=`${names[mode]}${geometry?' · 按实际面积统计':''} · 当前最大 ${label(stats.peak)} m/s${$('speedEnhance').checked?' · 平方根增强低速':''}${stats.peak>next?' · 超限颜色饱和':''}`;
 $('speedLimit').textContent=`${next?'≥ ':''}${label(next)} m/s`;$('speedMiddle').textContent=`${label(next*($('speedEnhance').checked?.25:.5))} m/s`;
 return changed;
}
function rgbSpeed(v){return speedColor(v,speedScale,$('speedEnhance').checked);}
function image(colors,leafColors=null){const [nz,nx]=scene.grid.shape,cv=document.createElement('canvas'),scale=leafColors&&geometry?2:1;cv.width=nx*scale;cv.height=nz*scale;
 const img=new ImageData(cv.width,cv.height);
 for(let i=0;i<nx*nz;i++){
  const color=[...colors(i).map(Math.round),255];
  if(scale===1)img.data.set(color,i*4);
  else{const p=meshPixel(i,0,nx);for(const offset of [0,1,2*nx,2*nx+1])img.data.set(color,(p+offset)*4);}
 }
 if(scale===2)for(const leaf of geometry.fine){const p=meshPixel(scene.M_mesh.parent[leaf],scene.M_mesh.child[leaf],nx);img.data.set([...leafColors(leaf).map(Math.round),255],p*4);}
 cv.getContext('2d').putImageData(img,0,0);return cv;}
function markers(ctx,sources=true,solids=scene.M_base.solid){
 const [nz,nx]=scene.grid.shape,sx=S/nx,sz=S/nz,b=scene.M_base;
 ctx.fillStyle='#414e4e';for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)if(solids[z][x])ctx.fillRect(x*sx,z*sz,sx,sz);
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
 markers(ctx,true,staticSolids(scene));
 const captions={elevation:`高程 ${fmt(terrainMin)}—${fmt(terrainMax)} m · 等高线与地物`,terrain:'独立地表类型图层 · 查点显示类别',roughness:'逐格 Manning 糙率 · 深色表示较大阻力',structures:'建筑与公共边阻水墙',sources:'绿色内源 · 紫色外源 · 青色排水口'};
 $('baseCaption').textContent='编辑预览 · '+captions[mode];
}
function arrows(ctx){
 if(!$('flowArrows').checked)return;
 const [nz,nx]=scene.grid.shape,N=nx*nz,sx=S/nx,sz=S/nz,stride=Math.max(1,Math.ceil(Math.max(nx,nz)/Number($('flowDensity').value)));
 const labelBoxes=[];ctx.font='17px sans-serif';ctx.lineWidth=2;ctx.textAlign='center';
 for(let z=0;z<nz;z+=stride)for(let x=0;x<nx;x+=stride){
  // Choose a real fastest wet cell in each tile; narrow rivers survive sampling.
  let best=-1,bestLeaf=-1,speed=0;
  for(let zz=z;zz<Math.min(nz,z+stride);zz++)for(let xx=x;xx<Math.min(nx,x+stride);xx++){
   const i=zz*nx+xx;
   if(geometry&&scene.M_base.refinement_level[zz][xx]){
    const n=geometry.n;for(let k=0;k<4;k++){const leaf=geometry.start[i]+k,v=Math.hypot(meshDisplay[n+leaf],meshDisplay[2*n+leaf]);if(meshDisplay[leaf]>=.01&&v>speed){best=i;bestLeaf=leaf;speed=v;}}
   }else{const v=Math.hypot(display[N+i],display[2*N+i]);if(display[i]>=.01&&v>speed){best=i;bestLeaf=-1;speed=v;}}
  }
  if(best<0||speed<.002)continue;
  const child=bestLeaf<0?-1:scene.M_mesh.child[bestLeaf],state=bestLeaf<0?display:meshDisplay,n=bestLeaf<0?N:geometry.n,index=bestLeaf<0?best:bestLeaf;
  const px=(best%nx+(child<0?.5:.25+.5*(child&1)))*sx,pz=(Math.floor(best/nx)+(child<0?.5:.25+.5*(child>>1)))*sz,u=state[n+index]/speed,v=state[2*n+index]/speed,len=Math.min(35,8+speed*18),dx=u*len,dz=v*len;
  ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(px-dx/2,pz-dz/2);ctx.lineTo(px+dx/2,pz+dz/2);ctx.lineTo(px+dx/2-u*7-v*4,pz+dz/2-v*7+u*4);ctx.moveTo(px+dx/2,pz+dz/2);ctx.lineTo(px+dx/2-u*7+v*4,pz+dz/2-v*7-u*4);ctx.stroke();
  if($('flowLabels').checked){const label=fmt(speed,speed<.1?3:2),w=ctx.measureText(label).width;const ly=Math.max(20,Math.min(S-5,pz+25));const box=[px-w/2-3,ly-16,px+w/2+3,ly+4];if(labelBoxes.some(b=>box[0]<b[2]+3&&box[2]>b[0]-3&&box[1]<b[3]+3&&box[3]>b[1]-3))continue;labelBoxes.push(box);ctx.fillStyle='#143e49b8';ctx.fillRect(box[0],box[1],w+6,20);ctx.fillStyle='#fff';ctx.fillText(label,px,ly);}
 }
}
function draw(id,kind){
 const ctx=$(id).getContext('2d'),[nz,nx]=scene.grid.shape,N=nz*nx,b=scene.M_base;ctx.imageSmoothingEnabled=false;
 if(kind==='base'){ctx.drawImage(baseImage,0,0);}
 else{
  const img=image(i=>{const h=display[i],z=Math.floor(i/nx),x=i%nx;
   if(kind==='maximum'&&display[3*N+i]>=.01)return rgbDepth(display[3*N+i]);
   if(kind==='speed'&&h>=.01)return rgbSpeed(Math.hypot(display[N+i],display[2*N+i]));
   if(kind==='depth'&&h>=.01)return rgbDepth(h);return rgbHeight(b.z[z][x]);},geometry?leaf=>{
    const i=scene.M_mesh.parent[leaf],z=Math.floor(i/nx),x=i%nx,n=geometry.n,h=meshDisplay[leaf];
    if(kind==='maximum'&&meshDisplay[3*n+leaf]>=.01)return rgbDepth(meshDisplay[3*n+leaf]);
    if(kind==='speed'&&h>=.01)return rgbSpeed(Math.hypot(meshDisplay[n+leaf],meshDisplay[2*n+leaf]));
    if(kind==='depth'&&h>=.01)return rgbDepth(h);return rgbHeight(b.z[z][x]);
   }:null);
  ctx.drawImage(img,0,0,S,S);markers(ctx,false);if(kind==='depth')arrows(ctx);
 }
 if($('showChannelAudit').checked&&$('channelAuditStale').hidden)drawChannelAudit(ctx,scene.M_structures?.channel_audit,scene.grid,S);
 externalMarkers(ctx,kind);
 structureEditor?.draw(ctx,S,time,kind==='base'?structureEditor.objects:(scene.M_structures?.objects||[]));
 peopleEditor?.draw(ctx,S,time,kind==='base'?peopleEditor.items:(scene.M_people?.items||[]),kind==='base'?peopleEditor.start:(scene.M_people?.start||null));
 const sx=S/nx,sz=S/nz,fine=selectedLeaf(geometry,scene.M_mesh,selected.z*nx+selected.x,selected.child)>=0;
 ctx.strokeStyle='#f9bd38';ctx.lineWidth=2;ctx.strokeRect((selected.x+(fine?(selected.child&1)/2:0))*sx,(selected.z+(fine?(selected.child>>1)/2:0))*sz,sx/(fine?2:1),sz/(fine?2:1));
 if(kind==='base'){ctx.strokeStyle='#da982b';ctx.setLineDash([4,3]);for(const p of patches)ctx.strokeRect(p.x*sx,p.z*sz,(p.width||1)*sx,(p.height||1)*sz);ctx.setLineDash([]);}
}
function inspect(){
 const [nz,nx]=scene.grid.shape,N=nz*nx,x=selected.x,z=selected.z,i=z*nx+x,b=scene.M_base;
 const leaf=selectedLeaf(geometry,scene.M_mesh,i,selected.child),fine=leaf>=0,n=fine?geometry.n:N,index=fine?leaf:i,state=fine?meshDisplay:display;
 const h=state[index],u=state[n+index],v=state[2*n+index],area=scene.grid.dx_m**2/(fine?4:1);
 const types=scene.scenario_metadata.terrain_ids||terrain.terrain_ids||{},label=types[b.terrain_class[z][x]]||b.terrain_class[z][x];
 const incoming=(scene.M_forcing?.external_inflows||[]).filter(p=>p.enabled&&p.cells.some(c=>c.x===x&&c.z===z)).map(p=>`${p.id} ${fmt(discharge(p.hydrograph,time))} m³/s`).join('；');
 const states=accountedWater().boundary_inflows||[],boundary=states.filter(p=>p.faces.some(f=>f.x===x&&f.z===z)).map(p=>`${fmt(p.discharge_m3_s)} m³/s · 面入速 ${boundarySpeed(p)} m/s`).join('；');
 $('mapCellDetail').textContent=`${fmt(time,1)} 秒 · 格 (${x}, ${z})${fine?` · 子格 ${selected.child}（边长 ${fmt(scene.grid.dx_m/2)} m，真实计算水情）`:geometry?' · 基础格面积平均':''} · 地面 ${fmt(b.z[z][x])} m · 地表 ${label} · 糙率 ${fmt(b.n[z][x],3)} · 建筑 ${b.solid[z][x]?'是':'否'}\n水深 ${fmt(h,3)} m · 水面 ${fmt(b.z[z][x]+h,3)} m · 流速 ${fmt(Math.hypot(u,v),3)} m/s · X / Z 速度 ${fmt(u,3)} / ${fmt(v,3)} m/s\n累计最大水深 ${fmt(state[3*n+index],3)} m · ${fine?'子格':'本格'}储水 ${fmt(h*area)} m³ · 排水能力 ${fmt(b.drain_capacity[z][x]/(fine?4:1))} m³/s · 水源编号 ${b.source_id[z][x]}${incoming?'\n外源来水：'+incoming:''}`;
 if(boundary)$('mapCellDetail').textContent+=`\n边界输入（最近计算子步 ${fmt(accountedWater().boundary_sample_time_s,3)} 秒）：${boundary}`;
}
function accountedWater(){const pair=framePair(process.frames,time);return pair?process.frames[pair.lo]:scene.M_water.summary;}
function boundaryInfo(id){return (accountedWater().boundary_inflows||scene.M_forcing?.boundary_inflows||[]).find(p=>p.id===id);}
function boundarySpeed(p){const v=p.faces.map(f=>f.inward_velocity_m_s),lo=Math.min(...v),hi=Math.max(...v);return hi-lo<.01?fmt(hi):`${fmt(lo)}—${fmt(hi)}`;}
function render(pair){
 if(!scene)return;updateSpeedScale();pair??=framePair(process.frames,time);draw('baseMap','base');draw('depthMap','depth');draw('speedMap','speed');draw('maximumMap','maximum');inspect();
 $('worldTime').textContent=`${fmt(time,1)} 秒`;$('mapTimeline').value=time;
 const count=scene.grid.shape[0]*scene.grid.shape[1];let maxDepth=0,maxSpeed=0,wet=0;
 for(let i=0;i<count;i++){maxDepth=Math.max(maxDepth,display[i]);if(display[i]>=.01){wet++;maxSpeed=Math.max(maxSpeed,Math.hypot(display[count+i],display[2*count+i]));}}
 $('metricDepth').textContent=`${fmt(maxDepth)} m`;$('metricSpeed').textContent=`${fmt(maxSpeed)} m/s`;
 $('metricArea').textContent=`${fmt(wet*scene.grid.dx_m**2/10000,2)} ha`;$('metricGrid').textContent=`${fmt(scene.grid.dx_m)} m · ${fmt(count,0)} 格`;
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
for(const button of document.querySelectorAll('button[data-panel]'))button.onclick=()=>showPanel(button.dataset.panel);
for(const button of document.querySelectorAll('button[data-view]'))button.onclick=()=>showView(button.dataset.view);
for(const tablist of document.querySelectorAll('[role=tablist]'))tablist.onkeydown=e=>{
 if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
 const tabs=[...tablist.querySelectorAll('[role=tab]')],i=tabs.indexOf(document.activeElement);
 const next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
 e.preventDefault();tabs[next].focus();tabs[next].click();
};
document.querySelector('.settings-scroll').addEventListener('input',e=>{if(e.target.id!=='crossingDisplay')pendingSettings();});
$('openExport').onclick=()=>{const details=document.querySelector('.export-area>details');details.open=true;details.scrollIntoView({behavior:'smooth',block:'nearest'});};
$('inflowDirectionMode').onchange=()=>{directionPreview();pendingSettings();};
$('inflowAngle').oninput=()=>{$('inflowAngleRange').value=$('inflowAngle').value;directionPreview();};
$('inflowAngleRange').oninput=()=>{$('inflowAngle').value=$('inflowAngleRange').value;directionPreview();};
$('saveInflowSettings').onclick=()=>{try{const p=externalDraft.find(p=>p.id===focusedPort);if(!p)throw Error('请先载入一个来水点');saveInflowEditor(p);}catch(e){status(e.message,true);}};
$('resetNumerics').onclick=()=>{loadNumerics();$('numericsApplied').textContent='已恢复默认设置；应用地图或重新模拟后生效。';};
for(const id of Object.keys(numericalFields))$(id).onchange=()=>{$('numericsApplied').textContent='数值设置已修改；应用地图或重新模拟后生效。';};
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
for(const id of ['flowArrows','flowLabels','flowDensity','depthScale','showPeople'])$(id).onchange=()=>render(framePair(process.frames,time));
$('crossingDisplay').onchange=()=>{inletControls();if(scene)render(framePair(process.frames,time));};
$('speedScaleMode').onchange=()=>{$('speedManualControl').hidden=$('speedScaleMode').value!=='manual';if(scene)render(framePair(process.frames,time));};
for(const id of ['speedManualLimit','speedEnhance'])$(id).onchange=()=>{if(!$('speedManualLimit').checkValidity()){$('speedManualLimit').reportValidity();return;}if(scene)render(framePair(process.frames,time));};
$('speedManualLimit').oninput=()=>{if($('speedManualLimit').checkValidity()&&scene)render(framePair(process.frames,time));};
$('depthScale').oninput=()=>{if($('depthScale').checkValidity()&&scene)render(framePair(process.frames,time));};
for(const id of ids){
 $(id).onclick=e=>{if(!scene||busy)return;const r=$(id).getBoundingClientRect(),[nz,nx]=scene.grid.shape;
  if(id==='baseMap'&&!pickInflow){
   if(structureDragClick){structureDragClick=false;return;}
   if(peopleEditor.pick&&peopleEditor.click([(e.clientX-r.left)/r.width*nx*scene.grid.dx_m,(e.clientY-r.top)/r.height*nz*scene.grid.dx_m]))return;
   if(structureEditor.click([(e.clientX-r.left)/r.width*nx*scene.grid.dx_m,(e.clientY-r.top)/r.height*nz*scene.grid.dx_m]))return;
  }
  if(id==='baseMap'&&!pickInflow&&$('editTool').value==='inspect'){
   const px=(e.clientX-r.left)/r.width*S,pz=(e.clientY-r.top)/r.height*S,hit=markerHits.find(m=>Math.abs(m.position[0]-px)<=24&&Math.abs(m.position[1]-pz)<=16);
   if(hit){focusPort(hit.p);return;}
  }
  selected={x:Math.max(0,Math.min(nx-1,Math.floor((e.clientX-r.left)/r.width*nx))),z:Math.max(0,Math.min(nz-1,Math.floor((e.clientY-r.top)/r.height*nz)))};
  if(geometry&&scene.M_base.refinement_level[selected.z][selected.x]){
   const px=(e.clientX-r.left)/r.width*nx,pz=(e.clientY-r.top)/r.height*nz;
   selected.child=(px-Math.floor(px)>=.5?1:0)+(pz-Math.floor(pz)>=.5?2:0);
  }
  if(pickInflow){const f=scene.scenario_metadata.coarsening_factor||1;$('sourceX').value=selected.x*f;$('sourceZ').value=selected.z*f;pickInflow=false;$('pickInflow').setAttribute('aria-pressed','false');status('入水位置已选定，点击添加手动强制来水点。');}
 else
  if($('editTool').value!=='inspect'){
   if(active()){status('固定环境需在计算结束或取消后编辑。',true);return;}
   patches.push({kind:$('editTool').value,...selected,width:Number($('patchWidth').value),height:Number($('patchHeight').value),value:Number($('patchValue').value),grid_factor:scene.scenario_metadata.coarsening_factor||1});draft();status('已加入地图编辑草稿，应用或重新模拟后生效。');
  }render(framePair(process.frames,time));
 };
 $(id).onkeydown=e=>{const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];if(!scene||!d)return;e.preventDefault();delete selected.child;selected.x=Math.max(0,Math.min(scene.grid.shape[1]-1,selected.x+d[0]));selected.z=Math.max(0,Math.min(scene.grid.shape[0]-1,selected.z+d[1]));render(framePair(process.frames,time));};
}
structureEditor=new StructureEditor({getScene:()=>scene,getTime:()=>time,
 redraw:()=>{if(scene&&baseImage)render();},dirty:()=>pendingSettings(),message:status,locked:()=>busy||active()});
peopleEditor=new PeopleEditor({getScene:()=>scene,getStructures:()=>structureEditor.objects,
 redraw:()=>{if(scene&&baseImage)render();},dirty:()=>pendingSettings(),message:status,locked:()=>busy||active()});
function structurePosition(e){const r=$('baseMap').getBoundingClientRect(),[nz,nx]=scene.grid.shape,dx=scene.grid.dx_m;return [Math.max(0,Math.min(nx*dx,(e.clientX-r.left)/r.width*nx*dx)),Math.max(0,Math.min(nz*dx,(e.clientY-r.top)/r.height*nz*dx))];}
$('baseMap').addEventListener('pointerdown',e=>{if(!scene||busy||pickInflow||peopleEditor?.pick)return;if(structureEditor.down(structurePosition(e))){$('baseMap').setPointerCapture(e.pointerId);e.preventDefault();}});
$('baseMap').addEventListener('pointermove',e=>{if(scene)structureEditor.move(structurePosition(e));});
$('baseMap').addEventListener('pointerup',e=>{if(scene&&structureEditor.up(structurePosition(e)))structureDragClick=true;});
$('baseMap').addEventListener('pointercancel',()=>structureEditor.cancel());
$('editTool').addEventListener('change',()=>{$('structureTool').value='none';structureEditor.cancel();});
$('editTool').onchange=()=>{$('patchValue').value=({roughness:.06,drain:1,source:0,terrain:0})[$('editTool').value]??1;};
$('clearDraft').onclick=()=>{patches=[];draft();render();};
$('mapSource').onchange=()=>{imported=null;patches=[];externalDraft=[];focusedPort=null;selectedRiverIds.clear();inflowDirty=true;peopleEditor.reset();inletControls();draft();$('hydrographs').value='{}';$('mapBoundary').value=$('mapSource').value==='demo'?'closed':'ocean';status('地图来源已更改，应用或开始模拟后生效。');};
function download(blob,name){const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function displayedForcing(){
 const [nz,nx]=scene.grid.shape,rate=Array.from({length:nz},()=>Array(nx).fill(0)),entries=scene.M_forcing?.external_inflows||[];
 const byId={};for(const p of entries){const q=p.enabled?discharge(p.hydrograph,time):0;byId[p.id]=q;for(const cell of p.cells)rate[cell.z][cell.x]+=q*cell.weight/scene.grid.dx_m**2;}
 const pair=framePair(process.frames,time),actual=pair?process.frames[pair.lo]:scene.M_water.summary;
 return {external_inflows:entries,river_crossings:scene.M_forcing?.river_crossings||[],
  inlet_count:scene.M_forcing?.inlet_count||Array.from({length:nz},()=>Array(nx).fill(0)),
  external_rate_m_s:rate,instantaneous_discharge_by_id_m3_s:byId,
  boundary_inflows:actual.boundary_inflows||[],boundary_sample_time_s:actual.boundary_sample_time_s??null,
  accumulated_m3:actual.external_inflow_by_id_m3||{},accounted_frame_time_s:actual.time_s};
}
function saveJson(data,name){download(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),name);}
$('saveMapConfig').onclick=()=>{try{saveJson(config(),'hyra-map-flood-scenario.json');}catch(e){status(e.message,true);}};
$('exportFrame').onclick=()=>{
 if(!scene)return;const [nz,nx]=scene.grid.shape,N=nz*nx,pair=framePair(process.frames,time),layers={};
 for(const [k,j] of ['depth','u','v','maximum_depth'].entries())layers[j]=Array.from({length:nz},(_,z)=>Array.from(display.slice(k*N+z*nx,k*N+(z+1)*nx)));
 saveJson({schema:'hyra-map-display-v1',time_s:time,grid:scene.grid,display_interpolated:Boolean(pair&&pair.alpha>1e-6&&pair.alpha<1-1e-6),
  source_frame_times_s:pair?[process.frames[pair.lo].time_s,process.frames[pair.hi].time_s]:[0],
  M_forcing:displayedForcing(),M_structures:scene.M_structures,layers,M_mesh:geometry?exportMesh(scene.M_mesh,meshDisplay):undefined,
  base_frame_projection:geometry?'area_averaged_conserved_state':undefined},'hyra-displayed-water.json');
};
$('exportPng').onclick=()=>$('depthMap').toBlob(blob=>download(blob,'hyra-map-water-depth.png'));
$('importMapConfig').onchange=async()=>{try{const file=$('importMapConfig').files[0];if(!file)return;if(file.size>8*1024*1024)throw Error('场景配置文件过大');const c=JSON.parse(await file.text());if(!c||!['current','demo'].includes(c.mode)||c.agents||c.legacy_exploration)throw Error('请选择纯地图与洪水场景配置');imported=c;structureEditor.load(c.structures||[]);peopleEditor.load(c.people||null);loadNumerics(c.flood);patches=[];externalDraft=normalizeInflows(c.external_inflows||[]);inflowDirty=true;draft();$('mapSource').value=c.mode;for(const [id,name] of Object.entries(fields))if(c.flood?.[name]!==undefined)$(id).value=c.flood[name];for(const [id,name] of [['mapBaseflow','baseflow'],['mapInitialWater','initial_water']])if(c.flood?.[name]!==undefined)$(id).checked=c.flood[name];if(c.flood?.boundary)$('mapBoundary').value=c.flood.boundary;if(c.flood?.inflow_m3_s>0&&!externalDraft.some(p=>p.id==='legacy-point'))externalDraft.push({id:'legacy-point',kind:'manual',x:c.flood.source_x,z:c.flood.source_z,enabled:true,hydrograph:[[0,c.flood.inflow_m3_s]]});$('hydrographs').value=JSON.stringify(c.layers?.hydrographs||{});inletControls();restoreInflowEditor();status('地图场景配置已载入，应用或开始模拟后生效。');}catch(e){status(e.message,true);}};
json('/api/current').then(async t=>{
 terrain=t;$('terrainInfo').textContent=t.imported?`导入地形 · ${t.imported.name||'外部高程'} · ${t.shape[1]} × ${t.shape[0]} 格 · 每格 ${t.step} 米。`:`当前种子 ${t.config.seed} · ${t.shape[1]} × ${t.shape[0]} 格 · 每格 ${t.step} 方块${t.step!==1?'（旧地图，重新生成可使用每方块一格）':''}。`;
 if(t.imported&&t.case_defaults){
  imported={...t.case_defaults,mode:'current',terrain_id:t.id};const f=imported.flood||{};
  loadNumerics(f);externalDraft=normalizeInflows(imported.external_inflows||[]);
  for(const [id,name] of Object.entries(fields))if(f[name]!==undefined)$(id).value=f[name];
  for(const [id,name] of [['mapBaseflow','baseflow'],['mapInitialWater','initial_water']])if(f[name]!==undefined)$(id).checked=f[name];
  if(f.boundary)$('mapBoundary').value=f.boundary;
 }
 for(const [id,size] of [['sourceX',t.shape[1]],['sourceZ',t.shape[0]]]){$(id).max=size-1;$(id).value=Math.min(size-1,Number($(id).value));}
 const entry=new URLSearchParams(location.search),fresh=entry.get('fresh')==='1';
 controls();let saved=location.hash.slice(1);if(!saved&&!fresh)try{saved=localStorage.getItem('hyra-map-process')||'';}catch{}
 if(/^(?:maps|merewether)-[a-z0-9-]{1,64}$/.test(saved)){busy=true;controls();try{await restore(saved);return;}catch{try{localStorage.removeItem('hyra-map-process');}catch{}history.replaceState(null,'',location.pathname);}finally{busy=false;controls();}}
 await create(false);
}).catch(e=>status(e.message,true));
requestAnimationFrame(animate);

function showChannelAudit(){
 const audit=scene.M_structures?.channel_audit;
 $('channelAuditStale').hidden=true;
 $('channelAuditSummary').textContent=auditSummary(audit);
 const list=$('channelAuditList');list.replaceChildren();
 for(const c of auditWarnings(audit)){
  const button=document.createElement('button');button.type='button';button.style.display='block';
  const label={closed:'网格内封闭',narrowed:'网格缩窄',not_evaluated:'未评估'}[c.status];
  button.textContent=`${c.building_ids.join(' ↔ ')}：${label}；几何宽 ${fmt(c.physical_width_m,3)} m，网格最小净宽 ${c.grid_min_width_m==null?'未计算':fmt(c.grid_min_width_m,3)+' m'}`;
  button.onclick=()=>{const [x,z,x1,z1]=c.bounds_m,dx=scene.grid.dx_m;selected={x:Math.min(scene.grid.shape[1]-1,Math.floor((x+x1)/2/dx)),z:Math.min(scene.grid.shape[0]-1,Math.floor((z+z1)/2/dx))};render();};
  list.append(button);
 }
}
$('showChannelAudit').onchange=()=>{if(scene)render();};
