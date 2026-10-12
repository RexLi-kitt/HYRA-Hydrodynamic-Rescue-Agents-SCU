'use strict';
(() => {
 const $=id=>document.getElementById(id),canvases=['baseMap','truthMap','knownMap','ageMap'];
 let terrain,state,patches=[],selected={x:7,z:16},busy=false,imported=null;
 const fmt=(v,n=2)=>Number(v).toLocaleString('zh-CN',{maximumFractionDigits:n});
 function status(text,error=false){$('mapStatus').textContent=text;$('mapStatus').classList.toggle('error',error);}
 async function json(url,options){const r=await fetch(url,options),j=await r.json();if(!r.ok)throw Error(j.error||'读取失败');return j;}
 function lock(v){busy=v;for(const id of ['createMap','advanceMap','moveAgent','observeMap','invalidObservation','shareMap','importMapConfig'])$(id).disabled=v||(!state&&id!=='createMap'&&id!=='importMapConfig');}
 function draft(){ $('draftInfo').textContent=`草稿：${patches.length} 项编辑；应用场景后生效。`; }
 function config(){
  if(imported)return {...imported,layers:{...imported.layers,patches:[...(imported.layers?.patches||[]),...patches]}};
  const mode=$('mapSource').value,rain=Number($('mapRain').value),until=Number($('mapRainUntil').value);
  return {mode,terrain_id:terrain.id,flood:{resolution:Number($('mapResolution').value),metres_per_block:Number($('mapUnit').value),
    rain_mm_h:rain,rain_duration_s:until,duration_s:Math.max(600,until),source_x:0,source_z:0,inflow_m3_s:0,
    baseflow:mode==='current'&&$('mapBaseflow').checked,warmup_s:0},layers:{patches:patches.slice(),hydrographs:JSON.parse($('hydrographs').value)}};
 }
 async function create(){lock(true);status('正在建立同一网格上的全部地图层…');try{
  state=await json('/api/maps',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config())});
  const keys=Object.keys(state.agents);for(const id of ['activeAgent','shareTarget','viewAgent']){
   const old=$(id).value;$(id).replaceChildren(...(id==='activeAgent'?keys:[...keys,'team']).map(k=>{const o=document.createElement('option');o.value=k;o.textContent=k==='team'?'团队共享':k;return o;}));
   $(id).value=keys.includes(old)||old==='team'&&id!=='activeAgent'?old:keys[0];
  }
  $('shareTarget').value='team';selected={x:state.agents[keys[0]].position.x,z:state.agents[keys[0]].position.z};
  $('targetX').value=selected.x;$('targetZ').value=selected.z;
  $('targetX').max=state.grid.shape[1]-1;$('targetZ').max=state.grid.shape[0]-1;
  status('场景已建立。已知水情初始为未知；请移动无人机并观察。');render();
 }catch(e){status(e.message,true);}finally{lock(false);}}
 async function act(action,data={}){if(!state||busy)return;lock(true);status(action==='advance'?'正在推进真实水情，观测记录保持原值…':'正在更新地图记录…');
  try{state=await json(`/api/maps/${state.id}/${action}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(action==='move')selected={x:data.x,z:data.z};render();
   const messages={advance:'真实水情已推进，旧观测保持原值，信息年龄增加。',observe:data.valid===false?'无效观测没有覆盖已有记录。':'当前视野已更新，未覆盖区域继续保留旧记录。',move:'无人机已移动，尚未重新观测。',share:'共享完成；旧记录不会覆盖更新的有效观测。'};status(messages[action]);
  }catch(e){status(e.message,true);}finally{lock(false);}}
 function knowledge(){return $('viewAgent').value==='team'?state.team:state.agents[$('viewAgent').value];}
 function terrainColor(z,min,max){const t=(z-min)/Math.max(.01,max-min);return [132+79*t,166+51*t,128+79*t];}
 function waterColor(h){const t=Math.min(1,Math.log1p(Math.max(0,h)*10)/Math.log1p(30));return [55-30*t,157-91*t,225-75*t];}
 function person(ctx,x,z,category,nx,nz){const px=(x+.5)*480/nx,pz=(z+.5)*480/nz;ctx.save();ctx.strokeStyle=category===1?'#ee554c':'#388bdc';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;ctx.beginPath();ctx.arc(px,pz-5,2.8,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(px,pz-2);ctx.lineTo(px,pz+5);ctx.moveTo(px-4,pz+1);ctx.lineTo(px+4,pz+1);ctx.moveTo(px,pz+5);ctx.lineTo(px-3,pz+9);ctx.moveTo(px,pz+5);ctx.lineTo(px+3,pz+9);ctx.stroke();ctx.restore();}
 function draw(id,kind,k){const canvas=$(id),ctx=canvas.getContext('2d'),[nz,nx]=state.grid.shape,b=state.M_base,t=state.M_water,obs=k.M_obs;
  let min=Infinity,max=-Infinity;for(const row of b.z)for(const z of row){min=Math.min(min,z);max=Math.max(max,z);}
  const img=new ImageData(nx,nz);for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
   let rgb=terrainColor(b.z[z][x],min,max);
   if(kind==='truth'&&t.h[z][x]>=.01)rgb=waterColor(t.h[z][x]);
   if(kind==='known')rgb=!obs.water_valid[z][x]?[203,209,208]:obs.water_last[z][x]>=.01?waterColor(obs.water_last[z][x]):[170,195,155];
   if(kind==='age'){const age=k.information_age[z][x],v=Math.min(1,(age||0)/120);rgb=age===null?[203,209,208]:[80+161*v,185-100*v,130-78*v];}
   if(kind==='base'&&b.n[z][x]>.05)rgb=[110,151,116];
   if(b.solid[z][x]&&(kind==='base'||kind==='truth'||k.M_exp.terrain_valid[z][x]))rgb=[69,79,79];
   img.data.set([...rgb.map(Math.round),255],(z*nx+x)*4);
  }
  const tmp=document.createElement('canvas');tmp.width=nx;tmp.height=nz;tmp.getContext('2d').putImageData(img,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(tmp,0,0,480,480);
  const sx=480/nx,sz=480/nz;
  if(kind==='base'){
   const interval=Math.max(.25,(max-min)/12);ctx.strokeStyle='#365b4b44';ctx.lineWidth=.6;ctx.beginPath();
   for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
    const band=Math.floor(b.z[z][x]/interval);
    if(x+1<nx&&Math.floor(b.z[z][x+1]/interval)!==band){ctx.moveTo((x+1)*sx,z*sz);ctx.lineTo((x+1)*sx,(z+1)*sz);}
    if(z+1<nz&&Math.floor(b.z[z+1][x]/interval)!==band){ctx.moveTo(x*sx,(z+1)*sz);ctx.lineTo((x+1)*sx,(z+1)*sz);}
    if(!b.flight_allowed[z][x]){ctx.fillStyle='#8152a477';ctx.fillRect(x*sx,z*sz,sx,sz);}
    if(b.drain_capacity[z][x]>0||b.source_id[z][x]>=0){ctx.fillStyle=b.source_id[z][x]>=0?'#1cbe87':'#17adc7';ctx.fillRect((x+.3)*sx,(z+.3)*sz,sx*.4,sz*.4);}
   }ctx.stroke();ctx.strokeStyle='#182d2c';ctx.lineWidth=2;ctx.beginPath();
   for(let z=0;z<nz;z++)for(let x=0;x<=nx;x++)if(b.wall_x[z][x]){ctx.moveTo(x*sx,z*sz);ctx.lineTo(x*sx,(z+1)*sz);}
   for(let z=0;z<=nz;z++)for(let x=0;x<nx;x++)if(b.wall_y[z][x]){ctx.moveTo(x*sx,z*sz);ctx.lineTo((x+1)*sx,z*sz);}ctx.stroke();
  }
  if(kind==='truth'||kind==='known'){
   const stride=Math.max(1,Math.ceil(nx/15));ctx.strokeStyle='#fff';ctx.lineWidth=1.3;ctx.beginPath();
   for(let z=0;z<nz;z+=stride)for(let x=0;x<nx;x+=stride){
    if(kind==='known'&&!obs.water_valid[z][x])continue;const h=kind==='truth'?t.h[z][x]:obs.water_last[z][x],u=kind==='truth'?t.u[z][x]:obs.u_last[z][x],v=kind==='truth'?t.v[z][x]:obs.v_last[z][x],speed=Math.hypot(u,v);
    if(h<.01||speed<.005)continue;const px=(x+.5)*sx,pz=(z+.5)*sz,dx=u/speed*10,dz=v/speed*10;
    ctx.moveTo(px-dx/2,pz-dz/2);ctx.lineTo(px+dx/2,pz+dz/2);ctx.lineTo(px+dx/2-dx*.3-dz*.2,pz+dz/2-dz*.3+dx*.2);ctx.moveTo(px+dx/2,pz+dz/2);ctx.lineTo(px+dx/2-dx*.3+dz*.2,pz+dz/2-dz*.3-dx*.2);
   }ctx.stroke();
   if(kind==='truth'){for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)if(state.M_ref.person_class[z][x])person(ctx,x,z,state.M_ref.person_class[z][x],nx,nz);}
   else{const positions=$('viewAgent').value==='team'?state.M_pos:k.M_pos;for(const p of positions.discovered_persons)person(ctx,p.x,p.z,p.category,nx,nz);}
  }
  for(const [name,a] of Object.entries(state.agents)){const p=a.position;ctx.strokeStyle=name===$('activeAgent').value?'#ce732d':'#8e9992';ctx.lineWidth=2;
   ctx.beginPath();ctx.arc((p.x+.5)*sx,(p.z+.5)*sz,5,0,Math.PI*2);ctx.stroke();
   if(name===$('activeAgent').value){ctx.setLineDash([4,4]);ctx.beginPath();ctx.ellipse((p.x+.5)*sx,(p.z+.5)*sz,p.radius*sx,p.radius*sz,0,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);}}
  ctx.strokeStyle='#f7e587';ctx.lineWidth=2;ctx.strokeRect(selected.x*sx,selected.z*sz,sx,sz);
  if(kind==='base'){ctx.strokeStyle='#dc9d30';ctx.setLineDash([4,3]);for(const p of patches)ctx.strokeRect(p.x*sx,p.z*sz,(p.width||1)*sx,(p.height||1)*sz);ctx.setLineDash([]);}
 }
 function inspect(){if(!state)return;const [nz,nx]=state.grid.shape,x=Math.min(nx-1,selected.x),z=Math.min(nz-1,selected.z),b=state.M_base,t=state.M_water,k=knowledge(),o=k.M_obs;
  const known=o.water_valid[z][x]?`${fmt(o.water_last[z][x],3)} m，采集于 ${fmt(o.t_water[z][x])} 秒，年龄 ${fmt(k.information_age[z][x])} 秒`:'未知（不是零水深）';
  const e=k.M_exp,terrainKnown=e.terrain_valid[z][x]?`${fmt(e.terrain_field[z][x])} m`:'未知',signalKnown=e.signal_valid[z][x]?fmt(e.signal_field[z][x],3):'未知';
  $('mapCellDetail').textContent=`格 (${x}, ${z}) · 地面 ${fmt(b.z[z][x])} m · 糙率 ${fmt(b.n[z][x],3)} · 建筑 ${b.solid[z][x]?'是':'否'} · ${b.flight_allowed[z][x]?'可飞行':'禁飞'}\n真实水深 ${fmt(t.h[z][x],3)} m · 流速 ${fmt(Math.hypot(t.u[z][x],t.v[z][x]),3)} m/s · 已知水情：${known}\n已知高程／地形估计 ${terrainKnown} · 已知信号／信号估计 ${signalKnown} · 已知流速 ${o.water_valid[z][x]?fmt(Math.hypot(o.u_last[z][x],o.v_last[z][x]),3)+' m/s':'未知'}\n排水能力 ${fmt(b.drain_capacity[z][x])} m³/s · 水源编号 ${b.source_id[z][x]} · 已探索 ${k.M_exp.explored_ever[z][x]?'是':'否'} · 原始坐标 x=${fmt(state.grid.origin_blocks[0]+x*state.grid.dx_m/(state.config.metres_per_block||1))}，z=${fmt(state.grid.origin_blocks[1]+z*state.grid.dx_m/(state.config.metres_per_block||1))}`;
 }
 function render(){if(!state)return;const k=knowledge(),count=k.M_obs.water_valid.flat().filter(Boolean).length;
  $('worldInfo').textContent=`${state.provenance==='synthetic_multilayer_demo'?'合成多层示例':'当前生成地图'} · ${state.grid.shape[1]} × ${state.grid.shape[0]} · 间距 ${state.grid.dx_m} 米 · 固定地图与动态层共用坐标`;
  $('worldTime').textContent=`${fmt(state.time_s,1)} 秒`;$('knownCount').textContent=`${$('viewAgent').value} 已知：${count} 格`;
  $('mapBalance').textContent=`收支误差：${Number(state.M_water.summary.balance.residual_m3).toExponential(2)} m³`;
  $('knownCaption').textContent=`${$('viewAgent').value} 最近的有效观测；灰色为未知，地面色为确认无水`;
  draw('baseMap','base',k);draw('truthMap','truth',k);draw('knownMap','known',k);draw('ageMap','age',k);inspect();
  const pos=$('viewAgent').value==='team'?state.M_pos:k.M_pos;
  $('positionInfo').textContent=`无人机：${pos.uavs.map(a=>`${a.id} (${a.x},${a.z})`).join('；')} · 已发现人员：${pos.discovered_persons.map(p=>`(${p.x},${p.z}) ${p.category===1?'未施救':'已施救'}`).join('；')||'暂无'} · 曾探索 ${k.M_exp.explored_ever.flat().filter(Boolean).length} 格`;
  const labels={created:'建立场景',advance:'推进洪水',move:'移动无人机',observe:'局部观测',share:'共享记录'};
  $('mapEvents').replaceChildren(...state.events.slice(-30).map(e=>{const li=document.createElement('li');li.textContent=`${fmt(e.time_s,1)} 秒 · ${labels[e.action]}${e.data.agent?' · '+e.data.agent:''}${e.data.updated!==undefined?' · 更新 '+e.data.updated+' 格':''}`;return li;}));
  $('downloadMap').hidden=false;$('downloadMap').href=state.archive_url;
  $('agentViewLink').hidden=false;$('agentViewLink').href=`/api/maps/${state.id}/agent/${$('viewAgent').value}`;
 }
 for(const id of canvases){$(id).onclick=e=>{if(!state||busy)return;const r=$(id).getBoundingClientRect(),[nz,nx]=state.grid.shape;
  selected={x:Math.max(0,Math.min(nx-1,Math.floor((e.clientX-r.left)/r.width*nx))),z:Math.max(0,Math.min(nz-1,Math.floor((e.clientY-r.top)/r.height*nz)))};
  $('targetX').value=selected.x;$('targetZ').value=selected.z;
  if($('editTool').value!=='inspect'){patches.push({kind:$('editTool').value,x:selected.x,z:selected.z,width:Number($('patchWidth').value),height:Number($('patchHeight').value),value:Number($('patchValue').value)});draft();status('已加入固定环境草稿；点击应用场景后生效。');}render();};
  $(id).onkeydown=e=>{const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];if(!state||!d)return;e.preventDefault();selected.x=Math.max(0,Math.min(state.grid.shape[1]-1,selected.x+d[0]));selected.z=Math.max(0,Math.min(state.grid.shape[0]-1,selected.z+d[1]));$('targetX').value=selected.x;$('targetZ').value=selected.z;render();};}
 $('createMap').onclick=create;
 $('advanceMap').onclick=()=>act('advance',{seconds:Number($('advanceSeconds').value)});
 $('observeMap').onclick=()=>act('observe',{agent:$('activeAgent').value});
 $('invalidObservation').onclick=()=>act('observe',{agent:$('activeAgent').value,valid:false});
 $('moveAgent').onclick=()=>act('move',{agent:$('activeAgent').value,x:Number($('targetX').value),z:Number($('targetZ').value)});
 $('shareMap').onclick=()=>act('share',{agent:$('activeAgent').value,target:$('shareTarget').value});
 $('viewAgent').onchange=render;$('activeAgent').onchange=()=>{const a=state.agents[$('activeAgent').value].position;$('targetX').value=a.x;$('targetZ').value=a.z;render();};
 $('editTool').onchange=()=>{$('patchValue').value=({roughness:.06,drain:1,source:0,no_fly:0,person:1})[$('editTool').value]??1;};
 $('clearDraft').onclick=()=>{patches=[];draft();render();};
 $('mapSource').onchange=()=>{imported=null;patches=[];draft();status('地图来源已更改，点击建立场景应用。');};
 $('saveMapConfig').onclick=()=>{try{const blob=new Blob([JSON.stringify(config(),null,2)],{type:'application/json'}),a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download='hyra-multilayer-scenario.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){status(e.message,true);}};
 $('importMapConfig').onchange=async()=>{try{const file=$('importMapConfig').files[0];if(!file)return;if(file.size>8*1024*1024)throw Error('场景配置文件过大');const c=JSON.parse(await file.text());if(!c||!['current','demo'].includes(c.mode))throw Error('请选择本工作台的场景配置 JSON');imported=c;patches=[];draft();$('mapSource').value=c.mode;status('场景配置已读入，点击应用场景建立全部地图层。');}catch(e){status(e.message,true);}};
 json('/api/current').then(t=>{terrain=t;$('terrainInfo').textContent=`当前种子 ${t.config.seed} · ${t.shape[1]} × ${t.shape[0]} 原始网格。`;$('createMap').disabled=false;}).catch(e=>status(e.message,true));
})();
