'use strict';
(() => {
 const $ = id => document.getElementById(id), form = $('floodControls');
 const canvas = $('floodMap'), ctx = canvas.getContext('2d');
 let terrain, rawBed, rawDepth, meta, frame, bed, selected, job, computing=false;
 let layer='depth', timer, follow=true, requestId=0, preview=true;
 const cache = new Map();
 const format = (n, digits=1) => Number(n).toLocaleString('zh-CN',{maximumFractionDigits:digits});
 const clock = s => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
 const status = (s,error=false) => {$('floodStatus').textContent=s;$('floodStatus').classList.toggle('error',error);};
 async function json(url, options) {const r=await fetch(url,options),j=await r.json();if(!r.ok)throw new Error(j.error||'读取失败');return j;}
 async function floats(url, length) {const r=await fetch(url);if(!r.ok)throw new Error('地图字段读取失败');const b=await r.arrayBuffer();if(b.byteLength!==length*4)throw new Error('地图数据不完整');return new Float32Array(b);}
 function config() {return {terrain_id:terrain.id,metres_per_block:Number($('unit').value),resolution:Number($('resolution').value),
  rain_mm_h:Number($('rain').value),rain_duration_s:Number($('rainDuration').value)*60,duration_s:Number($('duration').value)*60,
  inflow_m3_s:Number($('inflow').value),source_x:Number($('sourceX').value),source_z:Number($('sourceZ').value),
  infiltration_mm_h:Number($('infiltration').value),manning:Number($('manning').value),boundary:$('boundary').value,
  initial_water:$('initialWater').checked,frames:31};}
 function lock(value) {computing=value;for(const el of form.querySelectorAll('input,select,button'))el.disabled=value;
  $('cancel').disabled=false;$('cancel').hidden=!value;$('floodProgress').hidden=!value;$('refresh').disabled=value;}
 function stop() {clearInterval(timer);timer=null;$('play').textContent='播放 ▶';}
 function download(blob,name) {const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 function rows(a) {const [,nx]=meta.shape;return Array.from({length:meta.shape[0]},(_,z)=>Array.from(a.subarray(z*nx,(z+1)*nx)));}
 function makePreview() {
  if(!terrain)return;preview=true;const unit=Number($('unit').value)||1,n=rawBed.length;
  bed=Float32Array.from(rawBed,v=>v*unit);const depth=Float32Array.from(rawDepth,v=>$('initialWater').checked?v*unit:0);
  meta={shape:terrain.shape,dx_m:terrain.step*unit,origin_blocks:terrain.origin,step_blocks:terrain.step,
   coarsening_factor:1,frames:[],config:config()};
  frame={depth,u:new Float32Array(n),v:new Float32Array(n),maximum:depth,info:{time_s:0}};
  $('depthStat').textContent=format(depth.reduce((a,b)=>Math.max(a,b),0));
  $('areaStat').textContent='—';$('balanceStat').textContent='—';$('timeStat').textContent='0:00';
  $('timeLabel').textContent='0:00';$('timeline').value=0;$('timeline').max=0;$('timeline').disabled=true;
  for(const id of ['play','resetFlood','saveFrame','saveArchive'])$(id).disabled=true;
  $('savePng').disabled=false;$('waterBudget').replaceChildren();
  $('scenarioInfo').textContent=`预览初始水体 · 1 方块＝${unit} 米 · 尚未计算洪水。`;
  draw();
 }
 async function load() {
  stop();$('refresh').disabled=true;$('simulate').disabled=true;$('computeStatus').textContent='正在读取地图…';
  try {
   const m=await json('/api/current'),[nz,nx]=m.shape,n=nx*nz;
   const needed=m.full?['ground_height','water_surface']:['height','terrain','riverWaterLevel'];
   const values=Object.fromEntries(await Promise.all(needed.map(async key=>{
    const f=m.fields[key],a=await floats(f.url,n);return [key,f.dtype==='<i4'?new Int32Array(a.buffer):a];
   })));
   const heights=m.full?Float32Array.from(values.ground_height):Float32Array.from(values.height,v=>v*m.scale);
   if(!heights.every(Number.isFinite))throw new Error('地图包含无效高程');
   const depth=Float32Array.from(heights,(h,i)=>{
    if(m.full)return Number.isFinite(values.water_surface[i])?Math.max(0,values.water_surface[i]-h):0;
    const name=m.terrain_ids[values.terrain[i]]||'',river=['river','lake','wetland'].includes(name)?values.riverWaterLevel[i]*m.scale:0;
    return Math.max(0,m.sea_level-h,river-h);
   });
   terrain=m;rawBed=heights;rawDepth=depth;cache.clear();requestId++;
   $('sourceX').max=nx-1;$('sourceZ').max=nz-1;$('sourceX').value=Math.floor(nx/2);$('sourceZ').value=Math.floor(nz/2);
   selected={x:Math.floor(nx/2),z:Math.floor(nz/2)};
   $('mapInfo').textContent=`种子 ${m.config.seed} · ${nx} × ${nz} 原始网格 · ${m.full?'实际地面高程':'连续地形预览'} · 间距 ${m.step} 方块`;
   makePreview();$('computeStatus').textContent='地图已就绪，设置场景后计算。';status('点击地图选择来水位置；初始水体之外新增的淹没会单独统计。');
  } catch(e){status(e.message,true);$('computeStatus').textContent='读取失败，请重试。';}
  finally{$('refresh').disabled=false;$('simulate').disabled=!terrain;}
 }
 async function show(index) {
  if(preview||!meta?.frames.length)return;index=Math.max(0,Math.min(meta.frames.length-1,index));
  const id=++requestId,info=meta.frames[index],n=meta.shape[0]*meta.shape[1];
  let promise=cache.get(info.url);if(!promise){promise=floats(info.url,n*4);cache.set(info.url,promise);promise.catch(()=>cache.delete(info.url));}
  const a=await promise;if(id!==requestId)return;
  frame={depth:a.subarray(0,n),u:a.subarray(n,2*n),v:a.subarray(2*n,3*n),maximum:a.subarray(3*n,4*n),info};
  $('timeline').value=index;$('timeLabel').textContent=clock(info.time_s);$('timeStat').textContent=clock(info.time_s);
  $('depthStat').textContent=format(info.max_depth_m);$('areaStat').textContent=format(info.newly_flooded_area_m2,0);
  $('balanceStat').textContent=Number(info.balance.residual_m3).toExponential(2);
  const labels={initial_m3:'初始水量',rainfall_m3:'累计降雨',inflow_m3:'累计来水',infiltration_m3:'累计入渗',outflow_m3:'累计排水',current_m3:'当前水量',residual_m3:'水量收支误差'};
  $('waterBudget').replaceChildren(...Object.entries(labels).map(([k,label])=>{const item=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=`${format(info.balance[k],4)} m³`;item.append(dt,dd);return item;}));
  draw();
 }
 async function accept(result) {
  const isNew=preview||meta.bed_url!==result.bed_url;
  if(isNew){stop();requestId++;cache.clear();const n=result.shape[0]*result.shape[1];bed=await floats(result.bed_url,n);frame=null;}
  meta=result;preview=false;
  $('timeline').max=meta.frames.length-1;$('timeline').disabled=false;
  for(const id of ['play','resetFlood','saveFrame','savePng'])$(id).disabled=false;
  $('saveArchive').disabled=!meta.archive_url;
  const c=meta.config;$('scenarioInfo').textContent=`本次结果：${meta.shape[1]} × ${meta.shape[0]} 计算网格 · 间距 ${meta.dx_m} 米 · 1 方块＝${c.metres_per_block} 米 · 降雨 ${c.rain_mm_h} mm/h，持续 ${format(c.rain_duration_s/60)} 分 · 来水 ${c.inflow_m3_s} m³/s · ${c.boundary==='open'?'自由出流':'封闭边界'} · ${meta.terrain_source==='actual_ground'?'实际地面':'连续地形预览'}。`;
  if(follow||isNew)await show(meta.frames.length-1);
 }
 async function simulate(event) {
  event.preventDefault();if(computing||!terrain)return;stop();follow=true;lock(true);$('floodProgress').value=0;
  $('computeStatus').textContent='正在提交场景…';status('正在计算，可查看已完成的时间片。');
  try {
   const start=await json('/api/flood',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config())});job=start.id;
   while(true){const state=await json(`/api/flood-jobs/${job}`);$('computeStatus').textContent=state.message;$('floodProgress').value=state.progress;
    if(state.result?.frames.length)await accept(state.result);
    if(state.state==='failed')throw new Error(state.message);
    if(state.state==='cancelled'){status('计算已取消；已完成的时间片仍可查看。');break;}
    if(state.state==='done'){status('计算完成。播放或拖动时间进度查看演化，完整过程可下载。');break;}
    await new Promise(r=>setTimeout(r,1000));
   }
  } catch(e){status(e.message,true);$('computeStatus').textContent='本次计算未完成。';}
  finally{lock(false);job=null;}
 }
 function ramp(t) {const colors=[[44,139,210],[27,101,185],[34,60,141]],a=Math.min(1.999,Math.max(0,t)*2),k=Math.floor(a);return colors[k].map((c,j)=>Math.round(c+(colors[k+1][j]-c)*(a-k)));}
 function draw() {
  if(!frame||!bed||!meta)return;const [nz,nx]=meta.shape,n=nx*nz;
  let low=Infinity,high=-Infinity,maxSpeed=0,maxDepth=0;for(let i=0;i<n;i++){low=Math.min(low,bed[i]);high=Math.max(high,bed[i]);maxSpeed=Math.max(maxSpeed,Math.hypot(frame.u[i],frame.v[i]));maxDepth=Math.max(maxDepth,frame.maximum[i]);}
  const limit=layer==='speed'?Math.max(.1,maxSpeed):Math.max(1,maxDepth,...meta.frames.map(f=>f.max_depth_m));
  const image=new ImageData(nx,nz);
  for(let i=0;i<n;i++){
   const t=(bed[i]-low)/Math.max(1,high-low);let rgb=[151+65*t,180+32*t,141+62*t];
   const value=layer==='speed'?Math.hypot(frame.u[i],frame.v[i]):layer==='maximum'?frame.maximum[i]:frame.depth[i];
   if((layer==='maximum'?frame.maximum[i]:frame.depth[i])>=.01)rgb=ramp(layer==='speed'?value/limit:Math.log1p(value*10)/Math.log1p(limit*10));
   image.data.set([...rgb.map(Math.round),255],i*4);
  }
  const buffer=document.createElement('canvas');buffer.width=nx;buffer.height=nz;buffer.getContext('2d').putImageData(image,0,0);
  ctx.imageSmoothingEnabled=false;ctx.drawImage(buffer,0,0,canvas.width,canvas.height);
  if($('arrows').checked){ctx.strokeStyle='#ffffffdd';ctx.lineWidth=1.5;const stride=Math.max(1,Math.ceil(Math.max(nx,nz)/18));
   for(let z=Math.floor(stride/2);z<nz;z+=stride)for(let x=Math.floor(stride/2);x<nx;x+=stride){const i=z*nx+x,u=frame.u[i],v=frame.v[i],speed=Math.hypot(u,v);if(frame.depth[i]<.01||speed<.005)continue;
    const px=(x+.5)/nx*canvas.width,pz=(z+.5)/nz*canvas.height,len=9+9*Math.min(1,speed/Math.max(.1,maxSpeed)),dx=u/speed*len,dz=v/speed*len;
    ctx.beginPath();ctx.moveTo(px-dx/2,pz-dz/2);ctx.lineTo(px+dx/2,pz+dz/2);ctx.lineTo(px+dx/2-dx*.35-dz*.25,pz+dz/2-dz*.35+dx*.25);ctx.moveTo(px+dx/2,pz+dz/2);ctx.lineTo(px+dx/2-dx*.35+dz*.25,pz+dz/2-dz*.35-dx*.25);ctx.stroke();
   }
  }
  const sx=Number($('sourceX').value),sz=Number($('sourceZ').value);
  ctx.strokeStyle='#d36420';ctx.lineWidth=2;const px=(sx+.5)/terrain.shape[1]*canvas.width,pz=(sz+.5)/terrain.shape[0]*canvas.height;
  ctx.beginPath();ctx.arc(px,pz,7,0,Math.PI*2);ctx.moveTo(px-11,pz);ctx.lineTo(px+11,pz);ctx.moveTo(px,pz-11);ctx.lineTo(px,pz+11);ctx.stroke();
  const title={depth:'当前水深',speed:'流速',maximum:'累计最大水深'}[layer],unit=layer==='speed'?'m/s':'m';
  $('floodLegend').textContent=`绿色：地面 · 蓝色：${title}（≥ 0.01 米水深） · 浅蓝 → 深蓝：0 ～ ${format(limit)} ${unit} · 白箭头：当前流向 · 橙圈：来水位置${preview?' · 初始预览':''}`;
  canvas.setAttribute('aria-label',`${title}，${nx}乘${nz}计算网格，时间${clock(frame.info.time_s)}，点击指定来水位置`);
  inspect();
 }
 function inspect() {
  if(!frame||!selected)return;const factor=meta.coarsening_factor||1,[nz,nx]=meta.shape,x=Math.min(nx-1,Math.floor(selected.x/factor)),z=Math.min(nz-1,Math.floor(selected.z/factor)),i=z*nx+x;
  $('cellDetail').textContent=`原始格 (${selected.x}, ${selected.z})${factor>1?' → 计算格 ('+x+', '+z+')':''} · 地面 ${format(bed[i],2)} m · 水深 ${format(frame.depth[i],3)} m · 流速 ${format(Math.hypot(frame.u[i],frame.v[i]),3)} m/s · X / Z 速度 ${format(frame.u[i],3)} / ${format(frame.v[i],3)} m/s`;
 }
 function select(x,z) {if(!terrain||computing)return;selected={x:Math.max(0,Math.min(terrain.shape[1]-1,x)),z:Math.max(0,Math.min(terrain.shape[0]-1,z))};$('sourceX').value=selected.x;$('sourceZ').value=selected.z;draw();if(!preview)status('来水位置已修改；重新计算后应用，现有结果仍对应上一次场景。');}
 canvas.onclick=e=>{const r=canvas.getBoundingClientRect();if(terrain)select(Math.floor((e.clientX-r.left)/r.width*terrain.shape[1]),Math.floor((e.clientY-r.top)/r.height*terrain.shape[0]));};
 canvas.onkeydown=e=>{const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];if(d&&selected){e.preventDefault();select(selected.x+d[0],selected.z+d[1]);}};
 form.onsubmit=simulate;
 form.oninput=()=>{if(preview)makePreview();else{draw();status('参数已修改；现有结果仍对应上一次计算，重新计算后应用。');}};
 $('refresh').onclick=load;$('arrows').onchange=draw;
 $('cancel').onclick=async()=>{if(!job)return;try{await json(`/api/flood-jobs/${job}/cancel`,{method:'POST'});$('cancel').disabled=true;$('computeStatus').textContent='正在停止计算…';}catch(e){status(e.message,true);}};
 document.querySelectorAll('[data-layer]').forEach(b=>b.onclick=()=>{layer=b.dataset.layer;document.querySelectorAll('[data-layer]').forEach(a=>{const active=a===b;a.classList.toggle('active',active);a.setAttribute('aria-pressed',String(active));});draw();});
 $('timeline').oninput=()=>{stop();follow=false;show(Number($('timeline').value)).catch(e=>status(e.message,true));};
 $('resetFlood').onclick=()=>{stop();follow=false;show(0).catch(e=>status(e.message,true));};
 $('play').onclick=()=>{if(timer){stop();return;}follow=false;if(Number($('timeline').value)>=meta.frames.length-1)$('timeline').value=0;$('play').textContent='暂停 ❚❚';
  show(Number($('timeline').value)).catch(e=>{stop();status(e.message,true);});let pending=false;
  timer=setInterval(async()=>{if(pending)return;const next=Number($('timeline').value)+1;if(next>=meta.frames.length){stop();return;}pending=true;try{await show(next);}catch(e){stop();status(e.message,true);}finally{pending=false;}},600);
 };
 $('saveFrame').onclick=()=>{if(!frame||preview)return;const info={...meta,frames:undefined,bed_url:undefined,archive_url:undefined,time_s:frame.info.time_s,summary:frame.info,
  bed:rows(bed),depth:rows(frame.depth),u:rows(frame.u),v:rows(frame.v),maximum_depth:rows(frame.maximum)};
  download(new Blob([JSON.stringify(info)+'\n'],{type:'application/json'}),`flood-${terrain.config.seed}-${Math.round(frame.info.time_s)}s.json`);};
 $('saveArchive').onclick=()=>{if(!meta?.archive_url)return;const a=document.createElement('a');a.href=meta.archive_url;a.download=`flood-${terrain.config.seed}.zip`;a.click();};
 $('savePng').onclick=()=>{if(!frame)return;draw();const out=document.createElement('canvas');out.width=canvas.width;out.height=canvas.height+110;const c=out.getContext('2d');c.fillStyle='#f3f6f3';c.fillRect(0,0,out.width,out.height);c.drawImage(canvas,0,0);c.fillStyle='#183c3c';c.font='14px sans-serif';
  c.fillText(`HYRA · ${preview?'初始水体预览':'动态洪水'} · 种子 ${terrain.config.seed} · ${clock(frame.info.time_s)}`,18,canvas.height+25);
  c.font='12px sans-serif';c.fillText(`1 方块＝${meta.config.metres_per_block} m · 计算网格 ${meta.shape[1]} × ${meta.shape[0]} · 间距 ${meta.dx_m} m`,18,canvas.height+49);
  c.fillText(`${{depth:'当前水深 / m',speed:'流速 / m/s',maximum:'累计最大水深 / m'}[layer]} · 白箭头：当前流向 · 淹没阈值 0.01 m`,18,canvas.height+73);
  c.fillText('合成地形场景；初始水体与洪水过程分开记录。',18,canvas.height+96);
  out.toBlob(b=>{if(b)download(b,`flood-${terrain.config.seed}-${Math.round(frame.info.time_s)}s.png`);});};
 load();
})();
