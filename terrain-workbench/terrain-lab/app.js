'use strict';
const $ = id => document.getElementById(id);
const form = $('controls'), canvas = $('map'), ctx = canvas.getContext('2d');
let map = null, arrays = {}, layer = 'height', defaults = null, busy = false, generationJob = null;
let terrainProfiles = {};
const materialLabels = ['草本地表','土壤','砂质地表','砂砾','黏土','岩石'];
const materialColors = ['#6b9e58','#9b7049','#e1cd94','#a0a397','#9baeb7','#697b80'];
const biomeStyles = {
 'deep_ocean':['深海','#215586'],'ocean':['海洋','#358db4'],
 'cold_ocean':['冷水海洋','#357a9d'],'deep_cold_ocean':['冷水深海','#235a7e'],
 'lukewarm_ocean':['温水海洋','#438fae'],'deep_lukewarm_ocean':['温水深海','#2b7191'],
 'warm_ocean':['暖水海洋','#4fb5ba'],'dark_forest':['茂密森林','#326e48'],
 'forest':['森林','#55936c'],'birch_forest':['桦树林','#8eb580'],
 'plains':['平原','#b1bd78'],'sunflower_plains':['向日葵平原','#c7c982'],
 'beach':['沙滩','#e8d299'],'stony_shore':['石质海岸','#a4adb0'],
 'river':['河流','#58b4d6'],'frozen_river':['冰冻河流','#abcddd'],
 'taiga':['针叶林','#497775'],'jungle':['热带森林','#427b45'],
 'sparse_jungle':['稀疏热带森林','#84a95f'],'bamboo_jungle':['竹林','#679157'],
 'swamp':['湿地','#788b68'],'mangrove_swamp':['红树林湿地','#5e7f68'],
 'desert':['沙漠','#d5bb85'],'savanna':['稀树草原','#b5b277'],
 'badlands':['红色岩地','#bd876a'],'wooded_badlands':['林地红色岩地','#aa9270'],
 'eroded_badlands':['侵蚀红色岩地','#c19b76'],'dripstone_caves':['溶洞露头','#8a8e96'],
 'meadow':['山地草甸','#8ca571'],'windswept_hills':['丘陵','#8b9784'],
 'windswept_forest':['山地森林','#688676'],'stony_peaks':['裸岩山峰','#a29c94'],
 'snowy_plains':['积雪平原','#dae0d7'],'grove':['山地林地','#a1b5a5'],
 'snowy_slopes':['积雪山坡','#cdd6d2'],'frozen_peaks':['冰冻山峰','#c5d5d7']
};
const blockNames = {grass_block:'草方块',dirt:'泥土',rooted_dirt:'缠根泥土',sand:'沙子',gravel:'砂砾',clay:'黏土',stone:'石头',granite:'花岗岩',diorite:'闪长岩',andesite:'安山岩',tuff:'凝灰岩',smooth_basalt:'平滑玄武岩',deepslate:'深板岩',sandstone:'砂岩',coal_ore:'煤矿石',iron_ore:'铁矿石',copper_ore:'铜矿石',snow_block:'雪块',podzol:'灰化土',coarse_dirt:'砂土',mud:'泥巴'};
const short = s => s.split(':').pop();
const hex = s => [parseInt(s.slice(1,3),16),parseInt(s.slice(3,5),16),parseInt(s.slice(5,7),16)];
function biomeStyle(name) { const s = short(name); return biomeStyles[s] || [s, '#809785']; }
function materialClass(name) {
 const s = short(name);
 return s === 'grass_block' ? 0 : ['dirt','rooted_dirt','podzol','coarse_dirt','mud'].includes(s) ? 1 : s === 'sand' || s === 'red_sand' ? 2 : s === 'gravel' ? 3 : s === 'clay' ? 4 : 5;
}
function updateLabels() {
 for (const input of form.querySelectorAll('input[type=range]')) {
  const output = form.querySelector(`output[for="${input.id}"]`);
  output.value = input.id.endsWith('chance') ? `${Math.round(Number(input.value)*100)}%` : String(Number(input.value));
 }
 $('modeHelp').textContent = $('mode').value === 'quick' ? '快速预览只生成地形与河网。' : '完整生成耗时更长，包含实际方块与群系。';
 const side=Number($('span').value);
 $('gridHelp').textContent=`${side} × ${side} 格，共 ${(side*side).toLocaleString('zh-CN')} 格；每格 1 方块，洪水直接沿用。${side>=1024?'大地图会显著增加生成、模拟和存储开销，可先用 128 或 256 验证场景。':''}`;
 $('terrainModeHelp').textContent = $('terrain_mode').value === 'custom' ? '种子决定随机细节，参数决定地形倾向。' : '保留种子，最多检查附近 9 个区域；同一种子与坐标会重复相同搜索。符合目标后才替换地图，修改地形参数会回到自由调整。';
}
function getConfig() { const o = Object.fromEntries(new FormData(form)); for (const k of Object.keys(o)) if (!['seed','mode','terrain_mode'].includes(k)) o[k] = Number(o[k]); return o; }
function applyConfig(config) { for (const [k,v] of Object.entries(config)) if ($(k)) $(k).value = v; updateLabels(); }
function notify(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
async function json(url, options) { const r = await fetch(url, options); const j = await r.json(); if (!r.ok) throw new Error(j.error || '读取失败'); return j; }
async function loadMap(result) {
 const next = {};
 const needed = new Set(result.full ? ['ground_height','ground_block_id','biome_id','water_surface','water_depth'] : ['height','terrain','riverWaterLevel']);
 await Promise.all(Object.entries(result.fields).filter(([name])=>needed.has(name)).map(async ([name, f]) => {
  const r = await fetch(f.url); if (!r.ok) throw new Error(`读取 ${name} 失败`);
  const b = await r.arrayBuffer();
  const a = f.dtype === '<f4' ? new Float32Array(b) : new Int32Array(b);
  if (a.length !== result.shape[0] * result.shape[1]) throw new Error('地图数据不完整');
  next[name] = a;
 }));
 map = result; arrays = next;
 // A new terrain enters the current workbench, rather than restoring an old run.
 $('floodWorkbenchLink').href=`maps.html?ui=28&fresh=1&terrain_id=${encodeURIComponent(map.id)}`;
 map.blockNames = Object.fromEntries(Object.entries(map.block_ids).map(([n,i]) => [i,n]));
 map.biomeNames = Object.fromEntries(Object.entries(map.biome_ids).map(([n,i]) => [i,n]));
 const [nz,nx] = map.shape;
 $('mapInfo').textContent = `${nx} × ${nz} 格 · 每格 ${map.step} 方块 · 覆盖 ${nx*map.step} × ${nz*map.step} 方块${map.step!==1?'（旧地图，重新生成可使用每方块一格）':''} · ${map.full ? '完整地表' : '地形预览'} · 种子 ${map.config.seed}`;
 const profile=map.terrain_profile;
 $('mapTitle').textContent = profile?.accepted ? `${profile.label}地图` : '当前地形';
 $('terrainProfile').textContent = profile?.accepted ? `已通过${profile.label}检查 · 陆地 ${(profile.metrics.land_fraction*100).toFixed(0)}% · 陆地起伏 ${profile.metrics.land_relief_m.toFixed(1)} m${profile.terrain_mode==='water_network'?` · 河流水面 ${(profile.metrics.wet_river_fraction*100).toFixed(1)}%${Number.isFinite(profile.metrics.river_length_density_km_per_km2)?` · 河道密度 ${profile.metrics.river_length_density_km_per_km2.toFixed(2)} km/km²`:''}`:''} · 区域起点 (${map.origin.join(', ')}) · 检查了 ${profile.candidates_checked} 个区域` : '';
 for (const b of document.querySelectorAll('[data-layer]')) {
  b.disabled = !map.full && ['material','biome'].includes(b.dataset.layer);
  b.title = b.disabled ? '完整生成地表后可查看' : '';
 }
 if (!map.full && ['material','biome'].includes(layer)) setLayer('height'); else draw();
 $('cellDetail').textContent = '移动鼠标或点击网格，查看高程和地表信息。';
 const ports=map.river_inlets?.crossings||[];
 $('inletSummary').textContent=`河道边界：${ports.filter(p=>p.role==='inlet').length} 个推断入口、${ports.filter(p=>p.role==='outlet').length} 个推断出口、${ports.filter(p=>p.role==='uncertain').length} 个待确认交点。${map.river_inlets?.message||''} 外源来水可进入多层地图配置。`;
}
function height(i) { return map.full ? arrays.ground_height[i] : arrays.height[i] * map.scale; }
function wet(i) {
 if (map.full) return Number.isFinite(arrays.water_surface[i]);
 const name = map.terrain_ids[arrays.terrain[i]] || '';
 return name.includes('ocean') || height(i) < map.sea_level || ['river','lake','wetland'].includes(name) && height(i) < arrays.riverWaterLevel[i] * map.scale;
}
function ocean(i) {
 if (map.full) return (map.biomeNames[arrays.biome_id[i]] || '').includes('ocean');
 return (map.terrain_ids[arrays.terrain[i]] || '').includes('ocean');
}
function ramp(t) {
 const colors = [[50,98,103],[101,162,113],[177,178,117],[213,187,141],[232,221,200]];
 const x = Math.min(3.999999,Math.max(0,t)*4), k = Math.floor(x), f = x-k;
 return colors[k].map((c,j) => c*(1-f)+colors[k+1][j]*f);
}
function swatch(name,color) { const item=document.createElement('span'); const s=document.createElement('i'); s.className='swatch';s.style.background=color;item.append(s,document.createTextNode(name));return item; }
function draw() {
 if (!map) return;
 const [nz,nx] = map.shape, n = nx*nz;
 let min=Infinity,max=-Infinity; for(let i=0;i<n;i++){const v=height(i);min=Math.min(min,v);max=Math.max(max,v);}
 const image = new ImageData(nx,nz);
 for(let i=0;i<n;i++) {
  const x=i%nx,z=Math.floor(i/nx), h=height(i); let rgb;
  if(layer==='material') rgb=hex(materialColors[materialClass(map.blockNames[arrays.ground_block_id[i]])]);
  else if(layer==='biome') rgb=hex(biomeStyle(map.biomeNames[arrays.biome_id[i]])[1]);
  else {
   rgb=layer==='water'?[206,219,193]:ramp((h-min)/Math.max(1,max-min));
   const dx=(height(z*nx+Math.min(nx-1,x+1))-height(z*nx+Math.max(0,x-1)))/(map.step*2);
   const dz=(height(Math.min(nz-1,z+1)*nx+x)-height(Math.max(0,z-1)*nx+x))/(map.step*2);
   const lighting=0.72+0.35*Math.max(0,(-0.45*dx-0.55*dz+0.72)/Math.sqrt(dx*dx+dz*dz+1));
   rgb=rgb.map(c=>Math.min(255,c*lighting));
   if(wet(i)) rgb=ocean(i)?[43,117,166]:[74,177,208];
  }
  image.data.set([...rgb.map(Math.round),255],i*4);
 }
 const buffer=document.createElement('canvas');buffer.width=nx;buffer.height=nz;buffer.getContext('2d').putImageData(image,0,0);
 ctx.imageSmoothingEnabled=false;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(buffer,0,0,canvas.width,canvas.height);
 drawRiverPorts();
 const legend=$('legend');legend.replaceChildren();
 if(layer==='material') materialLabels.forEach((s,i)=>legend.append(swatch(s,materialColors[i])));
 else if(layer==='biome') Object.values(map.biomeNames).forEach(n=>{const[s,c]=biomeStyle(n);legend.append(swatch(s,c));});
 else {
  if(layer==='height'){const a=document.createElement('span');a.textContent=`当前范围 ${min.toFixed(1)}`;const r=document.createElement('span');r.className='color-ramp';const b=document.createElement('span');b.textContent=`${max.toFixed(1)} 高程 / 方块单位`;legend.append(a,r,b);}
  legend.append(swatch('河流与湖泊','#4ab1d0'),swatch('海洋','#2b75a6'));
 }
 legend.append(swatch('推断河流入口','#913db5'),swatch('推断出口／待确认','#777e87'));
 canvas.setAttribute('aria-label',`${{height:'地形高程',water:'水系',material:'地表类型',biome:'群系'}[layer]}，${nx}乘${nz}网格，最低高程${min.toFixed(1)}，最高${max.toFixed(1)}`);
}
function drawRiverPorts(){
 const [nz,nx]=map.shape;
 for(const p of map.river_inlets?.crossings||[]){
  const x=Math.max(10,Math.min(canvas.width-10,(p.x+.5)*canvas.width/nx)),z=Math.max(10,Math.min(canvas.height-10,(p.z+.5)*canvas.height/nz));
  ctx.save();ctx.strokeStyle=p.role==='inlet'?'#913db5':'#777e87';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,z,8,0,Math.PI*2);ctx.stroke();
  const label=p.role==='inlet'?'入口':p.role==='outlet'?'出口':'待确认';ctx.font='bold 15px sans-serif';ctx.textAlign=x>canvas.width/2?'right':'left';const tx=x+(x>canvas.width/2?-12:12),tz=Math.max(20,Math.min(canvas.height-7,z));ctx.strokeStyle='#fff';ctx.lineWidth=4;ctx.strokeText(label,tx,tz);ctx.fillText(label,tx,tz);ctx.restore();
 }
}
function setLayer(next) {
 layer=next;for(const b of document.querySelectorAll('[data-layer]')){const active=b.dataset.layer===layer;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));}draw();
}
function inspectCell(event) {
 if(!map||busy)return;
 const r=canvas.getBoundingClientRect(),[nz,nx]=map.shape;
 const x=Math.min(nx-1,Math.max(0,Math.floor((event.clientX-r.left)/r.width*nx)));
 const z=Math.min(nz-1,Math.max(0,Math.floor((event.clientY-r.top)/r.height*nz))),i=z*nx+x;
 let text=`格 (${x}, ${z}) · 高程 ${height(i).toFixed(1)}`;
 if(map.full){const bn=map.blockNames[arrays.ground_block_id[i]];text+=` · ${blockNames[short(bn)]||short(bn)} · ${biomeStyle(map.biomeNames[arrays.biome_id[i]])[0]}`;if(wet(i))text+=` · 水深 ${arrays.water_depth[i].toFixed(1)}`;}
 else text+=` · ${map.terrain_ids[arrays.terrain[i]] || ''}`;
 const port=(map.river_inlets?.crossings||[]).find(p=>Math.abs(p.x-x)<=2&&Math.abs(p.z-z)<=2);if(port)text+=` · ${port.id} · ${{inlet:'推断入口',outlet:'推断出口',uncertain:'方向待确认'}[port.role]}`;
 $('cellDetail').textContent=text;
}
async function generate(event) {
 event.preventDefault();if(busy)return;
 const config=getConfig();
 notify('正在生成新地图，可取消或查看生成记录。');
 busy=true;const start=Date.now();$('loading').hidden=false;$('progress').removeAttribute('value');$('phase').textContent='正在检查生成环境…';$('elapsed').textContent='';$('cancelGenerate').disabled=true;$('generationLog').hidden=true;$('lastGenerationLog').hidden=true;
 for(const el of form.querySelectorAll('button,input,select'))el.disabled=true;
 const timer=setInterval(()=>{$('elapsed').textContent=`已等待 ${Math.floor((Date.now()-start)/1000)} 秒`},1000);
 try {
  const job=await json('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});
  generationJob=job.id;$('cancelGenerate').disabled=false;
  const logUrl=`/generated/${job.id}/generation.log`;
  for(const id of ['generationLog','lastGenerationLog']){$(id).href=logUrl;$(id).hidden=false;}
  while(true){const state=await json(`/api/jobs/${job.id}`);$('phase').textContent=state.message;
   if(state.candidate_count)$('phase').textContent+=`（区域 ${state.candidate_index}/${state.candidate_count}）`;
   if(state.log_url)for(const id of ['generationLog','lastGenerationLog'])$(id).href=state.log_url;
   if(state.progress_known)$('progress').value=state.progress;else $('progress').removeAttribute('value');
   if(state.state==='cancelled'){notify(state.message);break;}
   if(state.state==='failed')throw new Error(state.message);
   if(state.state==='done'){await loadMap(state.result);applyConfig(map.config);notify(`生成完成。${map.full ? '当前可查看实际方块与群系。' : '切换“完整地表与群系”生成后可查看地表类型。'}`);break;}
   await new Promise(resolve=>setTimeout(resolve,1500));
  }
 }catch(error){notify(error.message,true);}
 finally{busy=false;generationJob=null;clearInterval(timer);$('loading').hidden=true;for(const el of form.querySelectorAll('button,input,select'))el.disabled=false;}
}
$('cancelGenerate').addEventListener('click',async()=>{
 if(!generationJob)return;$('cancelGenerate').disabled=true;
 try{await json(`/api/jobs/${generationJob}/cancel`,{method:'POST'});$('phase').textContent='正在停止生成…';}
 catch(error){notify(error.message,true);$('cancelGenerate').disabled=false;}
});
form.addEventListener('submit',generate);
form.addEventListener('input',event=>{if(!['seed','origin_x','origin_z','span','mode','terrain_mode'].includes(event.target.id))$('terrain_mode').value='custom';updateLabels();document.querySelectorAll('.preset').forEach(b=>b.classList.remove('active'));notify('参数已修改，点击“生成地图”应用。');});
$('terrain_mode').addEventListener('change',()=>{const profile=terrainProfiles[$('terrain_mode').value];if(profile)applyConfig(profile.parameters);updateLabels();notify('目标地形已选择，点击“生成地图”应用。');});
$('mode').addEventListener('change',updateLabels);
$('randomSeed').addEventListener('click',()=>{$('seed').value=String(crypto.getRandomValues(new Uint32Array(1))[0]);notify('种子已修改，点击“生成地图”应用。');});
$('rich').addEventListener('click',()=>{const current=getConfig();applyConfig({...defaults,seed:current.seed,origin_x:current.origin_x,origin_z:current.origin_z,span:current.span,mode:current.mode});$('rich').classList.add('active');$('original').classList.remove('active');notify('已载入水系丰富参数，点击“生成地图”应用。');});
$('original').addEventListener('click',()=>{const current=getConfig();applyConfig({...defaults,seed:current.seed,origin_x:current.origin_x,origin_z:current.origin_z,span:current.span,mode:current.mode,vertical_scale:1,continent_scale:3000,region_size:1200,river_count:8,main_width:8,branch_width:5,river_depth:5,main_bank_width:20,branch_bank_width:14,branch_depth:4,lake_chance:0.3,wetland_chance:0.6,lake_distance:0.03});$('original').classList.add('active');$('rich').classList.remove('active');notify('已载入原始参数，点击“生成地图”应用。');});
document.querySelectorAll('[data-layer]').forEach(b=>b.addEventListener('click',()=>setLayer(b.dataset.layer)));
canvas.addEventListener('pointermove',inspectCell);canvas.addEventListener('click',inspectCell);
$('exportPng').addEventListener('click',()=>{if(!map)return;$('previewImage').src=canvas.toDataURL('image/png');$('imageTitle').textContent=`${{height:'地形高程',water:'水系',material:'地表类型',biome:'群系'}[layer]} · 种子 ${map.config.seed}`;$('imagePreview').showModal();});
$('closePreview').addEventListener('click',()=>$('imagePreview').close());
(async()=>{for(const el of form.querySelectorAll('button,input,select'))el.disabled=true;try{const cfg=await json('/api/config');defaults=cfg.defaults;terrainProfiles=cfg.terrain_profiles||{};applyConfig(defaults);const example=new URLSearchParams(location.search).get('example')==='rich';const current=await json(example?'/rich-result.json':'/api/current');applyConfig({...current.config,mode:'quick'});await loadMap(current);}catch(error){notify(`读取地图失败：${error.message}`,true);}finally{for(const el of form.querySelectorAll('button,input,select'))el.disabled=false;}})();
