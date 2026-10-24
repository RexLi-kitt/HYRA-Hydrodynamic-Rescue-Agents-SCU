import {decodeRows} from '/shared/map-arrays.mjs';
const $=id=>document.getElementById(id);
const state={event:null,terrain:null,valid:null,packet:null,serial:0,playing:false,last:0,pending:false,lastQuery:''};
const nf=new Intl.NumberFormat('zh-CN');
const stamp=t=>new Date(t).toISOString().slice(0,16).replace('T',' ');
const local=t=>new Intl.DateTimeFormat('zh-CN',{timeZone:'America/Chicago',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(t));
async function read(url){const r=await fetch(url);const v=await r.json();if(!r.ok)throw Error(v.error||'读取失败');return v;}
function now(){return Date.parse(state.event.start_utc)+Number($('time').value)*60000;}
function failure(error){$('status').textContent=error.message;$('status').className='error';state.playing=false;$('play').textContent='播放';}
async function refresh(){
 if(!state.event)return;
 const query=`${now()}|${$('latency').value}`;
 if(query===state.lastQuery)return;
 state.lastQuery=query;const serial=++state.serial;
 try{
  const p=await read(`/api/observations?at=${encodeURIComponent(new Date(now()).toISOString())}&latency=${$('latency').value}`);
  if(serial!==state.serial)return;
  state.packet=p;$('clock').textContent=stamp(p.at_utc);$('local').textContent=`当地时间 ${local(p.at_utc)} · America/Chicago`;
  const latest=p.latest[0];$('reading').textContent=latest?.valid?`${latest.value.toFixed(3)} m`:'无可用记录';
  $('age').textContent=latest?`观测于 ${local(latest.observed_end_utc)} · 距今${Math.round(latest.age_s/60)}分钟${latest.stale?' · 已陈旧':''}`:'尚无已送达记录';
  $('count').textContent=nf.format(p.records.length);
  $('quality').textContent=latest?`最近记录：${latest.approval_status||'状态未知'}；来源标记：${latest.qualifier||'无额外标记'}。图上的点表示站点记录，不表示邻近格子水深。`:'';
  $('status').textContent='归档数据 / 因果回放';$('status').className='';drawChart();drawMap();
 }catch(e){if(serial===state.serial){state.lastQuery='';failure(e);}}
}
function drawChart(){
 const c=$('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);
 const pad={l:56,r:20,t:20,b:46}, x=t=>pad.l+(Date.parse(t)-Date.parse(state.event.start_utc))/(Date.parse(state.event.end_utc)-Date.parse(state.event.start_utc))*(w-pad.l-pad.r);
 // Fixed engineering axis, not an auto-scaled hint derived from future event measurements.
 const ymax=20,y=v=>h-pad.b-v/ymax*(h-pad.t-pad.b);ctx.font='15px sans-serif';
 for(let v=0;v<=20;v+=5){ctx.strokeStyle='#e5ece7';ctx.beginPath();ctx.moveTo(pad.l,y(v));ctx.lineTo(w-pad.r,y(v));ctx.stroke();ctx.fillStyle='#758781';ctx.fillText(v,18,y(v)+5);}
 for(let day=0;day<=8;day+=2){const date=new Date(Date.parse(state.event.start_utc)+day*86400000);ctx.fillStyle='#758781';ctx.fillText(`${date.getUTCMonth()+1}/${date.getUTCDate()}`,x(date.toISOString())-16,h-16);}
 const cutoff=x(state.packet.at_utc);ctx.fillStyle='#f1f3f1';ctx.fillRect(cutoff,pad.t,w-pad.r-cutoff,h-pad.t-pad.b);ctx.fillStyle='#82908b';if(cutoff<w-150)ctx.fillText('未来记录隐藏',cutoff+12,pad.t+26);
 ctx.strokeStyle='#287b70';ctx.lineWidth=2.2;ctx.beginPath();let previous=null;
 const nominal=Math.min(...Object.keys(state.event.interval_histogram_s).map(Number));
 for(const r of state.packet.records){if(!r.valid){previous=null;continue;}const px=x(r.observed_end_utc),py=y(r.value);if(previous&&(Date.parse(r.observed_end_utc)-Date.parse(previous.observed_end_utc))/1000<=nominal*1.5)ctx.lineTo(px,py);else ctx.moveTo(px,py);previous=r;}ctx.stroke();
 ctx.fillStyle='#287b70';for(const r of state.packet.records){if(!r.valid)continue;ctx.beginPath();ctx.arc(x(r.observed_end_utc),y(r.value),1.7,0,Math.PI*2);ctx.fill();}
 ctx.setLineDash([5,5]);ctx.strokeStyle='#b29d76';ctx.beginPath();ctx.moveTo(cutoff,pad.t);ctx.lineTo(cutoff,h-pad.b);ctx.stroke();ctx.setLineDash([]);
}
function drawMap(){
 const c=$('map'),ctx=c.getContext('2d'),w=c.width;ctx.clearRect(0,0,w,w);
 if(!state.terrain){ctx.fillStyle='#edf0ec';ctx.fillRect(0,0,w,w);ctx.fillStyle='#5b726b';ctx.font='20px sans-serif';ctx.fillText('未导入地形；地图水情未知',90,280);return;}
 const t=state.event.terrain,g=t.grid,rows=g.rows,cols=g.cols;
 const image=ctx.createImageData(cols,rows),span=t.max_m-t.min_m||1;
 for(let r=0;r<rows;r++)for(let col=0;col<cols;col++){
  const i=(r*cols+col)*4,z=(state.terrain[r][col]-t.min_m)/span;
  const height=Math.max(0,Math.min(1,z));
  const rgb=state.valid[r][col]?[Math.round(80+height*143),Math.round(119+height*97),Math.round(101+height*85)]:[194,199,194];
  image.data.set([...rgb,255],i);
 }
 const off=document.createElement('canvas');off.width=cols;off.height=rows;off.getContext('2d').putImageData(image,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(off,0,0,w,w);
 // Sparse station observations do not paint spatial wet/dry claims over a DEM.
 ctx.strokeStyle='#ffffff24';ctx.lineWidth=2;for(let p=-w;p<w;p+=25){ctx.beginPath();ctx.moveTo(p,0);ctx.lineTo(p+w,w);ctx.stroke();}
 const east=(g.station_east_m-g.west_m)/(g.cols*g.cell_m)*w,north=(g.north_m-g.station_north_m)/(g.rows*g.cell_m)*w;
 const width=Number($('width').value)/(cols*g.cell_m)*w;ctx.fillStyle='#fff7df1c';ctx.strokeStyle='#fff1c8';ctx.lineWidth=2.5;ctx.strokeRect(east-width/2,north-width/2,width,width);ctx.fillRect(east-width/2,north-width/2,width,width);
 const current=state.packet?.latest[0];ctx.beginPath();ctx.arc(east,north,9,0,Math.PI*2);ctx.fillStyle=current?.valid&&!current.stale?'#156d8e':'#8b8981';ctx.fill();ctx.lineWidth=3;ctx.strokeStyle='white';ctx.stroke();
 ctx.fillStyle='#143c35';ctx.fillRect(16,16,252,62);ctx.fillStyle='white';ctx.font='17px sans-serif';ctx.fillText('USGS 08075000',28,42);ctx.font='13px sans-serif';ctx.fillText('地形底图 · 全域水情未知',28,63);
 ctx.fillStyle='#143c35';ctx.fillRect(16,w-42,178,27);ctx.fillStyle='white';ctx.font='13px sans-serif';ctx.fillText('10米概览 / 2017年代待核',24,w-23);
 ctx.strokeStyle='white';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(w-120,w-30);ctx.lineTo(w-30,w-30);ctx.stroke();ctx.fillStyle='white';ctx.fillText('300 m',w-108,w-42);ctx.font='20px sans-serif';ctx.fillText('N ↑',w-65,40);
}
async function scale(){try{const plan=await read(`/api/grid-plan?width=${$('width').value}&cell=${$('cell').value}`);$('scale').innerHTML=`<strong>${nf.format(plan.cells)}</strong> 格<br>${plan.side} × ${plan.side} / ${plan.actual_width_m} m<br>四字段单帧约 ${plan.four_float32_fields_mib.toFixed(1)} MiB`;drawMap();}catch(e){failure(e);}}
async function init(){
 state.event=await read('/api/event');const e=state.event;$('title').textContent=e.title;
 const items=[['测站水位过程',true],['现代地形概览',e.capabilities.terrain_overview],['逐时淹没范围',false],['实测水深 / 流速',false],['真实事件求解条件',false]];
 $('inventory').replaceChildren(...items.map(([name,ready])=>{const d=document.createElement('div');const label=document.createElement('span');label.textContent=name;const badge=document.createElement('span');badge.className=ready?'yes':'pending';badge.textContent=ready?'已导入':'待补齐';d.append(label,badge);return d;}));
 $('cadence').textContent=`共${nf.format(e.counts.total)}条原始记录；相邻时间间隔：${Object.entries(e.interval_histogram_s).map(([s,n])=>`${Number(s)/60}分钟 × ${n}`).join('，')}。`;
 $('limits').replaceChildren(...e.limitations.map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
 $('sources').className='source-links';for(const source of e.sources){const a=document.createElement('a');a.textContent=source.title+' ↗';a.href=source.url;a.target='_blank';a.rel='noreferrer';$('sources').append(a);}
 if(e.terrain){$('terrain-note').textContent=e.terrain.warning;await Promise.all(Object.entries(e.terrain.fields).map(async([name,f])=>{const r=await fetch(f.url);if(!r.ok)throw Error('地形数组读取失败');state[name==='elevation'?'terrain':'valid']=decodeRows(await r.arrayBuffer(),f);}));}
 $('time').max=(Date.parse(e.end_utc)-Date.parse(e.start_utc))/60000;
 await scale();await refresh();
 $('time').addEventListener('input',()=>{state.playing=false;$('play').textContent='播放';refresh();});
 $('latency').addEventListener('change',refresh);$('width').addEventListener('change',scale);$('cell').addEventListener('change',scale);
 $('step').onclick=()=>{state.playing=false;$('play').textContent='播放';$('time').value=Math.min(Number($('time').max),Number($('time').value)+15);refresh();};
 $('play').onclick=()=>{state.playing=!state.playing;$('play').textContent=state.playing?'暂停':'播放';state.last=0;};
 setInterval(async()=>{if(!state.playing||state.pending)return;const current=performance.now();if(!state.last){state.last=current;return;}const delta=Math.min(.5,(current-state.last)/1000);state.last=current;const step=delta*Number($('speed').value)/60;const next=Math.min(Number($('time').max),Number($('time').value)+step);$('time').value=Math.round(next);if(next>=Number($('time').max)){state.playing=false;$('play').textContent='播放';}state.pending=true;try{await refresh();}finally{state.pending=false;}},250);
}
init().catch(failure);
