'use strict';
const $ = id => document.getElementById(id);
const colors=['#dbe1e5','#fafcfc','#e3f2fa','#b7d9ee','#7bb5d8','#4389b9','#235d92','#153863'];
const drones=['#d74848','#ef9f1b','#813cce','#127668'];
let catalog, replay, timer=null, busy=false;
const label=id=>id==='pure_planner'?'纯覆盖规划':id==='uniform_service'?'随机动作＋服务':id.toUpperCase().replaceAll('_',' + ');
const pct=x=>(100*x).toFixed(1)+'%';
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o;}
async function api(url,data){const r=await fetch(url,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});const j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);return j;}
function scores(){
  const ids=[...catalog.candidates.map(c=>c.id),'pure_planner','uniform_service'];
  for(const id of ids){const c=catalog.candidates.find(c=>c.id===id);const variant=id==='pure_planner'?'planner':id==='uniform_service'?'no-cover':'full';const mode=id==='uniform_service'?'sample':'fixed';const find=s=>catalog.summary.find(r=>r.candidate===id&&r.variant===variant&&r.mode===mode&&r.stratum===s);const a=find('iid50'),b=find('orientation_cluster20');if(!a)continue;
    const tr=document.createElement('tr');for(const value of [label(id),c?c.episode:'规则',pct(a.means.task_complete),b?pct(b.means.task_complete):'—',pct(a.means.RCR),(a.means.ST*5).toFixed(1)+'秒']){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('scores').append(tr);
  }
}
function filter(){
  stop(); const isDemo=$('mode').value==='demo';$('recordOptions').hidden=isDemo;$('demoOptions').hidden=!isDemo;
  const candidate=$('candidate').value,c=catalog.candidates.find(c=>c.id===candidate);$('assignment').disabled=!c||c.family!=='pair';if($('assignment').disabled)$('assignment').value='0';
  const rows=catalog.records.filter(r=>r.candidate===candidate&&($('stratum').value==='all'||r.stratum===$('stratum').value)&&($('variant').value==='all'||r.variant===$('variant').value)&&($('outcome').value==='all'||r.task_complete===($('outcome').value==='success')));
  const selected=$('record').value;
  $('record').replaceChildren(...rows.map(r=>option(r.index,`${r.case} · 分工${r.assignment} · ${r.variant==='full'?'完整':r.variant==='planner'?'纯规划':'无覆盖'} · ${r.mode==='fixed'?'固定':'随机'+r.action_seed} · ${r.task_complete?'完成':'未完成'} · ${r.rescued}/20人`)));
  if(rows.some(r=>String(r.index)===selected))$('record').value=selected;
  $('recordCount').textContent=`匹配 ${rows.length} 条；全部正式记录 ${catalog.records.length} 条。`;
  $('load').disabled=busy||!catalog.runtime_ready||(isDemo?(!c||!catalog.dev.length):!rows.length);
  $('load').textContent=isDemo?'加载最佳权重并演示':'加载并核验回放';
  for(const id of ['candidate','mode','stratum','variant','outcome','record','dev'])$(id).disabled=busy;
  if(busy)$('assignment').disabled=true;
  if(!catalog.runtime_ready)$('status').textContent='统计可查看；当前机器未配置本地权重、水场和轨迹，回放暂不可用。';
}
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='播放';}
function draw(){
  if(!replay)return;const i=Number($('time').value),f=replay.frames[i];const cv=$('map'),ctx=cv.getContext('2d'),scale=cv.width/1000,cell=cv.width/100;
  const grid=$('truth').checked?f.truth:f.observed;for(let r=0;r<100;r++)for(let c=0;c<100;c++){ctx.fillStyle=colors[Number(grid[r*100+c])];ctx.fillRect(c*cell,(99-r)*cell,cell+.1,cell+.1);}
  if($('trails').checked)for(let a=0;a<4;a++){ctx.strokeStyle=drones[a];ctx.lineWidth=2;ctx.beginPath();for(let n=0;n<=i;n++){const p=replay.frames[n].positions[a];if(n===0)ctx.moveTo(p[0]*scale,cv.height-p[1]*scale);else ctx.lineTo(p[0]*scale,cv.height-p[1]*scale);}ctx.stroke();}
  for(const p of f.people){const x=p.xy[0]*scale,y=cv.height-p.xy[1]*scale;ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fillStyle=p.state===2?'#159252':'#ed8615';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();if(p.progress>0&&p.state!==2){ctx.strokeStyle='#152d3d';ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,y,10,-Math.PI/2,-Math.PI/2+Math.PI*2*p.progress/10);ctx.stroke();}}
  for(let a=0;a<4;a++){const p=f.positions[a],h=f.headings[a],x=p[0]*scale,y=cv.height-p[1]*scale;ctx.save();ctx.translate(x,y);ctx.rotate(Math.atan2(-h[1],h[0]));ctx.beginPath();ctx.moveTo(12,0);ctx.lineTo(-8,-7);ctx.lineTo(-8,7);ctx.closePath();ctx.fillStyle=drones[a];ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke();ctx.restore();ctx.fillStyle='#142b39';ctx.font='bold 14px sans-serif';ctx.fillText(String(a+1),Math.max(3,Math.min(cv.width-18,x+12)),Math.max(17,Math.min(cv.height-4,y-8)));}
  $('clock').textContent=f.t+' / '+replay.frames.at(-1).t+'秒';
  $('metrics').replaceChildren();for(const [name,value] of [['首次覆盖',pct(f.coverage)],['已发现',f.people.length+'/20'],['已完成服务',f.rescued+'/20'],['累计冲突',f.conflicts]]){const d=document.createElement('div');d.textContent=name;const b=document.createElement('strong');b.textContent=value;d.append(b);$('metrics').append(d);}
}
async function load(){
  busy=true;stop();filter();$('status').textContent='正在核验来源并生成真实回放，请稍候…';
  try{const req=$('mode').value==='record'?{mode:'record',index:Number($('record').value)}:{mode:'demo',candidate:$('candidate').value,case:Number($('dev').value),assignment:Number($('assignment').value)};
    const run=await api('/api/six-study/run',req);const r=await api(run.url);replay=r;$('title').textContent=r.title;$('kind').textContent=r.evidence.kind==='record'?'正式动作重放 · 已核验':'开发演示 · 不计入评分';$('time').max=r.frames.length-1;$('time').value=0;$('play').disabled=false;$('truth').checked=false;$('evidence').textContent=JSON.stringify(r.evidence,null,2);$('detail').textContent=`终局：${r.metrics.task_complete?'完整任务完成':'任务未全部完成'}；服务 ${r.metrics.rescued}/20，覆盖 ${pct(r.metrics.coverage)}。${r.note}`;$('download').href=run.url;$('download').hidden=false;$('status').textContent='已就绪。点击播放，或拖动时间轴查看。';draw();
  }catch(e){$('status').textContent='未生成回放：'+e.message;}finally{busy=false;filter();}
}
$('load').onclick=load;$('time').oninput=()=>{stop();draw();};$('truth').onchange=draw;$('trails').onchange=draw;
$('play').onclick=()=>{if(timer){stop();return;}if(Number($('time').value)>=replay.frames.length-1)$('time').value=0;$('play').textContent='暂停';timer=setInterval(()=>{const i=Number($('time').value)+1;if(i>=replay.frames.length){stop();return;}$('time').value=i;draw();},1000/Number($('speed').value));};$('speed').onchange=stop;
for(const id of ['candidate','mode','stratum','variant','outcome'])$(id).onchange=filter;
api('/api/six-study/catalog').then(c=>{catalog=c;$('scope').textContent=c.scope;scores();$('candidate').replaceChildren(...[...c.candidates.map(c=>c.id),'pure_planner','uniform_service'].map(id=>option(id,label(id))));$('dev').replaceChildren(...c.dev.map(d=>option(d.index,d.name)));filter();}).catch(e=>{$('status').textContent='数据读取失败：'+e.message;$('scope').textContent='未读取到结果，请检查本地研究资料。';});
