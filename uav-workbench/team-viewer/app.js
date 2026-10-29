'use strict';
const $=id=>document.getElementById(id);
const colors={wet:'#3d84b8',dry:'#f2f6f8',unknown:'#788b96',agents:['#b85629','#6554a6'],wrong:'#a23123'};
let run=null,envelope=null,frame=0,timer=null,sourceName='';
function message(text,error=false){$('message').textContent=text;$('message').className=error?'error':'';}
function validate(value){
  if(!value||typeof value!=='object')throw Error('请选择包含双机回放的JSON对象。');
  const replay=value.replay||value;
  if(replay.schema!=='hyra-team-run-v1'||replay.n_agents!==2)throw Error('请选择双机回放 JSON；原单机回放格式不能在此页打开。');
  const shape=replay.grid?.shape;
  if(!Array.isArray(shape)||shape.length!==2||shape.some(v=>!Number.isInteger(v)||v<2||v>512))throw Error('回放网格尺寸无效。');
  const n=shape[0]*shape[1],snaps=replay.snapshots,truth=replay.truth;
  if(!Array.isArray(snaps)||snaps.length<2||snaps.length>5000||!Array.isArray(truth)||truth.length!==snaps.length)throw Error('回放缺少对应的实际帧或共享地图。');
  if(!Array.isArray(replay.observations)||replay.observations.length!==snaps.length*2)throw Error('每个时刻需有两架无人机的实际观测。');
  if(!Number.isFinite(replay.grid.dx_m)||replay.grid.dx_m<=0)throw Error('回放格距无效。');
  if(!replay.config||!Number.isFinite(replay.config.duration_s)||!Number.isFinite(replay.config.sample_s)||replay.config.sample_s<=0||!Number.isFinite(replay.threshold_m)||!/^\w{64}$/.test(replay.source_sha256||''))throw Error('回放缺少任务设置、阈值或来源摘要。');
  if(!Array.isArray(replay.changes)||replay.changes.some(c=>!Number.isFinite(c.changed_s)||(c.detected_s!==null&&(!Number.isFinite(c.detected_s)||c.detected_s<c.changed_s))))throw Error('回放变化区间无效。');
  for(let i=0;i<snaps.length;i++){
    const s=snaps[i];
    if(!Number.isFinite(s.time_s)||(i===0?s.time_s!==0:s.time_s<=snaps[i-1].time_s))throw Error('回放时间必须从零开始严格递增。');
    if(i&&Math.abs(s.time_s-snaps[i-1].time_s-replay.config.sample_s)>1e-7)throw Error('回放采样间隔与任务设置不一致。');
    if(!['distance_m','turn_s','collision_events'].every(k=>Number.isFinite(s[k])&&s[k]>=0))throw Error('回放行动成本记录无效。');
    if(!Array.isArray(s.labels)||s.labels.length!==n||!Array.isArray(s.observed_s)||s.observed_s.length!==n||!Array.isArray(truth[i])||truth[i].length!==n)throw Error('回放中的地图数组尺寸不一致。');
    if(!s.poses||s.poses.length!==2||s.poses.some(p=>!['x_m','z_m','heading_deg'].every(k=>Number.isFinite(p[k]))))throw Error('双机位置或航向无效。');
    for(let c=0;c<n;c++){
      if(![-1,0,1].includes(s.labels[c])||![0,1].includes(truth[i][c]))throw Error('水情标签必须为干燥、淹没或未知。');
      if(!Number.isFinite(s.observed_s[c])||s.observed_s[c]>s.time_s||(s.labels[c]===-1?s.observed_s[c]!==-1:s.observed_s[c]<0))throw Error('观测时间戳与记录状态不一致。');
    }
    for(let agent=0;agent<2;agent++){
      const o=replay.observations[2*i+agent];
      if(o.agent_id!==agent||o.observed_s!==s.time_s||o.arrived_s!==s.time_s||!Array.isArray(o.cells)||o.cells.some(c=>!Number.isInteger(c)||c<0||c>=n))throw Error('实际观测与两机共同时间轴不一致。');
    }
  }
  if(snaps.at(-1).time_s!==replay.config.duration_s)throw Error('回放未达到登记的任务截止时刻。');
  return replay;
}
function load(value,name){
  const valid=validate(value);pause();run=valid;envelope=value;sourceName=name;frame=0;
  $('time').max=run.snapshots.length-1;$('time').value=0;
  for(const id of ['play','previous','next','time'])$(id).disabled=false;
  message('已载入 '+name+'。');
  const method=run.execution?.method||value.method||'两机策略',seed=run.execution?.seed??value.seed;
  $('record').textContent=`${method}${seed===undefined?'':'，种子 '+seed}；${run.grid.shape[1]} × ${run.grid.shape[0]} 格，格距 ${run.grid.dx_m} 米，采样 ${run.config.sample_s} 秒，阈值 ${run.threshold_m} 米。来源摘要 ${run.source_sha256.slice(0,16)}…`;
  render();
}
function instantaneous(s,truth){
  let known=0,wrong=0;const totals=[0,0],correct=[0,0];
  for(let c=0;c<truth.length;c++){totals[truth[c]]++;if(s.labels[c]>=0){known++;if(s.labels[c]===truth[c])correct[truth[c]]++;else wrong++;}}
  const scores=[0,1].filter(c=>totals[c]>0).map(c=>correct[c]/totals[c]);
  return {quality:scores.reduce((a,b)=>a+b,0)/scores.length,unknown:1-known/truth.length,error:known?wrong/known:null};
}
function map(canvas,belief){
  const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,[nz,nx]=run.grid.shape,dx=run.grid.dx_m;
  const cw=w/nx,ch=h/nz,s=run.snapshots[frame],values=belief?s.labels:run.truth[frame];
  ctx.clearRect(0,0,w,h);
  for(let c=0;c<values.length;c++){
    const x=c%nx*cw,z=Math.floor(c/nx)*ch;
    ctx.fillStyle=values[c]<0?colors.unknown:values[c]?colors.wet:colors.dry;ctx.fillRect(x,z,cw+.3,ch+.3);
    if(belief&&values[c]>=0&&values[c]!==run.truth[frame][c]){ctx.strokeStyle=colors.wrong;ctx.lineWidth=1.5;ctx.strokeRect(x+1,z+1,cw-2,ch-2);}
  }
  const point=p=>[(p.x_m+dx/2)/(nx*dx)*w,(p.z_m+dx/2)/(nz*dx)*h];
  if($('paths').checked){
    for(let a=0;a<2;a++){
      ctx.beginPath();for(let i=0;i<=frame;i++){const [x,z]=point(run.snapshots[i].poses[a]);i?ctx.lineTo(x,z):ctx.moveTo(x,z);}
      ctx.strokeStyle=colors.agents[a];ctx.lineWidth=2.4;ctx.lineJoin='round';ctx.stroke();
    }
  }
  if($('sensing').checked){
    for(let a=0;a<2;a++){
      const cells=new Set(run.observations[frame*2+a].cells);ctx.strokeStyle=colors.agents[a];ctx.lineWidth=1.6;ctx.beginPath();
      for(const c of cells){const x=c%nx*cw,z=Math.floor(c/nx)*ch,col=c%nx,row=Math.floor(c/nx);
        if(!row||!cells.has(c-nx)){ctx.moveTo(x,z);ctx.lineTo(x+cw,z);}
        if(row===nz-1||!cells.has(c+nx)){ctx.moveTo(x,z+ch);ctx.lineTo(x+cw,z+ch);}
        if(!col||!cells.has(c-1)){ctx.moveTo(x,z);ctx.lineTo(x,z+ch);}
        if(col===nx-1||!cells.has(c+1)){ctx.moveTo(x+cw,z);ctx.lineTo(x+cw,z+ch);}
      }ctx.stroke();
    }
  }
  for(let a=0;a<2;a++){
    const p=s.poses[a],[x,z]=point(p);ctx.save();ctx.translate(x,z);ctx.rotate(p.heading_deg*Math.PI/180);
    ctx.beginPath();ctx.moveTo(10,0);ctx.lineTo(-7,-6);ctx.lineTo(-4,0);ctx.lineTo(-7,6);ctx.closePath();
    ctx.fillStyle=colors.agents[a];ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1.6;ctx.stroke();ctx.restore();
    ctx.fillStyle=colors.agents[a];ctx.font='bold 14px sans-serif';ctx.fillText(String(a+1),Math.max(4,Math.min(w-16,x+12)),Math.max(17,Math.min(h-5,z-9)));
  }
}
function render(){
  if(!run)return;const s=run.snapshots[frame],m=instantaneous(s,run.truth[frame]);
  $('clock').textContent=`${s.time_s} / ${run.config.duration_s} 秒`;$('time').value=frame;
  $('quality').textContent=(m.quality*100).toFixed(1)+'%';$('unknown').textContent=(m.unknown*100).toFixed(1)+'%';
  $('error').textContent='已知区错误 '+(m.error===null?'—':(m.error*100).toFixed(1)+'%');
  $('distance').textContent=s.distance_m.toFixed(1)+' m';$('turn').textContent='转向 '+s.turn_s.toFixed(0)+' 秒';
  const changed=(run.changes||[]).filter(c=>c.changed_s<=s.time_s),discovered=changed.filter(c=>c.detected_s!==null&&c.detected_s<=s.time_s);
  $('changes').textContent=`${discovered.length} / ${changed.length}`;$('collision').textContent=`已发现 / 已发生；碰撞 ${s.collision_events} 次`;
  $('previous').disabled=frame===0;$('next').disabled=frame===run.snapshots.length-1;
  map($('truth'),false);map($('belief'),true);
}
function pause(){if(timer)clearInterval(timer);timer=null;$('play').textContent='播放';}
function play(){
  if(timer){pause();return;}if(!run)return;if(frame===run.snapshots.length-1){frame=0;render();}
  const start=performance.now(),from=run.snapshots[frame].time_s,speed=Number($('speed').value);$('play').textContent='暂停';
  timer=setInterval(()=>{const target=from+(performance.now()-start)/1000*speed;let next=frame;
    while(next<run.snapshots.length-1&&run.snapshots[next+1].time_s<=target)next++;
    if(next!==frame){frame=next;render();}if(frame===run.snapshots.length-1)pause();},80);
}
$('play').addEventListener('click',play);
$('previous').addEventListener('click',()=>{pause();frame=Math.max(0,frame-1);render();});
$('next').addEventListener('click',()=>{pause();frame=Math.min(run.snapshots.length-1,frame+1);render();});
$('time').addEventListener('input',()=>{pause();frame=Number($('time').value);render();});
$('speed').addEventListener('change',()=>{if(timer){pause();play();}});
for(const id of ['paths','sensing'])$(id).addEventListener('change',render);
$('file').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;
  try{if(file.size>30*1024*1024)throw Error('文件超过30 MB，请选择单次回放。');load(JSON.parse(await file.text()),file.name);}
  catch(error){message(error.message,true);}finally{event.target.value='';}});
$('example').addEventListener('click',async()=>{try{const response=await fetch('/example.json');if(!response.ok)throw Error('当前没有工程演示，请先运行两机策略或打开本地回放。');load(await response.json(),'拟合集工程演示');}catch(error){message(error.message,true);}});
for(const id of ['truth','belief'])$(id).addEventListener('pointermove',event=>{
  if(!run)return;const rect=event.currentTarget.getBoundingClientRect(),[nz,nx]=run.grid.shape;
  const col=Math.min(nx-1,Math.max(0,Math.floor((event.clientX-rect.left)/rect.width*nx))),row=Math.min(nz-1,Math.max(0,Math.floor((event.clientY-rect.top)/rect.height*nz)));
  const c=row*nx+col,s=run.snapshots[frame],record=s.labels[c],actual=run.truth[frame][c],stamp=s.observed_s[c];
  $('cell').textContent=`格点 (${col}, ${row})，坐标 (${col*run.grid.dx_m}, ${row*run.grid.dx_m}) 米；实际${actual?'淹没':'干燥'}，${record<0?'尚未观测':`记录${record?'淹没':'干燥'}，实际观测于 ${stamp} 秒，距今 ${s.time_s-stamp} 秒`}。`;
});
