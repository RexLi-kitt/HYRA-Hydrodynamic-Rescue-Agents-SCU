const $ = id => document.getElementById(id);
let scenes = [], replay = null, frame = 0, playing = false, lastTick = null, accumulated = 0;
const pct = x => x == null ? '不适用' : `${(100*x).toFixed(1)}%`;
function status(message, error=false) { $('status').textContent=message; $('status').classList.toggle('error',error); }
async function api(url, options={}) { const response=await fetch(url,options); const value=await response.json(); if(!response.ok)throw new Error(value.error || `请求失败 ${response.status}`); return value; }
async function loadScenes(selected) {
  scenes=await api('/api/scenes'); $('scene').replaceChildren(...scenes.map(s=>new Option(s.name,s.id)));
  if(selected)$('scene').value=selected;
  $('run').disabled=!scenes.length;
  if(!scenes.length){status('尚无过程包。请生成开发过程或导入洪水工作台完整ZIP。');return;}
  sceneChanged();status('过程已就绪，选择策略后生成回放。');
}
function sceneChanged(){const s=scenes.find(s=>s.id===$('scene').value);if(!s)return;$('sceneInfo').textContent=`${s.grid.shape.join(' × ')} 格 / ${s.grid.dx_m} m格距 / ${s.duration_s} s。${s.kind==='hand_authored_not_hydraulic_evidence'?'人工验收时序，不作为洪水证据。':'合成或导入开发过程，须另行核查物理适用性。'}`;
  const config=s.recommended_config||{speed_m_s:2.5,angular_speed_deg_s:9,range_m:25,fov_deg:90,action_s:s.sample_s*2,sample_s:s.sample_s,duration_s:s.duration_s};
  for(const [id,key] of [['speed','speed_m_s'],['angular','angular_speed_deg_s'],['range','range_m'],['fov','fov_deg'],['actionSeconds','action_s'],['sample','sample_s'],['duration','duration_s']])$(id).value=config[key];
  const pose=s.demo_pose||{x_m:0,z_m:0,heading_deg:0};$('startX').value=pose.x_m;$('startZ').value=pose.z_m;$('heading').value=pose.heading_deg;
  $('method').value=s.demo_actions?'fixed':'R0';$('actionsLabel').hidden=!s.demo_actions;if(s.demo_actions)$('fixedActions').value=s.demo_actions.join(', ');
}
$('scene').addEventListener('change',sceneChanged);
$('method').addEventListener('change',()=>{$('actionsLabel').hidden=$('method').value!=='fixed';});
$('importFile').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{status('正在核查导入过程…');const result=await api('/api/import',{method:'POST',headers:{'Content-Type':'application/zip'},body:file});await loadScenes(result.id);}catch(e){status(e.message,true);}finally{event.target.value='';}});
$('setup').addEventListener('submit',async event=>{
  event.preventDefault();playing=false;$('play').textContent='播放';$('run').disabled=true;
  const config={speed_m_s:Number($('speed').value),angular_speed_deg_s:Number($('angular').value),range_m:Number($('range').value),fov_deg:Number($('fov').value),action_s:Number($('actionSeconds').value),sample_s:Number($('sample').value),duration_s:Number($('duration').value)};
  const data={scene:$('scene').value,method:$('method').value,config,pose:{x_m:Number($('startX').value),z_m:Number($('startZ').value),heading_deg:Number($('heading').value)}};
  if(data.method==='fixed'){data.actions=$('fixedActions').value.trim().split(/[\s,，]+/).filter(Boolean);data.method='R0';}
  status('正在执行动作、采集观测并保存日志…');
  try {
    const result=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    replay=await api(result.replay_url);frame=0;accumulated=0;
    $('timeline').max=replay.snapshots.length-1;$('timeline').value=0;$('timeline').disabled=false;$('play').disabled=false;
    $('downloads').hidden=false;
    for(const [id,name] of [['downloadReplay','replay'],['downloadObs','observations'],['downloadActions','actions'],['downloadAudit','audit']])$(id).href=`/runs/${result.run_id}/${name}.json`;
    $('summary').textContent=`全程按类正确覆盖 ${pct(replay.summary.time_average_correct_coverage)}；变化未发现 ${pct(replay.summary.undiscovered_change_fraction)}。`;
    $('provenance').textContent=`${replay.scene.name} / ${replay.method} / 过程 SHA256 ${replay.source_sha256.slice(0,12)} / 开发运行`;
    status(`完成 ${replay.actions.length} 个动作、${replay.observations.length} 次采样。可播放或拖动时间轴检查。`);render();
  }catch(e){status(e.message,true);}finally{$('run').disabled=false;}
});
function drawMap(id,kind){
  const canvas=$(id),ctx=canvas.getContext('2d'),s=replay.snapshots[frame],[rows,cols]=replay.grid.shape;
  const cell=Math.min(canvas.width/cols,canvas.height/rows),w=cell*cols,h=cell*rows,ox=(canvas.width-w)/2,oy=(canvas.height-h)/2;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#e7eef2';ctx.fillRect(0,0,canvas.width,canvas.height);
  const truth=replay.truth[frame];
  for(let i=0;i<rows*cols;i++){
    const value=kind==='truth'?truth[i]:s.labels[i];
    if(kind==='age'&&value>=0){const age=Math.min(1,(s.time_s-s.observed_s[i])/Math.max(1,replay.config.duration_s));ctx.fillStyle=`hsl(200 45% ${92-age*62}%)`;}
    else ctx.fillStyle=value<0?'#9caebb':value?'#1684b5':'#e3efea';
    const x=ox+(i%cols)*cell,y=oy+Math.floor(i/cols)*cell;ctx.fillRect(x,y,cell+.2,cell+.2);
    if(kind==='memory'&&value>=0&&value!==truth[i]){ctx.strokeStyle='#c35838';ctx.lineWidth=1.8;ctx.strokeRect(x+1,y+1,cell-2,cell-2);}
  }
  const toX=x=>ox+(x/replay.grid.dx_m+.5)*cell,toY=z=>oy+(z/replay.grid.dx_m+.5)*cell;
  ctx.save();ctx.beginPath();ctx.rect(ox,oy,w,h);ctx.clip();
  ctx.strokeStyle='#b77b10';ctx.lineWidth=2;ctx.beginPath();replay.snapshots.slice(0,frame+1).forEach((p,i)=>{const x=toX(p.x_m),y=toY(p.z_m);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  const x=toX(s.x_m),y=toY(s.z_m),angle=s.heading_deg*Math.PI/180,half=replay.config.fov_deg*Math.PI/360,r=replay.config.range_m/replay.grid.dx_m*cell;
  ctx.beginPath();ctx.moveTo(x,y);ctx.arc(x,y,r,angle-half,angle+half);ctx.closePath();ctx.fillStyle='#f8cb4b55';ctx.strokeStyle='#a8790c';ctx.lineWidth=1.5;ctx.fill();ctx.stroke();
  ctx.translate(x,y);ctx.rotate(angle);ctx.beginPath();ctx.moveTo(10,0);ctx.lineTo(-6,-6);ctx.lineTo(-3,0);ctx.lineTo(-6,6);ctx.closePath();ctx.fillStyle='#173c55';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1;ctx.stroke();ctx.restore();
}
function drawChart(){const c=$('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height,l=42,r=16,top=14,bottom=32,points=replay.snapshots,end=points.at(-1).time_s;ctx.clearRect(0,0,w,h);ctx.font='13px Segoe UI';ctx.fillStyle='#587180';ctx.lineWidth=1;
  for(const y of [0,.5,1]){const py=top+(1-y)*(h-top-bottom);ctx.strokeStyle='#e0e8ed';ctx.beginPath();ctx.moveTo(l,py);ctx.lineTo(w-r,py);ctx.stroke();ctx.fillText(`${y*100}%`,0,py+5);}
  for(const [key,color] of [['correct_coverage','#087f91'],['first_coverage','#95651d'],['unknown_fraction','#7b8793']]){ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.beginPath();points.forEach((p,i)=>{const x=l+p.time_s/end*(w-l-r),y=top+(1-p.metrics[key])*(h-top-bottom);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
  const x=l+points[frame].time_s/end*(w-l-r);ctx.strokeStyle='#16394d';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,h-bottom);ctx.stroke();ctx.fillStyle='#587180';ctx.fillText('0 s',l,h-8);ctx.fillText(`${end} s`,w-60,h-8);
}
function render(){if(!replay)return;const s=replay.snapshots[frame];$('clock').textContent=`${s.time_s} / ${replay.config.duration_s} s`;$('timeline').value=frame;$('quality').textContent=pct(s.metrics.correct_coverage);$('unknown').textContent=pct(s.metrics.unknown_fraction);$('error').textContent=pct(s.metrics.observed_error_rate);$('cost').textContent=`${s.distance_m.toFixed(0)} m / ${s.turn_s.toFixed(0)} s`;drawMap('truth','truth');drawMap('memory','memory');drawMap('age','age');drawChart();}
$('timeline').addEventListener('input',event=>{frame=Number(event.target.value);accumulated=0;render();});
$('play').addEventListener('click',()=>{if(!replay)return;playing=!playing;if(playing&&frame===replay.snapshots.length-1)frame=0;lastTick=null;accumulated=0;$('play').textContent=playing?'暂停':'播放';render();});
function tick(t){if(playing&&replay){if(lastTick!==null)accumulated+=(t-lastTick)/1000*Number($('playSpeed').value);while(frame<replay.snapshots.length-1&&accumulated>=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s){accumulated-=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s;frame++;render();}if(frame===replay.snapshots.length-1){playing=false;$('play').textContent='播放';}}lastTick=t;requestAnimationFrame(tick);}requestAnimationFrame(tick);
loadScenes().catch(e=>status(e.message,true));
