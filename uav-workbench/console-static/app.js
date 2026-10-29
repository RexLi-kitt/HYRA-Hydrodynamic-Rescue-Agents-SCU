const $=id=>document.getElementById(id);
let scenes=[],catalog=null,replay=null,frame=0,playing=false,lastTick=null,accumulated=0,busy=false;
const modelIds=new Set(['QMIX','Sur-QMIX','Sur-QMIX+Data','Sur-QMIX+Physics']);
const fields=[['speed','speed_m_s'],['angular','angular_speed_deg_s'],['range','range_m'],['fov','fov_deg'],['actionSeconds','action_s'],['sample','sample_s'],['duration','duration_s']];
const pct=x=>x==null?'不适用':`${(100*x).toFixed(1)}%`;
const isModel=()=>modelIds.has($('method').value),isTeam=()=>isModel()||$('method').value.startsWith('team:');
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
async function api(url,options={}){const response=await fetch(url,options);const value=await response.json();if(!response.ok)throw Error(value.error||`请求失败 ${response.status}`);return value;}
function controls(){
 const trained=isModel(),team=isTeam(),scene=scenes.find(s=>s.id===$('scene').value);
 for(const [id] of fields)$(id).disabled=busy||trained;
 for(const id of ['heading','startX','startZ'])$(id).disabled=busy||team;
 for(const id of ['method','modelSeed','scene','importFile'])$(id).disabled=busy;
 $('actionsLabel').hidden=$('method').value!=='fixed';$('seedLabel').hidden=!trained;$('modelInfo').hidden=!trained;
 let usable=true;
 if(trained){
  const model=catalog?.models.find(m=>m.id===$('method').value),seed=model?.seeds.find(s=>s.seed===Number($('modelSeed').value));
  usable=!!model?.available&&!!seed?.available&&!!scene?.model_compatible;
  $('modelInfo').textContent=seed?.available?`2 架协同，种子 ${seed.seed}，选定第 ${seed.selected_episode} 回合模型。${catalog.model_scope}`:(seed?.error||'模型尚未就绪。');
  $('policyHint').textContent=scene?.model_compatible?'使用已训练参数：600秒任务、5秒实际采样、10秒动作。两机从左上和左下出发。':'当前过程不匹配模型尺度。请选择32×32格、5米格距、5秒实际帧且不少于600秒的洪水过程。';
 }else $('policyHint').textContent=team?'两机从左上和左下出发，共享真实观测；可与训练模型按同一设置比较。':'前进、左转、右转、悬停由所选规则决定；固定序列可逐项指定。';
 $('run').disabled=busy||!scenes.length||!usable;
 return usable;
}
function methodChanged(){
 if(isModel()&&catalog){
  for(const [id,key] of fields)$(id).value=catalog.trained_config[key];
  const info=catalog.models.find(m=>m.id===$('method').value);
  for(const option of $('modelSeed').options){const row=info?.seeds.find(s=>s.seed===Number(option.value));option.disabled=!row?.available;}
  if($('modelSeed').selectedOptions[0]?.disabled){const first=[...$('modelSeed').options].find(o=>!o.disabled);if(first)$('modelSeed').value=first.value;}
 }
 if(isTeam())for(const id of ['heading','startX','startZ'])$(id).value=0;
 controls();
}
function sceneChanged(){
 const s=scenes.find(s=>s.id===$('scene').value);if(!s)return;
 $('sceneInfo').textContent=`${s.grid.shape.join(' × ')}格 / ${s.grid.dx_m}米格距 / ${s.duration_s}秒。${s.kind==='hand_authored_not_hydraulic_evidence'?'人工验收时序，不作为洪水证据。':'合成或导入过程，需单独核查物理适用性。'}`;
 const c=s.recommended_config||{speed_m_s:2.5,angular_speed_deg_s:9,range_m:25,fov_deg:90,action_s:s.sample_s*2,sample_s:s.sample_s,duration_s:s.duration_s};
 for(const [id,key] of fields)$(id).value=c[key];
 const pose=s.demo_pose||{x_m:0,z_m:0,heading_deg:0};$('startX').value=pose.x_m;$('startZ').value=pose.z_m;$('heading').value=pose.heading_deg;
 if(s.demo_actions&&!isTeam()){$('method').value='fixed';$('fixedActions').value=s.demo_actions.join(', ');}
 methodChanged();
}
async function loadScenes(selected){scenes=await api('/api/scenes');$('scene').replaceChildren(...scenes.map(s=>new Option(s.name,s.id)));if(selected)$('scene').value=selected;sceneChanged();controls();status(scenes.length?'过程已就绪，选择规则或已训练模型生成回放。':'尚无过程，请导入完整洪水ZIP。');}
$('scene').addEventListener('change',sceneChanged);$('method').addEventListener('change',methodChanged);$('modelSeed').addEventListener('change',controls);
$('importFile').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{status('正在核查导入过程…');const result=await api('/api/import',{method:'POST',headers:{'Content-Type':'application/zip'},body:file});await loadScenes(result.id);}catch(e){status(e.message,true);}finally{event.target.value='';}});
$('setup').addEventListener('submit',async event=>{
 event.preventDefault();if(busy||!controls())return;playing=false;$('play').textContent='播放';busy=true;controls();
 const config=Object.fromEntries(fields.map(([id,key])=>[key,Number($(id).value)]));
 const data={scene:$('scene').value,method:$('method').value,seed:Number($('modelSeed').value),config,pose:{x_m:Number($('startX').value),z_m:Number($('startZ').value),heading_deg:Number($('heading').value)}};
 const team=isTeam();
 if(data.method==='fixed'){data.actions=$('fixedActions').value.trim().split(/[\s,，]+/).filter(Boolean);data.method='R0';}
 status(isModel()?'正在加载选定模型，逐时刻执行两机动作与实际观测…':'正在执行动作、采集观测并保存日志…');
 try{
  const result=await api(team?'/api/team-run':'/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  replay=await api(result.replay_url);frame=0;accumulated=0;
  $('timeline').max=replay.snapshots.length-1;$('timeline').value=0;$('timeline').disabled=false;$('play').disabled=false;
  $('downloads').hidden=false;for(const [id,name] of [['downloadReplay','replay'],['downloadObs','observations'],['downloadActions','actions'],['downloadAudit','audit']])$(id).href=`/runs/${result.run_id}/${name}.json`;
  $('summary').textContent=`全程按类正确覆盖 ${pct(replay.summary.time_average_correct_coverage)}；变化未发现 ${pct(replay.summary.undiscovered_change_fraction)}。`;
  const model=replay.execution,agents=replay.n_agents||1;
  $('provenance').textContent=`${replay.scene.name} / ${replay.method}${model?.actor_sha256?` / 种子${model.seed} / 模型${model.actor_sha256.slice(0,12)}`:''} / 过程${replay.source_sha256.slice(0,12)} / 交互回放`;
  $('agentTwoLegend').hidden=agents<2;
  status(`完成${replay.actions.length}${agents>1?'次联合决策':'个动作'}、${replay.snapshots.length}个观测时刻、${replay.observations.length}条观测（${agents}架）。`);render();
 }catch(e){status(e.message,true);}finally{busy=false;controls();}
});
function poses(s){return s.poses||[{x_m:s.x_m,z_m:s.z_m,heading_deg:s.heading_deg}];}
function drawMap(id,kind){
 const canvas=$(id),ctx=canvas.getContext('2d'),s=replay.snapshots[frame],[rows,cols]=replay.grid.shape;
 const cell=Math.min(canvas.width/cols,canvas.height/rows),w=cell*cols,h=cell*rows,ox=(canvas.width-w)/2,oy=(canvas.height-h)/2;
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#e7eef2';ctx.fillRect(0,0,canvas.width,canvas.height);const truth=replay.truth[frame];
 for(let i=0;i<rows*cols;i++){
  const value=kind==='truth'?truth[i]:s.labels[i];
  if(kind==='age'&&value>=0){const age=Math.min(1,(s.time_s-s.observed_s[i])/Math.max(1,replay.config.duration_s));ctx.fillStyle=`hsl(200 45% ${92-age*62}%)`;}
  else ctx.fillStyle=value<0?'#9caebb':value?'#1684b5':'#e3efea';
  const x=ox+i%cols*cell,y=oy+Math.floor(i/cols)*cell;ctx.fillRect(x,y,cell+.2,cell+.2);
  if(kind==='memory'&&value>=0&&value!==truth[i]){ctx.strokeStyle='#c35838';ctx.lineWidth=1.8;ctx.strokeRect(x+1,y+1,cell-2,cell-2);}
 }
 const toX=x=>ox+(x/replay.grid.dx_m+.5)*cell,toY=z=>oy+(z/replay.grid.dx_m+.5)*cell,colors=['#b77b10','#6554a6'];
 ctx.save();ctx.beginPath();ctx.rect(ox,oy,w,h);ctx.clip();
 poses(s).forEach((p,agent)=>{
  ctx.strokeStyle=colors[agent%colors.length];ctx.lineWidth=2;ctx.beginPath();replay.snapshots.slice(0,frame+1).forEach((point,i)=>{const pose=poses(point)[agent],x=toX(pose.x_m),y=toY(pose.z_m);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  const x=toX(p.x_m),y=toY(p.z_m),angle=p.heading_deg*Math.PI/180,half=replay.config.fov_deg*Math.PI/360,r=replay.config.range_m/replay.grid.dx_m*cell;
  ctx.beginPath();ctx.moveTo(x,y);ctx.arc(x,y,r,angle-half,angle+half);ctx.closePath();ctx.fillStyle=agent?'#8f7fc444':'#f8cb4b55';ctx.strokeStyle=colors[agent%2];ctx.lineWidth=1.5;ctx.fill();ctx.stroke();
  ctx.save();ctx.translate(x,y);ctx.rotate(angle);ctx.beginPath();ctx.moveTo(10,0);ctx.lineTo(-6,-6);ctx.lineTo(-3,0);ctx.lineTo(-6,6);ctx.closePath();ctx.fillStyle=colors[agent%2];ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1;ctx.stroke();ctx.restore();
  if(poses(s).length>1){ctx.fillStyle=colors[agent%2];ctx.font='bold 12px sans-serif';ctx.fillText(String(agent+1),Math.max(ox+3,Math.min(ox+w-12,x+9)),Math.max(oy+13,Math.min(oy+h-3,y-8)));}
 });ctx.restore();
}
function drawChart(){const c=$('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height,l=42,r=16,top=14,bottom=32,points=replay.snapshots,end=points.at(-1).time_s;ctx.clearRect(0,0,w,h);ctx.font='13px Segoe UI';ctx.fillStyle='#587180';ctx.lineWidth=1;
  for(const y of [0,.5,1]){const py=top+(1-y)*(h-top-bottom);ctx.strokeStyle='#e0e8ed';ctx.beginPath();ctx.moveTo(l,py);ctx.lineTo(w-r,py);ctx.stroke();ctx.fillText(`${y*100}%`,0,py+5);}
  for(const [key,color] of [['correct_coverage','#087f91'],['first_coverage','#95651d'],['unknown_fraction','#7b8793']]){ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.beginPath();points.forEach((p,i)=>{const x=l+p.time_s/end*(w-l-r),y=top+(1-p.metrics[key])*(h-top-bottom);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
  const x=l+points[frame].time_s/end*(w-l-r);ctx.strokeStyle='#16394d';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,h-bottom);ctx.stroke();ctx.fillStyle='#587180';ctx.fillText('0 s',l,h-8);ctx.fillText(`${end} s`,w-60,h-8);
}
function render(){if(!replay)return;const s=replay.snapshots[frame],team=(replay.n_agents||1)>1;$('clock').textContent=`${s.time_s} / ${replay.config.duration_s} s`;$('timeline').value=frame;$('quality').textContent=pct(s.metrics.correct_coverage);$('unknown').textContent=pct(s.metrics.unknown_fraction);$('error').textContent=pct(s.metrics.observed_error_rate);$('costLabel').textContent=team?'全队距离 / 转向时间':'飞行距离 / 转向时间';$('cost').textContent=`${s.distance_m.toFixed(0)} m / ${s.turn_s.toFixed(0)} s`;$('teamCost').textContent=team?`2架合计；碰撞 ${s.collision_events} 次`:'';drawMap('truth','truth');drawMap('memory','memory');drawMap('age','age');drawChart();}
$('timeline').addEventListener('input',event=>{frame=Number(event.target.value);accumulated=0;render();});
$('play').addEventListener('click',()=>{if(!replay)return;playing=!playing;if(playing&&frame===replay.snapshots.length-1)frame=0;lastTick=null;accumulated=0;$('play').textContent=playing?'暂停':'播放';render();});
function tick(t){if(playing&&replay){if(lastTick!==null)accumulated+=(t-lastTick)/1000*Number($('playSpeed').value);while(frame<replay.snapshots.length-1&&accumulated>=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s){accumulated-=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s;frame++;render();}if(frame===replay.snapshots.length-1){playing=false;$('play').textContent='播放';}}lastTick=t;requestAnimationFrame(tick);}requestAnimationFrame(tick);
(async()=>{catalog=await api('/api/policies');for(const m of catalog.models){const option=[...$('method').options].find(o=>o.value===m.id);if(option)option.disabled=!m.available;}await loadScenes();})().catch(e=>{status(e.message,true);$('run').disabled=true;});
