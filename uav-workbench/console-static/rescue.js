const $=id=>document.getElementById(id);
let scenes=[],catalog=null,replay=null,frame=0,busy=false,playing=false,last=null,elapsed=0;
const names={west:'向左',east:'向右',north:'向上',south:'向下',northwest:'左上',southwest:'左下',northeast:'右上',southeast:'右下',initial:'初始'};
const reasons={move:'移动',boundary:'边界受阻',obstacle:'障碍受阻',collision:'冲突取消',rescuing:'施救停留',initial:'初始观测'};
async function api(url,body){const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const v=await r.json();if(!r.ok)throw Error(v.error||`请求失败 ${r.status}`);return v;}
function status(text,error=false){$('rescueStatus').textContent=text;$('rescueStatus').classList.toggle('error',error);}
function trained(){return $('rescueMethod').value!=='rescue:rule';}
function controls(){
 const model=catalog?.models.find(m=>m.id===$('rescueMethod').value),scene=scenes.find(s=>s.id===$('rescueScene').value);
 for(const id of ['rescueScene','rescueMethod'])$(id).disabled=busy;
 for(const id of ['rescueDuration','targetCount','serviceTime','obstaclePercent','scenarioSeed'])$(id).disabled=busy||trained();
 let available=!!scene;
 if(trained()){
  available=!!model?.available&&scene?.grid.shape[0]===32&&scene?.grid.shape[1]===32&&scene?.grid.dx_m===5&&scene.duration_s>=250;
  $('modelHint').textContent=available?`新八动作模型，种子${model.seed}，仅${model.episodes}回合工程贯通；尚未证明收敛或论文性能。`:'该八动作工程模型仅核验32×32、5米格距、250秒任务。大图请先用八向规则，短洪水过程需延长后再验证。';
 }else $('modelHint').textContent='使用已观测地图规划探索与施救，不加载神经网络模型；可用于核查八向运动、GMRF与目标交互。';
 $('rescueRun').disabled=busy||!available;
}
function methodChanged(){if(trained()){const c=catalog.default_config;$('rescueDuration').value=c.duration_s;$('targetCount').value=c.n_targets;$('serviceTime').value=c.service_s;$('obstaclePercent').value=c.obstacle_fraction*100;$('scenarioSeed').value=c.seed;}else{const scene=scenes.find(s=>s.id===$('rescueScene').value);if(scene&&Number($('rescueDuration').value)>scene.duration_s)$('rescueDuration').value=Math.floor(scene.duration_s/5)*5;}controls();}
function sceneChanged(){const s=scenes.find(s=>s.id===$('rescueScene').value);if(!s)return;$('sceneHint').textContent=`${s.grid.shape.join('×')}格，${s.grid.dx_m}米格距，实际洪水过程${s.duration_s}秒。${s.duration_s<250?'过程较短，仅适合短程接口检查；完整搜救需更长水力过程。':'从完整过程读取实际帧，不缩小地形。'}`;$('rescueDuration').value=Math.min(250,Math.floor(s.duration_s/5)*5);methodChanged();}
$('rescueMethod').addEventListener('change',methodChanged);$('rescueScene').addEventListener('change',sceneChanged);
$('rescueForm').addEventListener('submit',async event=>{
 event.preventDefault();if(busy)return;busy=true;playing=false;$('rescuePlay').textContent='播放';controls();status('正在求解双层GMRF、执行八向动作并记录实际传感…');
 try{
  const body={scene:$('rescueScene').value,method:$('rescueMethod').value,rescue_config:{step_s:5,duration_s:Number($('rescueDuration').value),sense_width:5,service_s:Number($('serviceTime').value),n_targets:Number($('targetCount').value),obstacle_fraction:Number($('obstaclePercent').value)/100,seed:Number($('scenarioSeed').value)}};
  const result=await api('/api/rescue/run',body);replay=await api(result.replay_url);frame=0;elapsed=0;
  $('rescueTimeline').max=replay.snapshots.length-1;$('rescueTimeline').value=0;$('rescueTimeline').disabled=false;$('rescuePlay').disabled=false;
  $('rescueDownloads').hidden=false;for(const [id,name] of [['rescueReplayDownload','replay'],['rescueActionsDownload','actions'],['rescueObservationsDownload','observations']])$(id).href=`/runs/${result.run_id}/${name}.json`;
  const m=replay.execution;$('rescueProvenance').textContent=`${replay.scene.name} / ${replay.method} / 过程${replay.source_sha256.slice(0,12)}${m.actor_sha256?` / 八动作模型${m.actor_sha256.slice(0,12)} / ${m.trained_episodes}回合工程检查点`:''} / 双层GMRF · 9通道 · 8动作`;
  status(`完成${replay.actions.length}次联合决策；施救${replay.summary.rescued}/${replay.summary.total_targets}人，探索${(100*replay.summary.explored_fraction).toFixed(1)}%。`);render();
 }catch(e){status(e.message,true);}finally{busy=false;controls();}
});
function draw(id,kind){
 const canvas=$(id),ctx=canvas.getContext('2d'),s=replay.snapshots[frame],[rows,cols]=replay.grid.shape;
 const cell=Math.min(canvas.width/cols,canvas.height/rows),ox=(canvas.width-cols*cell)/2,oy=(canvas.height-rows*cell)/2;ctx.clearRect(0,0,canvas.width,canvas.height);
 const showTruth=$('showTruth').checked,env=s.gmrf_environment,signal=s.gmrf_signal,signalMax=signal.reduce((a,b)=>Math.max(a,b),.001);
 for(let i=0;i<rows*cols;i++){
  if(kind==='environment'){const v=Math.max(-1,Math.min(1,env[i]));ctx.fillStyle=v>=0?`hsl(164 35% ${94-v*55}%)`:`hsl(266 35% ${94+v*55}%)`;}
  else if(kind==='signal'){const v=Math.max(0,Math.min(1,signal[i]/signalMax));ctx.fillStyle=s.explored[i]?`hsl(${48-v*35} 82% ${96-v*55}%)`:'#a6b6c0';}
  else{ctx.fillStyle=replay.truth[frame][i]?'#c5e2ee':'#edf2e9';if(!s.explored[i]&&!showTruth)ctx.fillStyle='#b2c2cd';if((showTruth?replay.scenario_truth.obstacles[i]:s.obstacles[i]))ctx.fillStyle='#394e5b';}
  ctx.fillRect(ox+i%cols*cell,oy+Math.floor(i/cols)*cell,cell+.25,cell+.25);
 }
 if(kind!=='mission')return;
 const x=c=>ox+(c+.5)*cell,y=r=>oy+(r+.5)*cell;
 const targets=showTruth?replay.scenario_truth.targets.map(([row,col])=>s.targets.find(p=>p.row===row&&p.col===col)||{row,col,status:'undiscovered'}):s.targets;
 targets.forEach(p=>{const xx=x(p.col),yy=y(p.row),size=Math.max(4,Math.min(9,cell*.3));ctx.strokeStyle=p.status==='rescued'?'#1a8a61':'#c43a38';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(xx-size,yy);ctx.lineTo(xx+size,yy);ctx.moveTo(xx,yy-size);ctx.lineTo(xx,yy+size);ctx.stroke();});
 s.poses.forEach((p,agent)=>{
  const color=agent?'#7254af':'#ad7610';ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.beginPath();replay.snapshots.slice(0,frame+1).forEach((point,j)=>{const a=point.poses[agent],xx=x(a.x_m/replay.grid.dx_m),yy=y(a.z_m/replay.grid.dx_m);j?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);});ctx.stroke();
  const xx=x(p.x_m/replay.grid.dx_m),yy=y(p.z_m/replay.grid.dx_m);ctx.fillStyle=color;ctx.beginPath();ctx.arc(xx,yy,Math.max(4,Math.min(9,cell*.28)),0,Math.PI*2);ctx.fill();ctx.fillStyle='#fff';ctx.font='bold 12px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(agent+1),xx,yy);
 });
}
function render(){if(!replay)return;const s=replay.snapshots[frame];$('rescueClock').textContent=`${s.time_s} / ${replay.config.duration_s} s`;$('rescueTimeline').value=frame;$('rescuedReadout').textContent=`${s.rescued} / ${replay.summary.total_targets}`;$('exploredReadout').textContent=`${(100*s.explored.reduce((a,b)=>a+b,0)/s.explored.length).toFixed(1)}%`;$('distanceReadout').textContent=`${s.distance_m.toFixed(1)} m`;$('collisionReadout').textContent=s.collision_events;$('rescueActions').textContent=s.actions.map((a,i)=>`无人机${i+1}：${names[a]||a} · ${reasons[s.motion_status[i]]||s.motion_status[i]}${s.busy[i]?`，施救剩余${s.busy[i]}秒`:''}`).join('\n');draw('rescueMap','mission');draw('environmentMap','environment');draw('signalMap','signal');}
$('rescueTimeline').addEventListener('input',e=>{frame=Number(e.target.value);elapsed=0;render();});$('showTruth').addEventListener('change',render);
$('rescuePlay').addEventListener('click',()=>{playing=!playing;if(playing&&frame===replay.snapshots.length-1)frame=0;elapsed=0;last=null;$('rescuePlay').textContent=playing?'暂停':'播放';render();});
function tick(t){if(playing&&replay){if(last!==null)elapsed+=(t-last)/1000*Number($('rescueSpeed').value);while(frame<replay.snapshots.length-1&&elapsed>=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s){elapsed-=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s;frame++;render();}if(frame===replay.snapshots.length-1){playing=false;$('rescuePlay').textContent='播放';}}last=t;requestAnimationFrame(tick);}requestAnimationFrame(tick);
(async()=>{[catalog,scenes]=await Promise.all([api('/api/rescue/policies'),api('/api/rescue/scenes')]);$('rescueScene').replaceChildren(...scenes.map(s=>new Option(s.name,s.id)));for(const model of catalog.models){const option=[...$('rescueMethod').options].find(o=>o.value===model.id);option.disabled=!model.available;}sceneChanged();status('双层GMRF、八方向动作和合成搜救目标已就绪。');})().catch(e=>status(e.message,true));
