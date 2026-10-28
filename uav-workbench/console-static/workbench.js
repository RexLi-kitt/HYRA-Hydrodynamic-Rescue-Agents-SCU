const $=id=>document.getElementById(id);
let catalog=null,scenes=[],replay=null,runId=null,frame=0,playing=false,lastTick=null,accumulated=0,busy=false,mapView='mission',lastRequest=null;
const taskNames={exploration:'自主探索',exploration_rescue:'探索—搜救'};
const actionNames={west:'向左',east:'向右',north:'向上',south:'向下',northwest:'左上',southwest:'左下',northeast:'右上',southeast:'右下',initial:'初始观测'};
const statuses={move:'移动中',boundary:'边界受阻',obstacle:'障碍受阻',collision:'冲突取消',rescuing:'施救停留',initial:'初始'};
const pct=v=>v==null?'—':`${(v*100).toFixed(1)}%`;
const selected=name=>document.querySelector(`input[name="${name}"]:checked`)?.value;
const group=()=>catalog?.groups.find(g=>g.id===selected('group'));
const scene=()=>scenes.find(s=>s.id===$('sceneSelect').value);
function node(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;}
async function api(path,body){const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const value=await response.json();if(!response.ok)throw Error(value.error||`请求失败 ${response.status}`);return value;}
function status(message,error=false){$('runStatus').textContent=message;$('runStatus').classList.toggle('error',error);}
function payload(){const task=selected('task');return {scene:$('sceneSelect').value,task,group:selected('group'),mode:selected('mode'),config:{step_s:5,duration_s:Number($('duration').value),sense_width:5,service_s:Number($('service').value),n_targets:task==='exploration'?0:Number($('targets').value),obstacle_fraction:Number($('obstacles').value)/100,seed:Number($('seed').value)}};}
function signature(value){return JSON.stringify(value,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))):v);}
function fillGroups(){const current=selected('group')||catalog.default_group;const labels=[];
 for(const g of catalog.groups){const label=node('label',undefined,'group-choice'),radio=node('input');radio.type='radio';radio.name='group';radio.value=g.id;radio.id=`group-${g.id}`;radio.checked=g.id===current;label.append(radio,node('span',g.id,'choice-code'));
  const title=node('span',g.label,'choice-name');const subtitle=g.id==='B0'?'原始观测基线':g.id==='B1'?'检验GMRF空间编码':g.id==='B2'?'检验惊奇增强':g.id==='B3'?'联合基础方法':g.id==='D'?'共同数据监督':'水动力物理约束';title.append(node('small',subtitle));label.append(title,node('span','','choice-status'));labels.push(label);
 }$('groupList').replaceChildren(...labels);
}
function setConfig(c){$('duration').value=c.duration_s;$('seed').value=c.seed;$('targets').value=c.n_targets||5;$('service').value=c.service_s;$('obstacles').value=c.obstacle_fraction*100;}
function applyCheckpoint(){const c=group()?.checkpoint?.config;if(c)setConfig(c);}
function taskChanged(){if(!catalog)return;setConfig(catalog.defaults[selected('task')]);clampDuration();if(selected('mode')==='model')applyCheckpoint();controls();}
function clampDuration(){const s=scene();if(s&&selected('mode')==='rule'&&Number($('duration').value)>s.duration_s)$('duration').value=Math.floor(s.duration_s/5)*5;}
function selectionChanged(){if(selected('mode')==='model')applyCheckpoint();else clampDuration();controls();}
function controls(){
 if(!catalog)return;const task=selected('task'),mode=selected('mode'),g=group(),s=scene(),rescue=task==='exploration_rescue',cap=g?.tasks[task];
 $('taskHelp').textContent=rescue?'搜索人员信号，接近目标并停留施救。':'不设置人员目标，检查探索覆盖与水情记录。';
 if(!replay){$('missionTitle').textContent=taskNames[task];$('missionDescription').textContent=rescue?'核查人员发现、接近与施救的完整动作过程。':'观察未知区域如何被探索，以及水情记录如何更新。';$('signalHint').textContent=rescue?'仅展示实际采集的信号':'自主探索无人员信号源';$('targetLegend').hidden=!rescue;}
 $('targetsLabel').hidden=!rescue;$('serviceLabel').hidden=!rescue;
 const locked=mode==='model'&&!!cap?.ready;
 for(const id of ['duration','seed','targets','service','obstacles'])$(id).disabled=busy||locked||(!rescue&&['targets','service'].includes(id));
 for(const input of document.querySelectorAll('input[name="task"],input[name="group"],input[name="mode"]'))input.disabled=busy;
 $('sceneSelect').disabled=busy;$('refreshScenes').disabled=busy;$('exportConfig').disabled=busy||!s;
 $('applyCheckpoint').hidden=!locked;$('applyCheckpoint').disabled=busy;
 $('parameterSummary').textContent=`${$('duration').value}秒 / 5秒一步`;
 if(s)$('sceneMeta').textContent=`${s.grid.shape.join('×')}格 · ${s.grid.dx_m}米格距 · 实际过程${s.duration_s}秒${s.duration_s<250?'。短过程仅适合接口核查。':'。保持原网格。'}`;
 for(const label of $('groupList').querySelectorAll('.group-choice')){const data=catalog.groups.find(x=>x.id===label.querySelector('input').value),badge=label.querySelector('.choice-status');const ready=data.tasks[task].ready;badge.textContent=ready?'工程模型':data.auxiliary==='none'?'待训练':'待接入';badge.classList.toggle('pilot',ready);}
 $('selectedGroupId').textContent=g?.id||'';$('selectedGroupLabel').textContent=g?.label||'';
 const tags=g?[g.representation==='dual_gmrf'?'双层GMRF':'同源原始观测',g.surprise?'惊奇增强':'基础QMIX',g.auxiliary==='none'?'无辅助':g.auxiliary==='data'?'数据辅助':'数据 + 物理约束']:[];
 $('componentTags').replaceChildren(...tags.map(x=>node('span',x)));
 let runnable=!!s&&!!g,reason=cap?.reason||'';
 if(mode==='model'){
  runnable=runnable&&!!cap?.ready;
  if(runnable&&(s.grid.shape.join(',')!=='32,32'||s.grid.dx_m!==5||s.duration_s<250)){runnable=false;reason='该工程模型需要32×32、5米格距、至少250秒过程；当前过程可切换规则核查。';}
  $('engineHelp').textContent=reason;
  $('groupAvailability').textContent=reason;
 }else{
  $('engineHelp').textContent='实际运行独立八向规则；所选组别仅用于方案配置，不计入模型结果。';
  $('groupAvailability').textContent=`${reason} 当前采用规则核查，不执行${g?.id}组模型。`;
 }
 $('runButton').disabled=busy||!runnable;$('runButton').textContent=busy?'正在生成回放…':mode==='model'?'运行所选组模型':'生成八向规则回放';
 if(!replay)$('selectionBadge').textContent=mode==='model'?`${g?.id} 模型运行`:'规则核查';
 dirtyNotice();
}
function dirtyNotice(){if(!replay)return;const changed=lastRequest&&signature(payload())!==lastRequest;$('replayNotice').hidden=!changed;$('replayNotice').textContent=`配置已更改。下方仍显示${replay.task_label}的上一次${replay.execution.mode==='rule'?'规则':'模型'}回放，重新生成后更新。`;}
async function loadScenes(){const keep=$('sceneSelect').value;scenes=await api('/api/workbench/scenes');$('sceneSelect').replaceChildren(...scenes.map(s=>new Option(s.name,s.id)));if(scenes.some(s=>s.id===keep))$('sceneSelect').value=keep;clampDuration();controls();}
async function history(){const rows=await api('/api/workbench/history');if(!rows.length)return;
 $('historyList').replaceChildren(...rows.map(r=>{const button=node('button',`${r.task_label} / ${r.mode==='rule'?'规则核查':r.group+'模型'}`);button.type='button';const value=r.task==='exploration'?`探索 ${pct(r.summary.explored_fraction)}`:`施救 ${r.summary.rescued}/${r.summary.total_targets}`;button.append(node('small',`${value} · ${r.run_id.slice(0,6)}`));button.addEventListener('click',async()=>{if(busy)return;try{await loadReplay(r.replay_url,r.run_id);status('已读取历史八向回放。');}catch(e){status(e.message,true);}});return button;}));
}
async function loadReplay(url,id){const value=await api(url);if(value.schema!=='hyra-eight-workbench-v1')throw Error('该回放不是当前双任务格式');replay=value;runId=id;frame=0;playing=false;lastTick=null;accumulated=0;
 lastRequest=signature({scene:value.scene.id,task:value.task,group:value.requested_group,mode:value.execution.mode,config:value.rescue_config});
 $('timeline').max=value.snapshots.length-1;$('timeline').disabled=value.snapshots.length<2;$('playButton').disabled=value.snapshots.length<2;$('playButton').textContent='播放';$('previousFrame').disabled=value.snapshots.length<2;$('nextFrame').disabled=value.snapshots.length<2;
 $('emptyState').hidden=true;$('mainMap').hidden=false;$('downloads').hidden=false;
 for(const [id,name] of [['downloadReplay','replay'],['downloadActions','actions'],['downloadObservations','observations'],['downloadAudit','audit']])$(id).href=`/runs/${runId}/${name}.json`;
 $('missionTitle').textContent=value.task_label;$('missionDescription').textContent=value.execution.mode==='rule'?'当前回放使用八向规则，记录实际探索、观测和任务交互。':'当前回放使用所选八动作工程检查点，尚未验证策略收敛。';
 $('selectionBadge').textContent=value.execution.mode==='rule'?'实际运行：规则核查':`实际运行：${value.group}工程模型`;
 $('resultScope').textContent=value.execution.mode==='rule'?'本次不是六组模型结果':'12回合工程检查点，未计入正式实验';
 $('provenance').textContent=`${value.scene.name} / ${value.task_label} / ${value.method} / 过程 ${value.source_sha256.slice(0,12)}${value.execution.actor_sha256?' / 模型 '+value.execution.actor_sha256.slice(0,12):''} / 运行 ${runId}`;
 render();dirtyNotice();
}
$('setupForm').addEventListener('submit',async event=>{event.preventDefault();if(busy||!catalog)return;const data=payload();busy=true;playing=false;$('playButton').textContent='播放';controls();status('正在按所选任务执行八方向动作，采集实际观测并保存回放…');
 try{const result=await api('/api/workbench/run',data);await loadReplay(result.replay_url,result.run_id);const s=replay.summary;status(replay.task==='exploration'?`自主探索完成：覆盖${pct(s.explored_fraction)}，${replay.actions.length}次联合决策。`:`探索—搜救完成：施救${s.rescued}/${s.total_targets}人，覆盖${pct(s.explored_fraction)}。`);await history();}catch(e){status(e.message,true);}finally{busy=false;controls();}
});
document.querySelectorAll('input[name="task"]').forEach(i=>i.addEventListener('change',taskChanged));document.querySelectorAll('input[name="mode"]').forEach(i=>i.addEventListener('change',selectionChanged));$('groupList').addEventListener('change',selectionChanged);
$('sceneSelect').addEventListener('change',()=>{clampDuration();controls();});$('refreshScenes').addEventListener('click',async()=>{try{await loadScenes();status('已刷新洪水工作台的完整过程。');}catch(e){status(e.message,true);}});
for(const id of ['duration','seed','targets','service','obstacles'])$(id).addEventListener('input',controls);
$('applyCheckpoint').addEventListener('click',()=>{applyCheckpoint();controls();});
$('exportConfig').addEventListener('click',()=>{const value={schema:'hyra-workbench-config-v1',...payload(),execution_note:selected('mode')==='rule'?'Independent rule check; selected group is a requested experiment plan only':'Requires a task/group-matched eight-action checkpoint'};const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download=`HyRA-${value.task}-${value.group}-config.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status('已导出当前任务与组别配置。');});
function drawField(id,kind){const c=$(id),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);if(!replay){ctx.fillStyle='#e5edf2';ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle='#849eae';ctx.font='18px Microsoft YaHei';ctx.textAlign='center';ctx.fillText('等待实际观测',c.width/2,c.height/2);return;}
 const s=replay.snapshots[frame],[rows,cols]=replay.grid.shape,cell=Math.min(c.width/cols,c.height/rows),ox=(c.width-cell*cols)/2,oy=(c.height-cell*rows)/2,max=s.gmrf_signal.reduce((a,b)=>Math.max(a,b),.001);
 for(let i=0;i<rows*cols;i++){if(kind==='environment'){const v=Math.max(-1,Math.min(1,s.gmrf_environment[i]));ctx.fillStyle=v>=0?`hsl(165 32% ${94-v*55}%)`:`hsl(267 30% ${94+v*55}%)`;}else{const v=Math.max(0,Math.min(1,s.gmrf_signal[i]/max));ctx.fillStyle=s.explored[i]?`hsl(${48-v*35} 78% ${96-v*53}%)`:'#a6b9c5';}ctx.fillRect(ox+i%cols*cell,oy+Math.floor(i/cols)*cell,cell+.2,cell+.2);}
}
function drawMap(){if(!replay)return;const c=$('mainMap'),ctx=c.getContext('2d'),s=replay.snapshots[frame],[rows,cols]=replay.grid.shape,cell=Math.min(c.width/cols,c.height/rows),ox=(c.width-cols*cell)/2,oy=(c.height-rows*cell)/2,truth=replay.truth[frame],showTruth=$('showTruth').checked;
 ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#e4edf2';ctx.fillRect(0,0,c.width,c.height);
 for(let i=0;i<rows*cols;i++){
  if(mapView==='mission'){ctx.fillStyle=s.explored[i]||showTruth?(truth[i]?'#c4e0ea':'#edf2e9'):'#a9bdc9';if(showTruth?replay.scenario_truth.obstacles[i]:s.obstacles[i])ctx.fillStyle='#3c5362';}
  else if(mapView==='memory')ctx.fillStyle=s.labels[i]<0?'#a9bdc9':s.labels[i]?'#2087aa':'#e1ece5';
  else{const age=Math.min(1,(s.time_s-s.observed_s[i])/Math.max(1,replay.config.duration_s));ctx.fillStyle=s.labels[i]<0?'#a9bdc9':`hsl(200 40% ${94-age*60}%)`;}
  const x=ox+i%cols*cell,y=oy+Math.floor(i/cols)*cell;ctx.fillRect(x,y,cell+.2,cell+.2);
  if(mapView==='memory'&&showTruth&&s.labels[i]>=0&&s.labels[i]!==truth[i]){ctx.strokeStyle='#b65c43';ctx.lineWidth=1.6;ctx.strokeRect(x+.5,y+.5,cell-1,cell-1);}
 }
 const x=col=>ox+(col+.5)*cell,y=row=>oy+(row+.5)*cell;
 if(mapView==='mission'){
  const targets=showTruth?replay.scenario_truth.targets.map(([row,col])=>s.targets.find(t=>t.row===row&&t.col===col)||{row,col,status:'undiscovered'}):s.targets;
  for(const p of targets){const size=Math.max(4,Math.min(8,cell*.3)),xx=x(p.col),yy=y(p.row);ctx.strokeStyle=p.status==='rescued'?'#208867':'#ae4d50';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(xx-size,yy);ctx.lineTo(xx+size,yy);ctx.moveTo(xx,yy-size);ctx.lineTo(xx,yy+size);ctx.stroke();}
 }
 s.poses.forEach((p,agent)=>{const color=agent?'#7460ab':'#b38025';ctx.strokeStyle=color;ctx.lineWidth=2.4;ctx.beginPath();replay.snapshots.slice(0,frame+1).forEach((point,j)=>{const a=point.poses[agent],xx=x(a.x_m/replay.grid.dx_m),yy=y(a.z_m/replay.grid.dx_m);j?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);});ctx.stroke();
  const xx=x(p.x_m/replay.grid.dx_m),yy=y(p.z_m/replay.grid.dx_m),size=Math.max(5,Math.min(10,cell*.34));ctx.save();ctx.translate(xx,yy);ctx.rotate(p.heading_deg*Math.PI/180);ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(size+3,0);ctx.lineTo(-size,-size*.7);ctx.lineTo(-size*.55,0);ctx.lineTo(-size,size*.7);ctx.closePath();ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();ctx.restore();
 });
 $('mapScale').textContent=`${cols}×${rows}格 / ${replay.grid.dx_m}米格距 / ${(cols*replay.grid.dx_m).toFixed(0)}米宽`;
}
function drawTrend(){const c=$('trend'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const left=35,right=10,top=12,bottom=24,w=c.width-left-right,h=c.height-top-bottom;ctx.font='12px Segoe UI';
 for(const v of [0,.5,1]){const y=top+(1-v)*h;ctx.strokeStyle='#dae5eb';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(c.width-right,y);ctx.stroke();ctx.fillStyle='#788e9c';ctx.fillText(`${v*100}%`,0,y+4);}
 if(!replay)return;const end=Math.max(1,replay.snapshots.at(-1).time_s);
 for(const [key,color] of [['first_coverage','#b38025'],['correct_coverage','#167d87']]){ctx.strokeStyle=color;ctx.lineWidth=2.3;ctx.beginPath();replay.snapshots.forEach((s,i)=>{const x=left+s.time_s/end*w,y=top+(1-s.metrics[key])*h;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
 const x=left+replay.snapshots[frame].time_s/end*w;ctx.strokeStyle='#8099a8';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,c.height-bottom);ctx.stroke();ctx.fillStyle='#788e9c';ctx.fillText('0 s',left,c.height-5);ctx.fillText(`${replay.snapshots.at(-1).time_s} s`,c.width-65,c.height-5);
}
function render(){if(!replay)return;const s=replay.snapshots[frame],rescue=replay.task==='exploration_rescue';$('timeline').value=frame;$('clock').textContent=`${s.time_s} / ${replay.config.duration_s} s`;
 const values=rescue?[['已施救',`${s.rescued} / ${replay.summary.total_targets}`],['已发现',`${s.targets.length} 人`],['已探索',pct(s.metrics.first_coverage)],['全队航程',`${s.distance_m.toFixed(0)} m`]]:[['已探索',pct(s.metrics.first_coverage)],['当前水情正确率',pct(s.metrics.correct_coverage)],['全队航程',`${s.distance_m.toFixed(0)} m`],['冲突计数',String(s.collision_events)]];
 values.forEach(([label,value],i)=>{$(`metricLabel${i+1}`).textContent=label;$(`metric${i+1}`).textContent=value;});$('targetLegend').hidden=!rescue;
 $('signalHint').textContent=rescue?'人员信号的实际传感记录':'无人员目标 / 无信号源';
 $('agentStates').replaceChildren(...s.poses.map((p,i)=>{const row=node('div',undefined,'agent-row');row.append(node('b',`无人机${i+1}`),node('span',`${actionNames[s.actions[i]]||s.actions[i]} / ${statuses[s.motion_status[i]]||s.motion_status[i]}`),node('small',`x ${p.x_m.toFixed(1)} m，z ${p.z_m.toFixed(1)} m${s.busy[i]?` · 施救剩余 ${s.busy[i]} s`:''}`));return row;}));
 $('previousFrame').disabled=frame===0;$('nextFrame').disabled=frame===replay.snapshots.length-1;drawMap();drawField('environmentMap','environment');drawField('signalMap','signal');drawTrend();
}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{mapView=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));drawMap();}));$('showTruth').addEventListener('change',drawMap);
function pause(){playing=false;$('playButton').textContent='播放';lastTick=null;accumulated=0;}
$('timeline').addEventListener('input',e=>{pause();frame=Number(e.target.value);render();});$('previousFrame').addEventListener('click',()=>{pause();frame=Math.max(0,frame-1);render();});$('nextFrame').addEventListener('click',()=>{pause();frame=Math.min(replay.snapshots.length-1,frame+1);render();});
$('playButton').addEventListener('click',()=>{if(!replay)return;playing=!playing;if(playing&&frame===replay.snapshots.length-1)frame=0;lastTick=null;accumulated=0;$('playButton').textContent=playing?'暂停':'播放';render();});
function tick(t){if(playing&&replay){if(lastTick!==null)accumulated+=(t-lastTick)/1000*Number($('playSpeed').value);while(frame<replay.snapshots.length-1&&accumulated>=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s){accumulated-=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s;frame++;render();}if(frame===replay.snapshots.length-1)pause();}lastTick=t;requestAnimationFrame(tick);}requestAnimationFrame(tick);
(async()=>{catalog=await api('/api/workbench/catalog');fillGroups();setConfig(catalog.defaults[selected('task')]);await loadScenes();await history();drawField('environmentMap','environment');drawField('signalMap','signal');drawTrend();status('八方向工作台已就绪。可选择任务、六组配置和运行方式。');})().catch(e=>status(e.message,true));
