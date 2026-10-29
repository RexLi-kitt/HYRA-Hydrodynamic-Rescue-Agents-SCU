const $=id=>document.getElementById(id);
let primaryCatalog=null,legacyCatalog=null,routeCatalog=null,catalog=null,allScenes=[],scenes=[],replay=null,runId=null,frame=0,playing=false,lastTick=null,accumulated=0,busy=false,mapView='mission',lastRequest=null;
const taskNames={exploration:'自主探索',exploration_rescue:'探索—搜救'};
const actionNames={west:'向左',east:'向右',north:'向上',south:'向下',northwest:'左上',southwest:'左下',northeast:'右上',southeast:'右下',initial:'初始观测'};
const statuses={move:'移动中',boundary:'边界受阻',obstacle:'障碍受阻',collision:'冲突取消',rescuing:'施救停留',initial:'初始'};
const pct=v=>v==null?'—':`${(v*100).toFixed(1)}%`;
const selected=name=>document.querySelector(`input[name="${name}"]:checked`)?.value;
const group=()=>catalog?.groups.find(g=>g.id===selected('group'));
const displayGroup=id=>isRoutes()?id:({B3:'A',D:'B',P:'C'}[id]||id);
const scene=()=>scenes.find(s=>s.id===$('sceneSelect').value);
const isRoutes=()=>scene()?.kind==='synthetic_signal_routes_swe';
const isScale=()=>['synthetic_scale100_swe','synthetic_signal_routes_swe'].includes(scene()?.kind);
function activateScene(reset=false){const next=isRoutes()?routeCatalog:isScale()?primaryCatalog:legacyCatalog;const changed=catalog!==next;catalog=next;if(reset||changed){setConfig(catalog.defaults[selected('task')]);if(!isScale())clampDuration();}fillGroups();controls();}
function requestedConfig(){const task=selected('task');const basic={step_s:5,duration_s:Number($('duration').value),sense_width:5,service_s:Number($('service').value),n_targets:task==='exploration'?0:Number($('targets').value),obstacle_fraction:Number($('obstacles').value)/100,seed:Number($('seed').value)};return isScale()?{...basic,n_agents:2,task:task==='exploration'?'E1':'E2',representation:'gmrf',service_depth_limit_m:.08,service_speed_limit_m_s:.3}:basic;}

function node(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;}
async function api(path,body){const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const value=await response.json();if(!response.ok)throw Error(value.error||`请求失败 ${response.status}`);return value;}
function status(message,error=false){$('runStatus').textContent=message;$('runStatus').classList.toggle('error',error);}
function payload(){return {scene:$('sceneSelect').value,task:selected('task'),group:selected('group'),mode:selected('mode'),config:requestedConfig(),...(isScale()?{training_seed:Number($('trainingSeed').value)}:{})};}
function signature(value){return JSON.stringify(value,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))):v);}
function fillGroups(){const current=catalog.groups.some(g=>g.id===selected('group'))?selected('group'):catalog.default_group;const labels=[];
 for(const g of catalog.groups){const label=node('label',undefined,'group-choice'),radio=node('input');radio.type='radio';radio.name='group';radio.value=g.id;radio.id=`group-${g.id}`;radio.checked=g.id===current;label.append(radio,node('span',displayGroup(g.id),'choice-code'));
  const title=node('span',g.label,'choice-name');const subtitle=g.description||({B3:'联合基础方法',D:'共同数据监督',P:'水动力物理约束'}[g.id]||'');title.append(node('small',subtitle));label.append(title,node('span','','choice-status'));labels.push(label);
 }$('groupList').replaceChildren(...labels);
}
function setConfig(c){$('duration').value=c.duration_s;$('seed').value=c.seed;$('targets').value=c.n_targets||5;$('service').value=c.service_s;$('obstacles').value=c.obstacle_fraction*100;}
function applyCheckpoint(){if(isScale()){const seed=Number($('seed').value);setConfig(catalog.defaults[selected('task')]);$('seed').value=seed;}else{const c=group()?.checkpoint?.config;if(c)setConfig(c);}}
function taskChanged(){if(!catalog)return;setConfig(catalog.defaults[selected('task')]);clampDuration();if(selected('mode')==='model')applyCheckpoint();controls();}
function clampDuration(){const s=scene();if(s&&!isScale()&&selected('mode')==='rule'&&Number($('duration').value)>s.duration_s)$('duration').value=Math.floor(s.duration_s/5)*5;}
function selectionChanged(){if(selected('mode')==='model')applyCheckpoint();else clampDuration();controls();}
function controls(){
 if(!catalog)return;const task=selected('task'),mode=selected('mode'),g=group(),s=scene(),rescue=task==='exploration_rescue',cap=g?.tasks[task];
 $('taskHelp').textContent=rescue?'搜索人员信号，接近目标并停留施救。':'不设置人员目标，检查探索覆盖与水情记录。';
 if(!replay){$('missionTitle').textContent=taskNames[task];$('missionDescription').textContent=rescue?'核查人员发现、接近与施救的完整动作过程。':'观察未知区域如何被探索，以及水情记录如何更新。';$('signalHint').textContent=rescue?'仅展示实际采集的信号':'自主探索无人员信号源';$('targetLegend').hidden=!rescue;}
 $('targetsLabel').hidden=!rescue;$('serviceLabel').hidden=!rescue;
 const locked=mode==='model'&&!!cap?.ready;
 for(const id of ['duration','seed','targets','service','obstacles'])$(id).disabled=busy||(isScale()?id!=='seed':locked)||(!rescue&&['targets','service'].includes(id));
 $('trainingSeedLabel').hidden=!isScale();$('trainingSeed').disabled=busy||mode==='rule';
 $('trainingProgress').hidden=!isScale();
 if(isScale()){const p=catalog.training_progress;$('trainingProgress').textContent=catalog.released?(isRoutes()?'三路径模型已完成24项训练、1944项评价与独立验收。':'正式模型已完成训练、留出评价与回放复核。'):`正式训练累计完成 ${p.completed}/${p.expected} 项；模型尚未发布，可先运行规则核查。`;}
 $('scopeText').textContent=isRoutes()?'三路径仅提供100米探索—搜救模型。本轮R1/R3拟合为零耦合；R2为标量均值校正，未证实方向性或可靠施救增益。这里运行的是已封存模型，交互回放不计入正式评价。':isScale()?'100×100米，双机、900秒、5秒一步；直向1米/秒，对角约1.41米/秒。场景种子生成本次任务；模型训练种子选择已有权重。人员、水流与障碍均为合成场景；施救阈值仅为实验条件。交互运行不计入正式评价。':'当前使用所选洪水过程的原始网格。已有32×32搜救模型为12回合工程检查点；大地图可运行规则核查。100米正式模型不用于未经验证的其他尺度。';
 for(const input of document.querySelectorAll('input[name="task"],input[name="group"],input[name="mode"]'))input.disabled=busy;
 $('studyProfile').disabled=busy;document.querySelector('input[name=task][value=exploration]').disabled=busy||isRoutes();$('groupCount').textContent=isRoutes()?'三条路径 + 基线与对照':'A / B / C';$('groupHelp').textContent=isRoutes()?'选择R1、R2或R3运行对应冻结模型。下方保留共同基线与水情置乱对照。':'组别可随时选择。能否运行模型取决于当前任务的检查点状态。';$('sceneSelect').disabled=busy;$('refreshScenes').disabled=busy;$('exportConfig').disabled=busy||!s;
 $('applyCheckpoint').hidden=!locked||isScale();$('applyCheckpoint').disabled=busy;
 $('parameterSummary').textContent=`${$('duration').value}秒 / 5秒一步`;
 if(s)$('sceneMeta').textContent=`${s.grid.shape.join('×')}格 · ${s.grid.dx_m}米格距 · 实际过程${s.duration_s}秒${s.duration_s<250?'。短过程仅适合接口核查。':'。保持原网格。'}`;
 for(const label of $('groupList').querySelectorAll('.group-choice')){const data=catalog.groups.find(x=>x.id===label.querySelector('input').value),badge=label.querySelector('.choice-status');const ready=data.tasks[task].ready&&(isScale()||(s?.grid.shape.join(',')==='32,32'&&s.grid.dx_m===5&&s.duration_s>=250));badge.textContent=ready?(isScale()?'已发布':'工程模型'):(isScale()?'训练/评价中':'无匹配模型');badge.classList.toggle('pilot',ready);}
 $('selectedGroupId').textContent=g?displayGroup(g.id):'';$('selectedGroupLabel').textContent=g?.label||'';
 const tags=g?.tags|| (g?[g.representation==='dual_gmrf'?'双层GMRF':'同源原始观测',g.surprise?'惊奇增强':'基础QMIX',g.auxiliary==='none'?'无辅助':g.auxiliary==='data'?'数据辅助':'数据 + 物理约束']:[]);
 $('componentTags').replaceChildren(...tags.map(x=>node('span',x)));
 let runnable=!!s&&!!g,reason=cap?.reason||'';if(!isScale()&&!cap?.ready)reason='当前原始网格没有匹配的组别模型；100米模型仅用于登记的20×20场景。';
 if(mode==='model'){
  runnable=runnable&&!!cap?.ready;
  if(!isScale()&&runnable&&(s.grid.shape.join(',')!=='32,32'||s.grid.dx_m!==5||s.duration_s<250)){runnable=false;reason='该工程模型需要32×32、5米格距、至少250秒过程；当前过程可切换规则核查。';}
  $('engineHelp').textContent=reason;
  $('groupAvailability').textContent=reason;
 }else{
  $('engineHelp').textContent=isScale()?'实际运行八向水情规则，使用已采集的水情记录。所选组别不计入模型结果。':'实际运行独立八向规则；所选组别仅用于方案配置，不计入模型结果。';
  $('groupAvailability').textContent=`${reason} 当前采用规则核查，不执行${g?.id}组模型。`;
 }
 $('runButton').disabled=busy||!runnable;$('runButton').textContent=busy?'正在生成回放…':mode==='model'?'运行所选组模型':'生成八向规则回放';
 if(!replay)$('selectionBadge').textContent=mode==='model'?`${g?.id} 模型运行`:'规则核查';
 dirtyNotice();
}
function dirtyNotice(){if(!replay)return;const changed=lastRequest&&signature(payload())!==lastRequest;$('replayNotice').hidden=!changed;$('replayNotice').textContent=`配置已更改。下方仍显示${replay.task_label}的上一次${replay.execution.mode==='rule'?'规则':'模型'}回放，重新生成后更新。`;}
function populateScenes(keep){scenes=allScenes.filter(s=>(s.kind==='synthetic_signal_routes_swe')===($('studyProfile').value==='signal_routes'));$('sceneSelect').replaceChildren(...scenes.map(s=>new Option(s.name,s.id)));if(scenes.some(s=>s.id===keep))$('sceneSelect').value=keep;else{const dev=scenes.find(s=>s.split==='development'&&['synthetic_scale100_swe','synthetic_signal_routes_swe'].includes(s.kind));if(dev)$('sceneSelect').value=dev.id;}}
async function loadScenes(){const keep=$('sceneSelect').value;allScenes=await api('/api/workbench/scenes');populateScenes(keep);activateScene();}
$('studyProfile').addEventListener('change',()=>{if($('studyProfile').value==='signal_routes'){document.querySelector('input[name=task][value=exploration_rescue]').checked=true;document.querySelector('input[name=mode][value=model]').checked=true;}populateScenes();activateScene(true);});
async function history(){const rows=await api('/api/workbench/history');if(!rows.length)return;
 $('historyList').replaceChildren(...rows.map(r=>{const button=node('button',`${r.task_label} / ${r.mode==='rule'?'规则核查':r.group+'模型'}`);button.type='button';const value=r.task==='exploration'?`探索 ${pct(r.summary.explored_fraction)}`:`施救 ${r.summary.rescued}/${r.summary.total_targets}`;button.append(node('small',`${value} · ${r.run_id.slice(0,6)}`));button.addEventListener('click',async()=>{if(busy)return;try{await loadReplay(r.replay_url,r.run_id);status('已读取历史八向回放。');}catch(e){status(e.message,true);}});return button;}));
}
async function loadReplay(url,id){const value=await api(url);if(value.schema!=='hyra-eight-workbench-v1')throw Error('该回放不是当前双任务格式');replay=value;runId=id;frame=0;playing=false;lastTick=null;accumulated=0;
 lastRequest=signature({scene:value.scene.id,task:value.task,group:value.requested_group,mode:value.execution.mode,config:value.requested_config||value.rescue_config,...(['scale100-900s-v1','signal-routes-900s-v1'].includes(value.contract)?{training_seed:value.execution.training_seed||Number($('trainingSeed').value)}:{})});
 $('timeline').max=value.snapshots.length-1;$('timeline').disabled=value.snapshots.length<2;$('playButton').disabled=value.snapshots.length<2;$('playButton').textContent='播放';$('previousFrame').disabled=value.snapshots.length<2;$('nextFrame').disabled=value.snapshots.length<2;
 $('emptyState').hidden=true;$('mainMap').hidden=false;$('downloads').hidden=false;
 for(const [id,name] of [['downloadReplay','replay'],['downloadActions','actions'],['downloadObservations','observations'],['downloadAudit','audit']])$(id).href=`/runs/${runId}/${name}.json`;
 $('missionTitle').textContent=value.task_label;$('missionDescription').textContent=value.execution.mode==='rule'?'当前回放使用八向规则，记录实际探索、观测和任务交互。':value.execution.model_formal?'当前回放使用经过统一预算训练与独立评价的所选组模型。':'当前回放使用八动作工程检查点，尚未验证策略收敛。';
 $('selectionBadge').textContent=value.execution.mode==='rule'?'实际运行：规则核查':`实际运行：${value.group}${value.execution.model_formal?'模型':'工程模型'}`;
 $('resultScope').textContent=value.execution.mode==='rule'?'规则核查，未计入模型结果':value.execution.model_formal?`训练种子 ${value.execution.training_seed} / 场景种子 ${value.execution.scenario_seed} / 选中第${value.execution.selected_episode}回合；本次为交互回放`:'12回合工程检查点，未计入正式实验';
 $('provenance').textContent=`${value.scene.name} / ${value.task_label} / ${value.method} / 过程 ${value.source_sha256.slice(0,12)}${value.execution.actor_sha256?' / 模型 '+value.execution.actor_sha256.slice(0,12):''} / 运行 ${runId}`;
 $('signalMapTitle').textContent=value.execution.signal_route?(value.execution.signal_route==='Braw'?'实际采集信号':'人员信号后验 · '+value.execution.signal_route):'人员信号 GMRF';$('signalHint').textContent=value.execution.signal_route&&value.execution.signal_route!=='Braw'?'策略使用的后验均值；估计不刷新实采记录':value.task==='exploration'?'自主探索无人员信号源':'实际采集记录对应的信号诊断';
 render();dirtyNotice();
}
$('setupForm').addEventListener('submit',async event=>{event.preventDefault();if(busy||!catalog)return;const data=payload();busy=true;playing=false;$('playButton').textContent='播放';controls();status('正在按所选任务执行八方向动作，采集实际观测并保存回放…');
 try{const result=await api('/api/workbench/run',data);await loadReplay(result.replay_url,result.run_id);const s=replay.summary;status(replay.task==='exploration'?`自主探索完成：覆盖${pct(s.explored_fraction)}，${replay.actions.length}次联合决策。`:`探索—搜救完成：施救${s.rescued}/${s.total_targets}人，覆盖${pct(s.explored_fraction)}。`);await history();}catch(e){status(e.message,true);}finally{busy=false;controls();}
});
document.querySelectorAll('input[name="task"]').forEach(i=>i.addEventListener('change',taskChanged));document.querySelectorAll('input[name="mode"]').forEach(i=>i.addEventListener('change',selectionChanged));$('groupList').addEventListener('change',selectionChanged);
$('sceneSelect').addEventListener('change',()=>{activateScene();clampDuration();controls();});$('refreshScenes').addEventListener('click',async()=>{try{await loadScenes();status('已刷新洪水工作台的完整过程。');}catch(e){status(e.message,true);}});
for(const id of ['duration','seed','targets','service','obstacles','trainingSeed'])$(id).addEventListener('input',controls);
$('applyCheckpoint').addEventListener('click',()=>{applyCheckpoint();controls();});
$('exportConfig').addEventListener('click',()=>{const value={schema:'hyra-workbench-config-v1',...payload(),execution_note:selected('mode')==='rule'?'Independent rule check; selected group is a requested experiment plan only':'Requires a task/group-matched eight-action checkpoint'};const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download=`HyRA-${value.task}-${value.group}-config.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status('已导出当前任务与组别配置。');});
function drawField(id,kind){const c=$(id),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);if(!replay){ctx.fillStyle='#e5edf2';ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle='#849eae';ctx.font='18px Microsoft YaHei';ctx.textAlign='center';ctx.fillText('等待实际观测',c.width/2,c.height/2);return;}
 const s=replay.snapshots[frame],[rows,cols]=replay.grid.shape,cell=Math.min(c.width/cols,c.height/rows),ox=(c.width-cell*cols)/2,oy=(c.height-cell*rows)/2,signal=s.signal_posterior||s.gmrf_signal,max=signal.reduce((a,b)=>Math.max(a,b),.001);
 for(let i=0;i<rows*cols;i++){if(kind==='environment'){const v=Math.max(-1,Math.min(1,s.gmrf_environment[i]));ctx.fillStyle=v>=0?`hsl(165 32% ${94-v*55}%)`:`hsl(267 30% ${94+v*55}%)`;}else{const v=Math.max(0,Math.min(1,signal[i]/max));ctx.fillStyle=(s.explored[i]||(s.signal_posterior&&replay.execution.signal_route!=='Braw'))?`hsl(${48-v*35} 78% ${96-v*53}%)`:'#a6b9c5';}ctx.fillRect(ox+i%cols*cell,oy+Math.floor(i/cols)*cell,cell+.2,cell+.2);}
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
 $('signalHint').textContent=replay.execution.signal_route&&replay.execution.signal_route!=='Braw'?'策略使用的后验均值；估计不刷新实采记录':rescue?'实际采集记录对应的信号诊断':'无人员目标 / 无信号源';
 $('agentStates').replaceChildren(...s.poses.map((p,i)=>{const row=node('div',undefined,'agent-row');row.append(node('b',`无人机${i+1}`),node('span',`${actionNames[s.actions[i]]||s.actions[i]} / ${statuses[s.motion_status[i]]||s.motion_status[i]}`),node('small',`x ${p.x_m.toFixed(1)} m，z ${p.z_m.toFixed(1)} m${s.busy[i]?` · 施救剩余 ${s.busy[i]} s`:''}`));return row;}));
 $('previousFrame').disabled=frame===0;$('nextFrame').disabled=frame===replay.snapshots.length-1;drawMap();drawField('environmentMap','environment');drawField('signalMap','signal');drawTrend();
}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{mapView=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));drawMap();}));$('showTruth').addEventListener('change',drawMap);
function pause(){playing=false;$('playButton').textContent='播放';lastTick=null;accumulated=0;}
$('timeline').addEventListener('input',e=>{pause();frame=Number(e.target.value);render();});$('previousFrame').addEventListener('click',()=>{pause();frame=Math.max(0,frame-1);render();});$('nextFrame').addEventListener('click',()=>{pause();frame=Math.min(replay.snapshots.length-1,frame+1);render();});
$('playButton').addEventListener('click',()=>{if(!replay)return;playing=!playing;if(playing&&frame===replay.snapshots.length-1)frame=0;lastTick=null;accumulated=0;$('playButton').textContent=playing?'暂停':'播放';render();});
function tick(t){if(playing&&replay){if(lastTick!==null)accumulated+=(t-lastTick)/1000*Number($('playSpeed').value);while(frame<replay.snapshots.length-1&&accumulated>=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s){accumulated-=replay.snapshots[frame+1].time_s-replay.snapshots[frame].time_s;frame++;render();}if(frame===replay.snapshots.length-1)pause();}lastTick=t;requestAnimationFrame(tick);}requestAnimationFrame(tick);
(async()=>{[primaryCatalog,legacyCatalog,routeCatalog]=await Promise.all([api('/api/workbench/scale100/catalog'),api('/api/workbench/catalog'),api('/api/workbench/signal-routes/catalog')]);catalog=primaryCatalog;fillGroups();setConfig(catalog.defaults[selected('task')]);await loadScenes();await history();drawField('environmentMap','environment');drawField('signalMap','signal');drawTrend();status('八方向工作台已就绪。默认100米、900秒；可在“研究方案”中切换三路径信号建模。');})().catch(e=>status(e.message,true));
setInterval(async()=>{if(busy)return;try{primaryCatalog=await api('/api/workbench/scale100/catalog');if(isScale()){if(isRoutes())routeCatalog=await api('/api/workbench/signal-routes/catalog');catalog=isRoutes()?routeCatalog:primaryCatalog;controls();}}catch(e){$('trainingProgress').textContent='训练记录暂时无法刷新；请稍后点击洪水过程刷新。';}},30000);

