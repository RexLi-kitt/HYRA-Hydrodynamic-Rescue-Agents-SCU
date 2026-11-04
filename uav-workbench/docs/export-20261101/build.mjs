import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const base=path.dirname(fileURLToPath(import.meta.url));
const d=JSON.parse(await fs.readFile(path.join(base,'data.json'),'utf8'));
const wb=Workbook.create(),sheets={};
const names=['当前21方案','严格测试结果','全部评估轮次','当前逐场验证','当前逐回合训练','历史逐场验证','历史逐回合训练','权重检查点','口径与来源'];
for(const name of names)sheets[name]=wb.worksheets.add(name);
const col=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
function clean(v){if(v===undefined||v===null)return null;if(typeof v==='object')return JSON.stringify(v);return v;}
function table(name,headers,rows,widths=[],start=1){
 const s=sheets[name],last=Math.max(start+rows.length,start),end=col(headers.length-1);
 s.showGridLines=false;
 s.getRange(`A${start}:${end}${last}`).values=[headers,...rows.map(r=>r.map(clean))];
 const used=s.getRange(`A${start}:${end}${last}`);used.format.font={name:'Microsoft YaHei',size:10,color:'#253342'};used.format.rowHeight=22;used.format.verticalAlignment='center';used.format.columnWidth=14;
 const h=s.getRange(`A${start}:${end}${start}`);h.format.fill='#244864';h.format.font={name:'Microsoft YaHei',size:10,color:'#FFFFFF',bold:true};h.format.rowHeight=34;h.format.wrapText=true;h.format.horizontalAlignment='center';
 for(let i=0;i<headers.length;i++)s.getRange(`${col(i)}${start}:${col(i)}${last}`).format.columnWidth=widths[i]||16;
 if(rows.length){const t=s.tables.add(`A${start}:${end}${last}`,true,`Records${names.indexOf(name)}`);t.showBandedRows=true;}
 if(last>30){s.freezePanes.freezeRows(start);s.freezePanes.freezeColumns(3);}
 return s;
}
function fmt(s,columns,start,end,format){if(end>=start)for(const c of columns)s.getRange(`${c}${start}:${c}${end}`).setNumberFormat(format);}
const erows=d.evaluations.map(v=>[v.id,v.experiment,v.candidate,v.episode,v.control,v.parents,v.n,v.completed,null,v.RCR,v.coverage,v.ACC,v.ST,v.max_conflicts,v.source]);
const ev=table('全部评估轮次',['评估ID','实验阶段','模型/组合','训练回合','控制/动作设置','验证父水场','回放次数','全任务完成次数','全任务完成比例','人员服务比例RCR','覆盖比例','潜在冲突/步ACC','平均步数ST','单场最多冲突','来源ID'],erows,[12,27,25,13,48,14,12,16,17,18,14,19,16,16,13]);
if(erows.length)ev.getRange(`I2:I${erows.length+1}`).formulas=erows.map((r,i)=>[`=H${i+2}/G${i+2}`]);
fmt(ev,['I','J','K'],2,erows.length+1,'0.00%');fmt(ev,['L'],2,erows.length+1,'0.00000');fmt(ev,['M'],2,erows.length+1,'0.0');
const eindex=new Map(d.evaluations.map((v,i)=>[v.id,i+2]));
const statusNames={waiting_for_pairs:'等待双模型队列',waiting_for_training:'等待训练完成并锁定权重',generating_test_fields:'生成独立测试场景',evaluating:'严格测试进行中',queued:'待前一模型完成',training:'训练中',stage_completed:'本阶段已完成',completed:'全部完成',preflight:'启动检查中',failed:'失败已停止',manual_stopped:'人工停止'};
const main=table('当前21方案',['类型','模型/组合','已训练回合','阶段目标','父权重回合','最佳回合','最佳完成比例','最佳人员比例','最佳覆盖比例','最佳冲突/步','最新评估回合','最新完成比例','2048决定','状态'],d.overview.map(r=>[r.experiment.replace('当前',''),r.candidate.toUpperCase().replaceAll('_',' + '),r.actual,r.target,r.parent_episodes,r.best_episode,null,null,null,null,r.latest_evaluation,null,r.extension,statusNames[r.status]??r.status]),[12,28,14,13,14,13,18,18,18,18,16,18,24,25],7);
main.getRange('E8:E28').format.horizontalAlignment='center';main.getRange('M8:M28').format.horizontalAlignment='center';
main.tabColor='#244864';main.getRange('A2').values=[['单模型与双模型｜全部训练数据']];main.getRange('A2').format.font={name:'Microsoft YaHei',size:16,bold:true,color:'#244864'};
main.getRange('A4').values=[['数据快照 '+d.timestamp+'；训练与5630条严格测试已完成，未记录项目留空。']];
main.getRange('A5').values=[['本页为重复使用的开发验证：单模型36回放、双模型72回放，只有6个父水场、4套布局；独立测试另页。']];
for(let i=0;i<d.overview.length;i++){
 const r=d.overview[i],n=i+8,b=eindex.get(r.best_id),l=eindex.get(r.latest_id);
 if(b)main.getRange(`G${n}:J${n}`).formulas=[[...['I','J','K','L'].map(c=>`='全部评估轮次'!${c}${b}`)]];
 if(l)main.getRange(`L${n}`).formulas=[[`='全部评估轮次'!I${l}`]];
}
fmt(main,['G','H','I','L'],8,28,'0.00%');fmt(main,['J'],8,28,'0.00000');
main.getRange('A31').values=[['最佳0：尚未证明本次学习优于父权重在同一控制规则下的表现。全任务完成要求覆盖全区域且20人完成服务。']];
main.getRange('A33').values=[['当前已记录训练回合（含兼容前缀，不加上父模型预训练）：']];main.getRange('F33').formulas=[['=SUM(C8:C28)']];
main.getRange('A35').values=[['历史结果在后续表中保留；旧三组合与15组合存在兼容前缀复用，不能相加算总工作量。']];
const detailsHead=['评估ID','实验阶段','模型/组合','训练回合','控制/动作设置','水场','分层','布局种子','分工编号','动作模式','动作种子','回放编号','全任务完成','获救人数','发现人数','人员服务RCR','覆盖比例','潜在冲突次数','潜在冲突/步','步数ST','首次服务步FRT','全员服务步CRT','覆盖规划占比','神经自由占比','超时','未服务目标','轨迹文件','来源ID'];
function detailRow(r){return [r.evaluation_id,r.experiment,r.candidate,r.episode,r.control,r.case,r.stratum,r.layout,r.assignment,r.mode,r.action_seed,r.row,r.task_complete,r.rescued,r.discovered,r.RCR,r.coverage,r.conflicts,r.ACC,r.ST,r.FRT,r.CRT,r.planner_fraction,r.neural_free_fraction,r.timed_out,r.unserved_ids,r.trace,r.source];}
for(const [name,rows]of [['当前逐场验证',d.details.filter(r=>r.experiment.startsWith('当前')||r.experiment.startsWith('严格'))],['历史逐场验证',d.details.filter(r=>!r.experiment.startsWith('当前')&&!r.experiment.startsWith('严格'))]]){
 const s=table(name,detailsHead,rows.map(detailRow),[13,27,25,13,48,24,12,14,14,14,14,14,14,12,12,17,16,18,19,13,18,18,18,18,12,28,70,12]);
 fmt(s,['P','Q','W','X'],2,rows.length+1,'0.00%');fmt(s,['S'],2,rows.length+1,'0.00000');
}
const trainHead=['实验阶段','模型/组合','训练回合','历史子阶段','水场','布局种子','分工','更新累计数','训练回报','全任务完成','获救人数','发现人数','人员服务RCR','覆盖比例','潜在冲突次数','潜在冲突/步','步数ST','首次服务步FRT','全员服务步CRT','覆盖规划占比','神经自由占比','阶段累计秒','损失诊断JSON','来源ID','原日志行'];
function trainRow(r){return [r.experiment,r.candidate,r.joint_episode,r.stage,r.case,r.layout,r.roles,r.optimizer_updates??r.updates,r.return_sum,r.task_complete,r.rescued,r.discovered,r.RCR,r.coverage,r.conflicts,r.ACC,r.ST,r.FRT,r.CRT,r.planner_fraction,r.neural_free_fraction,r.elapsed_stage_s??r.elapsed_s,r.loss??r.diagnostics,r.source,r.source_line];}
for(const [name,rows]of [['当前逐回合训练',d.training.filter(r=>r.experiment.startsWith('当前'))],['历史逐回合训练',d.training.filter(r=>!r.experiment.startsWith('当前'))]]){
 const s=table(name,trainHead,rows.map(trainRow),[27,25,13,18,23,16,16,16,16,14,12,12,18,16,18,19,14,18,18,18,18,18,80,13,14]);
 fmt(s,['M','N','T','U'],2,rows.length+1,'0.00%');fmt(s,['P'],2,rows.length+1,'0.00000');fmt(s,['I','V'],2,rows.length+1,'0.0');
}
const weights=table('权重检查点',['实验阶段','模型/组合','回合','文件字节数','权重绝对路径'],d.checkpoints.map(r=>[r.experiment,r.candidate,r.episode,r.bytes,r.path]),[25,25,14,20,115]);
weights.getRange(`E2:E${d.checkpoints.length+1}`).format.wrapText=true;weights.getRange(`A2:E${d.checkpoints.length+1}`).format.rowHeight=48;
const notes=[
 ['快照（UTC）',d.timestamp],['当前队列',Object.entries(d.queues).map(([k,v])=>`${k}：${statusNames[v.status]??v.status}，${v.current??'尚未选中首个模型'}，阶段${v.stage??'等待'}`).join('；')],['1024名单','HAPPO、HATRPO、HASAC、MAAC、MAT、COMA各一个学习器；双模型名单按原15组筛选保留。'],
 ['2048条件','768→896和896→1024均改善，1024比256实质改善，最佳在256之后；冲突与不退化门槛保持原协议。未满足则保留最佳，不强行延长。'],
 ['样本量','16训练/6验证合成父水场，1个训练种子。单模型36=6×2布局×(2随机动作种子+1固定)，双模型因两种分工为72。重复回放不是独立洪水。'],
 ['任务条件','1km×1km、4架无人机、20目标、100×100格；每步5秒，最多300步。'],
 ['全任务完成','完整覆盖全部网格，同时20名目标服务完成。'],['人员服务RCR','每场完成服务人数/20；不能与完整任务完成比例混用。'],
 ['ACC','潜在意图冲突对数/联合时间步，安全层可能拦截；不是实际碰撞百分比。'],['ST','完成或超时的步数。超时300步保留，不只平均成功场次。'],
 ['FRT/CRT','首次完成服务/全20人服务完成的步数。空白表示未发生或旧日志未记录，非0步。'],
 ['训练计数','父模型已训练256回合，HATRPO512；本次新增适配轮次另计。双模型一个环境回合只算一次。'],
 ['历史比较','历史单模型4训练/2验证父水场和控制器设置不同，不能直接与当前排名。旧三组合部分前缀在15组合复用，不重复计总量。'],
 ['学习器','同一算法控制4架无人机称单模型；异构两个算法各控制2架称双模型；不是把两份独立分数平均。'],
 ['规则控制','服务承诺、重试和覆盖规划会接管动作。成绩需结合no-cover对照，不能解释为神经模型独立能力。'],
 ['最佳与最新','最佳按预登记的分层冲突、任务完成、人员服务、最差分工、覆盖、冲突和步数顺序选；最佳0表示未证明新增学习收益。'],
 ['缺失与数值','空白为未运行、未评估或原日志未记录。所有比例保留底层精度；展示两位小数不代表统计置信度。'],
 ['诊断','损失不可跨算法比较；elapsed为阶段累计时间，不能逐行相加。旧双模型未记录的优化器累计次数留空。'],
 ['范围','包含全部可得单模型、15种组合、旧三组队列、旧联合学习、冻结权重组合、规划预选、关闭覆盖规划消融；原始轨迹及权重仍保留在源位置。'],
 ['未验证',(d.strict_summary?'严格测试结果另表；':'严格测试尚未完成；')+'未做多训练种子独立重复、真实人员和洪水、续航、通信、建筑和天气验证。'],
 ['统计核对',JSON.stringify(d.audit)],['来源','下方SHA256对应快照读取的原文件；追加中的训练日志随后改变是正常现象。'],
];
const ns=table('口径与来源',['项目','说明'],notes,[24,145]);ns.tabColor='#8A96A4';
const begin=notes.length+4;ns.getRange(`A${begin}:D${begin+d.sources.length}`).values=[['来源ID','绝对文件路径','SHA256','字节数'],...d.sources.map(r=>[r.id,r.path,r.sha256,r.bytes])];
ns.getRange(`A${begin}:D${begin}`).format.fill='#244864';ns.getRange(`A${begin}:D${begin}`).format.font={color:'#FFFFFF',bold:true};ns.getRange(`C${begin}:C${begin+d.sources.length}`).format.columnWidth=72;ns.getRange(`D${begin}:D${begin+d.sources.length}`).format.columnWidth=18;
ns.getRange(`A2:B${notes.length+1}`).format.rowHeight=38;ns.getRange(`B2:B${notes.length+1}`).format.wrapText=true;
ns.getRange('B2').setNumberFormat('yyyy-mm-dd hh:mm:ss');
ns.getRange(`B${begin+1}:B${begin+d.sources.length}`).format.wrapText=true;ns.getRange(`A${begin+1}:D${begin+d.sources.length}`).format.rowHeight=44;
const strictRows=(d.strict_summary?.summary??[]).map(r=>[r.candidate,r.variant,r.mode,r.stratum,r.independent_parents,r.rollouts,r.means.task_complete,r.means.RCR,r.means.coverage,r.means.ACC,r.means.ST,r.failed_rollouts,r.all_assignments_complete.completed_parents,r.all_assignments_complete.wilson_ci95[0],r.all_assignments_complete.wilson_ci95[1],r.max_conflicts]);
const ss=table('严格测试结果',['模型/组合','控制方式','动作模式','场景组','独立父场景','回放次数','完整任务比例','人员服务RCR','覆盖比例','潜在冲突/步','平均步数','失败回放数','全部重复成功父场','Wilson下限95%','Wilson上限95%','最多冲突'],strictRows,[26,22,16,29,16,14,18,18,16,19,16,16,23,19,19,17],6);
ss.getRange('A2').values=[['严格检验：50个新主场景 + 20个方向/布局压力场景']];ss.getRange('A2').format.font={name:'Microsoft YaHei',size:14,bold:true};
ss.getRange('A3').values=[['状态：'+(d.strict_summary?.status==='completed'?'全部5630条测试完成，完整性审计通过':(statusNames[d.strict_status?.status]??d.strict_status?.status??'等待训练完成、锁定权重后生成测试集'))]];
ss.getRange('A4').values=[['包含完整方法、关闭覆盖规划、纯规划与均匀随机探索；按父场景统计，不能把分工或动作重复当独立地图。']];
fmt(ss,['G','H','I','N','O'],7,strictRows.length+6,'0.00%');fmt(ss,['J'],7,strictRows.length+6,'0.00000');fmt(ss,['K'],7,strictRows.length+6,'0.0');
wb.recalculate();
const total=main.getRange('F33').values[0][0];if(total!==d.overview.reduce((a,r)=>a+r.actual,0))throw Error('Training count mismatch');
for(const [i,v]of d.evaluations.entries())if(Math.abs(ev.getRange(`I${i+2}`).values[0][0]-v.completion)>1e-12)throw Error('Completion ratio mismatch');
const inspection=await wb.inspect({kind:'table',range:'当前21方案!A7:N12',include:'values,formulas',tableMaxRows:6,tableMaxCols:14,maxChars:3500});
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1500});
const ranges={'当前21方案':'A2:N17','严格测试结果':'A2:L8','全部评估轮次':'A1:O7','当前逐场验证':'A1:K7','当前逐回合训练':'A1:K7','历史逐场验证':'A1:K7','历史逐回合训练':'A1:K7','权重检查点':'A1:E7','口径与来源':'A1:B9'};
for(const name of names){const png=await wb.render({sheetName:name,range:ranges[name],scale:1,format:'png'});await fs.writeFile(path.join(base,`preview-${name}.png`),new Uint8Array(await png.arrayBuffer()));}
await fs.writeFile(path.join(base,'verification.json'),JSON.stringify({audit:d.audit,totalCurrentEpisodes:total,inspection:inspection.ndjson,errors:errors.ndjson,sheets:names},null,2));
if(global.gc)global.gc();
console.log(JSON.stringify({stage:'export_start',memory:process.memoryUsage()}));
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(path.join(base,'单双模型训练全量数据-20261101.xlsx'));
console.log(JSON.stringify({sheets:names,evaluations:d.evaluations.length,validationRows:d.details.length,trainingRows:d.training.length,currentEpisodes:total,errors:errors.ndjson}));
