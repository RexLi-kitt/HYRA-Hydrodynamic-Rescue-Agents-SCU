import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const dir=path.dirname(fileURLToPath(import.meta.url));
const d=JSON.parse(await fs.readFile(path.join(dir,'data.json'),'utf8'));
const out=path.join(dir,'outputs','uav-data-20261101');await fs.mkdir(out,{recursive:true});
const wb=Workbook.create();
const names=['分组总览','组合试验','全部阶段验证','逐场验证','逐回合训练','检查点目录','口径与来源'];
const sh=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
const font='Microsoft YaHei';
const stageNames={pilot:'初训64',total256:'统一续至256',priority:'优先续训',full1024:'跑满1024'};
const algos=['happo','hatrpo','hasac','maac','mat','coma'];
const grp={happo:'A',hatrpo:'A',hasac:'B',maac:'B',mat:'C',coma:'C'};
function col(n){let s='';for(;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
function style(s,rows,cols,width=16){
 const r=s.getRange(`A1:${col(cols)}${rows}`);r.format.font={name:font,size:10,color:'#17243B'};r.format.rowHeight=23;r.format.columnWidth=width;r.format.verticalAlignment='center';s.showGridLines=false;
}
let tid=0;
function table(s,row,headers,rows,widths){
 const end=row+rows.length,c=col(headers.length);s.getRange(`A${row}:${c}${end}`).values=[headers,...rows];
 const t=s.tables.add(`A${row}:${c}${end}`,true,`DataTable${++tid}`);t.showFilterButton=true;
 const h=s.getRange(`A${row}:${c}${row}`);h.format={fill:'#253D62',font:{name:font,size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:42};
 if(widths)widths.forEach((w,i)=>s.getRange(`${col(i+1)}${row}:${col(i+1)}${end}`).format.columnWidth=w);
 return end;
}
function title(s,text){s.getRange('A2').values=[[text]];s.getRange('A2').format.font={name:font,size:15,bold:true,color:'#17243B'};s.getRange('A2:I2').format.borders={bottom:{style:'thin',color:'#A8B5C5'}};s.getRange('A2').format.rowHeight=28;}
function pct(s,c,start,end){s.getRange(`${c}${start}:${c}${end}`).setNumberFormat('0.00%');}
function num(s,c,start,end,fmt='0.00'){s.getRange(`${c}${start}:${c}${end}`).setNumberFormat(fmt);}
function evalAt(a,ep,mode,control,stage){return d.evaluations.find(x=>x.algorithm===a.toUpperCase()&&x.episode===ep&&x.mode===mode&&(!control||x.control===control)&&(!stage||x.stage===stage));}
function erow(r){return d.evaluations.indexOf(r)+2;}
function ref(r,c){return `='全部阶段验证'!${c}${erow(r)}`;}
// Flat scientific records retain source IDs and all original diagnostics.
const eh=['验证编号','组别','模型','累计训练回合','实验阶段','动作执行','控制方式','验证次数','区域覆盖率','人员获救比例','平均获救人数','全员获救次数','全任务完成次数','潜在冲突/步','平均任务步数','首次获救平均步','首次获救有效数','全员获救平均步','全员获救有效数','超时次数','父水场数','动作随机种子','潜在冲突总对数','口径说明','来源编号'];
const ev=d.evaluations.map(r=>[r.id,r.group??'组合',r.algorithm,r.episode,r.stage,r.mode,r.control,r.n,r.coverage,r.RCR,r.rescued,r.all_rescued,r.task_complete,r.ACC,r.ST,r.FRT,r.FRT_valid,r.CRT,r.CRT_valid,r.timed_out,r.water_parents,r.action_seeds,r.conflicts,r.note,r.source]);
style(sh['全部阶段验证'],ev.length+1,eh.length);table(sh['全部阶段验证'],1,eh,ev,[12,8,13,16,23,10,27,12,15,16,17,16,18,17,17,19,19,19,19,12,12,20,20,66,12]);
for(const c of ['I','J'])pct(sh['全部阶段验证'],c,2,ev.length+1);num(sh['全部阶段验证'],'N',2,ev.length+1,'0.00000');for(const c of ['K','O','P','R'])num(sh['全部阶段验证'],c,2,ev.length+1);
sh['全部阶段验证'].freezePanes.freezeRows(1);sh['全部阶段验证'].freezePanes.freezeColumns(3);
const dh=['验证编号','模型','训练回合','实验阶段','执行','控制方式','场次','水场','布局种子','动作种子','覆盖率','人员获救比例','获救人数','发现人数','全任务完成','超时','任务步数','首次获救步','全员获救步','潜在冲突/步','冲突对数','越界意图次数','回报和','强制施救机步','重试限制机步','来源编号','完整原始记录JSON'];
const details=d.details.map(r=>[r.validation_id,r.algorithm,r.episode,r.stage,r.mode,r.control,r.run,r.case,r.layout,r.action_seed??r.seed??null,r.coverage,r.RCR,r.rescued,r.discovered,r.task_complete,r.timed_out,r.ST,r.FRT,r.CRT,r.ACC,r.conflicts,r.boundaries,r.return_sum??null,r.forced_agent_steps??null,r.guard_agent_steps??null,r.source,JSON.stringify(r)]);
style(sh['逐场验证'],details.length+1,dh.length);table(sh['逐场验证'],1,dh,details);for(const c of ['K','L'])pct(sh['逐场验证'],c,2,details.length+1);num(sh['逐场验证'],'T',2,details.length+1,'0.00000');sh['逐场验证'].getRange(`H1:H${details.length+1}`).format.columnWidth=28;sh['逐场验证'].getRange(`AA1:AA${details.length+1}`).format.columnWidth=70;sh['逐场验证'].freezePanes.freezeRows(1);sh['逐场验证'].freezePanes.freezeColumns(3);
const th=['组别','模型','训练阶段','累计回合','训练水场','布局种子','覆盖率','人员获救比例','获救人数','发现人数','全任务完成','超时','任务步数','首次获救步','全员获救步','潜在冲突/步','冲突对数','越界意图次数','回报和','阶段累计耗时秒','阶段累计环境步','强制施救机步','重试限制机步','损失及优化诊断JSON','来源编号','原日志行','完整原始记录JSON'];
const tr=d.training.map(r=>[r.group,r.algorithm,stageNames[r.stage],r.total_episode,r.case,r.layout,r.coverage,r.RCR,r.rescued,r.discovered,r.task_complete,r.timed_out,r.ST,r.FRT,r.CRT,r.ACC,r.conflicts,r.boundaries,r.return_sum,r.elapsed_s,r.phase_steps??r.steps??null,r.forced_agent_steps??null,r.guard_agent_steps??null,JSON.stringify(r.loss??r.diagnostics??{}),r.source,r.source_line,JSON.stringify(r)]);
for(const row of [...details,...tr])if(row.some(v=>typeof v==='string'&&v.length>32767))throw Error('Excel text limit; do not truncate');
style(sh['逐回合训练'],tr.length+1,th.length);table(sh['逐回合训练'],1,th,tr);for(const c of ['G','H'])pct(sh['逐回合训练'],c,2,tr.length+1);num(sh['逐回合训练'],'P',2,tr.length+1,'0.00000');for(const c of ['S','T'])num(sh['逐回合训练'],c,2,tr.length+1);sh['逐回合训练'].getRange(`E1:E${tr.length+1}`).format.columnWidth=24;for(const c of ['X','AA'])sh['逐回合训练'].getRange(`${c}1:${c}${tr.length+1}`).format.columnWidth=65;sh['逐回合训练'].freezePanes.freezeRows(1);sh['逐回合训练'].freezePanes.freezeColumns(4);
// Output view: three groups, two independently trained models in each.
const s=sh['分组总览'];style(s,54,11,16);s.tabColor='#253D62';title(s,'无人机实验数据汇总　2026-11-01');
s.getRange('A3').values=[['原A/B/C为对照分组。组合试验另页列出，不把两个模型成绩的平均数称为组合效果。']];
const trainRows=algos.map(a=>[grp[a],a.toUpperCase(),d.stageCounts[a].pilot,d.stageCounts[a].total256,d.stageCounts[a].priority??0,d.stageCounts[a].full1024??0,null,d.identities[a].parameters]);
table(s,5,['组别','模型','初训回合','统一追加回合','优先追加回合','最后追加回合','累计回合','模型参数量'],trainRows,[9,17,17,18,18,18,16,19]);
for(let i=0;i<6;i++)s.getRange(`G${i+6}`).formulas=[[`=SUM(C${i+6}:F${i+6})`]];
s.getRange('A12').values=[['主线合计']];for(const c of ['C','D','E','F','G'])s.getRange(`${c}12`).formulas=[[`=SUM(${c}6:${c}11)`]];s.getRange('A12:H12').format.font={bold:true};s.getRange('C6:H12').setNumberFormat('#,##0');
s.getRange('A14').values=[['最新模型验证（自动施救，无覆盖补漏接管）']];s.getRange('A14').format.font={bold:true};
const latest=algos.map(a=>{const isLong=['hasac','hatrpo'].includes(a),ep=isLong?1024:256;return {a,ep,sample:evalAt(a,ep,'随机',a==='hasac'?'自动施救＋重试控制':'自动施救',isLong?'跑满1024':'256执行方式对照'),greedy:evalAt(a,ep,'固定',isLong?(a==='hasac'?'自动施救＋重试控制':'自动施救'):'自动施救',isLong?'跑满1024':'64与256同口径对照')};});
table(s,16,['组别','模型','累计回合','随机覆盖率','随机获救比例','固定覆盖率','固定获救比例','全员获救/随机','全任务/随机'],latest.map(x=>[grp[x.a],x.a.toUpperCase(),x.ep,null,null,null,null,`${x.sample.all_rescued}/${x.sample.n}`,`${x.sample.task_complete}/${x.sample.n}`]),[9,17,17,18,18,18,18,20,20]);
for(let i=0;i<latest.length;i++){const r=i+17,x=latest[i];s.getRange(`D${r}:G${r}`).formulas=[[ref(x.sample,'I'),ref(x.sample,'J'),ref(x.greedy,'I'),ref(x.greedy,'J')]];}for(const c of ['D','E','F','G'])pct(s,c,17,22);
s.getRange('A24').values=[['按既定综合规则保留的版本（随机执行）']];s.getRange('A24').format.font={bold:true};
const br=['hasac','hatrpo'].map(a=>({a,ep:d.best[a].episode,r:evalAt(a,d.best[a].episode,'随机',a==='hasac'?'自动施救＋重试控制':'自动施救',a==='hasac'?'256执行方式对照':'优先续训')}));
table(s,26,['模型','保留回合','覆盖率','人员获救比例','潜在冲突/步','全员获救次数','验证次数'],br.map(x=>[x.a.toUpperCase(),x.ep,null,null,null,x.r.all_rescued,x.r.n]));
for(let i=0;i<br.length;i++)s.getRange(`C${i+27}:E${i+27}`).formulas=[[ref(br[i].r,'I'),ref(br[i].r,'J'),ref(br[i].r,'N')]];for(const c of ['C','D'])pct(s,c,27,28);num(s,'E',27,28,'0.00000');
const notes=[
 '人员获救比例＝每场获救人数÷20。全员获救次数＝20人全部获救的场次。',
 '全任务完成要求区域覆盖100%且20人全部获救。它不等同于人员获救比例。',
 '随机验证：2个父水场×4种布局×2个动作种子＝16次。固定执行验证8次。',
 '这是反复使用的小规模合成开发验证集，未进行封存50场独立测试，也不是现实救援准确率。',
 'HASAC随机结果使用重试控制；其余主表随机结果使用自动施救。原始控制对照全部保留。',
 'HATRPO1024获救比例高，但潜在冲突/步0.02125超过原0.02门槛，未自动替换512保留版。',
 `已核对主线${d.audit.main_training_rows}个训练回合、${d.audit.validation_groups}个验证批次、${d.audit.validation_rollouts}条逐场验证。`,
 '旧无施救、旧施救实现、规则补漏、C组诊断和失败分支均在阶段验证中单独标注。',
 '3072回合为当前六条模型主线总量，不含诊断重放、单元测试和已放弃的独立改进分支。'
];notes.forEach((n,i)=>s.getRange(`A${31+i*2}`).values=[[n]]);
// Combination results are real jointly simulated fleets, not arithmetic score averaging.
const cs=sh['组合试验'];style(cs,36,11,18);title(cs,'双模型组合试验');
const comboPath=path.join(d.root,'uav6-combinations-20261101/results.json');
try{
 const c=JSON.parse(await fs.readFile(comboPath,'utf8'));
 cs.getRange('A3').values=[['4架无人机分成2+2，两个模型各管2架；两种对角分配均执行。权重冻结，没有新增训练。']];
 const rows=Object.entries(c.single_controls).map(([name,v])=>['单模型',name,'4架同模型',v.n,v.summary.coverage,v.summary.RCR,v.summary.ACC,v.summary.CRT.valid,v.whole_task_completed,'对照']);
 rows.push(...c.pairs.map(v=>[v.pair,`${v.first} + ${v.second}`,'两种分配汇总',v.n,v.summary.coverage,v.summary.RCR,v.summary.ACC,v.summary.CRT.valid,v.whole_task_completed,v.qualified_for_next_training?'满足预定候选条件':'未满足预定条件']));
 table(cs,5,['类型','模型组合','分配方式','验证次数','覆盖率','人员获救比例','潜在冲突/步','全员获救次数','全任务次数','预定条件判断'],rows,[12,44,23,14,16,18,18,18,17,26]);for(const x of ['E','F'])pct(cs,x,6,5+rows.length);num(cs,'G',6,5+rows.length,'0.00000');
 cs.getRange('A17').values=[['所有单模型和组合统一开启自动施救与重试控制，因此单模型分数可能不同于旧主表。']];
 cs.getRange('A19').values=[['32次组合重放仍只有2个父水场。两种分工均保留，不按结果挑选分工。']];
 cs.getRange('A21').values=[['合格条件同时约束获救、覆盖、潜在冲突及种子/分工稳定性，详见随附组合报告。']];
 const allocations=[];
 for(const v of c.pairs)for(let i=0;i<2;i++){const rr=JSON.parse(await fs.readFile(path.join(d.root,`uav6-combinations-20261101/${v.pair}-allocation${i}.json`),'utf8'));allocations.push([v.pair,i===0?'模型1管0/3号；模型2管1/2号':'模型1管1/2号；模型2管0/3号',rr.n,rr.summary.coverage,rr.summary.RCR,rr.summary.ACC,rr.summary.CRT.valid,rr.whole_task_completed]);}
 table(cs,24,['组合','分工','验证次数','覆盖率','人员获救比例','潜在冲突/步','全员获救次数','全任务次数'],allocations);for(const x of ['D','E'])pct(cs,x,25,32);num(cs,'F',25,32,'0.00000');
}catch(e){if(e.code!=='ENOENT')throw e;cs.getRange('A5').values=[['尚无完成结果，不填写组合成绩。']];}
const cp=d.checkpoints.map(r=>[r.algorithm,stageNames[r.stage],r.episode,r.bytes,r.file,r.modified]);style(sh['检查点目录'],cp.length+1,6);table(sh['检查点目录'],1,['模型','训练阶段','回合','文件字节数','保存位置（相对outputs）','修改时间UTC'],cp,[15,21,12,20,82,30]);sh['检查点目录'].freezePanes.freezeRows(1);
sh['检查点目录'].getRange(`F2:F${cp.length+1}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');sh['检查点目录'].getRange(`D2:D${cp.length+1}`).setNumberFormat('#,##0');
const ns=sh['口径与来源'];style(ns,d.sources.length+36,5,24);title(ns,'指标口径与来源');
const definitions=[
 ['数据日期','2026-11-01'],['任务','1km×1km，4架无人机，20人，100×100网格；每步5秒，最多300步。'],
 ['训练和验证','训练水场4个，验证水场2个。动作种子重复不等于独立训练种子或独立洪水样本。'],
 ['训练轮次','1轮指1个episode。64轮后续训192轮共256轮，不是256轮额外训练。'],
 ['覆盖率','区域网格累计被观察比例。覆盖并不代表全部人员获救。'],
 ['人员获救比例RCR','每场获救人数/20，表中验证汇总为场次平均。'],
 ['全任务完成','全部网格观察完成，同时全部20人完成服务。'],
 ['潜在冲突ACC','原始意图轨迹冲突对数/联合时间步；安全层会拦截，不是实际碰撞率。'],
 ['FRT与CRT','首次成功服务与全20人完成服务的步数。空白表示未发生；只对发生场次求平均。'],
 ['elapsed_s','训练日志中为该训练阶段的累计壁钟秒数，不应逐行求和。包含阶段内的验证/保存开销。'],
 ['损失值','优化器诊断并非准确率；不同算法损失定义不同，不能用损失绝对值直接排名。'],
 ['早期与后期','自动施救与执行方式曾改变；仅在控制口径一致的行之间比较训练收益。'],
 ['自动施救','发现人员后按已知信息接近并连续悬停服务。它是共享规则控制的一部分。'],
 ['覆盖补漏','覆盖规划器会接管探索，100%任务成绩不能解释为神经模型单独实现。'],
 ['保留版','依预登记的获救、覆盖和潜在冲突约束选择。并非最高RCR版或最新一轮。'],
 ['组合','新组合仅复用已有权重形成异构机群。没有把两个独立成绩平均冒充联合结果。'],
 ['记录范围','主线所有3072轮及已收集验证；独立失败分支仅保留验证结果，不计主线轮数。'],
 ['来源根目录',d.root],['来源核验','原始日志与JSON只读提取，验证平均值已从逐场数据重新计算并核对；下列SHA256用于追溯。']
];table(ns,4,['项目','说明'],definitions,[23,125]);
table(ns,27,['来源编号','原文件（相对outputs）','SHA256','字节数'],d.sources.map(r=>[r.id,r.file,r.sha256,r.bytes]),[14,110,72,16]);ns.freezePanes.freezeRows(1);
ns.getRange(`A1:A${d.sources.length+27}`).format.columnWidth=24;
sh['逐场验证'].getRange(`D1:D${details.length+1}`).format.columnWidth=26;sh['逐场验证'].getRange(`F1:F${details.length+1}`).format.columnWidth=30;
wb.recalculate();
const total=s.getRange('G12').values[0][0];if(total!==3072)throw Error(`Total failed: ${total}`);
for(const [cell,expected]of [['D18',.966275],['E18',.99375],['D19',.955475],['E19',.925]])if(Math.abs(s.getRange(cell).values[0][0]-expected)>1e-10)throw Error(`Headline ${cell}`);
const inspect=await wb.inspect({kind:'table',range:'分组总览!A16:I22',include:'values,formulas',tableMaxRows:7,tableMaxCols:9,maxChars:5000});
await fs.writeFile(path.join(dir,'verification.json'),JSON.stringify({audit:d.audit,inspect:inspect.ndjson},null,2));
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'Final formula errors',maxChars:1500});console.log(errors.ndjson);
const ranges={'分组总览':'A2:I22','组合试验':'A2:J14','全部阶段验证':'A1:O9','逐场验证':'A1:L9','逐回合训练':'A1:P9','检查点目录':'A1:F8','口径与来源':'A2:B15'};
for(const name of names){const preview=await wb.render({sheetName:name,range:ranges[name],scale:1,format:'png'});await fs.writeFile(path.join(dir,`preview-${name}.png`),new Uint8Array(await preview.arrayBuffer()));}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(path.join(out,'无人机训练与组合实验完整数据-20261101.xlsx'));
console.log(JSON.stringify({output:path.join(out,'无人机训练与组合实验完整数据-20261101.xlsx'),sheets:names,training:tr.length,validation:details.length}));
