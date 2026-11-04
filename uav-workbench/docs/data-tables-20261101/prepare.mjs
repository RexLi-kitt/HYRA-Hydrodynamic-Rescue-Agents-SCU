import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root='C:/Users/Lenovo/Documents/ChatGPT/无人机集群搜救大创/outputs';
const dest=path.dirname(fileURLToPath(import.meta.url));
const stages={pilot:'uav6-pilot-20261101',total256:'uav6-total256-20261101',priority:'uav6-priority-20261101',full1024:'uav6-full1024-20261101'};
const algos=['happo','hatrpo','hasac','maac','mat','coma'];
const group={happo:'A',hatrpo:'A',hasac:'B',maac:'B',mat:'C',coma:'C'};
const sources=[]; const sourceMap=new Map();
async function read(rel,lines=false){
 const b=await fs.readFile(path.join(root,rel));
 if(!sourceMap.has(rel)){const id=`S${String(sources.length+1).padStart(3,'0')}`; sourceMap.set(rel,id);sources.push({id,file:rel,sha256:crypto.createHash('sha256').update(b).digest('hex'),bytes:b.length});}
 const text=b.toString('utf8').replace(/^\uFEFF/,''); return lines?text.trim().split(/\r?\n/).filter(Boolean).map(JSON.parse):JSON.parse(text);
}
const training=[],stageCounts={},evaluations=[],details=[],checkpoints=[];
for(const a of algos){
 stageCounts[a]={};
 for(const [stage,dir] of Object.entries(stages)){
  if(['priority','full1024'].includes(stage)&&!['hasac','hatrpo'].includes(a))continue;
  const rel=`${dir}/${a}/episodes.jsonl`, rows=await read(rel,true);
  stageCounts[a][stage]=rows.length;
  for(let i=0;i<rows.length;i++)training.push({algorithm:a.toUpperCase(),group:group[a],stage,total_episode:rows[i].total_episode??rows[i].episode,...rows[i],source:sourceMap.get(rel),source_line:i+1});
 }
 const rows=training.filter(x=>x.algorithm===a.toUpperCase());
 const expected=['hasac','hatrpo'].includes(a)?1024:256;
 if(rows.length!==expected || rows.some((r,i)=>r.total_episode!==i+1))throw new Error(`Noncontiguous training ${a}`);
}
function mean(rows,key){const vals=rows.map(x=>x[key]).filter(x=>typeof x==='number');return vals.length?Number((vals.reduce((a,b)=>a+b,0)/vals.length).toPrecision(15)):null;}
function checkNear(v,b,label){if(v==null&&b==null)return;if(typeof v!=='number'||typeof b!=='number'||Math.abs(v-b)>1e-10)throw new Error(`Mismatch ${label}: ${v} vs ${b}`);}
async function addEval(rel,a,ep,stage,mode,control,note=''){
 const v=await read(rel); const rows=v.episodes??v.rows;
 if(!Array.isArray(rows))throw new Error(`Missing episodes ${rel}`);
 const summary=v.summary?.metrics??v.summary??{};
 for(const key of ['ST','ACC','RCR','coverage','rescued'])if(summary[key]!=null)checkNear(mean(rows,key),summary[key],`${rel} ${key}`);
 for(const key of ['FRT','CRT'])if(summary[key]&&typeof summary[key]==='object')checkNear(mean(rows,key),summary[key].mean,`${rel} ${key}`);
 const id=`V${String(evaluations.length+1).padStart(3,'0')}`;
 const rec={id,group:group[a],algorithm:a.toUpperCase(),episode:ep,stage,mode,control,n:rows.length,
  water_parents:new Set(rows.map(r=>r.case)).size,action_seeds:[...new Set(rows.map(r=>r.action_seed).filter(x=>x!=null))].join(','),
  coverage:mean(rows,'coverage'),RCR:mean(rows,'RCR'),ACC:mean(rows,'ACC'),ST:mean(rows,'ST'),rescued:mean(rows,'rescued'),discovered:mean(rows,'discovered'),
  all_rescued:rows.filter(r=>r.rescued===20).length,task_complete:rows.filter(r=>r.task_complete).length,timed_out:rows.filter(r=>r.timed_out).length,
  FRT:mean(rows,'FRT'),FRT_valid:rows.filter(r=>r.FRT!=null).length,CRT:mean(rows,'CRT'),CRT_valid:rows.filter(r=>r.CRT!=null).length,
  conflicts:rows.reduce((s,r)=>s+r.conflicts,0),note,source:sourceMap.get(rel),file:rel};
 evaluations.push(rec);
 for(let i=0;i<rows.length;i++) details.push({validation_id:id,algorithm:rec.algorithm,episode:ep,stage,mode,control,run:i+1,...rows[i],source:rec.source});
 return rec;
}
for(const a of algos)for(const ep of [64,256])for(const control of ['service','watchdog'])await addEval(`${stages.total256}/${a}/validation-${String(ep).padStart(4,'0')}-${control}.json`,a,ep,'64与256同口径对照','固定',control==='service'?'自动施救':'自动施救＋覆盖补漏',control==='watchdog'?'含规则接管，不能归因于模型学习':'无重试控制');
for(const a of algos)for(const variant of ['sample-original','sample-guarded','greedy-guarded'])await addEval(`${stages.priority}/compare-${a}/${variant}.json`,a,256,'256执行方式对照',variant.startsWith('sample')?'随机':'固定',variant.endsWith('original')?'自动施救':'自动施救＋重试控制');
for(const stage of ['priority','full1024'])for(const a of ['hasac','hatrpo']){
 const dir=`${stages[stage]}/${a}`;
 const names=(await fs.readdir(path.join(root,dir))).filter(x=>/^validation-\d+-(sample|greedy)\.json$/.test(x)).sort();
 for(const name of names){const m=name.match(/^validation-(\d+)-(sample|greedy)\.json$/);await addEval(`${dir}/${name}`,a,Number(m[1]),stage==='priority'?'优先续训':'跑满1024',m[2]==='sample'?'随机':'固定',a==='hasac'?'自动施救＋重试控制':'自动施救');}
}
// Separate earlier protocol revisions and diagnostic replays from the main comparison.
const supplemental=[];
const suppDirs=['uav6-service-commitment-20261101','uav6-group-c-diagnosis-20261101','uav6-service-improvement-20261101'];
for(const dir of suppDirs){
 for(const name of (await fs.readdir(path.join(root,dir))).filter(n=>n.endsWith('.json')).sort()){
  const rel=`${dir}/${name}`; const b=JSON.parse((await fs.readFile(path.join(root,rel),'utf8')).replace(/^\uFEFF/,''));
  if(!Array.isArray(b.episodes??b.rows))continue;
  let a=algos.find(a=>name.startsWith(a));if(!a&&dir.includes('service-improvement'))a='maac';if(!a)continue;
  let ep=name.includes('-256')?256:64;
  if(dir.includes('service-improvement'))ep=null;
  const note=dir.includes('commitment')?'早期控制实现，与后续因果动作掩码口径不同，不作直接进步对比':dir.includes('diagnosis')?'仅重放诊断，不增加训练回合':'独立失败改进分支，不计入当前主线轮次';
  try{const rec=await addEval(rel,a,ep,dir.includes('diagnosis')?'C组诊断':dir.includes('commitment')?'早期施救控制':'MAAC独立分支',name.includes('sample')?'随机':'固定',name,note);supplemental.push(rec.id);}catch(e){throw new Error(`Supplemental: ${e.message}`);}
 }
}
const best={};for(const a of ['hasac','hatrpo'])best[a]=await read(`${stages.full1024}/${a}/best.json`);
const identities={};for(const a of algos){identities[a]=await read(`${stages.pilot}/${a}/identity.json`);for(const [file,ep]of [['initial-validation.json',0],['final-validation.json',64]])await addEval(`${stages.pilot}/${a}/${file}`,a,ep,'早期无自动施救','固定','无自动施救','初期口径，不能与自动施救结果直接归因比较');}
const protocols={};for(const stage of ['total256','priority','full1024'])protocols[stage]=await read(`${stages[stage]}/protocol.json`);
const queue={};for(const stage of Object.keys(stages))queue[stage]=await read(`${stages[stage]}/queue-status.json`);
const comboDir='uav6-combinations-20261101';
try{
 for(const name of (await fs.readdir(path.join(root,comboDir))).filter(n=>/^(?:HASAC\d+|HATRPO\d+|MAT\d+|P\d-allocation\d)\.json$/.test(n)).sort())await addEval(`${comboDir}/${name}`,name.replace('.json',''),null,'新组合同口径重放','随机','统一自动施救＋重试控制','单模型/2+2分工对照；累计轮次见组合名，不新增训练');
 await read(`${comboDir}/results.json`);await read(`${comboDir}/protocol.json`);
}catch(e){if(e.code!=='ENOENT')throw e;}
// List every saved mainline checkpoint. These files are not loaded or changed.
for(const [stage,dir]of Object.entries(stages))for(const a of algos){
 if(['priority','full1024'].includes(stage)&&!['hasac','hatrpo'].includes(a))continue;
 for(const name of(await fs.readdir(path.join(root,dir,a))).filter(n=>n.endsWith('.pt')).sort()){
  const file=`${dir}/${a}/${name}`, stat=await fs.stat(path.join(root,file));
  const match=name.match(/(?:checkpoint[-_]?)(\d+)/);
  let ep=match?Number(match[1]):name==='latest.pt'?(stage==='pilot'?64:stage==='total256'?256:null):null;
  checkpoints.push({algorithm:a.toUpperCase(),stage,episode:ep,file,bytes:stat.size,modified:stat.mtime.toISOString()});
 }
}
const audit={main_training_rows:training.length,stageCounts,validation_groups:evaluations.length,validation_rollouts:details.length,supplemental_groups:supplemental.length,source_files:sources.length,checkpoints:checkpoints.length,summary_recomputed_and_agrees:true,contiguous_episode_ranges:true};
await fs.mkdir(dest,{recursive:true});
await fs.writeFile(path.join(dest,'data.json'),JSON.stringify({root,training,evaluations,details,checkpoints,sources,stageCounts,best,queue,supplemental,audit,identities,protocols},null,2));
console.log(JSON.stringify(audit,null,2));
console.log(JSON.stringify(evaluations.filter(r=>r.episode===1024&&r.mode==='随机').map(r=>({algo:r.algorithm,cov:r.coverage,rescue:r.RCR,all:r.all_rescued,n:r.n})),null,2));
