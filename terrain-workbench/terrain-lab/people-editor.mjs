// Rescue start and trapped people: four placement modes plus live legality checks.
import {buildModel,reachableFrom,defaultStart,legalIndices,placePeople,validate,cellFromMetres,GROUP_COLORS,MAX_PEOPLE} from './people-place.mjs';

const MODE_LABELS={uniform:'均匀分散',cluster:'聚簇分布',cluster_isolated:'聚簇 + 孤立人员',manual:'手动放置'};
const groupLabel=group=>group<0?'孤立':`聚簇 ${group+1}`;

export class PeopleEditor{
 constructor({getScene,getStructures,redraw,dirty,message,locked}){
  Object.assign(this,{getScene,getStructures,redraw,dirty,message,locked});
  this.mode='uniform';this.seed=20261018;this.parameters={count:6,clusters:1,radius_m:10,isolated:2};
  this.items=[];this.start=null;this.pick=null;this.bad=new Set();this.counter=0;
  this.$=id=>document.getElementById(id);
  this.$('peopleMode').onchange=()=>this.modeChanged();
  for(const id of ['peopleCount','peopleClusters','peopleRadius','peopleIsolated','peopleSeed'])
   this.$(id).onchange=()=>{this.readFields();this.dirty();this.summary();};
  this.$('placePeople').onclick=()=>this.generate();
  this.$('rerollPeople').onclick=()=>{this.readFields();this.seed=(this.seed%999999)+1;this.$('peopleSeed').value=this.seed;this.generate();};
  this.$('pickPeople').onclick=()=>this.togglePick('person');
  this.$('pickStart').onclick=()=>this.togglePick('start');
  this.$('autoStart').onclick=()=>this.autoStart();
  this.$('clearPeople').onclick=()=>{if(this.blocked())return;this.items=[];this.counter=0;this.changed('人员已清空。');};
  this.fields();
 }
 blocked(){return this.locked();}
 modeChanged(){
  if(this.blocked())return;
  this.readFields();this.fields();
  if(this.mode!=='manual'&&!this.items.length)this.generate();else this.changed('人员放置模式已切换。');
 }
 readFields(){
  const num=(id,fallback,low,high)=>{const v=Number(this.$(id).value);return Number.isFinite(v)?Math.max(low,Math.min(high,Math.round(v))):fallback;};
  this.mode=this.$( 'peopleMode').value;this.seed=num('peopleSeed',this.seed,1,999999);
  this.parameters={count:num('peopleCount',6,1,MAX_PEOPLE),clusters:num('peopleClusters',1,1,8),
   radius_m:Math.max(1,Math.min(120,Number(this.$('peopleRadius').value)||10)),isolated:num('peopleIsolated',2,1,30)};
 }
 writeFields(){
  this.$('peopleMode').value=this.mode;this.$('peopleSeed').value=this.seed;
  this.$('peopleCount').value=this.parameters.count;this.$('peopleClusters').value=this.parameters.clusters;
  this.$('peopleRadius').value=this.parameters.radius_m;this.$('peopleIsolated').value=this.parameters.isolated;
 }
 fields(){
  const manual=this.mode==='manual',cluster=this.mode!=='uniform';
  this.$('peopleClusterFields').hidden=!cluster;
  this.$('peopleIsolatedField').hidden=this.mode!=='cluster_isolated';
  this.$('placePeople').disabled=manual;
  this.$('rerollPeople').disabled=manual;
  this.$('pickPeople').disabled=!manual;
  this.$('pickPeople').setAttribute('aria-pressed',String(this.pick==='person'));
  this.$('pickStart').setAttribute('aria-pressed',String(this.pick==='start'));
  this.$('peopleHelp').textContent=manual?'在地图上逐点放置人员；点击“在地图上点选人员”开始，再点一次结束。':`${MODE_LABELS[this.mode]}：修改参数后点击“生成分布”，或“换一批”更换随机种子。`;
 }
 load(people){
  if(!people||typeof people!=='object'){this.items=[];this.start=null;this.fields();this.list();return;}
  this.mode=MODE_LABELS[people.mode]?people.mode:'manual';
  this.seed=Number(people.seed)||this.seed;
  const p=people.parameters||{};
  this.parameters={count:Number(p.count)||people.items?.length||6,clusters:Number(p.clusters)||1,
   radius_m:Number(p.radius_m)||10,isolated:Number(p.isolated)||2};
  this.start=people.start?{x_m:Number(people.start.x_m),z_m:Number(people.start.z_m),cell:people.start.cell}:null;
  this.items=(people.items||[]).map(item=>({id:item.id,x_m:Number(item.x_m),z_m:Number(item.z_m),group:Number(item.group)||0,kind:item.kind||'trapped',cell:item.cell}));
  this.counter=this.items.reduce((n,p)=>{const m=/(\d+)$/.exec(String(p.id||''));return m?Math.max(n,Number(m[1])):n;},0);
  this.writeFields();this.fields();this.list();
 }
 config(){
  return {schema:'hyra-people-v1',mode:this.mode,seed:this.seed,parameters:{...this.parameters},
   start:this.start?{x_m:this.start.x_m,z_m:this.start.z_m}:null,
   items:this.items.map(({id,x_m,z_m,group,kind})=>({id,x_m,z_m,group,kind}))};
 }
 reset(){this.items=[];this.start=null;this.bad.clear();this.pick=null;this.counter=0;this.fields();this.list();}
 model(){const scene=this.getScene();return scene?buildModel(scene,this.getStructures?.()||[]):null;}
 startCell(model){return this.start?{z:Math.floor(this.start.z_m/model.dx),x:Math.floor(this.start.x_m/model.dx)}:null;}
 ensureStart(model){
  if(this.start){const cell=cellFromMetres(this.start.x_m,this.start.z_m,model);this.start.cell=[cell.z,cell.x];return {z:cell.z,x:cell.x};}
  const auto=defaultStart(model);if(auto)this.start={x_m:auto.x_m,z_m:auto.z_m,cell:[auto.z,auto.x],automatic:true};
  return auto?{z:auto.z,x:auto.x}:null;
 }
 generate(){
  if(this.blocked()){this.message('计算期间不能修改人员放置。',true);return;}
  const model=this.model();if(!model){this.message('请先应用地图，再放置人员。',true);return;}
  this.readFields();
  const start=this.ensureStart(model);if(!start){this.message('无法确定救援起点。',true);return;}
  const reach=reachableFrom(model,start);
  const result=placePeople(model,reach,{mode:this.mode,count:this.parameters.count,clusters:this.parameters.clusters,
   radius:this.parameters.radius_m,isolated:this.parameters.isolated,seed:this.seed});
  if(result.error){this.message(result.error,true);return;}
  this.items=result.items.map(p=>({id:this.nextId(),x_m:(p.cell[1]+.5)*model.dx,z_m:(p.cell[0]+.5)*model.dx,group:p.group,kind:'trapped',cell:p.cell}));
  this.changed(`已生成 ${this.items.length} 名人员（${MODE_LABELS[this.mode]}）。`);
 }
 nextId(){return `person-${++this.counter}`;}
 autoStart(){
  if(this.blocked())return;
  const model=this.model();if(!model)return;
  const auto=defaultStart(model);if(!auto){this.message('地图上没有可用的救援起点。',true);return;}
  this.start={x_m:auto.x_m,z_m:auto.z_m,cell:[auto.z,auto.x],automatic:true};this.changed('已自动选择救援起点。');
 }
 togglePick(kind){
  if(this.blocked()){this.message('计算期间不能修改人员。',true);return;}
  if(kind==='person'&&this.mode!=='manual'){this.message('请先把放置模式切换为“手动放置”。',true);return;}
  this.pick=this.pick===kind?null:kind;this.fields();
  this.message(this.pick==='start'?'点击地图设置救援起点。':this.pick==='person'?'点击地图放置人员；再次点击按钮结束。':'已退出地图点选。');
 }
 click([x_m,z_m]){
  if(!this.pick)return false;
  const model=this.model();if(!model)return false;
  if(this.pick==='start'){
   const cell=cellFromMetres(x_m,z_m,model);
   if(model.solid[cell.z][cell.x]||model.ocean[cell.z][cell.x]){this.message('起点不能落在建筑、封闭占地或海域。',true);return true;}
   this.start={x_m:(cell.x+.5)*model.dx,z_m:(cell.z+.5)*model.dx,cell:[cell.z,cell.x]};this.pick=null;
   this.changed('救援起点已更新。');return true;
  }
  if(this.items.length>=MAX_PEOPLE){this.message(`最多${MAX_PEOPLE}名人员。`,true);return true;}
  const cell=cellFromMetres(x_m,z_m,model);
  if(model.solid[cell.z][cell.x]){this.message('这里位于建筑或封闭占地内，不能放置人员。',true);return true;}
  if(model.ocean[cell.z][cell.x]){this.message('这里属于海域，请移到陆地。',true);return true;}
  const start=this.ensureStart(model),reach=reachableFrom(model,start);
  if(!reach[cell.z][cell.x]){this.message('该位置与救援起点不连通，无法到达。',true);return true;}
  this.items.push({id:this.nextId(),x_m:(cell.x+.5)*model.dx,z_m:(cell.z+.5)*model.dx,group:0,kind:'trapped',cell:[cell.z,cell.x]});
  this.changed(`已放置 ${this.items.length} 名人员。`);
  return true;
 }
 validate(){
  const model=this.model();if(!model)return {ok:true,errors:[],unreachable:[]};
  const start=this.start?cellFromMetres(this.start.x_m,this.start.z_m,model):null;
  return validate(model,start,this.items.map(p=>{
   const cell=p.cell||[Math.floor(p.z_m/model.dx),Math.floor(p.x_m/model.dx)];return {...p,cell};
  }));
 }
 assertValid(){
  if(!this.items.length)return;
  const result=this.validate();
  if(!result.ok)throw Error(result.errors.join(' '));
 }
 changed(message){
  this.list();this.dirty();this.redraw();this.summary();if(message)this.message(message);
 }
 summary(){
  if(!this.items.length){this.$('peopleStatus').textContent='尚未放置人员。';this.$('peopleStatus').classList.remove('error');this.bad.clear();return;}
  const result=this.validate();
  this.bad=new Set(result.unreachable.map(i=>this.items[i-1]?.id).filter(Boolean));
  const total=this.items.length,parts=[`${total} 人`,this.start?'起点已设置':'起点将自动选择'];
  if(result.ok)parts.push('全部合法且与起点连通');else parts.push(...result.errors);
  this.$('peopleStatus').textContent=parts.join(' · ');
  this.$('peopleStatus').classList.toggle('error',!result.ok);
 }
 list(){
  const box=this.$('peopleList');box.replaceChildren();
  const counts=this.items.reduce((map,p)=>{const key=p.group??0;map[key]=(map[key]||0)+1;return map;},{});
  this.$('peopleCountLabel').textContent=`${this.items.length} 人${Object.keys(counts).length?` · ${Object.entries(counts).map(([g,n])=>`${groupLabel(Number(g))} ${n}`).join(' · ')}`:''}`;
  this.items.forEach((p,index)=>{
   const row=document.createElement('div');row.className='structure-row people-row'+(this.bad.has(p.id)?' is-invalid':'');
   const name=document.createElement('button');name.type='button';
   const dot=document.createElement('span');dot.className='people-dot';dot.style.background=p.group<0?'#6b7c8f':GROUP_COLORS[(p.group??0)%GROUP_COLORS.length];
   name.append(dot,document.createTextNode(`人员 ${index+1}`));
   name.onclick=()=>{this.redraw();this.message(`人员 ${index+1} · X ${p.x_m.toFixed(1)} m · Z ${p.z_m.toFixed(1)} m · ${groupLabel(p.group??0)}`);};
   const text=document.createElement('small');
   text.textContent=`X ${p.x_m.toFixed(1)} m · Z ${p.z_m.toFixed(1)} m · ${groupLabel(p.group??0)}${this.bad.has(p.id)?' · 不可达':''}`;
   const remove=document.createElement('button');remove.type='button';remove.textContent='删除';remove.setAttribute('aria-label',`删除人员 ${index+1}`);
   remove.onclick=()=>{if(this.blocked()){this.message('计算期间不能修改人员。',true);return;}this.items=this.items.filter(v=>v!==p);this.changed('已删除一名人员。');};
   row.append(name,text,remove);box.append(row);
  });
  this.summary();
 }
 draw(ctx,size,time,items=this.items,start=this.start){
  const scene=this.getScene();if(!scene||!document.getElementById('showPeople')?.checked)return;
  const [nz,nx]=scene.grid.shape,dx=scene.grid.dx_m,sx=size/(nx*dx),sz=size/(nz*dx);
  const startItem=start&&Number.isFinite(start.x_m)?start:null;
  for(const p of items){
   const x=p.x_m*sx,z=p.z_m*sz,group=Number(p.group)||0,color=group<0?'#5d6f80':GROUP_COLORS[((group%GROUP_COLORS.length)+GROUP_COLORS.length)%GROUP_COLORS.length];
   const bad=items===this.items&&this.bad.has(p.id);
   ctx.save();ctx.beginPath();ctx.arc(x,z,6,0,2*Math.PI);
   ctx.fillStyle=color;ctx.globalAlpha=.92;ctx.fill();
   ctx.lineWidth=bad?3:2;ctx.strokeStyle=bad?'#d21f1f':'#ffffff';ctx.stroke();
   if(bad){ctx.beginPath();ctx.arc(x,z,10,0,2*Math.PI);ctx.setLineDash([3,3]);ctx.strokeStyle='#d21f1f';ctx.lineWidth=1.5;ctx.stroke();ctx.setLineDash([]);}
   ctx.restore();
  }
  if(startItem){
   const x=startItem.x_m*sx,z=startItem.z_m*sz;
   ctx.save();ctx.beginPath();ctx.moveTo(x,z-10);ctx.lineTo(x+8,z);ctx.lineTo(x,z+8);ctx.lineTo(x-8,z);ctx.closePath();
   ctx.fillStyle='#12a86b';ctx.fill();ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.stroke();
   ctx.fillStyle='#0b3f2c';ctx.font='bold 9px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('起',x,z-1);
   ctx.restore();
  }
 }
}
