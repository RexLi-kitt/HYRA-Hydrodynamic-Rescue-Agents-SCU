// Metre geometry is retained independently of the chosen hydraulic grid.
export function snappedEdges(points, shape, dx){
 const [nz,nx]=shape,vertices=points.map(([x,z])=>[Math.max(0,Math.min(nx,Math.floor(x/dx+.5))),Math.max(0,Math.min(nz,Math.floor(z/dx+.5)))]),edges=new Map();
 for(let i=1;i<vertices.length;i++){
  let [x,z]=vertices[i-1],[ex,ez]=vertices[i];if(x>ex||(x===ex&&z>ez))[x,z,ex,ez]=[ex,ez,x,z];
  const [ax,az]=[x,z],[vx,vz]=[ex-x,ez-z];
  while(x!==ex||z!==ez){
   const sx=Math.sign(ex-x),sz=Math.sign(ez-z),vertical=!sx||(sz&&Math.abs(vx*(z+.5*sz-az)-vz*(x-ax))<=Math.abs(vx*(z-az)-vz*(x+.5*sx-ax)));
   let edge;if(vertical){edge=[1,Math.min(z,z+sz),x];z+=sz;}else{edge=[0,z,Math.min(x,x+sx)];x+=sx;}
   edges.set(edge.join(','),edge);
  }
 }
 return [...edges.values()];
}
export function isActive(o,time){return o.kind==='building'||o.kind==='permanent_levee'||(time+1e-10>=o.start_s&&(o.end_s==null||time<o.end_s-1e-10));}
export function rectangle(a,b,shape,dx){
 const [nz,nx]=shape,clip=(v,n)=>Math.max(0,Math.min(n,Math.floor(v/dx+.5)));
 let x0=Math.min(clip(a[0],nx),clip(b[0],nx)),x1=Math.max(clip(a[0],nx),clip(b[0],nx));
 let z0=Math.min(clip(a[1],nz),clip(b[1],nz)),z1=Math.max(clip(a[1],nz),clip(b[1],nz));
 if(x0===x1){if(x1<nx)x1++;else x0--;}
 if(z0===z1){if(z1<nz)z1++;else z0--;}
 return {x_m:x0*dx,z_m:z0*dx,width_m:(x1-x0)*dx,length_m:(z1-z0)*dx};
}

export function staticSolids(scene){
 if(scene.M_base.static_solid)return scene.M_base.static_solid;
 const buildings=(scene.M_structures?.objects||[]).filter(o=>o.kind==='building');
 if(!buildings.length)return scene.M_base.solid;
 const [nz,nx]=scene.grid.shape,dx=scene.grid.dx_m,rows=scene.M_base.solid.map(row=>Uint8Array.from(row));
 for(const o of buildings){
  const [x0,z0,x1,z1]=o.cell_bounds||[Math.floor(o.x_m/dx+1e-10),Math.floor(o.z_m/dx+1e-10),Math.ceil((o.x_m+o.width_m)/dx-1e-10),Math.ceil((o.z_m+o.length_m)/dx-1e-10)];
  for(let z=Math.max(0,z0);z<Math.min(nz,z1);z++)rows[z].fill(0,Math.max(0,x0),Math.min(nx,x1));
 }
 return rows;
}

export class StructureEditor{
 constructor({getScene,getTime,redraw,dirty,message,locked}){
  Object.assign(this,{getScene,getTime,redraw,dirty,message,locked});this.objects=[];this.points=[];this.selected=null;
  this.$=id=>document.getElementById(id);
  this.$('structureTool').onchange=()=>{this.cancel();this.selected=null;this.fields();};
  this.$('structureCrestMode').onchange=()=>this.fields();
  this.$('finishStructure').onclick=()=>this.finish();this.$('cancelStructure').onclick=()=>this.cancel();
  this.$('updateStructure').onclick=()=>this.update();
  this.fields();
 }
 load(objects){this.objects=structuredClone(objects||[]);this.selected=null;this.points=[];this.start=null;this.hover=null;this.list();}
 config(){return structuredClone(this.objects);}
 mode(){return this.$('structureTool').value;}
 cancel(){this.points=[];this.start=null;this.hover=null;this.redraw();}
 fields(){
  const selected=this.objects.find(o=>o.id===this.selected),kind=selected?.kind||this.mode(),building=kind==='building',temp=kind==='temporary_levee';
  this.$('buildingHeightField').hidden=!building;this.$('barrierFields').hidden=building||!kind.includes('levee');
  this.$('temporaryFields').hidden=!temp;
  this.$('structureCrestLabel').textContent=this.$('structureCrestMode').value==='relative'?'沿线地面以上（米）':'绝对堤顶高程（米）';
  this.$('updateStructure').disabled=!selected;
  this.$('finishStructure').hidden=!this.mode().includes('levee');
  this.$('structureHelp').textContent=building?'在地形图上拖动矩形。占地按当前计算格显示，建筑高度用于场景信息。':kind.includes('levee')?'在地形图上依次点击折线点，点击“完成堤防”。彩色实线表示实际计算格边。':'选择列表中的对象编辑属性或删除；“选择对象”模式也可点击地图选取。';
 }
 properties(kind){
  const val=id=>Number(this.$(id).value),name=this.$('structureName').value.trim();
  if(kind==='building'){
   const height=val('buildingHeight');if(!Number.isFinite(height)||height<.1||height>300)throw Error('建筑高度需要为0.1到300米');
   return {name:name||'矩形建筑',height_m:height};
  }
  const mode=this.$('structureCrestMode').value,crest=val('structureCrest'),factor=val('structureQfactor');
  if(!Number.isFinite(crest)||(mode==='relative'&&crest<=0))throw Error('请设置有效堤高／堤顶高程');
  if(!Number.isFinite(factor)||factor<.05||factor>2)throw Error('堰流系数倍数需要为0.05到2');
  const start=kind==='temporary_levee'?val('structureStart'):0,end=kind==='temporary_levee'&&this.$('structureEnd').value!==''?val('structureEnd'):null;
  if(!Number.isFinite(start)||start<0||(end!=null&&(!Number.isFinite(end)||end<=start)))throw Error('撤除时间必须晚于启用时间');
  return {name:name||(kind==='temporary_levee'?'临时防水堤':'永久堤防'),crest_mode:mode,crest_m:crest,qfactor:factor,start_s:start,end_s:end};
 }
 add(obj){
  try{if(this.locked())throw Error('请先结束或取消计算，再编辑建筑与堤防');
   if(obj.kind!=='building'){
    const scene=this.getScene(),[nz,nx]=scene.grid.shape,dx=scene.grid.dx_m,edges=snappedEdges(obj.points_m,scene.grid.shape,dx);
    if(!edges.length)throw Error('堤防短于当前计算格，请延长折线');
    if(edges.some(([axis,z,x])=>axis===1?x===0||x===nx:z===0||z===nz))throw Error('堤防不能沿地图外边界布置，请向地图内部移动');
   }
   const item={...obj,...this.properties(obj.kind),id:`structure-${crypto.randomUUID()}`};
   this.objects.push(item);this.selected=item.id;this.points=[];this.start=null;this.hover=null;this.changed();
  }catch(e){this.message(e.message,true);}
 }
 changed(){this.list();this.dirty();this.redraw();this.message('建筑与堤防草稿已更新，点击“应用设置”生效。');}
 finish(){if(this.points.length<2){this.message('请至少点击两个堤防折线点。',true);return;}this.add({kind:this.mode(),points_m:this.points.map(p=>[...p])});}
 update(){try{if(this.locked())throw Error('计算期间不能编辑结构');const o=this.objects.find(o=>o.id===this.selected);if(!o)return;Object.assign(o,this.properties(o.kind));delete o.edges;this.changed();}catch(e){this.message(e.message,true);}}
 choose(id){
  this.selected=id;const o=this.objects.find(o=>o.id===id);if(!o)return;
  this.$('structureTool').value='select';this.$('structureName').value=o.name;
  this.$('buildingHeight').value=o.height_m??6;this.$('structureCrestMode').value=o.crest_mode||'relative';
  this.$('structureCrest').value=o.crest_m??1;this.$('structureQfactor').value=o.qfactor??1;
  this.$('structureStart').value=o.start_s??0;this.$('structureEnd').value=o.end_s??'';
  this.list();this.redraw();
 }
 list(){
  const box=this.$('structureList');box.replaceChildren();
  this.$('structureCount').textContent=`${this.objects.filter(o=>o.kind==='building').length} 栋建筑 · ${this.objects.filter(o=>o.kind!=='building').length} 条堤防`;
  for(const o of this.objects){
   const row=document.createElement('div');row.className='structure-row'+(o.id===this.selected?' is-selected':'');
   const choose=document.createElement('button');choose.type='button';choose.textContent=o.name;choose.onclick=()=>this.choose(o.id);
   const text=document.createElement('small');text.textContent=o.kind==='building'?`${o.width_m} × ${o.length_m} m · 高 ${o.height_m} m`:`${o.kind==='temporary_levee'?'临时':'永久'} · ${o.crest_mode==='relative'?'增高':'堤顶'} ${o.crest_m} m${o.kind==='temporary_levee'?` · ${o.start_s}–${o.end_s??'结束'} s`:''}`;
   const remove=document.createElement('button');remove.type='button';remove.textContent='删除';remove.setAttribute('aria-label',`删除${o.name}`);
   remove.onclick=()=>{if(this.locked()){this.message('计算期间不能编辑结构',true);return;}this.objects=this.objects.filter(v=>v.id!==o.id);if(this.selected===o.id)this.selected=null;this.changed();};
   row.append(choose,text,remove);box.append(row);
  }this.fields();
 }
 down(p){if(this.mode()!=='building'||this.locked())return false;this.start=p;this.hover=p;return true;}
 move(p){if(!this.start)return;this.hover=p;this.redraw();}
 up(p){if(!this.start)return false;const scene=this.getScene(),rect=rectangle(this.start,p,scene.grid.shape,scene.grid.dx_m);this.add({kind:'building',...rect});return true;}
 click(p){
  const mode=this.mode();if(mode==='none'||mode==='building')return mode==='building';
  if(this.locked()){this.message('计算期间不能编辑结构',true);return true;}
  if(mode.includes('levee')){this.points.push(p);this.redraw();this.message(`已选 ${this.points.length} 个折线点，点击“完成堤防”。`);return true;}
  if(mode==='select'){
   const scene=this.getScene(),dx=scene.grid.dx_m;
   const hit=[...this.objects].reverse().find(o=>o.kind==='building'?p[0]>=o.x_m&&p[0]<=o.x_m+o.width_m&&p[1]>=o.z_m&&p[1]<=o.z_m+o.length_m:
    snappedEdges(o.points_m,scene.grid.shape,dx).some(([axis,z,x])=>Math.hypot(p[0]-(x+(axis===0?.5:0))*dx,p[1]-(z+(axis===1?.5:0))*dx)<dx));
   if(hit)this.choose(hit.id);return true;
  }return false;
 }
 draw(ctx,size,time,objects=this.objects){
  const scene=this.getScene();if(!scene)return;const [nz,nx]=scene.grid.shape,dx=scene.grid.dx_m,sx=size/(nx*dx),sz=size/(nz*dx);
  const draw=(o,preview=false)=>{
   ctx.save();const chosen=o.id===this.selected;ctx.strokeStyle=chosen?'#d79508':o.kind==='building'?'#293c51':o.kind==='temporary_levee'?'#db7f18':'#8b4c30';
   ctx.lineWidth=chosen?4:3;const active=isActive(o,time);if(!active||preview)ctx.setLineDash([7,5]);ctx.globalAlpha=active?1:.5;
   if(o.kind==='building'){
    ctx.fillStyle=preview?'#4d627955':'#536479';ctx.fillRect(o.x_m*sx,o.z_m*sz,o.width_m*sx,o.length_m*sz);ctx.strokeRect(o.x_m*sx,o.z_m*sz,o.width_m*sx,o.length_m*sz);
    ctx.strokeStyle='#ffffff55';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(o.x_m*sx,o.z_m*sz);ctx.lineTo((o.x_m+o.width_m)*sx,(o.z_m+o.length_m)*sz);ctx.stroke();
   }else{
    ctx.beginPath();for(const [axis,z,x] of snappedEdges(o.points_m,scene.grid.shape,dx)){ctx.moveTo(x*dx*sx,z*dx*sz);ctx.lineTo((x+(axis===0?1:0))*dx*sx,(z+(axis===1?1:0))*dx*sz);}ctx.stroke();
   }ctx.restore();
  };
  for(const o of objects)draw(o);
  if(objects!==this.objects)return;
  if(this.start&&this.hover)draw({kind:'building',...rectangle(this.start,this.hover,scene.grid.shape,dx)},true);
  if(this.points.length){ctx.save();ctx.strokeStyle='#de921b';ctx.lineWidth=3;ctx.setLineDash([6,4]);ctx.beginPath();this.points.forEach(([x,z],i)=>i?ctx.lineTo(x*sx,z*sz):ctx.moveTo(x*sx,z*sz));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#de921b';for(const [x,z] of this.points){ctx.beginPath();ctx.arc(x*sx,z*sz,4,0,2*Math.PI);ctx.fill();}ctx.restore();}
 }
}
