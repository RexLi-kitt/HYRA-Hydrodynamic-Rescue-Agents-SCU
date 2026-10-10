'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const c3 = $('terrain3d'), c2 = $('terrain2d'), profile = $('profile');
  const stops = [[40,94,104],[83,158,131],[172,195,145],[224,209,158],[180,138,102],[238,229,216]];
  let data, low, high, average, selected, view = '3d', busy = false, queued = false;
  let yaw = -.6, pitch = .72, distance = 1.7, drag, moved = false;
  let gl, program, mesh, vertices, indices, eye, target, right, up, forward;
  const status = (s, error = false) => { $('status').textContent = s; $('status').classList.toggle('error', error); };
  const notice = '高程单位尚未标定为米。水域高程为水底；初始水体来自地形生成器。';
  const color = h => { const t = Math.max(0, Math.min(.999999, (h - low) / Math.max(1, high - low))) * (stops.length - 1), k = Math.floor(t); return stops[k].map((v, j) => (v + (stops[k + 1][j] - v) * (t - k)) / 255); };
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const unit = a => { const n = Math.hypot(...a); return a.map(v => v / n); };
  const exaggeration = () => Number($('exaggeration').value);
  function fit(canvas) {
    const rect = canvas.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width * dpr)), height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    return [width, height, dpr];
  }
  function shader(type, source) {
    const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function initGL() {
    gl = c3.getContext('webgl', {antialias: true, preserveDrawingBuffer: true, alpha: true});
    if (!gl) { $('view3d').disabled = true; setView('2d'); status('此浏览器未提供三维渲染，已切换二维高程图。'); return; }
    program = gl.createProgram();
    gl.attachShader(program, shader(gl.VERTEX_SHADER, `attribute vec3 aPosition; attribute vec3 aNormal; attribute vec3 aColor; attribute float aHeight;
      uniform mat4 uMatrix; varying vec3 vNormal; varying vec3 vColor; varying float vHeight;
      void main(){gl_Position=uMatrix*vec4(aPosition,1.0);gl_PointSize=11.0;vNormal=aNormal;vColor=aColor;vHeight=aHeight;}`));
    gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `precision highp float; varying vec3 vNormal; varying vec3 vColor; varying float vHeight;
      uniform float uInterval; uniform bool uContours; uniform bool uMarker;
      void main(){if(uMarker){gl_FragColor=vec4(0.94,0.48,0.2,1.0);return;}
      float light=0.70+0.30*max(0.0,dot(normalize(vNormal),normalize(vec3(-0.5,1.0,0.4))));
      vec3 color=vColor*light;float band=abs(fract(vHeight/uInterval+0.5)-0.5);
      if(uContours && band<0.022)color=mix(color,vec3(0.18,0.30,0.23),0.25);gl_FragColor=vec4(color,1.0);}`));
    gl.linkProgram(program); if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program); gl.enable(gl.DEPTH_TEST); mesh = gl.createBuffer(); indices = gl.createBuffer();
    program.attrs = ['aPosition','aNormal','aColor','aHeight'].map(n => gl.getAttribLocation(program,n));
    program.uniforms = Object.fromEntries(['uMatrix','uInterval','uContours','uMarker'].map(n => [n, gl.getUniformLocation(program,n)]));
  }
  function point(x, z) {
    const [nz, nx] = data.shape, size = Math.max(nx - 1, nz - 1) * data.step;
    return [(x - (nx-1)/2)*data.step/size, (data.heights[z*nx+x]-low)*exaggeration()/size, (z-(nz-1)/2)*data.step/size];
  }
  function buildMesh() {
    if (!gl || !data) return;
    const [nz, nx] = data.shape;
    const xs = Array.from({length:Math.min(nx,256)},(_,i)=>Math.round(i*(nx-1)/(Math.min(nx,256)-1)));
    const zs = Array.from({length:Math.min(nz,256)},(_,i)=>Math.round(i*(nz-1)/(Math.min(nz,256)-1)));
    vertices = {xs,zs, points:[]}; const packed = [];
    for (const z of zs) for (const x of xs) {
      const pos = point(x,z), i = z*nx+x;
      const dx = (data.heights[z*nx+Math.min(nx-1,x+1)]-data.heights[z*nx+Math.max(0,x-1)]) / ((x>0&&x<nx-1?2:1)*data.step);
      const dz = (data.heights[Math.min(nz-1,z+1)*nx+x]-data.heights[Math.max(0,z-1)*nx+x]) / ((z>0&&z<nz-1?2:1)*data.step);
      const normal = unit([-dx*exaggeration(),1,-dz*exaggeration()]);
      packed.push(...pos,...normal,...color(data.heights[i]),data.heights[i]); vertices.points.push(pos);
    }
    const ids = []; for (let z=0;z<zs.length-1;z++) for(let x=0;x<xs.length-1;x++) {
      const i=z*xs.length+x; ids.push(i,i+xs.length,i+1,i+1,i+xs.length,i+xs.length+1);
    }
    vertices.ids=ids;
    gl.bindBuffer(gl.ARRAY_BUFFER,mesh); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(packed),gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indices); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(ids),gl.STATIC_DRAW);
  }
  function matrix() {
    const [nz,nx]=data.shape, size=Math.max(nx-1,nz-1)*data.step;
    target=[0,(average-low)*exaggeration()/size,0];
    eye=[Math.sin(yaw)*Math.cos(pitch)*distance,target[1]+Math.sin(pitch)*distance,Math.cos(yaw)*Math.cos(pitch)*distance];
    forward=unit(sub(target,eye));right=unit(cross(forward,[0,1,0]));up=cross(right,forward);
    const v=[right[0],up[0],-forward[0],0,right[1],up[1],-forward[1],0,right[2],up[2],-forward[2],0,-dot(right,eye),-dot(up,eye),dot(forward,eye),1];
    const f=1/Math.tan(Math.PI/8), near=.01,far=100;
    const p=[f/(c3.width/c3.height),0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0];
    const out=new Float32Array(16);
    for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)out[c*4+r]+=p[k*4+r]*v[c*4+k];
    return out;
  }
  function attributes(buffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    for(let i=0;i<4;i++){gl.enableVertexAttribArray(program.attrs[i]);gl.vertexAttribPointer(program.attrs[i],i===3?1:3,gl.FLOAT,false,40,[0,12,24,36][i]);}
  }
  function draw3D() {
    if(!gl||!data||!vertices)return;
    fit(c3);gl.viewport(0,0,c3.width,c3.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.useProgram(program);gl.uniformMatrix4fv(program.uniforms.uMatrix,false,matrix());
    gl.uniform1f(program.uniforms.uInterval,Number($('contourInterval').value));gl.uniform1i(program.uniforms.uContours,$('contours').checked);gl.uniform1i(program.uniforms.uMarker,0);
    attributes(mesh);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indices);gl.drawElements(gl.TRIANGLES,vertices.ids.length,gl.UNSIGNED_SHORT,0);
    if(selected){const p=point(selected.x,selected.z);p[1]+=.004;const marker=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,marker);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([...p,0,1,0,1,.5,.2,0]),gl.STREAM_DRAW);attributes(marker);gl.uniform1i(program.uniforms.uMarker,1);gl.drawArrays(gl.POINTS,0,1);gl.deleteBuffer(marker);}
  }
  function bounds2D() {
    const [nz,nx]=data.shape, padding=55, w=c2.width-2*padding, h=c2.height-2*padding;
    const scale=Math.min(w/nx,h/nz);return {x:(c2.width-nx*scale)/2,y:(c2.height-nz*scale)/2,w:nx*scale,h:nz*scale};
  }
  function draw2D() {
    if(!data)return;fit(c2);const ctx=c2.getContext('2d'), [nz,nx]=data.shape;
    const image=new ImageData(nx,nz),interval=Number($('contourInterval').value);
    for(let i=0;i<data.heights.length;i++){
      let rgb=color(data.heights[i]);
      if($('contours').checked){const x=i%nx,z=Math.floor(i/nx),h=Math.floor(data.heights[i]/interval);
        if(x>0&&h!==Math.floor(data.heights[i-1]/interval)||z>0&&h!==Math.floor(data.heights[i-nx]/interval))rgb=rgb.map(v=>v*.80);}
      image.data.set([...rgb.map(v=>Math.round(v*255)),255],i*4);
    }
    const buffer=document.createElement('canvas');buffer.width=nx;buffer.height=nz;buffer.getContext('2d').putImageData(image,0,0);
    const b=bounds2D();ctx.clearRect(0,0,c2.width,c2.height);ctx.imageSmoothingEnabled=false;ctx.drawImage(buffer,b.x,b.y,b.w,b.h);
    if(selected){const y=b.y+(selected.z+.5)*b.h/nz,x=b.x+(selected.x+.5)*b.w/nx;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.setLineDash([7,5]);ctx.beginPath();ctx.moveTo(b.x,y);ctx.lineTo(b.x+b.w,y);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#ef7b33';ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fill();ctx.stroke();}
    ctx.fillStyle='#536d60';ctx.font='12px sans-serif';ctx.fillText('X →',b.x+b.w-28,b.y+b.h+24);ctx.fillText('Z ↓',b.x-34,b.y+b.h/2);
  }
  function drawProfile() {
    if(!data||!selected)return;const [w,h,dpr]=fit(profile),ctx=profile.getContext('2d');ctx.clearRect(0,0,w,h);
    ctx.save();ctx.scale(dpr,dpr);const width=w/dpr,height=h/dpr,[,nx]=data.shape,row=selected.z;
    const margin={l:48,r:18,t:22,b:27},pw=width-margin.l-margin.r,ph=height-margin.t-margin.b,range=Math.max(1,high-low);
    const y=v=>margin.t+ph-(v-low)/range*ph,x=i=>margin.l+i/(nx-1)*pw;
    ctx.font='10px sans-serif';ctx.fillStyle='#798b80';ctx.strokeStyle='#e8ede6';ctx.lineWidth=1;
    for(let k=0;k<4;k++){const v=low+range*k/3;ctx.fillText(v.toFixed(0),5,y(v)+3);ctx.beginPath();ctx.moveTo(margin.l,y(v));ctx.lineTo(width-margin.r,y(v));ctx.stroke();}
    const gradient=ctx.createLinearGradient(0,margin.t,0,margin.t+ph);gradient.addColorStop(0,'#82ad8b80');gradient.addColorStop(1,'#82ad8b08');
    ctx.beginPath();ctx.moveTo(x(0),margin.t+ph);for(let i=0;i<nx;i++)ctx.lineTo(x(i),y(data.heights[row*nx+i]));ctx.lineTo(x(nx-1),margin.t+ph);ctx.closePath();ctx.fillStyle=gradient;ctx.fill();
    ctx.beginPath();for(let i=0;i<nx;i++){const v=data.heights[row*nx+i];i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v));}ctx.strokeStyle='#3f8065';ctx.lineWidth=1.6;ctx.stroke();
    ctx.fillStyle='#ef7b33';ctx.beginPath();ctx.arc(x(selected.x),y(data.heights[row*nx+selected.x]),3.5,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#798b80';ctx.textAlign='left';ctx.fillText(`X ${data.origin[0]}`,margin.l,height-6);ctx.textAlign='right';ctx.fillText(`X ${data.origin[0]+(nx-1)*data.step}`,width-margin.r,height-6);ctx.restore();
  }
  function render(){queued=false;if(!data)return;view==='3d'?draw3D():draw2D();drawProfile();}
  function requestRender(){if(!queued){queued=true;requestAnimationFrame(render);}}
  function select(x,z) {
    const [nz,nx]=data.shape;x=Math.max(0,Math.min(nx-1,x));z=Math.max(0,Math.min(nz-1,z));selected={x,z};const i=z*nx+x;
    $('cellIndex').textContent=`${x}, ${z}`;$('cellWorld').textContent=`${data.origin[0]+x*data.step}, ${data.origin[1]+z*data.step}`;
    $('cellHeight').textContent=data.heights[i].toFixed(2);const water=data.water?.[i];
    $('cellDepth').textContent=Number.isFinite(water)?`${Math.max(0,water-data.heights[i]).toFixed(2)}`:data.full?'0.00':'未导出';
    $('profileRow').value=z;$('rowValue').textContent=z;$('profileInfo').textContent=`世界 Z = ${data.origin[1]+z*data.step} · 橙色点为选中位置 · 原始高程`;
    requestRender();
  }
  function setView(next){view=next;c3.hidden=next!=='3d';c2.hidden=next!=='2d';for(const k of ['3d','2d']){$('view'+k).classList.toggle('active',k===next);$('view'+k).setAttribute('aria-pressed',String(k===next));}$('viewLabel').textContent=next==='3d'?'三维地形 / 透视视图':'二维高程 / 俯视图';$('interactionHelp').textContent=next==='3d'?'拖动旋转 · 滚轮缩放 · 点击查点':'点击查点 · 虚线表示剖面位置';$('exaggeration').disabled=next==='2d';requestRender();}
  async function read(url){const r=await fetch(url);if(!r.ok)throw new Error(`地图读取失败（${r.status}）`);return r;}
  async function load(){
    if(busy)return;busy=true;$('refresh').disabled=true;$('saveImage').disabled=true;$('saveMatrix').disabled=true;$('empty').hidden=false;$('empty').textContent='正在加载高程数据…';
    try{
      let next;
      if(window.ELEVATION_DATA) next=window.ELEVATION_DATA;
      else {
        const m=await (await read('/api/current')).json(),field=m.fields[m.full?'ground_height':'height'];
        const buf=await (await read(field.url)).arrayBuffer(),a=field.dtype==='<f4'?new Float32Array(buf):new Int32Array(buf);
        let water=null;if(m.full)water=new Float32Array(await (await read(m.fields.water_surface.url)).arrayBuffer());
        next={shape:m.shape,origin:m.origin,step:m.step,seed:m.config.seed,full:m.full,heights:Array.from(a,v=>v*(m.full?1:m.scale)),water};
      }
      const [nz,nx]=next.shape||[];
      if(!Number.isInteger(nx)||!Number.isInteger(nz)||nx<2||nz<2||next.heights?.length!==nx*nz||!next.heights.every(Number.isFinite)||!Number.isFinite(next.step)||next.step<=0||!Array.isArray(next.origin)||next.origin.length!==2||!next.origin.every(Number.isFinite)||next.water&&next.water.length!==nx*nz)throw new Error('高程数据格式不完整');
      data=next;low=Infinity;high=-Infinity;let sum=0;for(const v of data.heights){low=Math.min(low,v);high=Math.max(high,v);sum+=v;}average=sum/data.heights.length;
      for(const [id,v] of [['minimum',low],['maximum',high],['average',average],['relief',high-low]])$(id).textContent=v.toFixed(1);
      $('legendMin').textContent=low.toFixed(1);$('legendMax').textContent=high.toFixed(1);
      $('mapInfo').textContent=`种子 ${data.seed} · ${nx} × ${nz} 网格 · 采样间距 ${data.step} · X / Z 跨度 ${(nx-1)*data.step} / ${(nz-1)*data.step} 方块单位`;
      $('sourceLabel').textContent=data.full?'实际地面高程':'连续地形高程';$('profileRow').max=nz-1;
      buildMesh();select(Math.floor(nx/2),Math.floor(nz/2));$('empty').hidden=true;status(notice);
    }catch(e){$('empty').hidden=false;$('empty').textContent=e.message;status(`读取失败：${e.message}`,true);}
    finally{busy=false;$('refresh').disabled=Boolean(window.ELEVATION_DATA);$('saveImage').disabled=!data;$('saveMatrix').disabled=!data;}
  }
  function rayPick(event){
    if(!vertices||!data||busy)return;const rect=c3.getBoundingClientRect(),sx=(event.clientX-rect.left)/rect.width*2-1,sy=1-(event.clientY-rect.top)/rect.height*2;
    const t=Math.tan(Math.PI/8),ray=unit(forward.map((v,i)=>v+right[i]*sx*(rect.width/rect.height)*t+up[i]*sy*t));
    let nearest=Infinity, hit;
    for(let i=0;i<vertices.ids.length;i+=3){const a=vertices.points[vertices.ids[i]],b=vertices.points[vertices.ids[i+1]],c=vertices.points[vertices.ids[i+2]];
      const e1=sub(b,a),e2=sub(c,a),p=cross(ray,e2),det=dot(e1,p);if(Math.abs(det)<1e-8)continue;
      const s=sub(eye,a),u=dot(s,p)/det;if(u<0||u>1)continue;const q=cross(s,e1),v=dot(ray,q)/det;if(v<0||u+v>1)continue;
      const d=dot(e2,q)/det;if(d>0&&d<nearest){nearest=d;hit=eye.map((v,i)=>v+ray[i]*d);}
    }
    if(hit){const [nz,nx]=data.shape,size=Math.max(nx-1,nz-1)*data.step;select(Math.round(hit[0]*size/data.step+(nx-1)/2),Math.round(hit[2]*size/data.step+(nz-1)/2));}
  }
  c3.addEventListener('pointerdown',e=>{if(!data)return;drag={x:e.clientX,y:e.clientY};moved=false;c3.setPointerCapture(e.pointerId);});
  c3.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;yaw-=dx*.006;pitch=Math.max(.12,Math.min(1.45,pitch+dy*.006));drag={x:e.clientX,y:e.clientY};requestRender();});
  c3.addEventListener('pointerup',e=>{if(drag&&!moved)rayPick(e);drag=null;});c3.addEventListener('pointercancel',()=>{drag=null;});
  c3.addEventListener('wheel',e=>{if(!data)return;e.preventDefault();distance=Math.max(.65,Math.min(6,distance*Math.exp(e.deltaY*.001)));requestRender();},{passive:false});
  c3.addEventListener('keydown',e=>{if(!data)return;let handled=true;switch(e.key){case 'ArrowLeft':yaw-=.1;break;case 'ArrowRight':yaw+=.1;break;case 'ArrowUp':pitch=Math.min(1.45,pitch+.1);break;case 'ArrowDown':pitch=Math.max(.12,pitch-.1);break;case '+':case '=':distance=Math.max(.65,distance*.9);break;case '-':distance=Math.min(6,distance*1.1);break;default:handled=false;}if(handled){e.preventDefault();requestRender();}});
  c2.addEventListener('click',e=>{if(!data||busy)return;const rect=c2.getBoundingClientRect(),x=(e.clientX-rect.left)*c2.width/rect.width,y=(e.clientY-rect.top)*c2.height/rect.height,b=bounds2D();if(x<b.x||x>=b.x+b.w||y<b.y||y>=b.y+b.h)return;select(Math.floor((x-b.x)/b.w*data.shape[1]),Math.floor((y-b.y)/b.h*data.shape[0]));});
  c2.addEventListener('keydown',e=>{if(!selected)return;const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];if(d){e.preventDefault();select(selected.x+d[0],selected.z+d[1]);}});
  $('view3d').onclick=()=>setView('3d');$('view2d').onclick=()=>setView('2d');$('refresh').onclick=load;
  $('exaggeration').oninput=()=>{$('exaggerationValue').textContent=`${exaggeration().toFixed(1)}×`;buildMesh();requestRender();};
  $('contours').onchange=requestRender;$('contourInterval').onchange=requestRender;
  $('reset').onclick=()=>{yaw=-.6;pitch=.72;distance=1.7;requestRender();};
  $('profileRow').oninput=()=>{if(data)select(selected.x,Number($('profileRow').value));};
  function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $('saveMatrix').onclick=()=>{if(!data)return;const [,nx]=data.shape,rows=[];for(let z=0;z<data.shape[0];z++)rows.push(Array.from(data.heights.slice(z*nx,(z+1)*nx)));download(new Blob([JSON.stringify(rows)+'\n'],{type:'application/json'}),`height-matrix-${data.seed}.json`);status(`已导出 ${data.shape[1]} × ${data.shape[0]} 原始高程矩阵；H[行,列] = H[z,x]，单位为方块单位。`);};
  $('saveImage').onclick=()=>{if(!data)return;render();const source=view==='3d'?c3:c2,canvas=document.createElement('canvas');canvas.width=source.width;canvas.height=source.height+92;const ctx=canvas.getContext('2d');ctx.fillStyle='#f4f7f0';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(source,0,0);ctx.fillStyle='#254b3a';ctx.font='14px sans-serif';ctx.fillText(`HYRA · 种子 ${data.seed} · ${view==='3d'?'三维高程 / 夸张 '+exaggeration()+'×':'二维高程'}`,22,source.height+28);ctx.font='12px sans-serif';ctx.fillText(`高程 ${low.toFixed(1)} ～ ${high.toFixed(1)} 方块单位 · 尚未标定为米`,22,source.height+53);ctx.fillText(`网格 ${data.shape[1]} × ${data.shape[0]} · 间距 ${data.step} · ${data.full?'实际地面高程':'连续地形高程'}`,22,source.height+75);canvas.toBlob(blob=>{if(blob)download(blob,`elevation-${data.seed}-${view}.png`);});};
  new ResizeObserver(requestRender).observe($('stage'));new ResizeObserver(requestRender).observe(profile);
  $('saveImage').disabled=true;$('saveMatrix').disabled=true;
  try{initGL();}catch(e){$('view3d').disabled=true;gl=null;setView('2d');status('三维渲染初始化失败，已切换二维高程图。',true);}
  load();
})();
