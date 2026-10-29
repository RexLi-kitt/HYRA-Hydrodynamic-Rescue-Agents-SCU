'use strict';
const $ = id => document.getElementById(id);
const canvas = $('map');
const ctx = canvas.getContext('2d');
const offscreen = document.createElement('canvas');
const offctx = offscreen.getContext('2d');
let replay = null, playing = false, lastTick = null, accumulated = 0;
let states = null, applied = 0, appliedTime = -1;
const colors = ['#ed6b43','#d9a83b','#725da9','#338e85','#bf658c','#487bb7','#8b9a37','#996e4d','#db8a68','#8273bf','#588f9d','#b58138'];
function updateStates(t) {
  if (!states || t < appliedTime) { states = new Int8Array(240*240); states.fill(-1); applied = 0; }
  while (applied < replay.observations.length && replay.observations[applied].t <= t + 1e-6) {
    const o = replay.observations[applied++], [r0,r1,c0,c1] = o.bounds, bytes = atob(o.wet_bits);
    let k = 0;
    for (let r=r0; r<r1; r++) for (let c=c0; c<c1; c++,k++) states[r*240+c] = (bytes.charCodeAt(k>>3) >> (7-(k%8))) & 1;
  }
  appliedTime = t;
}
function render() {
  if (!replay) return;
  const index = Number($('time').value), frame = replay.trajectories[index], t = frame.t;
  updateStates(t);
  const n = 240, pixels = ctx.createImageData(n,n);
  for (let i=0; i<states.length; i++) {
    const rgb = states[i] < 0 ? [204,217,219] : states[i] === 1 ? [41,133,180] : [232,237,220];
    pixels.data[i*4]=rgb[0]; pixels.data[i*4+1]=rgb[1]; pixels.data[i*4+2]=rgb[2]; pixels.data[i*4+3]=255;
  }
  offscreen.width=n; offscreen.height=n; offctx.putImageData(pixels,0,0);
  ctx.imageSmoothingEnabled=false; ctx.clearRect(0,0,720,720); ctx.drawImage(offscreen,0,0,720,720);
  const scale=720/1200;
  for (let agent=0; agent<frame.positions.length; agent++) {
    ctx.beginPath();ctx.strokeStyle=colors[agent%colors.length];ctx.globalAlpha=.65;ctx.lineWidth=1.3;
    for (let j=0; j<=index; j++) { const [x,z]=replay.trajectories[j].positions[agent]; if(j===0)ctx.moveTo(x*scale,z*scale);else ctx.lineTo(x*scale,z*scale); }
    ctx.stroke(); ctx.globalAlpha=1;
    const [x,z]=frame.positions[agent];ctx.beginPath();ctx.arc(x*scale,z*scale,5,0,Math.PI*2);ctx.fillStyle=colors[agent%colors.length];ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
  }
  $('clock').textContent=`${t} / ${replay.config.duration_s} s`;
  $('previous').disabled=index===0; $('next').disabled=index===replay.trajectories.length-1;
}
function pause() { playing=false; lastTick=null; accumulated=0; $('play').textContent='播放'; }
$('time').addEventListener('input',()=>{pause();render()});
$('previous').addEventListener('click',()=>{pause();$('time').value=Math.max(0,Number($('time').value)-1);render()});
$('next').addEventListener('click',()=>{pause();$('time').value=Math.min(replay.trajectories.length-1,Number($('time').value)+1);render()});
$('play').addEventListener('click',()=>{if(!replay)return;if(playing){pause();return}if(Number($('time').value)>=replay.trajectories.length-1){$('time').value=0;render()}playing=true;lastTick=null;$('play').textContent='暂停'});
function tick(now){if(playing&&replay){if(lastTick!==null)accumulated+=(now-lastTick)/1000*Number($('rate').value);lastTick=now;let i=Number($('time').value);while(i<replay.trajectories.length-1&&accumulated>=replay.trajectories[i+1].t-replay.trajectories[i].t){accumulated-=replay.trajectories[i+1].t-replay.trajectories[i].t;i++}if(i!==Number($('time').value)){$('time').value=i;render()}if(i===replay.trajectories.length-1)pause()}requestAnimationFrame(tick)}requestAnimationFrame(tick);
$('form').addEventListener('submit',async e=>{
  e.preventDefault();pause();$('run').disabled=true;$('status').classList.remove('error');$('status').textContent='正在计算观测过程，请稍候…';$('run-state').textContent='运行中';
  try {
    const response=await fetch('/api/large-flood/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:$('input').value,agents:Number($('agents').value),speed:Number($('speed').value),footprint:Number($('footprint').value)})});
    const out=await response.json();if(!response.ok)throw new Error(out.error||'运行失败');
    const r=await fetch(out.replay_url);if(!r.ok)throw new Error('无法读取回放');replay=await r.json();states=null;appliedTime=-1;
    $('time').max=replay.trajectories.length-1;$('time').value=0;$('time').disabled=false;$('play').disabled=false;$('previous').disabled=true;$('next').disabled=false;$('empty').hidden=true;
    $('download').href=out.replay_url;$('download').download='large-flood-replay.json';$('download').hidden=false;
    const s=out.summary;$('metricCoverage').textContent=`${(s.coverage*100).toFixed(1)}%`;$('metricCorrect').textContent=`${(s.current_correct_coverage_mean*100).toFixed(1)}%`;$('metricAge').textContent=`${s.mean_observed_age_s.toFixed(0)} s`;$('metricDistance').textContent=`${(s.total_distance_m/1000).toFixed(1)} km`;
    $('status').textContent='回放已生成，可以播放或拖动时间轴。';$('run-state').textContent='回放已就绪';$('summary').textContent=`${replay.config.agents} 架无人机 · ${replay.config.speed_m_s} m/s · ${replay.config.footprint_m} 米观测宽度 · 计算用时 ${s.elapsed_s.toFixed(2)} 秒`;render();
  } catch(error) {$('status').textContent=error.message;$('status').classList.add('error');$('run-state').textContent='运行失败'}
  finally {$('run').disabled=false}
});
