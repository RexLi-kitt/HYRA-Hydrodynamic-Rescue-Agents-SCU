// Time-dependent external forcing display; uses the solver's piecewise constants.
export function discharge(points,time){let q=0;for(const [t,value] of points){if(t>time+1e-10)break;q=value;}return q;}
export function windowCurve(start,end,q){
 if(![start,end,q].every(Number.isFinite)||start<0||end<=start||end>3600||q<0||q>10000)throw Error('来水开始须早于结束，流量为0—10000 m³/s');
 return [[start,q],[end,0]];
}
export function portLabel(p){return `${p.label||p.id} · ${{inlet:'入口（推断）',outlet:'出口（推断）',uncertain:'方向待确认',forced_point:'手动点'}[p.role]||'手动点'} · (${p.x},${p.z})`;}
export function validateCurve(points){
 if(!Array.isArray(points)||!points.length||points.length>256)throw Error('来水过程需要1—256个[秒,m³/s]点');
 let previous=-1;
 return points.map(p=>{if(!Array.isArray(p)||p.length!==2||!p.every(Number.isFinite))throw Error('来水时间和流量需要是有限数值');
  const [t,q]=p;if(t<0||t>3600||t<=previous||q<0||q>10000)throw Error('时间需递增，流量为0—10000 m³/s');previous=t;return [t,q];});
}
