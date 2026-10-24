export function auditWarnings(audit){return (audit?.channels||[]).filter(c=>c.status!=='open');}
export function drawChannelAudit(ctx,audit,grid,size){
 const sx=size/(grid.shape[1]*grid.dx_m),sz=size/(grid.shape[0]*grid.dx_m);
 ctx.save();ctx.lineWidth=3;ctx.setLineDash([6,3]);
 for(const c of auditWarnings(audit)){
  const [x,z,x1,z1]=c.bounds_m;
  ctx.strokeStyle=c.status==='closed'?'#d62246':c.status==='narrowed'?'#c47900':'#676777';
  ctx.strokeRect(x*sx,z*sz,Math.max(2,(x1-x)*sx),Math.max(2,(z1-z)*sz));
 }ctx.restore();
}
export function auditSummary(audit){
 if(!audit)return '此地图尚无通道检查结果，请重新应用地图。';
 const counts={closed:0,narrowed:0,not_evaluated:0};
 for(const c of auditWarnings(audit))counts[c.status]++;
 return `检查了 ${audit.channels.length} 条建筑间候选通道：封闭 ${counts.closed}，缩窄至少 20% 的 ${counts.narrowed}，未评估 ${counts.not_evaluated}。${audit.complete?'':'检查达到资源上限，结果不完整。'}另有 ${audit.excluded_obstructed} 条因第三栋建筑实际占用而排除。`;
}
