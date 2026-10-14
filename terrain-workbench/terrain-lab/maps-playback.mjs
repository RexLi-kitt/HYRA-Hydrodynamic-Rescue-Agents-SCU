// Pure time selection and display interpolation; no simulator mutation.
export function framePair(frames,time) {
  if (!frames.length) return null;
  const t=Math.max(frames[0].time_s,Math.min(time,frames.at(-1).time_s));
  let lo=0,hi=frames.length-1;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].time_s<=t)lo=mid;else hi=mid-1;}
  const next=Math.min(lo+1,frames.length-1),span=frames[next].time_s-frames[lo].time_s;
  return {lo,hi:next,time:t,alpha:span>0?(t-frames[lo].time_s)/span:0};
}
export function blendFrames(a,b,alpha,out=new Float32Array(a.length)) {
  if(a.length!==b.length||out.length!==a.length)throw Error('过程帧尺寸不一致');
  const t=Math.max(0,Math.min(1,alpha));
  for(let i=0;i<a.length;i++)out[i]=a[i]+(b[i]-a[i])*t;
  return out;
}
export function playbackTime(time,elapsed,speed,available) {
  return Math.min(available,time+Math.max(0,Math.min(.25,elapsed))*speed);
}
