/* Long-horizon endpoint coverage beam. Reads only confirmed samples and known targets. */
const rl=require('node:readline').createInterface({input:process.stdin});
const h=Math.SQRT1_2;
const dx=[0,0,-1,1,-h,-h,h,h,0], dz=[-1,1,0,0,-h,h,-h,h,0,0];
const known=new Uint8Array(10000), zones=new Int16Array(100);
let queue=[[],[],[],[]],last=[null,null,null,null];
let forecast=[[],[],[],[]];
const weights=new Float64Array(10000).fill(1);
function motion(x,z,a,goals){
 const u=dx[a],v=dz[a];
 let len=Math.min(50,u>0?(1000-x)/u:u<0?-x/u:50,v>0?(1000-z)/v:v<0?-z/v:50);
 for(const [tx,tz] of goals){
  const px=tx-x,pz=tz-z, r=px*px+pz*pz, dot=px*u+pz*v,disc=100-r+dot*dot;
  if(r>100+1e-8 && dot>0 && disc>=-1e-8)len=Math.min(len,dot-Math.sqrt(Math.max(0,disc)));
 }
 return [x+u*len,z+v*len];
}
function footprint(x,z,a){
 let ids=[];
 for(let j=Math.max(0,Math.ceil((z-30)/10-1e-10));j<=Math.min(99,Math.floor((z+20)/10+1e-10));j++)
 for(let i=Math.max(0,Math.ceil((x-30)/10-1e-10));i<=Math.min(99,Math.floor((x+20)/10+1e-10));i++){
  const u=i*10+5-x,v=j*10+5-z;
  if(u*u+v*v<=625+1e-8 && u*dx[a]+v*dz[a]>=-1e-8)ids.push(100*j+i);
 }
 return ids;
}
function plan(input){
 let {p,step,fixed,targets,newIds}=input;
 if(input.weights){weights.fill(1);for(const [i,w] of input.weights)weights[i]=w}
 if(step===0){known.fill(0);queue=[[],[],[],[]];last=[null,null,null,null]}
 for(const i of newIds)known[i]=1;
 let result=fixed.slice(), used=known.slice(), occupied=[];
 for(let i=0;i<4;i++)if(fixed[i]>=0){queue[i]=[];occupied.push(motion(...p[i],fixed[i],targets))}
 zones.fill(0);for(let g=0;g<10000;g++)if(!known[g])zones[Math.floor(g/1000)*10+Math.floor((g%100)/10)]++;
 const D=Math.min(input.depth,300-step), width=input.width;
 for(let k=0;k<4;k++){
  const i=(k+Math.floor(step/6))%4;
  if(fixed[i]>=0)continue;
  let [x,z]=p[i],cached=queue[i],next=cached.length?motion(x,z,cached[0],targets):null;
  if(!last[i]||Math.hypot(last[i][0]-x,last[i][1]-z)>1e-6||next&&occupied.some(o=>Math.hypot(next[0]-o[0],next[1]-o[1])<30))cached=[];
  let shared=used;
  if(input.cooperate){
   shared=used.slice();
   for(let j=0;j<4;j++)if(j!==i && fixed[j]<0){
    let pt=p[j];for(const ac of forecast[j].slice(0,8)){
     pt=motion(...pt,ac,targets);for(const g of footprint(...pt,ac))shared[g]=1;
    }
   }
  }
  if(!cached.length){
   let rarity=null;
   if(input.rarity){
    const block=new Uint16Array(400);rarity=new Float64Array(400);
    for(let g=0;g<10000;g++)if(!shared[g])block[Math.floor(g/500)*20+Math.floor((g%100)/5)]++;
    for(let b=0;b<400;b++)rarity[b]=1+input.rarity*(1-Math.min(1,block[b]/25));
   }
   const band= new Float64Array(100).fill(1);
   if(input.band){
    const side=i%2,top=i>>1;let remaining=0;
    for(let n=0;n<5;n++){
     const row=top?9-n:n;
     for(let c=side*5;c<side*5+5;c++)remaining+=zones[10*row+c];
     for(let j=0;j<10;j++)band[10*row+j]=1+.85*Math.max(0,1-remaining/1100);
    }
   }
   let beam=[{x,z,score:0,seq:[],pts:new Set()}];
   const mapcache=new Map();
   let focus,high=-1;
   for(let j=0;j<100;j++){let tx=100*(j%10)+50,tz=100*Math.floor(j/10)+50;
    let d=Math.hypot(tx-x,tz-z),near=Math.min(...p.filter((_,n)=>n!==i).map(o=>Math.hypot(tx-o[0],tz-o[1])));
    let w=zones[j]/(1+d/300)/(1+Math.max(0,d-near)/120);
    if(w>high){high=w;focus=[tx,tz]}}
   for(let l=0;l<D;l++){
    const nodes=[];
    for(const b of beam)for(let a=0;a<8;a++){
     let [xx,zz]=motion(b.x,b.z,a,targets);
     if(Math.hypot(xx-b.x,zz-b.z)<1e-8)continue;
     if(!l && occupied.some(o=>Math.hypot(xx-o[0],zz-o[1])<5))continue;
     const hash=Math.round(xx*1000)+','+Math.round(zz*1000)+','+a;
     let ids=mapcache.get(hash);
     if(!ids){ids=footprint(xx,zz,a).filter(g=>!shared[g]);mapcache.set(hash,ids)}
     let gain=0;
     if(input.edge){
      const fresh=new Set(ids.filter(g=>!b.pts.has(g)));
      for(const g of fresh){
       gain++;
       for(const j of [g%100?g-1:-1,g%100<99?g+1:-1,g-100,g+100]){
        if(!fresh.has(j))gain+=input.edge*(j<0||j>=10000||used[j]||b.pts.has(j)?1:-1);
       }
      }
     }else for(const g of ids)if(!b.pts.has(g))gain+=weights[g]*(input.band?band[Math.floor(g/1000)*10+Math.floor((g%100)/10)]:1)*(rarity?rarity[Math.floor(g/500)*20+Math.floor((g%100)/5)]:1);
     let score=b.score+gain*Math.pow(.975,l);
     let priority=score-.012*Math.hypot(xx-focus[0],zz-focus[1])-.04*Math.max(0,110-Math.min(...p.filter((_,n)=>n!==i).map(o=>Math.hypot(xx-o[0],zz-o[1]))));
     nodes.push({x:xx,z:zz,a,ids,b,score,priority,hash});
    }
    if(!nodes.length)break;
    nodes.sort((a,b)=>b.priority-a.priority);
    let take=[],visited=new Set();
    for(const n of nodes){
     if(visited.has(n.hash))continue;visited.add(n.hash);
     let pts=new Set(n.b.pts);for(const g of n.ids)pts.add(g);
     take.push({x:n.x,z:n.z,score:n.score,seq:n.b.seq.concat(n.a),pts});
     if(take.length>=width)break;
    }
    beam=take;
   }
   let winning=beam[0].seq;
   if(input.refine && winning.length>2){
    const mark=new Int32Array(10000);let ticket=0;
    function evaluate(seq){
     let pt=[x,z],score=0,discount=1;ticket++;
     for(let k=0;k<seq.length;k++){
      const a=seq[k],q=motion(...pt,a,targets);
      if(!k && occupied.some(o=>Math.hypot(q[0]-o[0],q[1]-o[1])<5))return -1e9;
      if(Math.hypot(q[0]-pt[0],q[1]-pt[1])>1e-7){
       const hash=Math.round(q[0]*1000)+','+Math.round(q[1]*1000)+','+a;
       let ids=mapcache.get(hash);
       if(!ids){ids=footprint(...q,a).filter(g=>!shared[g]);mapcache.set(hash,ids)}
       for(const g of ids)if(mark[g]!=ticket){
        mark[g]=ticket;score+=discount*weights[g]*(rarity?rarity[Math.floor(g/500)*20+Math.floor(g%100/5)]:1);
       }
      }
      pt=q;discount*=.992;
     }
     return score-.009*Math.hypot(pt[0]-focus[0],pt[1]-focus[1]);
    }
    let current=winning,best=evaluate(winning),val=best,rand=(step+1)*10007+i;
    function rnd(){rand=(Math.imul(1664525,rand)+1013904223)>>>0;return rand/4294967296}
    for(let iter=0;iter<input.refine;iter++){
     let probe=current.slice(),k=Math.floor(rnd()*probe.length),j=Math.min(probe.length-1,k+1+Math.floor(rnd()*8));
     if(rnd()<.55){[probe[k],probe[j]]=[probe[j],probe[k]]}else probe[k]=Math.floor(rnd()*8);
     let alt=evaluate(probe);
     if(alt>best){best=alt;winning=probe}
     if(alt>=val||rnd()<Math.exp((alt-val)/(.05+.7*(1-iter/input.refine)))){val=alt;current=probe}
    }
   }
   cached=winning.slice(0,input.commit);forecast[i]=winning.slice();
  }
  let a=cached.shift()??8;forecast[i].shift();result[i]=a;queue[i]=cached;last[i]=motion(x,z,a,targets);occupied.push(last[i]);
  let pt=last[i];if(a!==8)for(const g of footprint(...pt,a))used[g]=1;
  for(const b of cached){pt=motion(...pt,b,targets);for(const g of footprint(...pt,b))used[g]=1}
 }
 return result;
}
rl.on('line',line=>{try{process.stdout.write(JSON.stringify(plan(JSON.parse(line)))+'\n')}catch(e){console.error(e);process.exit(1)}});

