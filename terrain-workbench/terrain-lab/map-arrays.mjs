// Binary transport changes storage/loading only; every original grid cell remains.
export function decodeRows(buffer,{dtype,shape}){
 const Type={'<f8':Float64Array,'<f4':Float32Array,'<i4':Int32Array,'|u1':Uint8Array,'|i1':Int8Array,'|b1':Uint8Array}[dtype];
 if(!Type||!Array.isArray(shape)||![1,2].includes(shape.length)||shape.some(n=>!Number.isSafeInteger(n)||n<1))throw Error('地图数组格式不正确');
 if(shape.length===1){if(buffer.byteLength!==shape[0]*Type.BYTES_PER_ELEMENT)throw Error('地图数组不完整');return new Type(buffer);}
 const [rows,cols]=shape;
 if(buffer.byteLength!==rows*cols*Type.BYTES_PER_ELEMENT)throw Error('地图数组不完整');
 const values=new Type(buffer);
 return Array.from({length:rows},(_,z)=>values.subarray(z*cols,(z+1)*cols));
}
export async function hydrateScene(scene,read=async url=>{
 const response=await fetch(url);if(!response.ok)throw Error('地图数组读取失败');return response.arrayBuffer();
}){
 const jobs=[];
 for(const name of ['M_base','M_water','M_forcing','M_mesh'])for(const [key,field] of Object.entries(scene[name]||{})){
  if(!field?.array_url)continue;
  if(!field.array_url.startsWith('/generated/'))throw Error('地图数组路径不正确');
  jobs.push((async()=>{scene[name][key]=decodeRows(await read(field.array_url),field);})());
 }
 await Promise.all(jobs);return scene;
}
