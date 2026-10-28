"""Read only requested actual hydraulic frames with a two-frame LRU."""
from collections import OrderedDict
from hashlib import sha256
import json,math,zipfile
from pathlib import Path
import numpy as np

def file_digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def metadata(path):
    with zipfile.ZipFile(path) as z:p=json.loads(z.read('process.json'))
    grid=p['grid'];shape=grid['shape'];frames=p['frames']
    if grid.get('axis_order')!=['z','x'] or len(shape)!=2 or any(type(n) is not int or not 2<=n<=1024 for n in shape):raise ValueError('Unsupported grid')
    if p.get('frame_fields')!=['depth','u','v','maximum_depth'] or p.get('frame_dtype')!='<f4':raise ValueError('Unsupported actual-frame format')
    if not math.isfinite(grid['dx_m']) or grid['dx_m']<=0:raise ValueError('Invalid metres per cell')
    times=np.asarray([f['time_s'] for f in frames],dtype=float)
    if len(times)<2 or times[0]!=0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid actual times')
    return p

class HydraulicFrames:
    def __init__(self,path):
        self.path=Path(path);self.process=metadata(path);self.shape=tuple(self.process['grid']['shape']);self.dx=float(self.process['grid']['dx_m'])
        self.times=np.asarray([f['time_s'] for f in self.process['frames']],dtype=float);self.digest=file_digest(path)
        self.cache=OrderedDict();self.read_indices=[]
    def index(self,time_s):
        match=np.flatnonzero(np.isclose(self.times,time_s,atol=1e-7,rtol=0))
        if len(match)!=1:raise ValueError(f'{time_s}s has no actual hydraulic frame; no interpolation or future substitution')
        return int(match[0])
    def labels(self,time_s):
        index=self.index(time_s)
        if index in self.cache:self.cache.move_to_end(index);return self.cache[index]
        row=self.process['frames'][index];name=row['url'].replace('\\','/').rsplit('/',1)[-1]
        with zipfile.ZipFile(self.path) as archive:
            if archive.getinfo(name).file_size!=16*math.prod(self.shape):raise ValueError('Invalid frame byte count')
            fields=np.frombuffer(archive.read(name),dtype='<f4').reshape(4,*self.shape)
        if not np.isfinite(fields).all() or fields[0].min()<-1e-7 or np.any(fields[3]+1e-6<fields[0]):raise ValueError('Invalid hydraulic values')
        value=fields[0]>=.05;value.setflags(write=False);self.cache[index]=value;self.read_indices.append(index)
        while len(self.cache)>2:self.cache.popitem(last=False)
        return value
