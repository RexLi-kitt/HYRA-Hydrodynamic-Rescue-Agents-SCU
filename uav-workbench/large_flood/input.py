"""Read actual workbench hydraulic frames without future interpolation."""
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np


class FloodInput:
    def __init__(self, path, shape=(240, 240), dx=5., duration=1800.):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as z:
            self.process = json.loads(z.read('process.json'))
            p = self.process; grid = p['grid']
            if tuple(grid['shape']) != tuple(shape) or grid.get('axis_order') != ['z','x'] or grid['dx_m'] != dx:
                raise ValueError('洪水网格必须为240×240、5米、[z,x]')
            if p.get('frame_fields') != ['depth','u','v','maximum_depth'] or p.get('frame_dtype') != '<f4':
                raise ValueError('洪水过程必须包含小端 float32 的 depth/u/v/maximum_depth')
            units = p.get('units')
            if units is not None:
                if units != dict(time='s', depth='m', velocity='m/s', distance='m'):
                    raise ValueError('洪水过程单位不匹配')
            else:
                dictionary = json.loads(z.read('data_dictionary.json'))
                f = dictionary.get('frames', {})
                if f.get('order') != '[field,z,x]' or not p.get('config', {}).get('metres_per_block'):
                    raise ValueError('缺少可确认的洪水单位和帧排列')
            self.shape = tuple(shape); self.dx = dx
            rows = p['frames']
            times = np.array([float(row['time_s']) for row in rows])
            if len(times) < 2 or not np.isfinite(times).all() or times[0] != 0 or not np.allclose(np.diff(times), 5., rtol=0, atol=1e-7) or times[-1] < duration:
                raise ValueError('需要从0秒开始、连续5秒间隔并覆盖1800秒的实际帧')
            names = [str(row['url']).replace('\\','/').rsplit('/',1)[-1] for row in rows]
            if len(set(names)) != len(names): raise ValueError('帧文件名重复')
            for i, (row, name) in enumerate(zip(rows,names)):
                if row.get('index', i) != i or z.getinfo(name).file_size != 16*math.prod(shape):
                    raise ValueError('洪水帧索引或长度错误')
            self.times, self.names = times, names
        h=hashlib.sha256()
        with self.path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
        self.sha256=h.hexdigest(); self._last = None

    def frame(self, t):
        if not math.isfinite(t) or t < 0 or t > self.times[-1]: raise ValueError('观测时刻超出洪水过程')
        index=int(np.searchsorted(self.times, t + 1e-9, side='right')-1)
        if self._last is None or self._last[0] != index:
            with zipfile.ZipFile(self.path) as z:
                a=np.frombuffer(z.read(self.names[index]),dtype='<f4').reshape((4,*self.shape))
            if not np.isfinite(a).all() or a[0].min() < -1e-7 or np.any(a[3]+1e-6<a[0]):
                raise ValueError('洪水帧包含非法数值')
            self._last=(index,a[0].copy())
        return index, float(self.times[index]), self._last[1]
