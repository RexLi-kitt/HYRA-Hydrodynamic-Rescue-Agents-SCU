"""Compile the unmodified pinned ANUGA C helper to make weir reference vectors.

The runtime solver does not need a C compiler or ANUGA installation.
"""
import argparse
import ctypes
import _ctypes
import os
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COMMIT='e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b'
WRAPPER='''
#include "gpu_device_helpers.h"
#ifdef _WIN32
__declspec(dllexport)
#endif
void reference_weir(const double *p, double *out) {
    out[0]=p[0]; out[1]=p[1]; out[2]=p[2]; out[3]=0.;
    gpu_adjust_edgeflux_with_weir(out,p[3],p[4],9.81,p[5],p[6],
                                  .9,.95,1.,1.5,out+3);
}
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compiler',default='gcc');args=p.parse_args()
    header=ROOT/'third_party/anuga/gpu_device_helpers.h';raw=header.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!='9f2c3a7619899955f42ce62848839c590fdc826c':
        raise ValueError('ANUGA header differs from pinned source')
    rng=np.random.default_rng(20261018);inputs=[]
    for left,right,height in [(0,0,1),(.5,0,2),(0,.5,2),(1,.92,2),(1,.96,2),(2,1.4,1),(3,2,1)]:
        inputs.append([.3,.4,-.2,left,right,height,1.])
    for _ in range(150):
        inputs.append([*rng.uniform(-3,3,3),*rng.uniform(0,4,2),float(rng.uniform(.01,4)),float(rng.uniform(.05,2))])
    inputs.extend([[0,1,2,.5,0,1,1],[1e-110,1,2,.5,0,1,1]])
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp);(folder/'gpu_device_helpers.h').write_bytes(raw)
        (folder/'anuga_typedefs.h').write_text('typedef long long anuga_int;\n',encoding='utf-8')
        (folder/'reference.c').write_text(WRAPPER,encoding='utf-8')
        lib=folder/'reference.dll'
        subprocess.run([args.compiler,'-O2','-shared','-static-libgcc',str(folder/'reference.c'),'-o',str(lib)],check=True)
        loaded=ctypes.CDLL(str(lib));call=loaded.reference_weir;ptr=ctypes.POINTER(ctypes.c_double);call.argtypes=[ptr,ptr]
        vectors=[]
        for row in inputs:
            values=np.asarray(row,dtype=float);out=np.zeros(4);call(values.ctypes.data_as(ptr),out.ctypes.data_as(ptr))
            vectors.append(dict(input=row,expected=out.tolist()))
        if os.name=='nt':_ctypes.FreeLibrary(loaded._handle)
        else:_ctypes.dlclose(loaded._handle)
    target=ROOT/'tests/data/anuga-weir-vectors.json'
    target.write_text(json.dumps(dict(commit=COMMIT,header_git_blob=blob,
        source='anuga/shallow_water/gpu/gpu_device_helpers.h:gpu_adjust_edgeflux_with_weir',vectors=vectors),indent=2)+'\n',encoding='utf-8')
    print(f'Generated {len(vectors)} original-C weir vectors')


if __name__=='__main__':main()
