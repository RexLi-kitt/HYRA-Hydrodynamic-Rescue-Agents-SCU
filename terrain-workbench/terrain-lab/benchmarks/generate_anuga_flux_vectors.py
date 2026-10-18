"""Generate scalar KNP regression vectors with the original ANUGA C helper.

Pass --header pointing at the unchanged gpu_device_helpers.h from commit
e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b, together with the upstream license.
The file is hashed and compiled; only the integer mode typedef is supplied
by a small harness. ANUGA's flux/rotation/Froude code is not rewritten.
Requires a C compiler; the resulting JSON tests need only NumPy.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

COMMIT='e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b'
URL='https://raw.githubusercontent.com/anuga-community/anuga_core/'+COMMIT+'/anuga/shallow_water/gpu/gpu_device_helpers.h'
WRAPPER='''
#include "gpu_device_helpers.h"
#ifdef _WIN32
__declspec(dllexport)
#endif
void reference_flux(const double *p, int mode, double *out) {
    double ql[3] = {p[0], p[2]*p[4], p[2]*p[5]};
    double qr[3] = {p[1], p[3]*p[6], p[3]*p[7]};
    gpu_flux_function_central(ql,qr,p[0],p[1],p[2],p[3],p[8],p[9],
                             1e-12,0.0,9.81,out,out+4,out+3,mode);
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header',type=Path,required=True)
    parser.add_argument('--compiler',default='gcc')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'tests/data/anuga-knp-vectors.json')
    args=parser.parse_args()
    raw=args.header.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!='9f2c3a7619899955f42ce62848839c590fdc826c':
        raise ValueError('Header does not match the pinned ANUGA source')
    rng=np.random.default_rng(20261015)
    inputs=[]
    for i in range(16):
        original=10**rng.uniform(-4,1,2)
        clipped=original*rng.uniform(0,1,2)
        if i==0: original[:]=0.;clipped[:]=0.
        if i==1: clipped[1]=0.
        velocities=rng.uniform(-3,3,4)
        if i==0: velocities[:]=0.
        angle=rng.uniform(-np.pi,np.pi)
        normal=(np.cos(angle),np.sin(angle))
        inputs.append([*clipped,*original,*velocities,*normal])
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        (folder/'gpu_device_helpers.h').write_bytes(raw)
        (folder/'anuga_typedefs.h').write_text('typedef long long anuga_int;\n',encoding='utf-8')
        (folder/'reference.c').write_text(WRAPPER,encoding='utf-8')
        lib=folder/'reference.dll'
        subprocess.run([args.compiler,'-O2','-shared','-static-libgcc',str(folder/'reference.c'),'-o',str(lib)],check=True)
        loaded=ctypes.CDLL(str(lib))
        call=loaded.reference_flux
        ptr=ctypes.POINTER(ctypes.c_double)
        call.argtypes=[ptr,ctypes.c_int,ptr];call.restype=None
        vectors=[]
        for values in inputs:
            p=np.array(values,dtype=np.float64)
            for mode in (0,1,2):
                out=np.empty(5,dtype=np.float64)
                call(p.ctypes.data_as(ptr),mode,out.ctypes.data_as(ptr))
                vectors.append(dict(input=p.tolist(),low_froude=mode,expected=out.tolist()))
        # Release the DLL before TemporaryDirectory attempts to remove it.
        if hasattr(ctypes,'windll'):
            ctypes.windll.kernel32.FreeLibrary.argtypes=[ctypes.c_void_p]
            ctypes.windll.kernel32.FreeLibrary(loaded._handle)
        del call,loaded
    result=dict(source_url=URL,source_commit=COMMIT,
        source_sha256=hashlib.sha256(raw).hexdigest(),g=9.81,epsilon=1e-12,
        input_fields=['h_left','h_right','hle','hre','u_left','v_left','u_right','v_right','normal_x','normal_z'],
        output_fields=['mass','advective_xmomentum','advective_zmomentum','pressure','max_speed'],
        vectors=vectors)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Generated '+str(len(vectors))+' original-C reference vectors')


if __name__=='__main__':
    main()
