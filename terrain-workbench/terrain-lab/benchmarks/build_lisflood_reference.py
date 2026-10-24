"""Compile checksum-verified official LISFLOOD-FP 8.0 CPU source, unchanged."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess

LAB = Path(__file__).resolve().parents[1]
WORK = LAB/'.runtime/lisflood-comparison'
SRC = LAB/'.runtime/merewether/boundary-audit/source/code/LISFLOOD-FP-trunk'

def main():
    package = SRC.parents[1]/'LISFLOOD-FP-8.zip'
    assert hashlib.md5(package.read_bytes()).hexdigest() == '00602cbba617a50a34478495d2688add'
    build=WORK/'build-manual';build.mkdir(parents=True,exist_ok=True)
    cmake=(SRC/'CMakeLists.txt').read_text()
    block=cmake.split('target_sources(lisflood-base INTERFACE',1)[1].split(')',1)[0]
    sources=block.split()+['lisflood.cpp']
    flags=['-O2','-std=c++14','-fopenmp','-D_NETCDF=0','-D_NUMERIC_MODE=0','-D_ONLY_RECT=0',
           '-D_PROFILE_MODE=0','-D_DISABLE_WET_DRY=0','-D_CALCULATE_Q_MODE=1','-D_SGM_BY_BLOCKS=0','-D_BALANCE_TYPE=0']
    def compile_one(name):
        obj=build/(name.replace('/','_')+'.o')
        platform_flags=['-D__INTEL_COMPILER','-include','mm_malloc.h'] if name=='utility.cpp' else []
        args=['C:/msys64/ucrt64/bin/g++.exe',*flags,*platform_flags,'-c',name,'-o',os.path.relpath(obj,SRC)]
        r=subprocess.run(args,cwd=SRC,capture_output=True)
        return name,r.returncode,(r.stdout+r.stderr).decode('utf-8',errors='replace')
    results=list(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(compile_one,sources))
    (build/'compile.log').write_text('\n'.join(f'{n}: {rc}\n{o}' for n,rc,o in results),encoding='utf-8')
    errors=[(n,o) for n,rc,o in results if rc]
    if errors:
        for n,o in errors:print(n,o[-2500:])
        raise RuntimeError(f'{len(errors)} source files failed; see compile.log')
    objects=[os.path.relpath(build/(n.replace('/','_')+'.o'),SRC) for n in sources]
    args=['C:/msys64/ucrt64/bin/g++.exe','-fopenmp',*objects,'-lpsapi','-o',os.path.relpath(build/'lisflood.exe',SRC)]
    r=subprocess.run(args,cwd=SRC,capture_output=True)
    (build/'link.log').write_bytes(r.stdout+r.stderr)
    if r.returncode:print((r.stdout+r.stderr).decode(errors='replace'));raise RuntimeError('link failed')
    (build/'build-provenance.json').write_text(json.dumps(dict(zip_md5='00602cbba617a50a34478495d2688add',
        flags=flags,cpu_only=True,netcdf=False,source_changes=False,
        allocator_portability='utility.cpp selects existing _mm_malloc/_mm_free branch via __INTEL_COMPILER and mm_malloc.h; no solver changes',
        source_sha256={n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in sources},
        executable_sha256=hashlib.sha256((build/'lisflood.exe').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print('Built official CPU executable',flush=True)

if __name__=='__main__':main()
