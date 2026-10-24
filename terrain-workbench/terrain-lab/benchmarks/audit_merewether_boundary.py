"""Read-only boundary audit, using compiled official 8.0 boundary/HLL functions.

Requires the checksum-verified official source ZIP in .runtime/merewether/
boundary-audit/source. Does not alter scenarios or run a full flood simulation.
"""
from pathlib import Path
import array
import ast
import hashlib
import io
import json
import math
import struct
import subprocess
import zipfile

LAB = Path(__file__).resolve().parents[1]
WORK = LAB / '.runtime/merewether/boundary-audit'
SOURCE = WORK / 'source/code/LISFLOOD-FP-trunk'
OUT = LAB.parent / 'docs/experiments/2026-10-20/merewether/input-audit'
Q = .5184
G = 9.80665  # HYRA constant: checked below, overridden from archived source.


def npy(data):
    stream = io.BytesIO(data)
    assert stream.read(6) == b'\x93NUMPY'
    version = stream.read(2)
    size = struct.unpack('<H' if version[0] == 1 else '<I', stream.read(2 if version[0] == 1 else 4))[0]
    header = ast.literal_eval(stream.read(size).decode('latin1'))
    assert not header['fortran_order']
    assert header['descr'] in ('<f8', '<f4', '|b1', '|u1')
    code = {'<f8': 'd', '<f4': 'f', '|b1': 'B', '|u1': 'B'}[header['descr']]
    values = array.array(code)
    values.frombytes(stream.read())
    return header['shape'], values


def main():
    global G
    run = LAB / '.runtime/merewether/layered-results/f4-n0.042-layered-v1'
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    archived = (run / 'source-code/flood_solver.py').read_text(encoding='utf-8')
    for line in archived.splitlines():
        if line.startswith('G ='):
            G = float(line.split('=')[1].strip())
    assert 'depth=max(float(self.h[z,x]),critical,DRY)' in archived
    assert "pressure[index] = pressure[index]*keep+.5*G*self.h[ci]**2*fraction" in archived
    # Compile the actual official function body and HLL routine, with minimal data structs.
    original = (SOURCE / 'swe/dg2.cpp').read_text()
    function = original[original.index('void dg2::set_boundary_values'):original.index('void dg2::update_intermediate_coefficients')]
    function = function.replace('void dg2::set_boundary_values', 'void set_boundary_values')
    hll = (SOURCE / 'swe/hll.cpp').read_text().replace('#include "hll.h"', '').replace('#include "flux.h"', '')
    prototype = hll[hll.index('void HLL\n'):hll.index('{', hll.index('void HLL\n'))] + ';\n'
    preamble = '''#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
using namespace std;
#define NUMERIC_TYPE double
#define C(x) (x)
#define FMAX fmax
#define FABS fabs
#define SQRT sqrt
struct Pars {};
struct TimeSeries {};
struct Solver { double DepthThresh=0.001,g=9.81,t=0; };
enum { NONE0,FREE1,HFIX2,HVAR3,QFIX4,QVAR5 };
struct BoundCs { int BC_Ident[1]={QFIX4}; double BC_Val[1]={0.5184}; TimeSeries* BC_TimeSeries[1]={nullptr}; };
double linear_interpolate(TimeSeries*,double) { return 0.; }
'''
    driver = '''int main(){
Pars p; Solver s; BoundCs b; double h,g;
cout<<setprecision(17);
while(cin>>h>>g){s.g=g; double hi=h, qi=0,ti=0,ho=0,qo=0,to=0,f=0,m=0,t=0;
set_boundary_values(&p,&s,&b,0,1,0,h,0,0,hi,qi,ti,ho,qo,to);
HLL( &s,ho,qo,to,hi,qi,ti,f,m,t);
cout<<h<<" "<<hi<<" "<<ho<<" "<<qo<<" "<<f<<" "<<m<<"\\n";}
}
'''
    build = WORK / 'build'
    build.mkdir(parents=True, exist_ok=True)
    cpp = build / 'probe.cpp'
    cpp.write_text(preamble + prototype + function + hll + driver, encoding='utf-8')
    exe = build / 'probe.exe'
    subprocess.run(['C:/msys64/ucrt64/bin/g++.exe', '-O2', '-I', '../source/code/LISFLOOD-FP-trunk/swe', 'probe.cpp', '-o', 'probe.exe'], cwd=build, check=True)
    depths = [0, .001, .01, .05, .1, .2, .3, .4, .5, 1.]
    response = subprocess.run([str(exe)], input=''.join(f'{h} {G}\n' for h in depths), text=True, capture_output=True, check=True)
    hc = (Q*Q/G)**(1/3)
    probes = []
    for line in response.stdout.splitlines():
        h, hi, ho, qo, mass, momentum = map(float, line.split())
        local_adv = Q*Q/max(h, hc)
        local_pressure = .5*G*h*h
        assert math.isclose(mass, Q, abs_tol=1e-12)
        assert math.isclose(momentum, Q*Q/hi + .5*G*hi*hi, rel_tol=1e-12)
        probes.append(dict(interior_depth_m=h, original_boundary_depth_m=hi,
            original_mass_flux_m2_s=mass, original_normal_momentum_flux_m3_s2=momentum,
            local_mass_flux_m2_s=Q, local_advective_momentum_flux_m3_s2=local_adv,
            local_pressure_flux_m3_s2=local_pressure, local_total_normal_momentum_flux_m3_s2=local_adv+local_pressure,
            local_to_original_total_ratio=(local_adv+local_pressure)/momentum))
    raw = LAB / 'generated/merewether-f4-layered-v1-n0p042/raw'
    desc = json.loads((raw / 'imported.json').read_text(encoding='utf-8'))
    faces = desc['crossings'][0]['faces']
    with zipfile.ZipFile(run / 'final.npz') as z:
        shape, depth = npy(z.read('depth.npy'))
    nz, nx = shape
    def inlet_summary(values):
        selected = [(values[f['offset']], .7*f['fraction']) for f in faces]
        local_total = sum(w*(Q*Q/max(h,hc)+.5*G*h*h) for h,w in selected)
        original_total = sum(w*(Q*Q/max(h,.0011)+.5*G*max(h,.0011)**2) for h,w in selected)
        return dict(depth_min_m=min(h for h,w in selected), depth_max_m=max(h for h,w in selected),
            width_below_critical_m=sum(w for h,w in selected if h < hc),
            width_fraction_below_critical=sum(w for h,w in selected if h < hc)/38,
            local_width_integrated_normal_momentum_flux_m4_s2=local_total,
            original_formula_on_same_cell_depths_m4_s2=original_total,
            local_to_original_on_same_depths_ratio=local_total/original_total)
    result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    replay_dir = LAB / 'generated' / result['replay_id'] / 'maps'
    replay = json.loads((replay_dir / 'replay.json').read_text(encoding='utf-8'))
    frame_summaries = []
    for frame in replay['process']['frames']:
        file = replay_dir / Path(frame['url']).name
        with file.open('rb') as stream:
            stream.seek((nz-1)*nx*4)
            row = array.array('f'); row.frombytes(stream.read(nx*4))
        frame_summaries.append(dict(time_s=frame['time_s'], **inlet_summary(row)))
    report = dict(scope='compiled official CPU DG2 boundary/HLL probes and archived inlet depth snapshots; no full-domain counterfactual',
        gravity_m_s2=G, critical_depth_m=hc, critical_velocity_m_s=Q/hc,
        original_version='LISFLOOD-FP 8.0, Zenodo 4073011; exact paper-run binary not archived',
        original_zip_md5=hashlib.md5((WORK/'source/LISFLOOD-FP-8.zip').read_bytes()).hexdigest(),
        original_source_sha256={str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'swe/dg2.cpp',SOURCE/'swe/hll.cpp',SOURCE/'swe/hll_include.h']},
        archived_local_solver_sha256=hashlib.sha256((run/'source-code/flood_solver.py').read_bytes()).hexdigest(),
        probes=probes, final_inlet=inlet_summary(depth[(nz-1)*nx:nz*nx]), snapshots=frame_summaries,
        limitations=['CPU DG2 boundary probe uses homogeneous face/interior depths and zero tangential discharge',
            'snapshots do not measure time-integrated momentum or isolate whole-domain effects',
            '8.0 reference source is verified; exact binary/commit used for 2023 paper is unverified',
            'no direct transplant of original near-dry boundary into production solver'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'boundary-checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
