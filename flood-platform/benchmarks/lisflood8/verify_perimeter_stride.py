"""Compile the actual perimeter kernel against bounds-checked storage.

Only the test function name is changed. No equation is reimplemented by the
test; its oracle specifies boundary values and preservation of interior cells.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess

PREAMBLE=r'''
#include <vector>
#include <array>
#include <iostream>
#include <cstddef>
#define C(x) (x)
struct Buffer {
    std::vector<double> v;
    int bad=0;
    double sink=0;
    double& operator[](int k) {
        if (k<0 || k>=static_cast<int>(v.size())) { ++bad; return sink; }
        return v[k];
    }
};
struct Pars {int xsz,ysz;};
struct Arrays {std::vector<double> DEM1x,DEM1y;};
struct FlowCoefficients {Buffer &H1x,&HU1x,&HV1x,&H1y,&HU1y,&HV1y;};
'''
HARNESS=r'''
using Kernel=void(*)(Pars*,Arrays*,FlowCoefficients const&);
void check(Kernel kernel,const char* label,int nx,int ny) {
    Pars p{nx,ny}; Arrays a;
    a.DEM1x.resize(nx*ny);a.DEM1y.resize(nx*ny);
    std::array<Buffer,6> f;
    for (int c=0;c<6;c++) {
        f[c].v.resize(nx*ny);
        for(int k=0;k<nx*ny;k++) f[c].v[k]=100000.0*(c+1)+k;
    }
    for(int k=0;k<nx*ny;k++){a.DEM1x[k]=1+k/1000.;a.DEM1y[k]=2+k/1000.;}
    auto expected=f;
    for(int j=0;j<ny;j++) for(int i=0;i<nx;i++) {
        int k=j*nx+i;
        if(i==0 || i==nx-1) {expected[0].v[k]=-a.DEM1x[k];expected[1].v[k]=expected[2].v[k]=0;}
        if(j==0 || j==ny-1) {expected[3].v[k]=-a.DEM1y[k];expected[4].v[k]=expected[5].v[k]=0;}
    }
    FlowCoefficients u{f[0],f[1],f[2],f[3],f[4],f[5]};kernel(&p,&a,u);
    int bad=0,mismatch=0,interior=0;
    for(int c=0;c<6;c++) {
        bad+=f[c].bad;
        for(int k=0;k<nx*ny;k++) if(f[c].v[k]!=expected[c].v[k]) {
            mismatch++;
            int j=k/nx,i=k%nx;
            bool edge=c<3?(i==0||i==nx-1):(j==0||j==ny-1);
            if(!edge)interior++;
        }
    }
    std::cout << "{\"variant\":\""<<label<<"\",\"nx\":"<<nx<<",\"ny\":"<<ny
        <<",\"out_of_bounds_accesses\":"<<bad<<",\"mismatched_coefficients\":"<<mismatch
        <<",\"unexpected_interior_changes\":"<<interior<<"}\n";
}
int main(){
    for(auto dims: {std::array<int,2>{7,3},std::array<int,2>{3,7},std::array<int,2>{5,5},std::array<int,2>{476,306}}){
        check(before_kernel,"before",dims[0],dims[1]);check(after_kernel,"after",dims[0],dims[1]);
    }
}
'''


def extract(path,name):
    text=Path(path).read_text()
    a=text.index('void dg2::zero_perimeter_slopes')
    b=text.index('void dg2::apply_slope_limiter',a)
    return text[a:b].replace('void dg2::zero_perimeter_slopes','void '+name,1)


def verify(before,after,out,cmake):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    before,after=Path(before),Path(after)
    (out/'kernel.cpp').write_text(PREAMBLE+extract(before,'before_kernel')+extract(after,'after_kernel')+HARNESS)
    (out/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\nproject(perimeter_contract LANGUAGES CXX)\nadd_executable(perimeter_contract kernel.cpp)\nset_target_properties(perimeter_contract PROPERTIES CXX_STANDARD 17)\n')
    build=out/'build'
    commands=[[cmake,'-S',str(out),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64'],
              [cmake,'--build',str(build),'--config','Release','--parallel','1']]
    with (out/'build.log').open('w') as log:
        for cmd in commands:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    proc=subprocess.run([str(build/'Release/perimeter_contract.exe')],capture_output=True,text=True,check=True,timeout=20)
    (out/'raw.jsonl').write_text(proc.stdout)
    rows=[json.loads(x) for x in proc.stdout.splitlines()]
    fixed=[r for r in rows if r['variant']=='after']
    checks={'all_fixed_match_boundary_contract':all(r['out_of_bounds_accesses']==r['mismatched_coefficients']==0 for r in fixed),
            'old_wide_corrupts_interior':any(r['variant']=='before' and r['nx']>r['ny'] and r['unexpected_interior_changes']>0 for r in rows),
            'old_tall_out_of_bounds':any(r['variant']=='before' and r['nx']<r['ny'] and r['out_of_bounds_accesses']>0 for r in rows),
            'square_unchanged':all(r['out_of_bounds_accesses']==r['mismatched_coefficients']==0 for r in rows if r['nx']==r['ny'])}
    result=dict(checks=checks,passed=all(checks.values()),cases=rows,commands=commands,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [before,after]},
        scope='actual function text, serial bounds-checked arrays; not full solver or OpenMP race test')
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');p.add_argument('out');p.add_argument('cmake')
    a=p.parse_args();verify(a.before,a.after,a.out,a.cmake)
