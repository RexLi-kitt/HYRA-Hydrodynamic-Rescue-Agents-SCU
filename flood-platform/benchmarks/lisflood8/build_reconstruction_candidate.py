"""One isolated candidate; scale DG slopes without changing cell averages."""
from pathlib import Path
import difflib, hashlib, json, shutil, subprocess, time

HELPER=r'''
// Experimental reconstruction-only candidate. This is not a proof of mean
// positivity: the existing CFL and hydrostatic discretisation are unchanged.
static void limit_nonnegative_reconstruction(Pars* p, dg2::FlowCoefficients const& u)
{
    for(int k=0;k<p->xsz*p->ysz;k++) {
        const NUMERIC_TYPE span=SQRT(C(3.0))*(FABS(u.H1x[k])+FABS(u.H1y[k]));
        if(span>u.H[k] && span>C(0.0)) {
            if(u.H[k]<C(0.0)){fprintf(stderr,"Negative mean cannot be repaired by slope scaling\n");exit(45);}
            const NUMERIC_TYPE theta=(u.H[k]/span)*(C(1.0)-C(8.0)*DBL_EPSILON);
            u.H1x[k]*=theta;u.H1y[k]*=theta;
            u.HU1x[k]*=theta;u.HU1y[k]*=theta;
            u.HV1x[k]*=theta;u.HV1y[k]*=theta;
        }
    }
}
'''
out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs'); r=out/'round75'
target=r/'lisflood8-reconstruction-candidate'
shutil.copytree(r/'lisflood8-negative-trace',target)
p=target/'swe/dg2.cpp';before=p.read_text();after=before
assert after.count('void dg2::solve')==1
after=after.replace('void dg2::solve','#include <cfloat>\n'+HELPER+'\nvoid dg2::solve')
for u in ['U','U_int']:
 needle=f'update_Hstar(Parptr, Arrptr, {u});'
 assert after.count(needle)==1
 after=after.replace(needle,f'limit_nonnegative_reconstruction(Parptr,{u});\n\t'+needle)
p.write_text(after)
(r/'nonnegative-reconstruction-candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/swe/dg2.cpp',tofile='b/swe/dg2.cpp')))
protocol=dict(hypothesis='Overlarge DG depth slopes create excessive one-sided face depth at millimetre mean water; mean-preserving reconstruction may remove the observed trigger.',
 change='Before each RK flux calculation scale H/HU/HV slope coefficients by the same theta; leave all cell averages, bed, inputs, generic limiter and CFL unchanged.',
 theta='min(1,hbar/[sqrt(3)*(abs(h1x)+abs(h1y))]); eight machine eps safety only when limiting; negative means rejected, never clipped.',
 scope='Isolated CPU64, single-thread research candidate; NOT a verbatim implementation or theorem transfer from Xing et al. 2010.',
 source='https://people.math.osu.edu/xing.205/research/xzsAWR2010.pdf',
 criteria=['full fixed 60s case finishes, all saved and traced means finite/nonnegative and water accounting closes','fully wet sloped lake remains at rest','partially dry step lake remains at rest','flat multi-source volume remains correct'],
 rejection='Any mean negativity, mass error above 1e-8 m3 in small tests / 1e-6 in Carlisle, or stationary depth drift above 1e-10m rejects as general repair. No threshold search, no acceptance relaxation.',
 cost='one build and one bounded 60s startup; three small controls; 180s wall cap for actual startup',
 field_validation='None: this isolates a numerical prerequisite and does not establish realistic inundation or full-event applicability.')
(r/'candidate-protocol.json').write_text(json.dumps(protocol,indent=2))
cmake=r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
build=r/'build-reconstruction-cpu64'
commands=[[cmake,'-S',str(target),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64','-D_CONFIG='+str(out/'round65/cpu64.cmake'),'-DCMAKE_CUDA_COMPILER=NOTFOUND'],[cmake,'--build',str(build),'--config','Release','--target','lisflood','--parallel','1']]
start=time.perf_counter()
with (r/'build-candidate.log').open('w') as f:
 for cmd in commands:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
exe=build/'Release/lisflood.exe'
result=dict(commands=commands,elapsed_s=time.perf_counter()-start,exe=str(exe),exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(r/'build-candidate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
