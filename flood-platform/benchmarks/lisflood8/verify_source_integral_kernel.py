"""Compile actual source routines with a controlled CFL environment."""
from pathlib import Path
import json,subprocess
out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');r=out/'round76';work=r/'kernel';work.mkdir(exist_ok=False)
src=(r/'lisflood8-integrated-sources/swe/dg2.cpp').read_text()
start=src.index('static NUMERIC_TYPE positive_series_integral');end=src.index('// Diagnostic observer only.')
functions=src[start:end]
prefix=r'''
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
using NUMERIC_TYPE=double;
#define C(x) (x)
struct TimeSeries{double *time,*value;int count;};
struct Pars{int xsz,ysz;double dx,dA,MassTotal,SaveTotal;};
struct Solver{double t,Sim_Time,Tstep;};
struct BoundCs{int numPS;int *xpi,*ypi,*PS_Ident;double *PS_Val;TimeSeries **PS_TimeSeries;double Qin;};
struct Arrays{double *H;};
enum{QFIX4=4,QVAR5=5};
struct TracePointSource{int index;double before,delta;};
static std::vector<TracePointSource> trace_point_sources;
static int cfl_calls=0,commit_calls=0;
namespace dg2 {
void update_boundary_values(Pars*,Solver*,BoundCs*,Arrays*){}
double Tstep_from_cfl(Pars*,Solver*,Arrays*){return cfl_calls++==0 ? 1.0 : .4;}
}
void accumulate_point_flux_stats(BoundCs* b,double q){b->Qin+=q;commit_calls++;}
void close(double a,double b){if(!std::isfinite(a) || std::abs(a-b)>1e-12){fprintf(stderr,"contract %.17g != %.17g\n",a,b);exit(99);}}
'''
main=r'''
int main(int argc,char**argv){
 double tt[]={0,1},vv[]={0,2};TimeSeries line{tt,vv,2};
 if(argc>1){
   if(!strcmp(argv[1],"negative"))vv[1]=-1;
   if(!strcmp(argv[1],"duplicate"))tt[1]=0;
   if(!strcmp(argv[1],"reverse_interval")){positive_series_integral(&line,1,0);return 98;}
   positive_series_integral(&line,0,1);return 98;
 }
 double tc[]={0},vc[]={3};TimeSeries constant{tc,vc,1};
 close(positive_series_integral(&constant,-2,4),18);
 close(positive_series_integral(&line,0,1),1);
 close(positive_series_integral(&line,0,2),3);
 double knots[]={0,.25,.75,1},vals[]={0,2,2,0};TimeSeries pulse{knots,vals,4};
 close(positive_series_integral(&pulse,0,1),1.5);
 close(positive_series_integral(&pulse,.1,.9),1.42);
 double divisions[]={0,.1,.2,.6,.9,1},sum=0;
 for(int k=0;k<5;k++)sum+=positive_series_integral(&pulse,divisions[k],divisions[k+1]);
 close(sum,1.5);
 close(positive_series_integral(&pulse,1,1),0);
 int xx[]={0,0},yy[]={0,0},ids[]={4,5};double fixed[]={1,0};TimeSeries* series[]={nullptr,&line};
 double hh[]={.2,0,0,0};Pars p{2,2,2,4,1,1};Solver s{0,1,0};BoundCs b{2,xx,yy,ids,fixed,series,0};Arrays a{hh};
 accepted_positive_point_step(&p,&s,&b,&a);
 // Forced rejection of dt=1; retry dt=.36. Analytic QFIX + ramp area.
 const double v=2*.36+2*.36*.36;
 close(s.Tstep,.36);close(hh[0],.2+v/4);close(b.Qin*s.Tstep,v);
 if(commit_calls!=1 || cfl_calls!=3 || trace_point_sources.size()!=1)return 97;
 close(trace_point_sources[0].before,.2);close(trace_point_sources[0].delta,v/4);
 printf("{\"analytic_and_partition_pass\":true,\"retry_commits_once\":true,\"duplicate_cell_sum_pass\":true,\"accepted_dt\":%.17g,\"accepted_volume\":%.17g,\"post_H\":%.17g}\n",s.Tstep,v,hh[0]);
}
'''
(work/'source_contract.cpp').write_text(prefix+functions+main)
(work/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.15)\nproject(source_contract LANGUAGES CXX)\nadd_executable(source_contract source_contract.cpp)\ntarget_compile_features(source_contract PRIVATE cxx_std_14)\n')
cmake=r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
with (work/'build.log').open('w') as f:
 subprocess.run([cmake,'-S',str(work),'-B',str(work/'build'),'-G','Visual Studio 17 2022','-A','x64'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
 subprocess.run([cmake,'--build',str(work/'build'),'--config','Release','--parallel','1'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
exe=work/'build/Release/source_contract.exe';p=subprocess.run([str(exe)],cwd=work,text=True,capture_output=True)
(work/'run.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
result=json.loads(p.stdout);result['invalid_rejections']={}
for mode in ['negative','duplicate','reverse_interval']:
 p=subprocess.run([str(exe),mode],cwd=work,text=True,capture_output=True);result['invalid_rejections'][mode]=dict(exit_code=p.returncode,stderr=p.stderr.strip());assert p.returncode==46
result['scope']='Actual integral/source C++ routines; mocked CFL schedule only, not a hydrodynamic solver test.'
(work/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
