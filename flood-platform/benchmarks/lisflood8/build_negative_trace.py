"""Build a read-only RK-stage observer; never alters a numerical update."""
from pathlib import Path
import difflib,hashlib,json,shutil,subprocess,time

GLOBALS=r'''
struct TracePointSource { int index; NUMERIC_TYPE before, delta; };
static std::vector<TracePointSource> trace_point_sources;
'''
HELPER=r'''
// Diagnostic observer only. Single thread; stops before committing a material
// negative update. It does not clip, limit, recompute or replace any flux.
static void trace_negative_update(int stage, Pars* p, Solver* s, Arrays* a,
    BoundCs* b, dg2::FlowCoefficients const& u, int i, int j,
    dg2::Increment const& inc)
{
    static bool saw_negative=false;
    const int k=j*p->xsz+i;
    const NUMERIC_TYPE predicted=stage==1 ? a->H[k]+s->Tstep*inc.H :
        C(0.5)*(a->H[k]+a->H_int[k]+s->Tstep*inc.H);
    if (!(predicted<C(0.0))) return;
    const bool material=predicted<C(-1e-12);
    if(saw_negative && !material) return;
    FILE* f=fopen("negative-depth.jsonl", "a");
    if(!f){perror("negative-depth.jsonl");exit(44);}
    const NUMERIC_TYPE fw=a->FHx[j*(p->xsz+1)+i];
    const NUMERIC_TYPE fe=a->FHx[j*(p->xsz+1)+i+1];
    const NUMERIC_TYPE fn=a->FHy[j*(p->xsz+1)+i];
    const NUMERIC_TYPE fs=a->FHy[(j+1)*(p->xsz+1)+i];
    double before_source=double(a->H[k]),source_delta=0;
    for(const auto& src:trace_point_sources)if(src.index==k){before_source=double(src.before);source_delta=double(src.delta);}
    fprintf(f,"{\"stage\":%d,\"t_s\":%.17g,\"dt_s\":%.17g,\"i\":%d,\"j\":%d,",stage,double(s->t),double(s->Tstep),i,j);
    fprintf(f,"\"material\":%s,\"first_negative\":%s,\"dx_m\":%.17g,\"dy_m\":%.17g,",material?"true":"false",saw_negative?"false":"true",double(p->dx),double(p->dy));
    fprintf(f,"\"H_before_source\":%.17g,\"source_delta_h\":%.17g,\"H_n_postsource\":%.17g,\"H_stage\":%.17g,\"H_after_predicted\":%.17g,\"rhs_H\":%.17g,",before_source,source_delta,double(a->H[k]),double(u.H[k]),double(predicted),double(inc.H));
    fprintf(f,"\"mass_flux_w_e_n_s\":[%.17g,%.17g,%.17g,%.17g],",double(fw),double(fe),double(fn),double(fs));
    fprintf(f,"\"bed_mean_slope_x_slope_y\":[%.17g,%.17g,%.17g],",double(a->DEM[k]),double(a->DEM1x[k]),double(a->DEM1y[k]));
    fprintf(f,"\"H_mean_slope_x_slope_y\":[%.17g,%.17g,%.17g],",double(u.H[k]),double(u.H1x[k]),double(u.H1y[k]));
    fprintf(f,"\"HU_coefficients\":[%.17g,%.17g,%.17g],\"HV_coefficients\":[%.17g,%.17g,%.17g],",double(u.HU[k]),double(u.HU1x[k]),double(u.HU1y[k]),double(u.HV[k]),double(u.HV1x[k]),double(u.HV1y[k]));
    fprintf(f,"\"Hstar_w_e_n_s\":[%.17g,%.17g,%.17g,%.17g],",double(a->Hstar_pos_x[k]),double(a->Hstar_neg_x[k]),double(a->Hstar_neg_y[k]),double(a->Hstar_pos_y[k]));
    fprintf(f,"\"H_raw_faces_w_e_n_s\":[%.17g,%.17g,%.17g,%.17g],",double(u.H[k]-SQRT(C(3.0))*u.H1x[k]),double(u.H[k]+SQRT(C(3.0))*u.H1x[k]),double(u.H[k]+SQRT(C(3.0))*u.H1y[k]),double(u.H[k]-SQRT(C(3.0))*u.H1y[k]));
    fprintf(f,"\"neighbors\":[");bool first=true;
    for(int jj=(std::max)(0,j-1);jj<=(std::min)(p->ysz-1,j+1);jj++)
    for(int ii=(std::max)(0,i-1);ii<=(std::min)(p->xsz-1,i+1);ii++){
        int kk=jj*p->xsz+ii;
        if(!first)fprintf(f,",");first=false;
        fprintf(f,"{\"i\":%d,\"j\":%d,\"h\":%.17g,\"h1x\":%.17g,\"h1y\":%.17g,\"hu\":%.17g,\"hv\":%.17g,\"z\":%.17g,\"z1x\":%.17g,\"z1y\":%.17g}",ii,jj,double(u.H[kk]),double(u.H1x[kk]),double(u.H1y[kk]),double(u.HU[kk]),double(u.HV[kk]),double(a->DEM[kk]),double(a->DEM1x[kk]),double(a->DEM1y[kk]));
    }
    fprintf(f,"]}\n");fclose(f);saw_negative=true;
    if(material){fprintf(stderr,"Trace reached material negative depth; expected diagnostic exit 42\n");fflush(NULL);exit(42);}
}
'''

out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
r=out/'round75';r.mkdir(exist_ok=False)
baseline=out/'round74/lisflood8-perimeter-stride';target=r/'lisflood8-negative-trace'
shutil.copytree(baseline,target)
p=target/'swe/dg2.cpp';before=p.read_text();after=before
needle='// Experimental adapter: nonnegative QFIX/QVAR, no rain, from t=0 only.'
assert after.count(needle)==1;after=after.replace(needle,GLOBALS+'\n'+needle)
needle='s->Tstep=dt;\n            accumulate_point_flux_stats(b,total_q);'
assert after.count(needle)==1
after=after.replace(needle,'s->Tstep=dt;\n            trace_point_sources.clear();\n            for(const auto& item:initial_h) trace_point_sources.push_back({item.first,item.second,a->H[item.first]-item.second});\n            accumulate_point_flux_stats(b,total_q);')
needle='void dg2::solve';assert after.count(needle)==1;after=after.replace(needle,HELPER+'\n'+needle)
needle='    if (Statesptr->rainfall == ON';assert after.count(needle)==1
after=after.replace(needle,'    if(!getenv("OMP_NUM_THREADS") || strcmp(getenv("OMP_NUM_THREADS"),"1")){fprintf(stderr,"Trace requires OMP_NUM_THREADS=1\\n");exit(43);}\n'+needle)
needle='L(Parptr, Solverptr, Arrptr, i, j, U, U_inc);'
assert after.count(needle)==1
after=after.replace(needle,needle+'\n            trace_negative_update(1,Parptr,Solverptr,Arrptr,BCptr,U,i,j,U_inc);')
needle='L(Parptr, Solverptr, Arrptr, i, j, U_int, U_inc);'
assert after.count(needle)==1
after=after.replace(needle,needle+'\n            trace_negative_update(2,Parptr,Solverptr,Arrptr,BCptr,U_int,i,j,U_inc);')
p.write_text(after)
(r/'negative-depth-observer.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/swe/dg2.cpp',tofile='b/swe/dg2.cpp')))
protocol=dict(baseline='round74 stride fix; original 60s archived startup, limiter OFF',
    change='read-only RK observer and source-depth snapshots; no formula/input changes; diagnostic exit42 before material negative update',
    trace_trigger_m=-1e-12,strict_nonnegative_acceptance_unchanged=True,
    criteria=['independently reconstruct RHS from face fluxes and RK depth update','prefix of accepted timesteps identical to baseline','square nonnegative control finishes with identical depth field'],
    rejection='If prefix/control differ, diagnose observer before drawing physics conclusions. Do not rerun full 60s without new information.',
    expected_cost_s=90,evidence_class='internal numerical diagnosis, not field validation')
(r/'protocol.json').write_text(json.dumps(protocol,indent=2))
cmake=r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
build=r/'build-trace-cpu64'
commands=[[cmake,'-S',str(target),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64','-D_CONFIG='+str(out/'round65/cpu64.cmake'),'-DCMAKE_CUDA_COMPILER=NOTFOUND'],[cmake,'--build',str(build),'--config','Release','--target','lisflood','--parallel','1']]
start=time.perf_counter()
with (r/'build-trace.log').open('w') as f:
    for cmd in commands:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
exe=build/'Release/lisflood.exe'
result=dict(commands=commands,elapsed_s=time.perf_counter()-start,exe=str(exe),exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
            source_before_sha256=hashlib.sha256((baseline/'swe/dg2.cpp').read_bytes()).hexdigest(),source_after_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(r/'build-trace.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
