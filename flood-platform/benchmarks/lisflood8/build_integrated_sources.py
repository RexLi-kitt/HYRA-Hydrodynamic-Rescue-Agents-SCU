"""Round76: exact prescribed piecewise-linear volume over accepted dt."""
from pathlib import Path
import difflib,hashlib,json,shutil,subprocess,time

FUNCTIONS=r'''
static NUMERIC_TYPE positive_series_integral(const TimeSeries* ts,
    NUMERIC_TYPE begin, NUMERIC_TYPE end)
{
    if(!ts || ts->count<1 || !std::isfinite(double(begin)) ||
       !std::isfinite(double(end)) || end<begin){fprintf(stderr,"Invalid source interval/series\n");exit(46);}
    for(int k=0;k<ts->count;k++) {
        if(!std::isfinite(double(ts->time[k])) || !std::isfinite(double(ts->value[k])) ||
           ts->value[k]<C(0.0) || (k>0 && !(ts->time[k]>ts->time[k-1]))) {
            fprintf(stderr,"Source requires finite increasing times and nonnegative values\n");exit(46);
        }
    }
    // Same constant extension outside the data range as linear_interpolate.
    NUMERIC_TYPE area=C(0.0);
    if(begin<ts->time[0])area+=ts->value[0]*((std::min)(end,ts->time[0])-begin);
    for(int k=0;k+1<ts->count;k++) {
        const NUMERIC_TYPE a=(std::max)(begin,ts->time[k]);
        const NUMERIC_TYPE z=(std::min)(end,ts->time[k+1]);
        if(z<=a)continue;
        const NUMERIC_TYPE span=ts->time[k+1]-ts->time[k];
        const NUMERIC_TYPE fa=(a-ts->time[k])/span,fz=(z-ts->time[k])/span;
        const NUMERIC_TYPE qa=(C(1.0)-fa)*ts->value[k]+fa*ts->value[k+1];
        const NUMERIC_TYPE qz=(C(1.0)-fz)*ts->value[k]+fz*ts->value[k+1];
        area+=C(0.5)*(qa+qz)*(z-a);
    }
    if(end>ts->time[ts->count-1])area+=ts->value[ts->count-1]*(end-(std::max)(begin,ts->time[ts->count-1]));
    if(!std::isfinite(double(area)) || area<C(0.0)){fprintf(stderr,"Invalid integrated source volume\n");exit(46);}
    return area;
}

static void accepted_positive_point_step(Pars* p, Solver* s, BoundCs* b,
                                        Arrays* a)
{
    std::map<int, NUMERIC_TYPE> initial_h;
    for(int k=0;k<b->numPS;k++) {
        const int i=b->xpi[k],j=b->ypi[k];
        if(i<0 || i>=p->xsz || j<0 || j>=p->ysz){fprintf(stderr,"Point outside raster\n");exit(EXIT_FAILURE);}
        initial_h.emplace(j*p->xsz+i,a->H[j*p->xsz+i]);
    }
    dg2::update_boundary_values(p,s,b,a);
    NUMERIC_TYPE dt=dg2::Tstep_from_cfl(p,s,a);
    dt=(std::min)(dt,s->Sim_Time-s->t);
    dt=(std::min)(dt,p->MassTotal-s->t);
    dt=(std::min)(dt,p->SaveTotal-s->t);
    for(int attempt=0;attempt<64;attempt++) {
        if(!(dt>C(0.0)) || !std::isfinite(double(dt)) || !(s->t+dt>s->t)) {
            fprintf(stderr,"Invalid point-source timestep\n");exit(EXIT_FAILURE);
        }
        std::map<int,NUMERIC_TYPE> depth_increment;
        NUMERIC_TYPE total_volume=C(0.0);
        for(int k=0;k<b->numPS;k++) {
            NUMERIC_TYPE unit_width_volume;
            if(b->PS_Ident[k]==QFIX4) {
                const NUMERIC_TYPE q=b->PS_Val[k];
                if(!std::isfinite(double(q)) || q<C(0.0)){fprintf(stderr,"Invalid fixed point flow\n");exit(46);}
                unit_width_volume=q*dt;
            }
            else if(b->PS_Ident[k]==QVAR5)
                unit_width_volume=positive_series_integral(b->PS_TimeSeries[k],s->t,s->t+dt);
            else {fprintf(stderr,"Only nonnegative QFIX/QVAR supported\n");exit(EXIT_FAILURE);}
            const NUMERIC_TYPE volume=unit_width_volume*p->dx;
            if(!std::isfinite(double(volume)) || volume<C(0.0)){fprintf(stderr,"Invalid point volume\n");exit(46);}
            depth_increment[b->ypi[k]*p->xsz+b->xpi[k]]+=volume/p->dA;
            total_volume+=volume;
        }
        for(const auto& item:initial_h)a->H[item.first]=item.second+depth_increment[item.first];
        dg2::update_boundary_values(p,s,b,a);
        const NUMERIC_TYPE allowed=dg2::Tstep_from_cfl(p,s,a);
        if(dt<=allowed) {
            s->Tstep=dt;
            trace_point_sources.clear();
            for(const auto& item:initial_h)trace_point_sources.push_back({item.first,item.second,a->H[item.first]-item.second});
            accumulate_point_flux_stats(b,total_volume/dt);
            static FILE* source_audit=fopen("accepted_source_volumes.txt","w");
            if(!source_audit){perror("accepted_source_volumes.txt");exit(47);}
            fprintf(source_audit,"%.17g %.17g %.17g %d\n",double(s->t),double(dt),double(total_volume),attempt+1);
            return;
        }
        dt=C(0.9)*allowed;
    }
    fprintf(stderr,"Point-source CFL iteration failed\n");exit(EXIT_FAILURE);
}
'''

out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');r=out/'round76';r.mkdir(exist_ok=False)
base=out/'round75/lisflood8-reconstruction-candidate';src=r/'lisflood8-integrated-sources';shutil.copytree(base,src)
p=src/'swe/dg2.cpp';old=p.read_text();begin=old.index('// Experimental adapter:');end=old.index('// Diagnostic observer only.')
new=old[:begin]+'// Experimental nonnegative point sources: accepted-interval integral.\n'+FUNCTIONS+'\n\n'+old[end:];p.write_text(new)
(r/'accepted-source-integral.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/swe/dg2.cpp',tofile='b/swe/dg2.cpp')))
protocol=dict(baseline='round75 isolated reconstruction candidate',change='Exact piecewise-linear nonnegative QVAR area on each proposed interval; recompute source on retry from original H; commit interval-average rate to existing volume ledger; QFIX unchanged.',
 criteria=['Analytic constant/linear/knot-crossing volumes within 1e-12','Partition invariance within 1e-12','Mock forced CFL retry with two points in same cell preserves initial H and commits once','Invalid negative values or non-increasing times rejected','Constant-source field identical to saved round75','Static wet/sloped shore depth drift <1e-10m','Fixed60s original-data run finite/nonnegative, ledger and prescribed integral errors <1e-6m3; if passes, one max-dt-halved check verifies prescribed volume invariance'],
 scope='Nonnegative QFIX/QVAR, no rain/restart, CPU64 one thread, source-only exact time integration; no second-order operator-splitting claim or field acceptance.',
 rejection='Do not widen thresholds or alter source/observation arrays. Stop on nonnegative/conservation/static failures. Keep complete negative results.',
 costs='One isolated full compile, actual-function kernel, light regressions, at most two60s startup runs (~15s expected).',
 original_inputs_unchanged=True,production_default_changed=False)
(r/'protocol.json').write_text(json.dumps(protocol,indent=2))
cmake=r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe';build=r/'build-integrated-cpu64'
commands=[[cmake,'-S',str(src),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64','-D_CONFIG='+str(out/'round65/cpu64.cmake'),'-DCMAKE_CUDA_COMPILER=NOTFOUND'],[cmake,'--build',str(build),'--config','Release','--target','lisflood','--parallel','1']]
start=time.perf_counter()
with (r/'build.log').open('w') as f:
 for cmd in commands:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
exe=build/'Release/lisflood.exe';result=dict(commands=commands,wall_s=time.perf_counter()-start,exe=str(exe),exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(r/'build.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
