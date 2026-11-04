from pathlib import Path
import hashlib,json,shutil,subprocess,time,difflib
OUT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
R=OUT/'round83'
HELPER=r'''
// Isolated experiment: whole-domain RK attempt, no negative-depth clipping.
struct RejectRK { int stage, i, j; NUMERIC_TYPE proposed; };
static NUMERIC_TYPE pending_source_volume=C(0.0);
static int pending_source_iterations=0;
struct SavedFlow83 {
    std::vector<NUMERIC_TYPE*> fields;
    std::vector<std::vector<NUMERIC_TYPE>> saved;
    SavedFlow83(Pars* p,Arrays* a):fields{a->H,a->HU,a->HV,a->H1x,a->HU1x,a->HV1x,a->H1y,a->HU1y,a->HV1y} {
        const int n=p->xsz*p->ysz;
        for(auto f:fields)saved.emplace_back(f,f+n);
    }
    void restore(){for(size_t n=0;n<fields.size();n++)std::copy(saved[n].begin(),saved[n].end(),fields[n]);}
    void dump(const char* path,Pars* p,Solver* s)const {
        FILE* f=fopen(path,"wb");if(!f){perror(path);exit(53);}
        const int dims[3]={p->xsz,p->ysz,9};fwrite(dims,sizeof(int),3,f);
        const double clock[2]={double(s->t),double(s->Tstep)};fwrite(clock,sizeof(double),2,f);
        for(const auto& v:saved)fwrite(v.data(),sizeof(NUMERIC_TYPE),v.size(),f);
        fclose(f);
    }
};
static std::vector<dg2::Increment> checked_increments83(int stage,Pars* p,Solver* s,
    Arrays* a,dg2::FlowCoefficients const& u) {
    std::vector<dg2::Increment> incs(p->xsz*p->ysz);
    bool bad=false;int bi=0,bj=0;
    NUMERIC_TYPE next_dt=C(0.5)*s->Tstep;
    for(int j=0;j<p->ysz;j++)for(int i=0;i<p->xsz;i++){
        int k=j*p->xsz+i;auto& v=incs[k];dg2::L(p,s,a,i,j,u,v);
        const NUMERIC_TYPE euler=u.H[k]+s->Tstep*v.H;
        const NUMERIC_TYPE final=stage==1?euler:C(0.5)*(a->H[k]+euler);
        const NUMERIC_TYPE vals[]={v.H,v.HU,v.HV,v.H1x,v.HU1x,v.HV1x,v.H1y,v.HU1y,v.HV1y,
          euler,final,u.HU[k]+s->Tstep*v.HU,u.HV[k]+s->Tstep*v.HV};
        bool finite=true;for(auto x:vals)finite=finite&&std::isfinite(double(x));
        // Both forward-Euler substeps must be admissible before SSP averaging.
        if(!finite || euler<C(0.0) || final<C(0.0)){
            if(!bad){bi=i;bj=j;}bad=true;
            if(std::isfinite(double(v.H)) && v.H<C(0.0) && u.H[k]>C(0.0))
                next_dt=(std::min)(next_dt,C(0.9)*u.H[k]/(-v.H));
        }
    }
    if(bad)throw RejectRK{stage,bi,bj,next_dt};
    return incs;
}
static void commit_source_audit83(Solver* s,BoundCs* b){
    if(b->numPS<=0)return;
    static FILE* f=fopen("accepted_source_volumes.txt","w");
    if(!f){perror("accepted_source_volumes.txt");exit(47);}
    fprintf(f,"%.17g %.17g %.17g %d\n",double(s->t),double(s->Tstep),double(pending_source_volume),pending_source_iterations);
}
'''

def main():
    R.mkdir(exist_ok=False)
    protocol={'baseline':'round77 immutable CPU64; round81 failed at2179.0073s',
      'change':'isolated full-domain RK attempt/reject/restore; no flux/geometry/roughness change',
      'acceptance':'finite nonnegative Euler means in both SSP stages before committing any cell of each stage; snapshot nine original fields, rederive boundary and flux arrays; defer source and step audit until acceptance',
      'backtracking':'at most8 attempts; min(half previous dt,0.9*local frozen depletion bound) and recompute all state-dependent quantities, never reuse old flux as prediction truth',
      'limits':'same no-rain/no-restart/positive-QFIX-QVAR/OMP1 scope; decline production; below1e-10s or retry cap saves preattempt fields and exits',
      'gate':'3 no-rejection regression cases and source-corner retry case before one2400sCarlisle run,15minwallcap; all original observations preserved',
      'status':'prepared; not field validated'}
    (R/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    source=R/'lisflood8-rk-retry'
    shutil.copytree(OUT/'round77/lisflood8-rk-ledger',source)
    p=source/'swe/dg2.cpp';old=p.read_text();new=old
    def replace(a,b):
        nonlocal new
        assert new.count(a)==1,(new.count(a),a[:90])
        new=new.replace(a,b,1)
    replace('struct TracePointSource',HELPER+'\nstruct TracePointSource')
    replace('Arrays* a)\n{\n    std::map<int, NUMERIC_TYPE> initial_h;',
            'Arrays* a, NUMERIC_TYPE retry_cap)\n{\n    std::map<int, NUMERIC_TYPE> initial_h;')
    replace('NUMERIC_TYPE dt=dg2::Tstep_from_cfl(p,s,a);','NUMERIC_TYPE dt=(std::min)(retry_cap,dg2::Tstep_from_cfl(p,s,a));')
    start=new.index('            static FILE* source_audit=')
    end=new.index('            return;',start)
    new=new[:start]+'            pending_source_volume=total_volume;pending_source_iterations=attempt+1;\n'+new[end:]
    # Precompute/check all cells before the stage writes any updated cell.
    for stage,which in [(1,'U'),(2,'U_int')]:
        start=new.index('void dg2::rk_stage'+str(stage)+'\n')
        end=new.index('\nvoid dg2::',start+10)
        block=new[start:end]
        anchor='\n#pragma omp parallel for\n'
        assert block.count(anchor)==1
        block=block.replace(anchor,f'\n    const auto increments83=checked_increments83({stage},Parptr,Solverptr,Arrptr,{which});\n'+anchor,1)
        a='\t\t\tIncrement U_inc = {};\n\t\t\tL(Parptr, Solverptr, Arrptr, i, j, '+which+', U_inc);\n            trace_negative_update('+str(stage)+',Parptr,Solverptr,Arrptr,BCptr,'+which+',i,j,U_inc);'
        assert a in block
        block=block.replace(a,'            const Increment& U_inc=increments83[j*Parptr->xsz+i];')
        new=new[:start]+block+new[end:]
    replace('\t\tzero_flux_stats(BCptr);\n\n        if (BCptr->numPS > 0)',
      '''        SavedFlow83 saved83(Parptr,Arrptr);
        NUMERIC_TYPE retry_cap83=Solverptr->InitTstep;
        bool accepted83=false;
        for(int attempt83=0;attempt83<8;attempt83++){
        saved83.restore();zero_flux_stats(BCptr);
        pending_source_volume=C(0.0);pending_source_iterations=0;
        try {
        if (BCptr->numPS > 0)''')
    replace('accepted_positive_point_step(Parptr, Solverptr, BCptr, Arrptr);',
            'accepted_positive_point_step(Parptr, Solverptr, BCptr, Arrptr,retry_cap83);')
    replace('        if (!(Solverptr->Tstep > C(0.0)))',
            '        Solverptr->Tstep=(std::min)(Solverptr->Tstep,retry_cap83);\n        if (!(Solverptr->Tstep > C(0.0)))')
    replace('        fprintf(step_audit, "%.17g %.17g\\n", double(Solverptr->t), double(Solverptr->Tstep));','')
    marker='\t\taccumulate_rk_boundary_stats(Solverptr,BCptr,boundary_rk1,boundary_rk2);'
    replace(marker,marker+'''
        commit_source_audit83(Solverptr,BCptr);
        fprintf(step_audit,"%.17g %.17g\\n",double(Solverptr->t),double(Solverptr->Tstep));
        accepted83=true;break;
        }catch(const RejectRK& r){
            static FILE* f=fopen("rejected_rk_attempts.txt","w");
            if(!f){perror("rejected_rk_attempts.txt");exit(52);}
            fprintf(f,"%.17g %.17g %d %d %d %d %.17g\\n",double(Solverptr->t),double(Solverptr->Tstep),attempt83+1,r.stage,r.i,r.j,double(r.proposed));fflush(f);
            if(attempt83==0)saved83.dump("last-rejection-prestate.bin",Parptr,Solverptr);
            retry_cap83=r.proposed;
            if(!std::isfinite(double(retry_cap83)) || retry_cap83<C(1e-10) || !(Solverptr->t+retry_cap83>Solverptr->t))break;
        }
        }
        if(!accepted83){saved83.dump("retry-exhausted-prestate.bin",Parptr,Solverptr);saved83.restore();fprintf(stderr,"RK retry limit; complete preattempt fields saved\\n");exit(51);}
''')
    p.write_text(new)
    (R/'rk-step-transaction.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/swe/dg2.cpp',tofile='b/swe/dg2.cpp')))
    cmake=r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    build=R/'build-rk-retry-cpu64'
    commands=[[cmake,'-S',str(source),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64','-D_CONFIG='+str(OUT/'round65/cpu64.cmake'),'-DCMAKE_CUDA_COMPILER=NOTFOUND'],[cmake,'--build',str(build),'--config','Release','--target','lisflood','--parallel','1']]
    t=time.perf_counter()
    with (R/'build.log').open('w') as log:
        for cmd in commands: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    exe=build/'Release/lisflood.exe'
    (R/'build.json').write_text(json.dumps({'wall_s':time.perf_counter()-t,'commands':commands,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()},indent=2))
    print('Built',exe,flush=True)

if __name__=='__main__': main()
