from pathlib import Path
import difflib,json,shutil,subprocess,time
from prepare_round77 import OUT,R,write,digest,run_case
import numpy as np

HELPER=r'''
// DG2 RK2 audit only: integrate each stage's gross boundary rates equally.
// Do not modify the physical flux arrays or count point sources a second time.
struct RKBoundaryRates { NUMERIC_TYPE in=C(0.0), out=C(0.0); };
static RKBoundaryRates rk_boundary_rates(const Pars* p, const Arrays* a)
{
    RKBoundaryRates q;
    const auto add = [&q](NUMERIC_TYPE inward) {
        if(inward>C(0.0)) q.in+=inward; else q.out-=inward;
    };
    for(int i=0;i<p->xsz;i++) add(-a->FHy[i]*p->dx);
    for(int i=0;i<p->xsz;i++) add(a->FHy[p->ysz*(p->xsz+1)+i]*p->dx);
    for(int j=0;j<p->ysz;j++) add(a->FHx[j*(p->xsz+1)]*p->dy);
    for(int j=0;j<p->ysz;j++) add(-a->FHx[j*(p->xsz+1)+p->xsz]*p->dy);
    return q;
}
static void accumulate_rk_boundary_stats(Solver* s, BoundCs* b,
    const RKBoundaryRates& q1, const RKBoundaryRates& q2)
{
    const NUMERIC_TYPE point_in=b->Qin, point_out=b->Qout;
    b->Qin+=C(0.5)*(q1.in+q2.in);
    b->Qout+=C(0.5)*(q1.out+q2.out);
    b->VolInMT+=b->Qin*s->Tstep;
    b->VolOutMT+=b->Qout*s->Tstep;
    static FILE* audit=fopen("accepted_boundary_exchange.txt","w");
    if(!audit){perror("accepted_boundary_exchange.txt");exit(48);}
    fprintf(audit,"%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
        double(s->t),double(s->Tstep),double(q1.in),double(q1.out),
        double(q2.in),double(q2.out),double(point_in),double(point_out));
}

'''

if __name__=='__main__':
    baseline=json.loads((R/'baseline-results.json').read_text())
    assert all(x['exit_code']==0 and x['finite_nonnegative'] and not x['negative_trace_present'] for x in baseline.values())
    assert all(abs(v['total_ledger_residual_m3'])>1e-8 for k,v in baseline.items() if k!='closed')
    src=R/'lisflood8-rk-ledger';shutil.copytree(OUT/'round76/lisflood8-integrated-sources',src)
    p=src/'swe/dg2.cpp';old=p.read_text();new=old.replace('struct TracePointSource',HELPER+'struct TracePointSource',1)
    before='\t\trk_stage1(Parptr, Solverptr, BCptr, Arrptr);\n\t\trk_stage2(Parptr, Solverptr, BCptr, Arrptr);'
    after='\t\trk_stage1(Parptr, Solverptr, BCptr, Arrptr);\n        const RKBoundaryRates boundary_rk1=rk_boundary_rates(Parptr,Arrptr);\n\t\trk_stage2(Parptr, Solverptr, BCptr, Arrptr);\n        const RKBoundaryRates boundary_rk2=rk_boundary_rates(Parptr,Arrptr);'
    assert new.count(before)==1
    new=new.replace(before,after)
    before='accumulate_boundary_flux_stats(Parptr, Solverptr, BCptr, Arrptr);'
    assert new.count(before)==1
    new=new.replace(before,'accumulate_rk_boundary_stats(Solverptr,BCptr,boundary_rk1,boundary_rk2);')
    p.write_text(new)
    (R/'rk-boundary-ledger.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/swe/dg2.cpp',tofile='b/swe/dg2.cpp')))
    cmake=r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe';build=R/'build-rk-ledger-cpu64'
    commands=[[cmake,'-S',str(src),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64','-D_CONFIG='+str(OUT/'round65/cpu64.cmake'),'-DCMAKE_CUDA_COMPILER=NOTFOUND'],[cmake,'--build',str(build),'--config','Release','--target','lisflood','--parallel','1']]
    start=time.perf_counter()
    with (R/'build.log').open('w') as log:
        for cmd in commands:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    exe=build/'Release/lisflood.exe'
    write(R/'build.json',dict(commands=commands,wall_s=time.perf_counter()-start,exe=str(exe),exe_sha256=digest(exe),source_sha256=digest(p)))
    results={}
    for name,base_result in baseline.items():
        base=R/'baseline'/name;case=R/'candidate'/name;case.mkdir(parents=True,exist_ok=False)
        for file in base.iterdir():
            if file.name in ['case.par','input.bci'] or file.name.startswith(('bed.asc','water.asc')):shutil.copy2(file,case/file.name)
        result=run_case(exe,case,base_result['initial_volume_m3'])
        if result['exit_code']==0:
            physical=[x.name for x in (base/'results').iterdir() if x.suffix!='.mass']
            result['emitted_field_byte_identical']={n:(case/'results'/n).read_bytes()==(base/'results'/n).read_bytes() for n in physical}
            result['steps_byte_identical']=(case/'accepted_timesteps.txt').read_bytes()==(base/'accepted_timesteps.txt').read_bytes()
            audit=np.loadtxt(case/'accepted_boundary_exchange.txt',ndmin=2)
            steps=np.loadtxt(case/'accepted_timesteps.txt',ndmin=2)
            result['audit_steps_exact']=bool(np.array_equal(audit[:,:2],steps))
            source=float(np.sum(audit[:,1]*(audit[:,6]-audit[:,7])))
            exchange=float(np.sum(audit[:,1]*.5*(audit[:,2]-audit[:,3]+audit[:,4]-audit[:,5])))
            rk2only=float(np.sum(audit[:,1]*(audit[:,4]-audit[:,5])))
            delta=result['final_volume_m3']-result['initial_volume_m3']
            result.update(point_volume_m3=source,rk_boundary_volume_m3=exchange,
                independent_volume_residual_m3=source+exchange-delta,
                reconstructed_old_ledger_residual_m3=source+rk2only-delta,
                old_ledger_reproduced=bool(abs(source+rk2only-delta-base_result['total_ledger_residual_m3'])<1e-5),
                point_volume_expected_m3=.005 if name=='point_and_outlet' else 0.)
            result['passed']=bool(result['finite_nonnegative'] and not result['negative_trace_present'] and result['steps_byte_identical'] and result['audit_steps_exact']
                and all(result['emitted_field_byte_identical'].values()) and result['old_ledger_reproduced']
                and abs(result['total_ledger_residual_m3'])<1e-8 and abs(result['independent_volume_residual_m3'])<1e-8
                and abs(source-result['point_volume_expected_m3'])<1e-12)
        else:result['passed']=False
        write(case/'comparison.json',result);results[name]=result
    write(R/'verification.json',dict(cases=results,passed=all(x['passed'] for x in results.values()),field_qualitative_passes_added=0,production_adopted=False))
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ['exe_sha256','emitted_field_byte_identical']} for k,v in results.items()}),flush=True)
