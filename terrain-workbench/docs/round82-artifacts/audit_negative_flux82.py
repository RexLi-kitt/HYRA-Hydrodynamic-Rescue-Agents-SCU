"""Independent algebraic replay of a saved RK1 failure, not a PDE solver."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def replay(r):
    if r['stage'] != 1:
        raise ValueError('This replay supports RK1 only; RK2 needs the convex-combination state.')
    h, dt, dx, dy = (r[k] for k in ('H_stage','dt_s','dx_m','dy_m'))
    fw, fe, fn, fs = r['mass_flux_w_e_n_s']
    # j increases southwards, whereas FHy is positive northwards.
    outward = [-fw*dy, fe*dy, fn*dx, -fs*dx]
    rhs = -math.fsum(outward)/(dx*dy)
    predicted = h+dt*rhs
    outgoing = math.fsum(max(0, q) for q in outward)
    incoming = math.fsum(max(0, -q) for q in outward)
    available = h*dx*dy
    drain = available/outgoing if outgoing else None
    root = math.sqrt(3)
    def faces(v):
        m,x,y=v
        return [m-root*x,m+root*x,m+root*y,m-root*y]
    depths=faces(r['H_mean_slope_x_slope_y'])
    hu=faces(r['HU_coefficients']); hv=faces(r['HV_coefficients'])
    u=[q/z for q,z in zip(hu,depths)]
    v=[q/z for q,z in zip(hv,depths)]
    # Confirmed from pars.cpp dg2 branch, not the generic 0.7 default.
    cfl,g=.33,9.80655
    mean_u=r['HU_coefficients'][0]/h
    mean_v=r['HV_coefficients'][0]/h
    mean_scale=min(cfl*dx/(abs(mean_u)+math.sqrt(g*h)),cfl*dy/(abs(mean_v)+math.sqrt(g*h)))
    face_scales=[min(cfl*dx/(abs(a)+math.sqrt(g*z)),cfl*dy/(abs(b)+math.sqrt(g*z))) for a,b,z in zip(u,v,depths)]
    # Independent Decimal volume balance confirms binary-float arithmetic.
    from decimal import Decimal as D, localcontext
    with localcontext() as ctx:
        ctx.prec=45
        d=lambda a:D(str(a))
        decimal_h=d(h)-d(dt)*((d(fe)-d(fw))/d(dx)+(d(fn)-d(fs))/d(dy))
    checks={
        'rhs_matches_trace':abs(rhs-r['rhs_H'])<1e-14,
        'prediction_matches_trace':abs(predicted-r['H_after_predicted'])<1e-14,
        'decimal_balance_matches':abs(float(decimal_h)-predicted)<1e-14,
        'all_four_faces_positive':min(depths)>0,
        'recorded_raw_faces_match':max(abs(a-b) for a,b in zip(depths,r['H_raw_faces_w_e_n_s']))<1e-13,
        'source_increment_zero':r['source_delta_h']==0,
        'all_four_fluxes_outward':min(outward)>0,
        'dt_exceeds_frozen_drain_bound':drain is not None and dt>drain,
        'below_bound_nonnegative':h+(drain*(1-1e-8))*rhs>=0,
        'above_bound_negative':h+(drain*(1+1e-8))*rhs<0,
    }
    return {'checks':checks,'pass':all(checks.values()),'outward_m3_s_w_e_n_s':outward,
        'outgoing_m3_s':outgoing,'incoming_m3_s':incoming,'available_m3':available,
        'requested_outgoing_m3':dt*outgoing,'mean_rhs_m_s':rhs,'replayed_mean_m':predicted,
        'frozen_drain_bound_s':drain,'excess_fraction':dt/drain-1,
        'face_depth_m_w_e_n_s':depths,'face_hu_m2_s_w_e_n_s':hu,
        'face_u_m_s_w_e_n_s':u,'face_v_m_s_w_e_n_s':v,
        'mean_u_m_s':mean_u,'mean_v_m_s':mean_v,
        'local_mean_cfl_scale_s':mean_scale,'face_cfl_scales_s':face_scales,
        'gravity_used_for_diagnostic':g,'cfl_from_dg2_parser':cfl,
        'limitations':['Frozen recorded flux replay; does not reconstruct missing neighbour momentum slopes or rerun the Riemann solver.',
        'Face CFL scales are diagnostic, not a proof of an admissible DG time step.',
        'No full-state checkpoint; no PDE recovery or field validation was performed.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    raw=a.trace.read_bytes();rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=1: raise ValueError('Expected the single retained first-failure record')
    result=replay(rows[0]);result['trace_sha256']=hashlib.sha256(raw).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if not result['pass']: raise SystemExit(1)

if __name__=='__main__': main()
