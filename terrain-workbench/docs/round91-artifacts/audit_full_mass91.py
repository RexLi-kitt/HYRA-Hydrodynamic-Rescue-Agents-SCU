"""Independent CSV inflow / reported outflow budget; respects output precision."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
def audit(massfile):
 m=np.loadtxt(massfile,skiprows=1);t=m[:,0];start=np.r_[0,t[:-1]];dt=t-start
 csv=O/'round89/original/Upstream_Flows_Run1.csv'
 c=np.genfromtxt(csv,delimiter=',',names=True);q=c['Upstream1']+c['Upstream2']+c['Upstream3'];qt=c['Time']
 qi=np.interp(start,qt,q);vi=float(qi@dt);vo=float(m[:,8]@dt)
 slopes=abs(np.diff(q)/np.diff(qt));qin_tol=.0005+.0005*float(slopes.max())+1e-5
 qerr=abs(m[:,6]-qi)
 knots=np.r_[qt[qt<t[-1]],t[-1]];rates=np.interp(knots,qt,q)
 continuous=float(np.trapezoid(rates,knots))
 source_time_bound=.001*(2*float(abs(np.diff(rates)).sum())+abs(float(rates[0]))+abs(float(rates[-1])))
 outflow_time_bound=.0005*(float(abs(np.diff(m[:,8])).sum())+abs(float(m[0,8]))+abs(float(m[-1,8])))
 flux_print_bound=.0005*float(t[-1]);input_export_bound=1e-5*float(t[-1])
 last_vol=float(m[-1,5]);volume_print_bound=.5*10**(np.floor(np.log10(abs(last_vol)))-4) if last_vol else 0
 residual=vi-vo-last_vol;internal_abs=float(abs(m[:,10]).sum())
 budget_bound=float(source_time_bound+outflow_time_bound+flux_print_bound+input_export_bound+volume_print_bound+internal_abs+1e-5)
 checks={'all_steps_recorded':bool(np.array_equal(m[:,3],np.arange(1,len(m)+1))),'strictly_increasing_times':bool(np.all(dt>0)),'finite_mass':bool(np.isfinite(m).all()),'inflow_matches_independent_csv_with_print_precision':bool(np.all(qerr<=qin_tol)),'integral_budget_within_conservative_print_bound':abs(residual)<=budget_bound,'no_rain_loss_used':bool(np.all(m[:,11]==0))}
 return {'checks':checks,'gate':all(checks.values()),'actual_end_s':float(t[-1]),'steps':len(m),'independent_left_inflow_m3':vi,'continuous_csv_inflow_m3':continuous,'left_vs_continuous_difference_m3':vi-continuous,'outflow_from_every_step_printed_rate_m3':vo,'last_printed_volume_m3':last_vol,'independent_budget_residual_m3':residual,'conservative_precision_bound_m3':budget_bound,'precision_bound_terms_m3':{'source_time':float(source_time_bound),'outflow_time':outflow_time_bound,'outflow_rate_rounding':flux_print_bound,'bdy_csv_export':input_export_bound,'volume_print':float(volume_print_bound),'sum_abs_internal_residual':internal_abs},'max_qin_csv_difference_m3_s':float(qerr.max()),'qin_precision_bound_m3_s':qin_tol,'source_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),'mass_sha256':hashlib.sha256(Path(massfile).read_bytes()).hexdigest(),'limits':['dry initial state and no rain/infiltration/evaporation as this archived configuration','uses every-step printed time/rate precision bounds, not exact boundary volume instrumentation','passes internal accounting only; no observational accuracy or full-time field positivity claim']}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--startup-check',action='store_true');args=ap.parse_args()
 if args.startup_check:
  mass=O/'round90/cost7200_omp4/results/check.mass';name='independent-mass-startup-check.json'
 else:
  state=json.loads((O/'round91/status.json').read_text());assert state['state']=='completed' and state['exit_code']==0,'Do not audit incomplete full event'
  mass=O/'round91/full_event/results/check.mass';name='independent-full-mass-budget.json'
 result=audit(mass);(O/'round91'/name).write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['gate'],'Independent mass audit failed'
