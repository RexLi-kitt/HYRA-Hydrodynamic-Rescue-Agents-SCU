from pathlib import Path
import json,hashlib,numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round90';C=R/'cost7200'
m=np.loadtxt(C/'results/check.mass',skiprows=1);t=m[:,0];a=np.r_[0,t[:-1]];dt=t-a
c=np.genfromtxt(O/'round89/original/Upstream_Flows_Run1.csv',delimiter=',',names=True);q=c['Upstream1']+c['Upstream2']+c['Upstream3']
left=float(np.sum(np.interp(a,c['Time'],q)*dt));knots=np.r_[c['Time'][c['Time']<t[-1]],t[-1]];v=np.interp(knots,c['Time'],q);exact=float(np.trapezoid(v,knots))
cs=json.loads((R/'cost-status.json').read_text());volume=cs['frames'][-1]['volume_m3'];tv=float(abs(np.diff(v)).sum());time_round_bound=.001*(2*tv+abs(v[0])+abs(v[-1]))
result={'actual_end_reported_s':float(t[-1]),'target_s':7200,'end_overshoot_s':float(t[-1]-7200),'all_steps_recorded':bool(np.array_equal(m[:,3],np.arange(1,len(m)+1))),'independent_left_rule_volume_using_rounded_step_times_m3':left,'binary_volume_m3':volume,'difference_m3':volume-left,'reported_time_rounding_bound_m3':time_round_bound,'left_rule_match_within_time_precision':abs(volume-left)<=time_round_bound+1e-5,'piecewise_linear_exact_volume_at_reported_end_m3':exact,'integration_shortfall_m3':exact-volume,'step_count':int(m[-1,3]),'steps_min_median_max_s':[float(m[:,1].min()),float(np.median(m[:,1])),float(m[:,1].max())],'sum_abs_internal_mass_residual_m3':float(abs(m[:,10]).sum()),'sum_signed_internal_mass_residual_m3':float(m[:,10].sum()),'zero_outflow':bool(np.all(m[:,8]==0)),'finite_nonnegative_all_saved_frames':all(f['finite_nonnegative'] for f in cs['frames']),'wall_s':cs['wall_s'],'linear_whole_event_estimate_minutes_not_prediction':cs['wall_s']/t[-1]*245700/60,'limitations':['7200s is early, wetted footprint and runtime grow later; linear cost is not a promise','ASCII3-decimal limitation avoided by binary; every-step time labels still rounded to .001s','fast backend writes no verified restart; no complete event launched','mass residual does not establish observed fit; endpoint slightly beyond target retained, not shifted']}
(R/'cost-analysis.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()))
src=O/'round86/lisflood8-isotropic-friction'
snips=[('pars.cpp',730,737),('pars.cpp',658,663),('input.cpp',1306,1314),('input.cpp',2155,2170),('lisflood.cpp',453,467),('lisflood.cpp',733,740),('lisflood2/sgm_fast.cpp',2378,2394),('lisflood2/sgm_fast.cpp',3019,3028),('lisflood2/sgm_fast.cpp',3714,3729),('lisflood2/sgm_fast.cpp',4613,4627),('lisflood2/file_tool.cpp',661,670),('lisflood2/file_tool.cpp',690,707)]
e=[]
for n,lo,hi in snips:
 p=src/n;lines=p.read_text(errors='replace').splitlines();e.append({'file':n,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'lines':[{'line':i+1,'text':lines[i]} for i in range(lo-1,hi)]})
(R/'source-path-evidence.json').write_text(json.dumps(e,indent=2))
print(json.dumps(result,default=lambda x:x.item()))
