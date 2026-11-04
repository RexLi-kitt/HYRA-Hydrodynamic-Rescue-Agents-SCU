from pathlib import Path
import json,numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round90'
c=np.genfromtxt(O/'round89/original/Upstream_Flows_Run1.csv',delimiter=',',names=True);q=c['Upstream1']+c['Upstream2']+c['Upstream3'];slope=(q[1]-q[0])/900
rows=[]
for cap in [2,1]:
 m=np.loadtxt(R/f'cap{cap}/results/check.mass',skiprows=1);t=m[:,0];starts=np.r_[0,t[:-1]];dt=t-starts
 discrete=float(np.sum(np.interp(starts,c['Time'],q)*dt));exact=float(q[0]*t[-1]+.5*slope*t[-1]**2)
 vol=float(m[-1,5]);err=abs(vol-discrete);printed_bound=5e-6*10**np.floor(np.log10(abs(vol))+1)
 checks={'time60':bool(t[-1]==60),'all_steps_seen':bool(np.array_equal(m[:,3],np.arange(1,len(m)+1))),'fixed_steps':bool(np.all(dt==cap)),'outflow_zero':bool(np.all(m[:,8]==0)),'discrete_volume_matches_mass_text_precision':err<=printed_bound+1e-5,'source_quadrature_bound':abs(discrete-exact)<=abs(slope)*t[-1]*cap/2+1e-5,'all_step_mass_residual_small':bool(abs(m[:,10]).sum()<1e-5)}
 rows.append({'cap':cap,'steps':len(m),'exact_piecewise_volume_m3':exact,'left_rule_volume_m3':discrete,'reported_volume_m3':vol,'mass_print_rounding_bound_m3':float(printed_bound),'sum_abs_mass_residual_m3':float(abs(m[:,10]).sum()),'quadrature_shortfall_m3':exact-discrete,'checks':checks})
result={'rows':rows,'gate':all(all(x['checks'].values()) for x in rows),'ascii_precision_issue':'fast ASCII writer hardcodes 3 decimals despite output_precision12; raster volume is not a mass audit at 1e-5m3. Next unique longer cost probe uses binary output, not a physical modification.','checkpoint':'Fast writer checkpoint call commented; no validated restart','field_passes_added':0}
(R/'startup-analysis.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));print(json.dumps(result,default=lambda x:x.item()))

