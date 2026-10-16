"""Independent metric geometry, scalar action, mask, and training screen audit."""
from itertools import combinations
from pathlib import Path
import argparse
import sys
import numpy as np

from audit_pulse_service_alignment import read,load,digest,maximum
from audit_physical_scale import visible
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hyra_validation.artifacts import write_json


def reference(times,depth,order,travel,now,deadline):
    t=now;maxima=[];starts=[]
    for point,dt in zip(order,travel):
        begin=t+dt;assert begin+12<=deadline+1e-10
        m=maximum(times,depth,point,begin,begin+12)
        first=maximum(times,depth,point,begin,begin+6)
        maxima.append(m);starts.append(begin);t=begin+(6 if first>.8 else 12)
    return dict(count=sum(m<=.8 for m in maxima),margin=min(.8-m for m in maxima),maxima=maxima,starts=starts)


def winner(scores,orders):
    if scores[1]['count']!=scores[0]['count']:return int(scores[1]['count']>scores[0]['count'])
    if abs(scores[1]['margin']-scores[0]['margin'])>1e-12:return int(scores[1]['margin']>scores[0]['margin'])
    return int(orders[1][0]<orders[0][0])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve();config=read(run/'config.json');design=config['design']
    assert read(run/'completion.json')['completed']
    for rel,checksum in read(run/'provenance.json')['files'].items():assert digest(ROOT/rel)==checksum,rel
    parent=Path(config['hydraulic_parent']);registry=read(parent/'field_registry.json')
    assert digest(parent/'field_registry.json')==config['hydraulic_registry_sha256']
    assert read(parent/'independent_audit.json')['passed']
    families=read(parent/'config.json')['design']['family_ids'];geometries=read(run/'geometry.json')
    counters=dict(contexts=0,geometric_cases=0,scalar_order_evaluations=0,visibility_flags=0)
    additional=read(run/'field_registry.json');assert len(additional)==24
    for info in additional.values():
        assert info['kind']=='nominal_forecast' and info['anchor_s'] in (320,330)
        assert digest(info['raw_path'])==info['raw_sha256'] and digest(info['projected_path'])==info['projected_sha256']
        restart=info['restart'];assert restart['source_kind']=='same_mesh_independent_nominal_base'
        assert 'nominal_base' in Path(restart['source_path']).name and digest(restart['source_path'])==restart['source_sha256']
        raw=load(info['raw_path']);base=load(restart['source_path'])
        index=int(np.flatnonzero(base['time_s']==restart['state_time_s'])[0])
        for name in ('stage','xmomentum','ymomentum'):np.testing.assert_allclose(raw[name][0],base[name][index],atol=1e-12,rtol=0)
        projection=load(info['projected_path']);v,w=projection['vertices'],projection['weights']
        depth=np.maximum(raw['stage']-raw['bed_m'],0)
        np.testing.assert_allclose(sum(depth[:,v[:,j]]*w[:,j] for j in range(3)),projection['linear'],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(raw['time_s'],np.arange(info['anchor_s'],info['anchor_s']+61,2))
    registry.update(additional);counters['additional_nominal_restarts']=24
    cache={};all_rows=[]
    def field(family,length,kind,anchor=0,center=0,amp=0.):
        key=f'{family}_L{length}_n40_{kind}_a{anchor}_y{center}_mm{round(amp*1000)}'
        if key not in cache:
            info=registry[key];assert digest(info['projected_path'])==info['projected_sha256']
            cache[key]=load(info['projected_path'])
        return cache[key]
    for length in design['candidate_lengths_x_m']:
        xy=np.array([((5+10*x)*length/100,5+10*y) for y in range(10) for x in range(10)])
        layouts=[];excluded=0
        for a,b in combinations(sorted(design['target_ids']),2):
            for origin in design['origin_ids']:
                for speed in design['flight_speed_m_per_s']:
                    between=np.linalg.norm(xy[a]-xy[b])/speed
                    initial=[np.linalg.norm(xy[origin]-xy[p])/speed for p in (a,b)]
                    duration=max(initial)+between+24
                    if duration>48+1e-12:excluded+=1;continue
                    layouts.append(dict(targets=[a,b],origin=origin,speed_m_per_s=speed,
                        cross_half=(a//10<5)!=(b//10<5),orders=[([a,b],[initial[0],between]),([b,a],[initial[1],between])]))
        cases=[dict(c,layout=i,now_s=offset,decision_offset_s=offset) for i,c in enumerate(layouts)
               for offset in design['decision_offsets_s'] if offset+max(c['orders'][0][1][0],c['orders'][1][1][0])+c['orders'][0][1][1]+24<=60+1e-12]
        saved_geometry=geometries[str(length)]
        assert len(cases)==len(saved_geometry['cases']) and saved_geometry['coverage']['excluded_layouts']==excluded
        for actual,saved in zip(cases,saved_geometry['cases']):
            for k in ('targets','origin','speed_m_per_s','cross_half','layout','now_s'):assert actual[k]==saved[k]
            for (order,dt),(saved_order,saved_dt) in zip(actual['orders'],saved['orders']):
                assert order==saved_order;np.testing.assert_allclose(dt,saved_dt,atol=1e-12,rtol=0)
        counters['geometric_cases']+=len(cases)
        for family in families:
            result=read(run/(family+f'_L{length}.json'));assert digest(result['scores_path'])==result['scores_sha256']
            dumps=load(result['scores_path']);base=field(family,length,'truth_base');n=len(cases);ix=np.arange(n)
            for anchor in design['anchors_s']:
                prefix=f'a{anchor}';af=field(family,length,'nominal_forecast',anchor)
                chosen=[];critical=[];base_rows=[]
                for c in cases:
                    now=c['now_s']+anchor;release=int(now//10*10);f=field(family,length,'nominal_forecast',release)
                    pred=[reference(f['time_s'],f['linear'],*order,now,anchor+60) for order in c['orders']]
                    ac=[reference(af['time_s'],af['linear'],*order,now,anchor+60) for order in c['orders']]
                    truth=[reference(base['time_s'],base['linear'],*order,now,anchor+60) for order in c['orders']]
                    chosen.append(winner(pred,c['orders']));critical.append(min(abs(.8-m) for row in ac for m in row['maxima'])<=.05)
                    base_rows.append(truth);counters['scalar_order_evaluations']+=6
                chosen=np.array(chosen);critical=np.array(critical)
                np.testing.assert_array_equal(chosen,dumps[prefix+'_nominal_actions'])
                np.testing.assert_array_equal(critical,dumps[prefix+'_critical'])
                for name in ('count','margin','starts','maxima'):
                    manual=np.array([[r[name] for r in row] for row in base_rows])
                    np.testing.assert_allclose(manual,dumps[prefix+'_E0_'+name],atol=1e-12,rtol=0)
                e0=dumps[prefix+'_E0_count'];g0=e0[ix,1-chosen]-e0[ix,chosen]
                for center in (20,70):
                    for amp in design['candidate_amplitudes_m']:
                        pp=prefix+f'_y{center}_mm{round(amp*1000)}';pulse=field(family,length,'truth_pulse',anchor,center,amp)
                        scores=[[reference(pulse['time_s'],pulse['linear'],*order,c['now_s']+anchor,anchor+60)
                                 for order in c['orders']] for c in cases]
                        counters['scalar_order_evaluations']+=2*n
                        for name in ('count','margin','starts','maxima'):
                            manual=np.array([[r[name] for r in row] for row in scores])
                            np.testing.assert_allclose(manual,dumps[pp+'_E1_'+name],atol=1e-12,rtol=0)
                        e1=dumps[pp+'_E1_count'];gap=e1[ix,1-chosen]-e1[ix,chosen];specific=(gap>0)&(g0<=0)
                        np.testing.assert_array_equal(specific,dumps[pp+'_pulse_specific'])
                        availability={offset:np.array([[visible(pulse['time_s'],pulse['linear']-base['linear'],anchor+offset,phase,delay)
                                      for delay in (0,10)] for phase in (0,.25,.5,.75)]) for offset in design['decision_offsets_s']}
                        flags=np.array([availability[c['decision_offset_s']] for c in cases])&specific[:,None,None]
                        np.testing.assert_array_equal(flags,dumps[pp+'_visible']);counters['visibility_flags']+=n*8
                        tables=[r for r in result['rows'] if r['anchor_s']==anchor and r['center_y_m']==center and r['amplitude_m']==amp]
                        for row in tables:
                            keep=critical if row['population']=='critical' else np.ones(n,bool)
                            counts=e0 if row['condition']=='E0' else e1;actual=counts[ix,chosen];oracle=counts.max(axis=1)
                            assert row['cases']==int(keep.sum()) and row['nominal_sum']==int(actual[keep].sum())
                            assert row['oracle_sum']==int(oracle[keep].sum()) and row['headroom_sum']==int((oracle-actual)[keep].sum())
                            for d,delay in enumerate((0,10)):
                                expected=int((flags[:,:,d]*np.maximum(gap,0)[:,None])[keep].sum()) if row['condition']=='E1' else 0
                                assert row[f'visible_D{delay}_sum']==expected
            all_rows.extend(result['rows']);counters['contexts']+=1
    summary=read(run/'summary.json');selected=None
    for item in summary['conditions']:
        for fam in item['families']:
            rows=[r for r in all_rows if r['family']==fam['family'] and r['length_x_m']==item['length_x_m'] and r['amplitude_m']==item['amplitude_m'] and r['population']==item['population']]
            count=sum(r['cases'] for r in rows);assert fam['cases']==count and fam['unique_design_cases']==count//2
            assert fam['nonempty_anchor_center_cells']==sum(r['cases']>0 for r in rows if r['condition']=='E0')
            if count:
                assert abs(fam['headroom_per100']-sum(r['headroom_sum'] for r in rows)/count*100)<1e-12
                for delay in (0,10):assert abs(fam[f'visible_D{delay}_per100']-sum(r[f'visible_D{delay}_sum'] for r in rows)/(count*4)*100)<1e-12
        nonempty=[r for r in item['families'] if r['cases']]
        if nonempty:
            assert abs(item['headroom_family_macro_per100']-np.mean([r['headroom_per100'] for r in nonempty]))<1e-12
            for delay in (0,10):assert abs(item[f'visible_D{delay}_family_macro_per100']-np.mean([r[f'visible_D{delay}_per100'] for r in nonempty]))<1e-12
        if item['population']=='critical':
            passed=all(r['unique_design_cases']>=10 and r['nonempty_anchor_center_cells']>=2 for r in item['families'])
            passed=passed and (item['headroom_family_macro_per100'] or 0)>=4 and (item['visible_D0_family_macro_per100'] or 0)>=2
            assert passed==item['training_screen_passed']
            if passed and selected is None:selected=dict(length_x_m=item['length_x_m'],amplitude_m=item['amplitude_m'])
    assert selected==summary['selected_for_next_stage'] and counters['contexts']==12
    write_json(run/'independent_audit.json',dict(passed=True,**counters,audit_source_sha256=digest(__file__),
        scope='all metric geometry and actions, masks, source visibility, complete service scores and candidate selection; no policy efficacy'))
    pointer=read(ROOT/'results/latest_metric_task_screen.json');assert Path(pointer['run'])==run
    pointer.update(audit_pending=False,independent_audit_passed=True)
    write_json(ROOT/'results/latest_metric_task_screen.json',pointer)
    print('AUDIT_PASSED '+str(counters))


if __name__=='__main__':main()
