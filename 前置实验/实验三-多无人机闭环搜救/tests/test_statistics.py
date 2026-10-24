import pytest
from hyra_exp3.statistics import paired_effect,select_bstar


def rows(methods=('H','E','P')):
    return [dict(family_id=f'family{f}',training_seed=s,layout=l,method=m,side=50,delay_s=0.,
                 summary=dict(success_rate=.9 if m!='P' else 1.))
            for f in range(2) for s in (101,202,303) for l in (0,1) for m in methods]


def test_paired_intervals_average_layouts_and_cross_resample_family_and_seed():
    r=rows()
    result=paired_effect(r,'P','H',replicates=100)
    assert len(result['family_seed_differences_pp'])==2
    assert result['delta_percentage_points']==pytest.approx(10)
    assert result['ci95_percentage_points']==pytest.approx([10,10])
    assert result['engineering_target_3pp_met'] and result['primary_positive']
    for row in r:
        row['summary']['success_rate']=1.
    zero=paired_effect(r,'P','H',replicates=100)
    assert zero['ci95_percentage_points']==[0.,0.] and not zero['primary_positive']
    with pytest.raises(ValueError,match='Missing paired'):
        paired_effect(r[:-1],'P','H',replicates=10)


def test_B_star_ignores_P_and_uses_one_validation_only_tie_rule():
    r=rows();fixed=[dict(method=x['method'],delay_s=0.,window_error_rate=.02 if x['method']=='H' else .01,mse_m2=.001) for x in r]
    selected=select_bstar(r,fixed,expected_families=2)
    assert selected['B_star']=='E' and not selected['P_or_test_results_used']
    for row in r:
        if row['method']=='H':row['summary']['success_rate']=.95
    assert select_bstar(r,fixed,expected_families=2)['B_star']=='H'
