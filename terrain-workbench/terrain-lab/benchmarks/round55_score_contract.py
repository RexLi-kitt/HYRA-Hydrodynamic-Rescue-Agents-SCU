"""Independent, strict full-window scorer. Never shift, drop or detrend samples."""
from pathlib import Path
import ast, hashlib, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def score(t, predicted, observed):
    expected=np.arange(451)*.05
    if t.shape!=(451,) or not np.allclose(t,expected,rtol=0,atol=1e-12):
        raise ValueError('Expected original 0--22.5 s times at 0.05 s')
    if predicted.shape!=observed.shape or observed.shape!=(451,3):
        raise ValueError('All three gauges and all 451 samples are required')
    if not all(np.isfinite(a).all() for a in (t,predicted,observed)):
        raise ValueError('Missing/nonfinite data must not be silently removed')
    d=predicted-observed
    result={'rmse_mm':(np.sqrt((d*d).mean(axis=0))*1000).tolist(),
            'mae_mm':(np.abs(d).mean(axis=0)*1000).tolist(),
            'bias_mm':(d.mean(axis=0)*1000).tolist(),
            'peak_error_mm':((predicted.max(axis=0)-observed.max(axis=0))*1000).tolist(),
            'peak_time_error_s':(t[predicted.argmax(axis=0)]-t[observed.argmax(axis=0)]).tolist()}
    result['segments']={}
    for lo,hi in [(0,10),(10,15),(15,20),(20,22.5)]:
        m=(t>=lo)&((t<=hi) if hi==22.5 else t<hi)
        result['segments'][f'{lo}-{hi}']={'n':int(m.sum()),'rmse_mm':(np.sqrt((d[m]**2).mean(axis=0))*1000).tolist()}
    return result

def self_test():
    t=np.arange(451)*.05;o=np.zeros((451,3));p=np.ones_like(o)*.002
    assert np.allclose(score(t,p,o)['rmse_mm'],2)
    assert np.allclose(score(t,p,o)['mae_mm'],2)
    for tt,pp,oo in [(t+.05,p,o),(t,p[:-1],o),(t,np.full_like(p,np.nan),o)]:
        try:score(tt,pp,oo)
        except ValueError:pass
        else:raise AssertionError('Invalid scoring input accepted')
    return {'constant_error':True,'reject_time_shift':True,'reject_missing_row':True,'reject_nan':True}

def main():
    data=ROOT/'round39';prov=json.loads((data/'provenance.json').read_text())
    for f in prov['files']:
        assert hashlib.sha256((data/f['file']).read_bytes()).hexdigest()==f['sha256']
    source=data/'test_results.py';tree=ast.parse(source.read_text())
    expected=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='expected_difference' for n in node.targets):expected=ast.literal_eval(node.value)
    obs=np.loadtxt(data/'output_ch5-7-9.txt',skiprows=1)[:451,1:]/100
    results={};hashes={}
    for name,rel in [('round44','round44/original_coordinates'),('round48','round48/fixed_domain_018'),('round47','round47/exact_full')]:
        path=ROOT/rel/'timeseries.npz'
        with np.load(path) as z:
            assert np.array_equal(z['observed_m'],obs)
            results[name]=score(z['time_s'],z['predicted_m'],obs)
        saved=json.loads((path.parent/'result.json').read_text())
        assert np.allclose(results[name]['rmse_mm'],[m['rmse_m']*1000 for m in saved['metrics']],rtol=0,atol=1e-10)
        assert np.allclose(results[name]['mae_mm'],[m['mae_m']*1000 for m in saved['metrics']],rtol=0,atol=1e-10)
        hashes[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
    report={'tests':self_test(),'input_sha256':hashes,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_commit':prov['commit'],'source_expected_difference_metric':'MAE, not RMSE',
            'historical_anuga_mae_mm':{g:expected[g]*1000 for g in ['ch5','ch7','ch9']},
            'warning':'Historical software regression constants only; NOT a fresh ANUGA run, independent experiment, or acceptance target.',
            'results':results}
    (ROOT/'round55/scoring-audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--outputs',type=Path,default=ROOT)
    ROOT=parser.parse_args().outputs.resolve()
    main()

