from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.artifacts import new_run,finish_run,write_json,sha256
from hyra_exp3.data_audit import audit_common_data
from hyra_exp3.protocol import load_protocol
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-run',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3,choices=range(1,5))
    a=p.parse_args();data=a.data_run.resolve()
    directory=new_run(ROOT,'common_data_independent_audit',dict(data_run=str(data),workers=a.workers))
    print(f'Complete common-record audit: {directory}',flush=True)
    try:
        result=audit_common_data(ROOT,data,a.workers)
        write_json(directory/'audit_result.json',result);finish_run(ROOT,directory)
        ledger=load_protocol(ROOT/'COMPLETION.json')
        proof=data/'common_data_integrity_audit.json'
        ledger['requirements']['common_causal_training_records']=dict(status='verified',
            evidence=[dict(path=proof.relative_to(ROOT).as_posix(),sha256=sha256(proof))])
        write_json(ROOT/'COMPLETION.json',ledger)
        print(result,flush=True)
    except BaseException as error:
        import traceback
        (directory/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(ROOT,directory,status='failed',error=repr(error));raise
