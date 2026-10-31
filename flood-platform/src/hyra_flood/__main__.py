import argparse,json
from .contracts import load_case,write_json
from .engines import CAPABILITIES,run
from .evaluation import compare_runs

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    s.add_parser('backends')
    v=s.add_parser('validate');v.add_argument('case')
    r=s.add_parser('run');r.add_argument('case');r.add_argument('--backend',required=True,choices=list(CAPABILITIES));r.add_argument('--output',required=True);r.add_argument('--legacy-lab')
    c=s.add_parser('compare');c.add_argument('first');c.add_argument('second');c.add_argument('--output',required=True)
    q=s.add_parser('score-merewether');q.add_argument('archive');q.add_argument('--source',required=True);q.add_argument('--output',required=True)
    a=p.parse_args()
    if a.cmd=='backends':result=CAPABILITIES
    elif a.cmd=='validate':case,_,fingerprint=load_case(a.case);result={'id':case['id'],'sha256':fingerprint,'validated_input':True}
    elif a.cmd=='run':result=run(a.case,a.backend,a.output,a.legacy_lab)
    elif a.cmd=='score-merewether':
        from .archive import score_merewether
        result=score_merewether(a.archive,a.source);write_json(a.output,result)
    else:result=compare_runs(a.first,a.second);write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
