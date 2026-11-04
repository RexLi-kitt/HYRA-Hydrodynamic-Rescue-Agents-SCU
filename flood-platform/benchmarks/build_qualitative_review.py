"""Build the local, offline round69 qualitative evidence snapshot (no server/API).

This is a review interface, not a new flood simulator or live decision system.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build(base, template, destination):
    comparison=json.loads((base/'round69/analysis/comparison.json').read_text(encoding='utf-8'))
    primary=comparison['cases'][0]
    if primary['case']!='round47/exact_full' or primary['screening_all_rows_consistent']:
        raise ValueError('This snapshot template requires review for changed primary evidence')
    opposite={(r['station'],r['start_s'],r['end_s']) for r in primary['phase_rows'] if r['comparison']=='inconsistent'}
    if opposite!={('ch9',7.0,8.0),('ch7',21.0,22.0)}:
        raise ValueError('Issue text no longer matches the underlying archive; update deliberately')
    pairs=primary['peak_order_rows']
    if pairs[-1]['comparison']!='inconsistent' or pairs[-1]['pair']!=['ch7','ch9']:
        raise ValueError('Peak-order headline does not match source')
    data={'rows':primary['phase_rows'],
          'labCards':[
              ['哪里被淹','未验证','三站水位不能证明全域淹没或未淹范围。',''],
              ['上涨还是退去','存在不一致','尾段趋势与部分主峰先后仍有相反判断。','bad'],
              ['主要流向','无法判断','当前曲线未提供独立的方向观测。',''],
              ['相对深浅与急缓','无法判断','水面高程不等于水深，也不是流速。',''],
              ['地面通道与起降点','未验证','没有同期可通行或起降条件证据。',''],
              ['搜索区域与顺序','暂不作可靠判断','先解决水情反向判断，再验证任务影响。','']],
          'fieldCards':[
              ['哪里被淹','现场观测已备','历史最大淹没记录已保存；尚未与模型比较。','ready'],
              ['上涨还是退去','连续观测已备','保留原始测站过程，编辑记录待审核。','ready'],
              ['主要流向','无法判断','仅凭淹没范围和水位不能推出街区流向。',''],
              ['相对深浅与急缓','无法判断','调查水面高程需配套可靠地面和速度证据。',''],
              ['地面通道与起降点','未验证','受淹与能否安全使用需要分别验证。',''],
              ['搜索区域与顺序','尚未验证','没有实际受困人员和当时可见信息的记录。','']],
          'outlines':json.loads((base/'round68/prepared/outlines-wgs84.geojson').read_text(encoding='utf-8')),
          'marks':json.loads((base/'round68/prepared/marks-wgs84.geojson').read_text(encoding='utf-8'))}
    raw=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    html=template.read_text(encoding='utf-8').replace('__EVIDENCE_DATA__',raw)
    if '__EVIDENCE_DATA__' in html:raise ValueError('template token was not filled')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(html,encoding='utf-8')
    return {'output':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'primary_case':primary['case'],'all_phase_rows':len(data['rows']),
            'all_survey_points':len(data['marks']['features']),'all_outlines':len(data['outlines']['features']),
            'source_comparison_sha256':hashlib.sha256((base/'round69/analysis/comparison.json').read_bytes()).hexdigest(),
            'template_sha256':hashlib.sha256(template.read_bytes()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.base,a.template,a.output),ensure_ascii=False,indent=2))
