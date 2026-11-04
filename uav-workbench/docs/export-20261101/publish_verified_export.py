import json,hashlib,shutil,importlib.util
from pathlib import Path
from datetime import datetime
base=Path(__file__).resolve().parent;root=base.parent;run=root/'uav-strict-eval-20261101/run01'
out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
repo=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
desktop=Path(r'C:\Users\Lenovo\Desktop\HYRA_完整训练与严格测试_20261101.xlsx')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
audit=read(base/'export-audit.json');assert audit['status']=='passed'
book=base/'单双模型训练全量数据-20261101.xlsx';assert sha(book)==audit['xlsx_sha256']
integrity=read(run/'integrity-audit.json');assert integrity['status']=='passed' and integrity['records']==5630
for p,h in read(run/'registration.json')['source_hashes'].items():assert sha(Path(p))==h,p
archive=out/'uav-export-20261101';archive.mkdir(exist_ok=True)
if not (archive/'status-before-export-recovery.json').exists():shutil.copy2(run/'status.json',archive/'status-before-export-recovery.json')
for p in (root/'uav-parallel-more-test-20261101/failure.json',root/'uav-parallel-more-test-20261101/stderr.log'):
    if p.exists():shutil.copy2(p,archive/p.name)
shutil.copy2(book,desktop);assert sha(desktop)==audit['xlsx_sha256']
v=module('strict_final_export',root/'uav-strict-eval-20261101/strict_eval.py')
v.emit('completed',records=5630,completed_records=5630,total_records=5630,evaluation_completed=True,workbook_export='completed',
    workbook=str(desktop),export_sha256=audit['xlsx_sha256'],prior_export_failure_preserved=str(archive/'status-before-export-recovery.json'))
v.mirror()
refresh=module('refresh_publish',base/'refresh.py');refresh.publish()
report=f'''2026-11-01 Excel导出修复与交付
原始失败：严格测试5630条、聚合和完整性审计均已完成，随后Node默认堆上限触发JavaScript heap out of memory，外层状态因此被写为failed。原始异常保留，未重跑模拟或模型。
更改：build.mjs在exportXlsx前显式回收临时对象，修正首页与严格测试页完成状态文字；refresh.py的Node参数新增--expose-gc --max-old-space-size=6144。保留原9表、全部数值、公式、来源和历史记录。不修改全局内存设置、训练源、模型权重、评价或协议。
原因与可行性：默认约4GiB堆在生成大表导出对象时不足；可用物理内存约6.6GiB，本次仅提高进程上限并主动释放临时渲染对象。导出前GC后heapUsed约456MiB，修复后成功导出。没有删数据、分文件或安装其他环境。
回归：9张渲染预览已检查；导出ZIP结构有效。独立只读校验所有8张数据表的每个导出数据字段与来源相符，公式缓存与累计9600当前训练回合核对通过，全部工作表无Excel错误单元格。源冻结哈希复核通过。Excel/WPS交互打开未代操作。
记录：430评估汇总，16642逐场评估（含全部5630严格测试），13184训练记录，90严格分组结果，21当前方案，232权重检查点。历史部分存在兼容前缀复用，不能将所有历史训练与当前回合直接相加。
文件：{desktop}
SHA256：{audit['xlsx_sha256']}
回退：去除Node新增参数和GC调用可恢复原导出行为；原测试/数据均未改变。原内存限制下可能再次导出失败。
下一步：可以直接审阅严格测试表、全量轮次与对照，不按测试重新选检查点；仅一个训练种子与合成环境，不能解释为实地搜救准确率。
'''
(base/'export-report.txt').write_text(report,encoding='utf8')
files=[book,base/'export-audit.json',base/'export-report.txt',base/'verification.json',base/'build.mjs',base/'refresh.py',base/'verify_export.py',base/'publish_verified_export.py',base/'data.json']
files+=list(base.glob('preview-*.png'))
manifest={p.name:sha(p) for p in files};(base/'export-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');files.append(base/'export-manifest.json')
for folder in (archive,repo/'uav-workbench/docs/export-20261101'):
    folder.mkdir(parents=True,exist_ok=True)
    for p in files:
        shutil.copy2(p,folder/p.name);assert sha(folder/p.name)==sha(p)
entry=f'\n\n## 2026-11-01 完整Excel导出完成\n\n5630严格测试与完整性审计原已完成，失败仅来自后续Excel导出达到默认JS堆上限。按用户要求提高本次Node堆至6144MiB并在导出前回收临时对象，保留9表全部数据。独立逐表字段/公式缓存/ZIP核验通过，430评估汇总、16642逐场评估、13184训练记录（历史复用不累计）、90严格分组、232检查点完整；9个预览检查完成。桌面HYRA_完整训练与严格测试_20261101.xlsx，4,811,805字节，SHA256={audit["xlsx_sha256"]}。旧失败保留在outputs/uav-export-20261101，当前状态修正为测试与导出均completed，未重跑/改权重/推送。输出及uav-workbench/docs/export-20261101双份保存配置、来源、核验与报告。\n'
for folder in (out,repo/'terrain-workbench/docs'):
    for name in ('HYRA自动续作进度.md','HYRA累计更改日志.md'):
        with (folder/name).open('a',encoding='utf8') as f:f.write(entry)
print(json.dumps(dict(desktop=str(desktop),bytes=desktop.stat().st_size,verified=True),ensure_ascii=False))
