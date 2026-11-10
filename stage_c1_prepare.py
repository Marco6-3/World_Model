"""Freeze the C.1 protocol and update report wording without changing old evidence."""
from pathlib import Path
import json
from datetime import datetime
from hashlib import sha256

root=Path(__file__).resolve().parent
out=root/'outputs/stage_c_bag'
p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
for key in ('additional_collection_and_retraining_cancelled_by_user','physical_slip_audit','frozen_models_unchanged','source_sha256'):
    p.pop(key,None)
p.update(version='stage_c1',created_at=datetime.now().isoformat(),
         orders=['A-left then B-right','A-right then B-left'],world_span_frames=444,
         training_data_version='stage_c1_forced_finger_release_v2',
         history_formula_note='Spec formula produces 67 tokens; use 60 spaced plus last 4 consecutive, 444-frame inclusive span',
         slot_hint='Ground-truth one-hot for flow training only; ShortPast predicts BC/Joint/Short, MambaPast predicts GPC/Full/Four at evaluation',
         guidance='0.5 * 10 / remaining flow steps; same emergency priors for all searches with >=8 particles',
         original_archive='outputs/stage_c_bag_pre_c1_20261110')
p['limitations']=[s for s in p['limitations'] if 'Fixed A-left' not in s]
p['limitations'].append('Physics may retain small history-dependent differences in H4; chance-level short-memory performance is tested, not mathematically assumed')
p['source_sha256']={f:sha256((root/f).read_bytes()).hexdigest() for f in ('bag_scene.py','bag_common.py','bag_learning.py','bag_audit.py','inference.py')}
(out/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')

path=root/'bag_report.py'
s=path.read_text(encoding='utf-8')
replacements={
 'clear.axhline(12,':'clear.axhline(28,',
 '12 mm planning margin':'28 mm planning margin',
 '规范固定 A→左槽、B→右槽，当前物品身份因此能直接推断目标槽位。':'本轮固定 A→B 抓取顺序，order 决定 A 左/B 右或 A 右/B 左；pre_entry 均从 x=0 开始，第二阶段指令清零。',
 '主任务的高分槽准确率不能单独证明长程记忆必要，也不能证明只有 Mamba 能记住。':'这消除了固定身份映射，但真实动力学可能留下微小短历史差异；不能先验宣称 H4 输入完全相同或其成绩必定为 50%。',
 '覆盖约 12.1 s':'覆盖约 14.8 s',
 '且仍可能使用物品身份捷径；不能替代闭环分槽结果。':'固定物品身份捷径已移除；回放成绩仍不能替代闭环分槽结果。',
 '冻结模型不因这次扰动修复而重训。采集时恢复专家会主动松开左指，评测时此释放由外部故障强制发生。训练示教中专家会主动松爪，因此“仍命令闭爪却被外部张开”的状态覆盖有限；不能把泛化失败单独归因于记忆或隐空间维度。':'本轮重新采集并训练全部模型，采集和评测统一使用物理释放 v2；实际强制开指目标同步写回观测。恢复示教先垂直撤至 z=245 mm 再重撑；occupied 提前至物品下表面低于170 mm。',
 '早期仅解除辅助抓持但真实手指仍夹持的扰动回合已归档，全部扰动组按物理释放 v2 重新评测；常规组物理逻辑未变并保留原始回合。':'首轮数据与模型整体归档，本轮常规和扰动组均重新评测。',
 '本次复核应直接使用 dataset 与 trained 中冻结的 v1 数据和模型，评测施加物理释放 v2；这是扰动状态分布差异下的鲁棒性评测，不能写成同分布恢复优势。重新采集或训练属于新实验，不保证得到本次数值。':'本轮 Stage C.1 按新授权重新采集192回合并训练全部模型，统一v2扰动后重新评测288回合。结果按实际数据报告，未达到的验收目标不改写为通过。',
 '当前物品身份可透露固定目标槽位，因此主表不能独立证明长程记忆必要性。':'固定身份到槽位映射已移除，槽位提示来自各方法自己的历史编码器；记忆效果以实测为准。',
 '部分回合按种子分片共享 GPU，进程中断后其余回合顺序续跑，闭环延迟含不同机器负载':'评测顺序运行；闭环延迟仍受模拟器负载影响',
 '下方含并行负载的闭环延迟':'下方含模拟器负载的闭环延迟',
 'Stage C · 双臂':'Stage C.1 · 双臂',
 }
for old,new in replacements.items():
    if old not in s:raise ValueError('Missing report text: '+old)
    s=s.replace(old,new)
path.write_text(s,encoding='utf-8')
print('C.1 protocol and report wording prepared')
