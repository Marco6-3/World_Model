"""Final provenance checks for C.1, separate from scientific performance acceptance."""
from pathlib import Path
from hashlib import sha256
import json
from datetime import datetime
from PIL import Image

root=Path(__file__).resolve().parent;out=root/'outputs/stage_c_bag'
def digest(p):return sha256(p.read_bytes()).hexdigest()
data=json.loads((out/'results.json').read_text(encoding='utf-8'))
train=json.loads((out/'training.json').read_text(encoding='utf-8'))
rows=data['episodes'];assert len(rows)==288
keys=[tuple(r[k] for k in ('method','mode','order','seed','perturbed')) for r in rows]
assert len(set(keys))==288
assert sum(r['perturbed'] for r in rows)==144
assert all(r['protocol_version']=='stage_c1' for r in rows)
hashes={p.name:digest(p) for p in (out/'trained').glob('*.pt')}
assert all(r['checkpoint_hashes']==hashes for r in rows)
assert all(digest(out/r['trace_file'])==r['trace_sha256'] for r in rows)
assert len(train['files'])==192
assert all(digest(out/'dataset'/r['file'])==r['sha256'] for r in train['files'])
assert train['flow_updates']==5000 and train['world_updates']==3500 and train['device']=='cuda'
assert set(train['models'])=={'full','short','four'}
obs=json.loads((out/'observation_audit.json').read_text(encoding='utf-8'))
assert not obs['violations']
assert all(obs['schemes'][s]==dict(n=48,success=48,collision=0) for s in ('spread_and_hold','slip_withdraw_reopen'))
assert obs['recovery_physics']==dict(slipped=48,slip_effective=48,safe_recovery=48)
def strip_section(s):
    a=s.find('<!-- STAGE_C_BAG_BEGIN -->');b=s.find('<!-- STAGE_C_BAG_END -->')
    return (s[:a]+s[b+len('<!-- STAGE_C_BAG_END -->'):]).rstrip() if a>=0 else s.rstrip()
for p in (root/'outputs/Phase2_GPC_Mamba_实验报告.md',root/'outputs/evaluation/results.md'):
    text=p.read_text(encoding='utf-8');prior=(out/'previous_reports'/p.name).read_text(encoding='utf-8')
    assert text.count('<!-- STAGE_C_BAG_BEGIN -->')==1
    assert strip_section(text)==strip_section(prior)
reports=[root/'outputs/StageC_Bag_Packing_实验报告.md',root/'outputs/Phase2_GPC_Mamba_实验报告.md',root/'outputs/evaluation/results.md']
for p in reports:
    s=p.read_text(encoding='utf-8');assert '\ufffd' not in s and '???' not in s
assert Image.open(out/'bag_packing_demo.png').size==(4680,3120)
fig=json.loads((out/'figure_manifest.json').read_text(encoding='utf-8'))
assert all(digest(out/r['trace_file'])==r['trace_sha256'] for r in fig['cases'])
timing=json.loads((out/'isolated_timing.json').read_text(encoding='utf-8'))
assert len(timing['histories'])==12 and len(timing['rows'])==96
record=dict(time=datetime.now().isoformat(),status='passed',dataset_episodes=192,evaluation_episodes=288,
            checkpoint_hashes=hashes,source_sha256={p.name:digest(p) for p in root.glob('bag_*.py')},
            reports_sha256={p.name:digest(p) for p in reports},png_sha256=digest(out/'bag_packing_demo.png'),
            scientific_acceptance=json.loads((out/'acceptance.json').read_text(encoding='utf-8')),
            note='Provenance consistency is not proof that performance targets were achieved')
(out/'audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k in ('time','status','dataset_episodes','evaluation_episodes')},indent=2))
