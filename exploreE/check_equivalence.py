"""Verify explore_decide.decide(prior=on) == bag_learning.decide exactly, and that prior=off
changes outputs only in emergency states for multi-particle planners.
Uses real recorded C.1 perturbed traces (histories around the slip)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, r"C:\isaacsim_env\phase2_cuda"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np, torch
torch.set_num_threads(1); torch.set_float32_matmul_precision('high')
from bag_learning import load_models, decide as ref_decide
from bag_common import history_indices
from explore_decide import decide as new_decide

OUT = ROOT / 'outputs/stage_c_bag'
models = load_models(OUT, 'cuda')
data = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
rows = [r for r in data['episodes'] if r['perturbed'] and r['method'] == 'Ours tau=0.9'][:4]
methods = ['GPC-RANK', 'GPC-OPT', 'Ours 4D-WM', 'Ours tau=0.7', 'Ours tau=0.9', 'BC H4']
report = []
for r in rows:
    with np.load(OUT / r['trace_file']) as raw: obs = raw['observations']
    slip = [e['t'] for e in r['events'] if e['event'] == 'collar_slip'][0]
    for t in (slip - 20, slip + 3, slip + 8):
        hist = obs[history_indices(t)]
        emergency = bool(hist[-1, 4] < .085 or hist[-1, 4] - hist[-2, 4] < -.006)
        for m in methods:
            a, _ = ref_decide(models, hist, m, torch.Generator().manual_seed(7))
            b, _ = new_decide(models, hist, m, torch.Generator().manual_seed(7))
            c, _ = new_decide(models, hist, m + '|noprior', torch.Generator().manual_seed(7))
            report.append(dict(trace=r['trace_file'], t=int(t), emergency=emergency, method=m,
                               same_prior_on=float(np.abs(a - b).max()), diff_prior_off=float(np.abs(a - c).max())))
bad = [x for x in report if x['same_prior_on'] > 1e-6]
print(f'{len(report)} checks; prior=on mismatches: {len(bad)}')
for x in report:
    print(f"{x['method']:<14} t={x['t']:<4} emergency={x['emergency']!s:<5} on==ref:{x['same_prior_on']:.2e}  off-vs-on:{x['diff_prior_off']:.2e}")
(HERE / 'outputs').mkdir(exist_ok=True)
(HERE / 'outputs/equivalence_check.json').write_text(json.dumps(report, indent=1))
assert not bad, 'explore_decide diverges from bag_learning.decide with prior on'
