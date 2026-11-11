"""Analyse exploreE runs: re-audit outcomes from recorded poses (bag_audit.audit, same criterion as C.1),
then table + paired comparisons. Also adds the existing C.1 prior-on rows for seeds 2000-2002 when present.

Usage: python exploreE/analyze.py exploreE/outputs/run_A exploreE/outputs/run_B ...
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from bag_audit import audit


def load(folder):
    if folder == 'C1':  # existing, already-audited C.1 prior-on episodes (seeds 2000-2002)
        data = json.loads((ROOT / 'outputs/stage_c_bag/results.json').read_text(encoding='utf-8'))
        keep = ('GPC-RANK', 'Ours 4D-WM', 'Ours tau=0.7', 'Ours tau=0.9')
        rows = [r for r in data['episodes'] if r['perturbed'] and r['method'] in keep]
        for r in rows: r['_folder'] = 'C1'
        return rows
    folder = Path(folder)
    rows = [json.loads(l) for l in (folder / 'episodes.jsonl').read_text().splitlines()]
    data = audit(folder, dict(episodes=rows), write=False)
    for r in data['episodes']: r['_folder'] = str(folder)
    return data['episodes']


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(folders):
    rows = [r for f in folders for r in load(f)]
    keys = ('success', 'collision', 'withdraw_reopen', 'safe_recovery', 'collar_grazing_before_withdrawal')
    methods = sorted({r['method'] for r in rows}, key=lambda m: (m.split('|')[0], '|' in m))
    lines = ['| variant | n | success | collision | withdraw+reopen | **safe recovery** [95% Wilson] | grazing before withdraw | P50 ms |',
             '|---|---|---|---|---|---|---|---|']
    for m in methods:
        R = [r for r in rows if r['method'] == m]; n = len(R)
        c = {k: sum(bool(r.get(k)) for r in R) for k in keys}
        lo, hi = wilson(c['safe_recovery'], n)
        ms = np.median([v for r in R for v in r['decision_ms']])
        lines.append(f"| {m} | {n} | {c['success']} | {c['collision']} | {c['withdraw_reopen']} | "
                     f"**{c['safe_recovery']}** [{lo:.2f}, {hi:.2f}] | {c['collar_grazing_before_withdrawal']} | {ms:.0f} |")
    modes = ('relaxed', 'elastic_slow', 'elastic_fast')
    lines += ['', '| variant | ' + ' | '.join(f'safe {m}' for m in modes) + ' |', '|---|---|---|---|']
    for m in methods:
        cells = []
        for md in modes:
            R = [r for r in rows if r['method'] == m and r['mode'] == md]
            cells.append(f"{sum(bool(r['safe_recovery']) for r in R)}/{len(R)}")
        lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    # paired comparisons on identical (seed, mode, order)
    look = {(r['method'], r['seed'], r['mode'], r['order']): r for r in rows}
    def paired(a, b, k='safe_recovery'):
        cells = sorted({(s, m, o) for (meth, s, m, o) in look if meth == a} & {(s, m, o) for (meth, s, m, o) in look if meth == b})
        if not cells: return None
        d = np.array([int(look[a, *c][k]) - int(look[b, *c][k]) for c in cells])
        seeds = sorted({c[0] for c in cells})
        per_seed = np.array([d[[c[0] == s for c in cells]].mean() for s in seeds])
        bs = per_seed[np.random.default_rng(731).integers(0, len(seeds), (10000, len(seeds)))].mean(1)
        wins = int((d > 0).sum()); losses = int((d < 0).sum())
        from math import comb
        nn_ = wins + losses
        p = min(1., 2 * sum(comb(nn_, i) for i in range(0, min(wins, losses) + 1)) / 2 ** nn_) if nn_ else 1.
        return len(cells), d.mean(), np.quantile(bs, [.025, .975]), wins, losses, p
    comps = []
    bases = sorted({m.split('|')[0] for m in methods})
    for b in bases:
        comps.append((b, b + '|noprior'))
    for pr in ('', '|noprior'):
        for o in ('Ours tau=0.9', 'Ours tau=0.7', 'Ours 4D-WM'):
            comps.append((o + pr, 'GPC-RANK' + pr))
    lines += ['', '| comparison (A − B), paired cells | n | Δ safe recovery | seed-cluster bootstrap 95% CI | A wins / B wins | sign test p |', '|---|---|---|---|---|---|']
    for a, b in comps:
        res = paired(a, b)
        if res is None: continue
        n, mu, (lo, hi), w, l, p = res
        lines.append(f'| {a} − {b} | {n} | {mu:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {w} / {l} | {p:.3f} |')
    text = '\n'.join(lines)
    print(text)
    (HERE / 'outputs').mkdir(exist_ok=True)
    name = 'summary_' + '_'.join(Path(f).name for f in folders) + '.md'
    (HERE / 'outputs' / name).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1:])
