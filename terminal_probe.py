"""Offline risk-ranking and handoff posterior checks on grouped trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from inference import TERMINAL_CLEARANCE, TERMINAL_WEIGHT, discounted_risk
from learn import REGIMES, load_model


def episode(raw, seed, regime, scheme):
    with np.load(raw / f"{seed:04d}_{regime}_{scheme}.npz", allow_pickle=False) as data:
        meta = json.loads(data["metadata"].item())
        return meta["branch_starts"], data["belief"], data["action"]


@torch.no_grad()
def probe(models, raw, seeds):
    checks, classifications = [], []
    for seed in seeds:
        for index, regime in enumerate(REGIMES):
            starts, beliefs, hold = episode(raw, seed, regime, "hold")
            encoded = models.encode(torch.from_numpy(beliefs[starts]))
            posterior = models.xi_prob(encoded)
            for item, (start, p) in enumerate(zip(starts, posterior, strict=True)):
                classifications.append({
                    "seed": seed, "item": item, "regime": regime, "correct": int(p.argmax()) == index,
                    "probabilities": p.tolist(), "angle_rad": float(beliefs[start, -1, 32]),
                    "omega_rad_s": float(beliefs[start, -1, 33]),
                })
            if regime == "fast":
                continue
            direct_starts, _, direct = episode(raw, seed, regime, "direct")
            horizon = models.flow.horizon
            candidates = {}
            for scheme, actions, start in (("hold", hold, starts[0]), ("direct", direct, direct_starts[0])):
                action = torch.from_numpy(actions[start:start + horizon])
                states = models.world(encoded[:1].expand(3, -1, -1), torch.eye(4)[1].repeat(3, 1),
                                      models.actions(action).expand(3, -1, -1), torch.eye(3)) * models.state_scale
                base = discounted_risk(states, action, terminal=False) @ posterior[0]
                total = discounted_risk(states, action) @ posterior[0]
                candidates[scheme] = {"step_cost": float(base), "with_terminal": float(total),
                                      "terminal": float(total - base)}
            preferred = "direct" if regime == "static" else "hold"
            other = "hold" if preferred == "direct" else "direct"
            checks.append({
                "seed": seed, "regime": regime, "preferred": preferred, **candidates,
                "passed": candidates[preferred]["with_terminal"] < candidates[other]["with_terminal"],
            })
    return {
        "terminal_weight": TERMINAL_WEIGHT, "terminal_clearance_m": TERMINAL_CLEARANCE,
        "passed": all(row["passed"] for row in checks),
        "pairs": checks, "handoff": classifications,
        "handoff_accuracy": sum(row["correct"] for row in classifications) / len(classifications),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = probe(load_model(args.model), args.raw, args.seeds)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print({k: v for k, v in results.items() if k != "handoff"}, flush=True)
    return 0 if results["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
