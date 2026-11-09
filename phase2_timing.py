"""Measure Push-T planners on identical held-out observations after other jobs finish."""

import pusht_benchmark as p
import argparse
from pathlib import Path
import json
from time import perf_counter
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2_1/pusht"))
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    model = p.PushModels().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(args.out / "model.pt", map_location="cuda", weights_only=True))
    train = json.loads((args.out / "training.json").read_text())
    expert = p.read_expert(args.out / "pusht.zip")
    histories = []
    for episode in train["val_expert_ids"]:
        states = expert[episode][0]
        obs = p.features(states, np.concatenate((states[:1], states[:-1])))
        t = max(p.HISTORY, len(states) // 2)
        histories.append(obs[t-p.HISTORY:t])
    methods = list(p.METHODS)
    for method in methods:
        p.decide(model, histories[0], p.METHODS[method], torch.Generator().manual_seed(0))
    rows = []
    for repeat in range(args.repeats):
        for method in methods[repeat % len(methods):] + methods[:repeat % len(methods)]:
            rng = torch.Generator().manual_seed(50000 + repeat)
            torch.cuda.synchronize()
            start = perf_counter()
            _, calls, rollouts = p.decide(model, histories[repeat % len(histories)], p.METHODS[method], rng)
            torch.cuda.synchronize()
            rows.append(dict(method=method, repeat=repeat, ms=(perf_counter() - start) * 1000,
                             calls=calls, rollouts=rollouts))
        print(f"Timing {repeat + 1}/{args.repeats}", flush=True)
    summary = {method: dict(p50_ms=float(np.median([r["ms"] for r in rows if r["method"] == method])),
                            mean_ms=float(np.mean([r["ms"] for r in rows if r["method"] == method])))
               for method in methods}
    p.dump(args.out / "isolated_timing.json", dict(device=torch.cuda.get_device_name(), torch=str(torch.__version__),
           protocol="Same held-out expert histories; warmed planners; rotating order; synchronized end-to-end calls",
           repeats=args.repeats, summary=summary, rows=rows))


if __name__ == "__main__":
    main()
