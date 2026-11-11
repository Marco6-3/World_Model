"""exploreE factorial evaluation: recovery prior {on, off} x planner, perturbed (collar-slip) episodes only.

Mirrors bag_scene.evaluate exactly (same reset, same rng seeding seed+12000, same 216-step budget,
same 6-step commit and aperture interrupt), but calls explore_decide.decide and writes to exploreE/outputs.
Checkpoints are the frozen C.1 models in outputs/stage_c_bag/trained (read-only).

Usage (Isaac python):
  C:\\isaacsim_env\\Scripts\\python.exe exploreE/explore_eval.py --out exploreE/outputs/run_A --seeds 3000 3001
"""
import os, sys, json, argparse, traceback
from pathlib import Path
from time import perf_counter
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, r"C:\isaacsim_env\phase2_cuda"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
os.chdir(ROOT)
import torch

DEFAULT_METHODS = ['GPC-RANK', 'Ours 4D-WM', 'Ours tau=0.7', 'Ours tau=0.9']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    p.add_argument('--priors', nargs='+', default=['on', 'off'], choices=['on', 'off'])
    p.add_argument('--modes', nargs='+', default=['relaxed', 'elastic_slow', 'elastic_fast'])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.set_float32_matmul_precision('high')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 640, "height": 480, "multi_gpu": False,
                         "limit_cpu_threads": 3, "disable_viewport_updates": True,
                         "extra_args": ["--/persistent/physics/numThreads=2"]})
    status = 0
    try:
        from hashlib import sha256
        from bag_packing import ASSET, dump
        from bag_scene import BagTask
        from bag_learning import load_models
        from bag_common import CONTROL_DT, THRESHOLD
        from explore_decide import decide
        model_root = ROOT / 'outputs/stage_c_bag'
        models = load_models(model_root, 'cuda')
        hashes = {q.name: sha256(q.read_bytes()).hexdigest() for q in (model_root / 'trained').glob('*.pt')}
        task = BagTask(ASSET)
        variants = [m if pr == 'on' else m + '|noprior' for m in args.methods for pr in args.priors]
        journal = args.out / 'episodes.jsonl'
        prior_rows = [json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []
        done = {(r['method'], r['mode'], r['order'], r['seed']) for r in prior_rows}
        with journal.open('a', encoding='utf-8') as stream:
            for seed in args.seeds:
                for mode in args.modes:
                    for order in (0, 1):
                        for method in variants:
                            key = (method, mode, order, seed)
                            if key in done: continue
                            task.reset(mode, order, seed)
                            rng = torch.Generator().manual_seed(seed + 12000)
                            latency = []; commits = []; interrupts = 0; emergencies = 0
                            for phase, item in enumerate(task.sequence):
                                task.phase = phase; task.active = item; task.observations[-1] = task.observe()
                                task.spread([0, .22, .215], .032); task.pickup(item)
                                start = task.tick
                                while task.tick - start < 216:
                                    begin = perf_counter()
                                    actions, diag = decide(models, task.history(), method, rng)
                                    latency.append((perf_counter() - begin) * 1000); emergencies += diag['emergency']
                                    executed = 0
                                    for action in actions[:min(6, 216 - (task.tick - start))]:
                                        prev = task.env.aperture
                                        task.step(action, learn=True, perturb=True); executed += 1
                                        if task.release_ready(): break
                                        if prev - task.env.aperture > .006: interrupts += 1; break
                                    commits.append(executed)
                                    if task.release_ready(): break
                                task.learn_ranges.append((start, task.tick)); task.end_item()
                            task.phase = 2; task.close()
                            result = task.summary(method)
                            result.update(perturbed=True, perturbation_version='forced_finger_release_v2',
                                          recovery_prior=('|noprior' not in method), decision_ms=latency,
                                          committed_steps=commits, aperture_interrupts=interrupts,
                                          emergency_decisions=emergencies, checkpoint_hashes=hashes,
                                          execution_context='exploreE_factorial')
                            name = f"{seed}_{mode}_{order}_{method.replace('|', '_').replace(' ', '_').replace('=', '')}.npz"
                            result['trace_file'] = 'traces/' + name
                            task.save(args.out / result['trace_file'], result)
                            stream.write(json.dumps(result, allow_nan=False) + '\n'); stream.flush(); done.add(key)
                            print(f"E {len(done)} {method} {mode}/{order}/{seed}: success={result['success']} "
                                  f"collision={result['collision']} wd_ro={result['withdraw_reopen']} safe={result['safe_recovery']}", flush=True)
        rows = [json.loads(l) for l in journal.read_text().splitlines()]
        dump(args.out / 'results.json', dict(seeds=args.seeds, modes=args.modes, methods=variants, episodes=rows,
                                            control_dt=CONTROL_DT, collision_threshold_Ns=THRESHOLD,
                                            protocol_version='stage_c1', note='exploreE prior ablation'))
    except Exception:
        traceback.print_exc(); status = 1
    sys.stdout.flush(); sys.stderr.flush(); os._exit(status)


if __name__ == '__main__':
    main()
