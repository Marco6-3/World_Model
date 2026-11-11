"""Stage C: CUDA learning and IsaacLab bimanual elastic-collar bag packing."""
import os
import sys
from pathlib import Path

CUDA_SITE = r"C:\isaacsim_env\phase2_cuda"
if "torch" not in sys.modules and os.path.isdir(CUDA_SITE):
    sys.path.insert(0, CUDA_SITE)

import argparse
import json
import traceback
import torch

OUT = Path(__file__).resolve().parent / "outputs/stage_c_bag"
ASSET = Path("C:/isaac_ws/FlexiTac-IsaacSim-Simulation/Isaacsim_tactile_env/output/aloha_urdf/aloha_tactile.usd")
MODES = ("relaxed", "elastic_slow", "elastic_fast")
METHODS = ("BC H4", "Joint-FM", "GPC-RANK", "GPC-OPT", "Ours Short-MLP H4",
           "Ours 4D-WM", "Ours tau=0.7", "Ours tau=0.9")


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("smoke", "calibrate", "collect", "fit", "evaluate", "probe", "report"))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--robot-usd", type=Path, default=ASSET)
    p.add_argument("--model-root", type=Path)
    p.add_argument("--execution-context", default="unspecified")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--scheme", choices=("direct_insert","spread_and_hold","slip_withdraw_reopen","exploration","hold_through_slip"), default="spread_and_hold")
    p.add_argument("--order", type=int, choices=(0,1), default=0)
    p.add_argument("--flow-steps", type=int, default=5000)
    p.add_argument("--world-steps", type=int, default=3500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    if args.command in ("fit", "evaluate") and args.device != "cuda":
        raise RuntimeError("Stage C training and policy evaluation require the requested CUDA environment")
    if args.command == "fit":
        from bag_learning import fit
        fit(args)
        return
    if args.command == "report":
        from bag_report import build_report
        build_report(args.out)
        return
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 640, "height": 480, "multi_gpu": False,
                         "limit_cpu_threads": 3, "disable_viewport_updates": True,
                         "extra_args": ["--/persistent/physics/numThreads=2"]})
    status = 0
    try:
        from bag_scene import BagTask, collect, evaluate
        task = BagTask(args.robot_usd)
        if args.command == "calibrate":
            import numpy as np
            from bag_common import CONTROL_DT
            rows=[]
            for mode in args.modes:
                task.reset(mode,0,901)
                task.spread([0,.22,.23],.032)
                start=task.tick
                task.do("release_response",task.hold,[0,.22,.23],.032,.032,60)
                aperture=np.asarray(task.frames)[start:,1]
                crossing=np.flatnonzero(aperture<.070)
                row=dict(mode=mode,initial_aperture_m=float(task.frames[start-1][1]),
                         aperture_m=aperture.tolist(),dt=CONTROL_DT,
                         time_below_70mm_s=float((crossing[0]+1)*CONTROL_DT) if len(crossing) else None)
                rows.append(row)
                task.save(args.out/f"calibration_{mode}.npz",row)
            dump(args.out/"calibration.json",rows)
        elif args.command == "smoke":
            rows = []
            for mode in args.modes:
                row = task.expert_episode(mode, args.order, 900, args.scheme)
                task.save(args.out / f"smoke_{mode}_{args.scheme}_{args.order}.npz", row)
                rows.append(row)
                print(row, flush=True)
            dump(args.out / f"smoke_{args.scheme}_{args.order}.json", rows)
        elif args.command == "collect":
            collect(task, args)
        elif args.command == "probe":
            from bag_probe import physical_probe
            physical_probe(task,args)
        else:
            evaluate(task, args)
    except Exception:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    main()
