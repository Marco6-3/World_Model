"""Headless entry point for Phase 1."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


ASSET = Path(
    "C:/isaac_ws/FlexiTac-IsaacSim-Simulation/Isaacsim_tactile_env/"
    "output/aloha_urdf/aloha_tactile.usd"
)


def calibration(out, robot_usd):
    import numpy as np

    from scene import COLLISION_THRESHOLD, DT, DYNAMICS, PackingScene

    # The calibration fixture isolates lid dynamics before the ALOHA
    # integration. Its small force-controlled workpiece is a real rigid body.
    env = PackingScene(robot_usd, fixture=True)
    if env.robot is not None:
        print("ALOHA articulation", env.robot.joint_names, flush=True)
    report = {"physics_dt_s": DT, "backend": "IsaacLab / PhysX CPU", "runs": []}
    for regime, dynamics in DYNAMICS.items():
        for mode in ("release", "direct", "delayed", "withdraw"):
            open_angle = 0.83
            env.reset(dynamics, open_angle, ((-0.047, 0.125, 0.108), (0.047, 0.31, 0.05)))
            data = []
            for step in range(300):
                t = step * DT
                y, z = 0.125, 0.108
                if mode in ("direct", "delayed"):
                    start = 0.02 if mode == "direct" else 1.25
                    y = float(np.interp(t, [start, start + 0.46], [0.125, -0.010]))
                    z = float(np.interp(t, [start + 0.30, start + 0.49], [0.108, 0.029]))
                if mode == "withdraw":
                    y = float(np.interp(t, [0.0, 0.10, 0.20], [0.125, 0.105, 0.155]))
                env.step(fixture_target=(-0.047, y, z))
                state = env.labels(0)
                pose = env.objects[0].data.root_link_pos_w[0].tolist()
                support = env.support_contact.data.force_matrix_w[0, 0].norm(dim=-1).tolist()
                centre = env.lid.data.root_link_pos_w[0].tolist()
                omega = float(env.lid.data.root_com_ang_vel_w[0, 0])
                data.append([t + DT, env.angle, *pose, *map(float, state), *centre, omega, *support])
            rows = np.asarray(data)
            low = np.flatnonzero(rows[:, 5] < 0.005)
            summary = {
                "regime": regime, "stiffness_Nm_rad": dynamics.stiffness,
                "damping_Nms_rad": dynamics.damping, "mode": mode,
                "initial_angle_rad": open_angle, "final_angle_rad": float(rows[-1, 1]),
                "low_clearance_s": float(rows[low[0], 0]) if low.size else None,
                "occupied_steps": int(rows[:, 6].sum()),
                "occupied_at_end": bool(rows[-1, 6]),
                "peak_lid_object_impulse_Ns": float(rows[:, 8].max()),
                "collided": bool(np.any(rows[:, 8] >= COLLISION_THRESHOLD)),
                "final_workpiece_xyz_m": rows[-1, 2:5].tolist(),
            }
            np.savez_compressed(out / f"{regime}_{mode}.npz", trace=rows)
            print(summary, flush=True)
            report["runs"].append(summary)
    report["trace_columns"] = [
        "time_s", "angle_rad", "object_x_m", "object_y_m", "object_z_m",
        "clearance_m", "occupancy", "progress", "impulse_Ns",
        "lid_x_m", "lid_y_m", "lid_z_m", "lid_omega_rad_s",
        "front_force_N", "back_force_N", "west_force_N", "east_force_N",
    ]
    if env.robot is None:
        report["trace_columns"].append("table_force_N")
    if env.robot is not None:
        report["robot_dofs"] = env.robot.num_joints
        report["robot_bodies"] = env.robot.body_names
    (out / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "robot-probe", "collect", "evaluate", "gate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--robot-usd", type=Path, default=ASSET)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--regimes", nargs="+", choices=("static", "slow", "fast"),
                        default=("static", "slow", "fast"))
    parser.add_argument("--schemes", nargs="+", choices=("direct", "hold", "delayed", "recover"),
                        default=("direct", "hold", "delayed", "recover"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=(60, 61, 62))
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command != "calibrate" and not args.robot_usd.is_file():
        parser.error(f"Missing ALOHA asset at {args.robot_usd}")
    if args.command in ("evaluate", "gate") and (args.model is None or not args.model.is_file()):
        parser.error("Evaluation requires an existing --model checkpoint")

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import torch

    torch.set_num_threads(1)
    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": True, "width": 320, "height": 240, "multi_gpu": False,
        "limit_cpu_threads": 4, "disable_viewport_updates": True,
        "extra_args": ["--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false", "--/log/flush=true",
                       "--/persistent/physics/numThreads=2"],
    })
    exit_code = 0
    try:
        if args.command == "robot-probe":
            from robot import probe

            probe(args.out, args.robot_usd)
        elif args.command == "collect":
            from collect import collect

            collect(args.out, args.robot_usd, args.groups, args.first_seed, args.schemes, args.regimes)
        elif args.command == "evaluate":
            from evaluate import evaluate

            evaluate(args.out, args.robot_usd, args.model, args.eval_seeds,
                     exact=args.exact, regimes=args.regimes)
        elif args.command == "gate":
            from evaluate import gate

            exit_code = 0 if gate(args.out, args.robot_usd, args.model, args.eval_seeds) else 2
        else:
            calibration(args.out, None)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    # The headless jobs are synchronous and have written their outputs. Kit's
    # normal teardown can hang on this Windows installation after simulation.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
