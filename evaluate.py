"""Closed-loop evaluation of the common IsaacLab packing task."""

from math import asin
from time import perf_counter
import json

import numpy as np
import torch

from collect import CONTROL_DT, GRASP_GAP, HOLD_SETTLE_STEPS, LEFT_HOLD, LEFT_STANDBY, OPEN_GAP, PackingTask
from control import RELEASE_STEPS
from inference import TAU_START, discounted_risk, propose
from learn import load_model
from report import ABLATIONS, METHODS, RECOVERY_METHODS, write_report
from scene import COLLISION_THRESHOLD, DEPTH, DYNAMICS, TARGETS


COMMIT = 8
MAX_CHUNKS = 26


def learned_insert(task, model, rng, method, tau, exact):
    samples = []
    released = False
    for _ in range(MAX_CHUNKS):
        start = perf_counter()
        result = propose(model, np.asarray(task.history), task.stage, method, rng, tau_start=tau, exact=exact)
        milliseconds = (perf_counter() - start) * 1000
        planned = result.action[:COMMIT]
        offset = len(task.trace)
        for action in planned.numpy().copy():
            action[6] = GRASP_GAP
            task.step(action)
            if task.release_ready(task.active) or task.placed(task.active):
                released = True
                break
        executed = task.trace[offset:]
        sample = {
            "ms": milliseconds, "flow_batches": result.flow_batches, "flow_particles": result.flow_particles,
            "world_batches": result.world_batches, "world_rollouts": result.world_rollouts,
            "committed_steps": len(executed),
        }
        if result.predicted is not None:
            pred = result.predicted[:len(executed)].clone()
            pred[:, 1:3].clamp_(0, 1)
            pred[:, 3].clamp_min_(0)
            actual_actions = torch.as_tensor(np.stack([row[1] for row in executed]))
            truth = torch.as_tensor(np.stack([row[2] for row in executed]))
            sample.update(predicted_cost=float(discounted_risk(pred, actual_actions, terminal=False)),
                          actual_cost=float(discounted_risk(truth, actual_actions, terminal=False)),
                          predicted_peak_impulse=float(pred[:, 3].max()),
                          actual_peak_impulse=float(truth[:, 3].max()))
        samples.append(sample)
        if released:
            last = task.last_command.copy()
            task.do("latched_release", last[:3], last[3:6], OPEN_GAP, RELEASE_STEPS)
            task.end_insertion()
            task.exit_box(task.active, last[:3])
            return samples, task.placed(task.active)
    task.end_insertion()
    return samples, False


def timer_branch(task, pre):
    angles = [task.env.angle]
    for _ in range(5):
        task.do("timer_release", LEFT_STANDBY, pre, GRASP_GAP, 1)
        angles.append(task.env.angle)
    omega = (angles[-1] - angles[-3]) / (2 * CONTROL_DT)
    angle_safe = asin((0.024 + 0.012) / DEPTH)
    close_time = max(0.0, angles[-1] - angle_safe) / (-omega) if omega < -0.02 else float("inf")
    hold = angles[-1] <= angle_safe or close_time < 0.85
    if hold:
        task.open_lid(pre, GRASP_GAP)
    return hold


def baseline_insert(task, item, method, pre, speed):
    hold = method == "B1" or timer_branch(task, pre)
    x = TARGETS[item][0]
    left = LEFT_HOLD if hold else LEFT_STANDBY
    if hold:
        task.do("baseline_reacquire_lid", left, pre, GRASP_GAP, HOLD_SETTLE_STEPS)
    task.do("baseline_enter", left, [x, 0.032, 0.130], GRASP_GAP, speed)
    task.do("baseline_lower", left, [x, -0.010, 0.045], GRASP_GAP, 29)
    task.do("baseline_release", left, [x, -0.010, 0.045], OPEN_GAP, RELEASE_STEPS)
    task.end_insertion()
    task.exit_box(item, left)
    return hold


def recovered(trace, initial_y):
    """Detect an outward retreat followed by reopening in the physical trace."""
    low_angle = None
    for belief, *_ in trace:
        last = belief[-1]
        angle, right_y = last[32], last[38]
        if right_y >= max(0.145, initial_y + 0.020):
            low_angle = angle if low_angle is None else min(low_angle, angle)
        if low_angle is not None and angle >= 0.55 and angle - low_angle >= 0.18:
            return True
    return False


def run_episode(task, model, regime, seed, method, tau, exact, *, perturbed=False):
    rng = torch.Generator().manual_seed(5000 + seed)
    task.reset(regime, seed)
    park = np.array([0.0, 0.21, 0.17])
    task.open_lid(park, OPEN_GAP)
    samples, baseline, placed = [], [], []
    recovery = {}
    for item in (0, 1):
        task.stage = item + 1
        task.open_lid(park, OPEN_GAP)
        task.pick(item)
        pre = task.pre_entry(item)
        task.begin_insertion()
        if perturbed and item == 0:
            task.perturb_entry(pre)
            recovery = {"handoff_angle_rad": task.env.angle, "handoff_right_y_m": float(task.controller.tip(1)[1])}
        offset = len(task.trace)
        if method in ("B1", "B2"):
            baseline.append(baseline_insert(task, item, method, pre, 26 + seed % 9))
            placed.append(task.placed(item))
        else:
            chunks, success = learned_insert(task, model, rng, method, tau, exact)
            samples.extend(chunks)
            placed.append(success)
        if recovery and item == 0:
            _, stop = task.insertion_ranges[-1]
            recovery["withdrew_and_reopened"] = recovered(
                task.trace[offset:stop], recovery["handoff_right_y_m"],
            )
        if not placed[-1]:
            break
    if len(placed) == 2 and all(placed):
        task.stage = 3
        task.close_lid()
    peak = float(max(record[2][3] for record in task.trace))
    seconds = CONTROL_DT * len(task.trace)
    left_seconds = CONTROL_DT * task.left_steps
    success = len(placed) == 2 and all(placed) and task.placed(0) and task.placed(1) and task.env.angle < 0.075
    if recovery:
        recovery["safe_recovery"] = success and peak < COLLISION_THRESHOLD and recovery["withdrew_and_reopened"]
    return {
        "method": method, "regime": regime, "seed": seed, "tau_start": tau,
        "placed_a": task.placed(0), "placed_b": task.placed(1), "final_angle": task.env.angle,
        "success": success,
        "pinch_or_collision": peak >= COLLISION_THRESHOLD, "peak_impulse_Ns": peak,
        "elapsed_s": seconds, "left_engaged_s": left_seconds, "cost": seconds + 0.35 * left_seconds,
        "chunks": samples, "baseline_hold": baseline,
        "final_object_xyz_m": [obj.data.root_link_pos_w[0].tolist() for obj in task.env.objects],
        **recovery,
    }


def evaluate(out, robot_usd, model_path, seeds, *, exact=False, regimes=tuple(DYNAMICS)):
    task = PackingTask(robot_usd)
    model = load_model(model_path)
    rows, ablation, recovery = [], [], []

    def record(stream, row):
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        print({k: v for k, v in row.items() if k != "chunks"}, flush=True)

    with (out / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for seed in seeds:
            for regime in regimes:
                default = {}
                for method in METHODS:
                    tau = 1.0 if method == "M3" else TAU_START
                    row = run_episode(task, model, regime, seed, method, tau, exact)
                    default[method] = row
                    rows.append(row)
                    record(stream, row)
                    write_report(out, rows, ablation, recovery)
                for tau in ABLATIONS:
                    if tau in (TAU_START, 1.0):
                        row = default["M4-b" if tau == TAU_START else "M3"]
                    else:
                        row = run_episode(task, model, regime, seed, "M4-b", tau, exact)
                        record(stream, row)
                    ablation.append(row)
                    write_report(out, rows, ablation, recovery)
                if regime in ("slow", "fast"):
                    for method in RECOVERY_METHODS:
                        tau = 1.0 if method == "M3" else TAU_START
                        row = run_episode(task, model, regime, seed, method, tau, exact, perturbed=True)
                        recovery.append(row)
                        record(stream, row)
                    write_report(out, rows, ablation, recovery)
    (out / "matrix.json").write_text(json.dumps({"main": rows, "ablation": ablation, "recovery": recovery,
                                               "exact_jacobian": exact},
                                               indent=2), encoding="utf-8")


def gate(out, robot_usd, model_path, seeds):
    if len(set(seeds)) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("The gate requires at least three distinct evaluation seeds")
    task = PackingTask(robot_usd)
    model = load_model(model_path)
    static = []
    fast = []
    for seed in seeds:
        base = run_episode(task, model, "static", seed, "M1", TAU_START, False)
        hold = run_episode(task, model, "fast", seed, "B1", TAU_START, False)
        static.append(base)
        fast.append(hold)
        for row in (base, hold):
            print({k: v for k, v in row.items() if k != "chunks"}, flush=True)
    static_rate = sum(row["success"] for row in static) / len(static)
    fast_collisions = sum(row["pinch_or_collision"] for row in fast)
    report = {
        "passed": static_rate > 0.8 and fast_collisions == 0,
        "m1_static_success_rate": static_rate, "b1_fast_collisions": fast_collisions,
        "m1_static": static, "b1_fast": fast,
    }
    (out / "gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report["passed"]
