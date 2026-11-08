"""Scripted PhysX rollouts for the spring-lid packing benchmark."""

from math import cos, sin

import json
import numpy as np

from control import ARM_STEP_M, LEFT_HOLD_POSITION, RELEASE_STEPS
from robot import WaypointController, handle_top
from scene import COLLISION_THRESHOLD, DEPTH, DT, DYNAMICS, HINGE, PackingScene, STAGING, TARGETS, envelope


SUBSTEPS = 4
CONTROL_DT = DT * SUBSTEPS
HORIZON = 24
HISTORY = 12
LEFT_STANDBY = handle_top(0) + [0, -0.045, 0.06]
LEFT_HOVER = handle_top(0) + [0, 0, 0.03]
# Press the rear lever on its outer half to get enough opening torque in fast.
LEFT_HOLD = np.asarray(LEFT_HOLD_POSITION)
OPEN_GAP = 0.032
GRASP_GAP = 0.010
SCHEMES = ("direct", "hold", "delayed", "recover")
RELEASE_OBSERVATION_STEPS = 4
HOLD_SETTLE_STEPS = 12


class PackingTask:
    """One simulator/control path for data generation and evaluation."""

    def __init__(self, robot_usd):
        self.env = PackingScene(robot_usd)
        self.controller = WaypointController(self.env)

    def reset(self, regime, seed):
        rng = np.random.default_rng(seed)
        self.pickups = np.asarray(STAGING) + rng.uniform(
            [-0.005, -0.004, 0], [0.005, 0.004, 0], size=(2, 3),
        )
        self.env.reset(DYNAMICS[regime], items=self.pickups)
        self.stage = 0
        self.memory = np.zeros(2, dtype=np.int8)  # outside, grasped, channel, inside, uncertain
        self.trace = []
        self.checkpoints = []
        self.branch_starts = []
        self.insertion_ranges = []
        self.insertion_step = None
        self.left_steps = 0
        for _ in range(24):
            self.env.step()
        self.last_command = np.array([*self.controller.tip(0), *self.controller.tip(1), OPEN_GAP],
                                     dtype=np.float32)
        self.history = [self.observe()] * HISTORY

    @property
    def active(self):
        return min(1, max(0, self.stage - 1))

    def observe(self):
        q = self.env.robot.data
        features = [q.joint_pos[0].numpy(), q.joint_vel[0].numpy()]
        features.append([self.env.angle, float(self.env.lid.data.root_com_ang_vel_w[0, 0])])
        features.extend((self.controller.tip(0), self.controller.tip(1)))
        for index, obj in enumerate(self.env.objects):
            pose = obj.data.root_link_pose_w[0].numpy()
            # Geometric observation with the contents masked under the opaque
            # lid. The state memory is based on the last visible placement.
            outside = pose[1] >= DEPTH / 2
            visible = bool(outside or self.env.angle > 0.60)
            if visible:
                lo, hi = envelope(pose)
                at_bottom = hi[2] <= 0.060 and hi[1] < DEPTH / 2 - 0.008
                in_channel = lo[1] < DEPTH / 2 + 0.004 and hi[1] > DEPTH / 2 - 0.008
                near_tip = np.linalg.norm(self.controller.tip(1) - pose[:3]) < 0.037
                gripper = q.joint_pos[0, self.controller.fingers[1]].abs().mean().item()
                self.memory[index] = 3 if at_bottom else 2 if in_channel else 1 if near_tip and gripper < 0.026 else 0
            elif self.memory[index] != 3:
                self.memory[index] = 4
            features.extend((pose[:3] if visible else np.zeros(3), [visible], np.eye(5)[self.memory[index]]))
        features.extend(([-1.0 if self.insertion_step is None else self.insertion_step / 100.0],
                         self.last_command))
        return np.concatenate(features).astype(np.float32)

    def begin_insertion(self):
        self.branch_starts.append(len(self.trace))
        self.insertion_step = 0
        self.history[-1] = self.observe()

    def end_insertion(self):
        self.insertion_ranges.append((self.branch_starts[-1], len(self.trace)))
        self.insertion_step = None
        self.history[-1] = self.observe()

    def step(self, action, *, expert=False):
        if self.insertion_step is not None:
            action = np.array(action, dtype=np.float32, copy=True)
            delta = (action[:6] - self.last_command[:6]).reshape(2, 3)
            ratio = ARM_STEP_M / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), ARM_STEP_M)
            action[:6] = self.last_command[:6] + (ratio * delta).ravel()
        left, right, gap = action[:3], action[3:6], action[6]
        gap = float(np.clip(gap, 0.006, 0.038))
        start_belief = np.asarray(self.history, dtype=np.float32)
        impulse = 0.0
        for _ in range(SUBSTEPS):
            self.env.step(self.controller.action(left, right, right_gap=gap))
            impulse = max(impulse, self.env.contact_impulse(closing=self.stage == 3))
        state = self.env.labels(self.active, impulse=impulse)
        self.trace.append((start_belief, action, state, self.stage, expert and self.insertion_step is not None))
        self.left_steps += int(np.linalg.norm(self.controller.tip(0) - handle_top(self.env.angle)) < 0.059)
        self.last_command = action
        if self.insertion_step is not None:
            self.insertion_step += 1
        self.history = self.history[1:] + [self.observe()]

    def do(self, name, left, right, gap, steps, *, expert=False):
        action = np.asarray([*left, *right, gap], dtype=np.float32)
        for _ in range(steps):
            self.step(action, expert=expert)
        self.checkpoints.append((
            name, self.env.angle, self.controller.tip(0).tolist(), self.controller.tip(1).tolist(),
            [o.data.root_link_pos_w[0].tolist() for o in self.env.objects],
        ))

    def open_lid(self, right, gap, *, recovery_expert=False):
        self.do("approach_lid", LEFT_HOVER, right, gap, 22, expert=recovery_expert)
        self.do("press_lid", LEFT_HOLD, right, gap, 31, expert=recovery_expert)

    def pick(self, item):
        pick = self.pickups[item]
        hover = pick + [0, 0, 0.068]
        self.do("approach_pick", LEFT_HOLD, hover, OPEN_GAP, 36)
        self.do("lower_pick", LEFT_HOLD, pick, OPEN_GAP, 29)
        self.do("grasp", LEFT_HOLD, pick, GRASP_GAP, 19)
        self.do("lift", LEFT_HOLD, hover, GRASP_GAP, 31)

    def pre_entry(self, item):
        pre = np.array([TARGETS[item][0], 0.126, 0.133])
        self.do("pre_entry", LEFT_HOLD, pre, GRASP_GAP, 22)
        self.do("release_observation", LEFT_HOVER, pre, GRASP_GAP, RELEASE_OBSERVATION_STEPS)
        return pre

    def perturb_entry(self, pre):
        release_steps = 18 if self.env.dynamics == DYNAMICS["slow"] else 4
        self.do("unsupported", LEFT_HOVER, pre, GRASP_GAP, release_steps - RELEASE_OBSERVATION_STEPS)
        self.do("probe_entry", LEFT_HOVER, [pre[0], 0.117, 0.133], GRASP_GAP, 3)

    def insert(self, item, scheme, *, speed):
        x, _, _ = TARGETS[item]
        pre = self.pre_entry(item)
        inside = np.array([x, 0.032, 0.130])
        lower = np.array([x, -0.010, 0.045])
        self.begin_insertion()
        insert_start = len(self.trace)
        left = LEFT_HOLD if scheme == "hold" else LEFT_STANDBY
        recovery_expert = scheme == "recover" and self.env.dynamics != DYNAMICS["static"]
        if scheme == "delayed":
            self.do("unsupported", left, pre, GRASP_GAP, 26, expert=False)
        if scheme == "recover":
            self.perturb_entry(pre)
            self.do("withdraw", left, [x, 0.17, 0.15], GRASP_GAP, 22, expert=recovery_expert)
            self.open_lid([x, 0.17, 0.15], GRASP_GAP, recovery_expert=recovery_expert)
            left = LEFT_HOLD
            self.do("resume_entry", left, pre, GRASP_GAP, 19, expert=recovery_expert)
        if scheme == "hold":
            self.do("reacquire_lid", LEFT_HOLD, pre, GRASP_GAP, HOLD_SETTLE_STEPS, expert=True)
        candidate = scheme in ("hold", "direct") or recovery_expert
        self.do("enter", left, inside, GRASP_GAP, speed, expert=candidate)
        self.do("lower_inside", left, lower, GRASP_GAP, 29, expert=candidate)
        self.do("release_item", left, lower, OPEN_GAP, RELEASE_STEPS, expert=candidate)
        self.end_insertion()
        branch = self.trace[insert_start:]
        valid = self.placed(item) or (
            scheme == "direct" and any(row[2][3] >= COLLISION_THRESHOLD for row in branch)
        )
        for index, record in enumerate(branch, start=insert_start):
            if record[2][3] >= COLLISION_THRESHOLD and scheme == "direct":
                valid = False
            self.trace[index] = (*record[:4], bool(record[4] and valid and record[2][3] < COLLISION_THRESHOLD))
        self.exit_box(item, left)

    def exit_box(self, item, left):
        x = TARGETS[item][0]
        self.do("lift_empty", left, [x, -0.010, 0.155], OPEN_GAP, 27)
        self.do("return_empty", left, [x, 0.19, 0.16], OPEN_GAP, 36)

    def release_ready(self, index):
        pose = self.env.objects[index].data.root_link_pose_w[0].numpy()
        lo, hi = envelope(pose)
        target = TARGETS[index]
        return bool(
            hi[1] <= DEPTH / 2 - 0.008 and lo[1] >= -DEPTH / 2 + 0.008
            and 0.012 <= lo[2] and hi[2] <= 0.098
            and np.linalg.norm(pose[:2] - target[:2]) < 0.034
        )

    def placed(self, index):
        obj = self.env.objects[index]
        pose = obj.data.root_link_pose_w[0].numpy()
        lo, hi = envelope(pose)
        target = TARGETS[index]
        return bool(
            hi[2] < 0.063 and lo[2] >= 0.011 and hi[1] <= DEPTH / 2 - 0.008
            and lo[1] >= -DEPTH / 2 + 0.008
            and np.linalg.norm(pose[:2] - target[:2]) < 0.039
        )

    def close_lid(self):
        x = 0.030
        self.do("free_lid", LEFT_STANDBY, [x, 0.155, 0.265], OPEN_GAP, 27)
        if self.env.angle > 0.08:
            initial = self.env.angle
            for i, angle in enumerate(np.linspace(initial, 0.0, 33)):
                y = HINGE[1] + (DEPTH - 0.02) * cos(angle) - 0.007 * sin(angle)
                z = HINGE[2] + (DEPTH - 0.02) * sin(angle) + 0.007 * cos(angle)
                right = [x, y, z + (0.027 if i == 0 else 0.002)]
                self.do("close_panel", LEFT_STANDBY, right, 0.009, 21 if i == 0 else 2)
        self.do("end", LEFT_STANDBY, [x, 0.19, 0.20], OPEN_GAP, 30)

    def rollout(self, regime, seed, scheme):
        self.reset(regime, seed)
        right_park = np.array([0.0, 0.21, 0.17])
        self.open_lid(right_park, OPEN_GAP)
        opened = bool(self.env.labels(0)[0] >= 0.035)
        placed_a = False
        for item in (0, 1):
            self.stage = item + 1
            self.open_lid(right_park, OPEN_GAP)
            self.pick(item)
            self.insert(item, scheme, speed=26 + seed % 9)
            if item == 0:
                placed_a = self.placed(0)
            if not self.placed(item):
                break
        placed_b = self.placed(1)
        if placed_a and placed_b:
            self.stage = 3
            self.close_lid()
        collision = max(record[2][3] for record in self.trace)
        success = bool(placed_a and self.placed(0) and self.placed(1) and self.env.angle < 0.075)
        summary = {
            "regime": regime, "seed": seed, "scheme": scheme, "opened": opened,
            "placed_a": placed_a, "placed_b": placed_b, "final_angle_rad": self.env.angle,
            "success": success, "peak_impulse_Ns": float(collision), "collided": bool(collision >= COLLISION_THRESHOLD),
            "steps": len(self.trace), "branch_starts": self.branch_starts, "insertion_ranges": self.insertion_ranges,
            "expert_steps": sum(record[4] for record in self.trace),
        }
        return summary

    def save(self, path, summary):
        histories, actions, states, stages, expert = map(np.asarray, zip(*self.trace, strict=True))
        if not all(np.isfinite(x).all() for x in (histories, actions, states)):
            raise RuntimeError(f"Non-finite rollout {summary}")
        np.savez_compressed(path, belief=histories, action=actions, next_state=states,
                            stage=stages, expert=expert, metadata=json.dumps(summary))


def collect(out, robot_usd, groups, first_seed, schemes, regimes=tuple(DYNAMICS)):
    task = PackingTask(robot_usd)
    summaries = []
    for seed in range(first_seed, first_seed + groups):
        for regime in regimes:
            for scheme in schemes:
                result = task.rollout(regime, seed, scheme)
                task.save(out / f"{seed:04d}_{regime}_{scheme}.npz", result)
                checkpoint_path = out / f"{seed:04d}_{regime}_{scheme}_waypoints.json"
                checkpoint_path.write_text(json.dumps(task.checkpoints, indent=2), encoding="utf-8")
                print(result, flush=True)
                summaries.append(result)
    (out / "rollouts.json").write_text(json.dumps({
        "control_dt_s": CONTROL_DT, "horizon": HORIZON, "history": HISTORY, "episodes": summaries,
    }, indent=2), encoding="utf-8")
