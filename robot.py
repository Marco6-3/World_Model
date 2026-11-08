"""Cartesian waypoint control for the simulated ALOHA articulation."""

from math import atan2, cos, sin

import numpy as np
import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from control import LEFT_HOLD_POSITION
from scene import DYNAMICS, HINGE, STAGING, PackingScene


class WaypointController:
    """Track the midpoint of the tactile pads with an inward-facing gripper."""

    def __init__(self, env):
        self.robot = env.robot
        self.ids = []
        self.fingers = []
        self.bodies = []
        self.controllers = []
        self.bases = ((-0.005, -0.525), (0.005, 0.525))
        for side in ("left", "right"):
            self.ids.append(self.robot.find_joints([
                f"{side}_{name}" for name in ("waist", "shoulder", "elbow", "forearm_roll",
                                               "wrist_angle", "wrist_rotate")
            ], preserve_order=True)[0])
            self.fingers.append(self.robot.find_joints(
                [f"{side}_left_finger", f"{side}_right_finger"], preserve_order=True,
            )[0])
            self.bodies.append(self.robot.find_bodies(f"{side}_vx300s_fingers_link")[0][0])
            self.controllers.append(DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls",
                    ik_params={"lambda_val": 0.065},
                ),
                num_envs=1, device="cpu",
            ))
        if not self.robot.is_fixed_base:
            raise RuntimeError("The ALOHA asset needs a fixed base for waypoint IK")
        self.pad_offset = torch.tensor([[0.100, 0.0, 0.0]])

    def tip(self, index):
        body = self.bodies[index]
        pos = self.robot.data.body_link_pos_w[:, body]
        quat = self.robot.data.body_link_quat_w[:, body]
        return (pos + quat_apply(quat, self.pad_offset))[0].numpy()

    def action(self, left, right, left_gap=0.008, right_gap=0.035):
        jacobians = self.robot.root_physx_view.get_jacobians()
        joint_pos = self.robot.data.joint_pos
        result = joint_pos.clone()
        for index, (goal, gap) in enumerate(((left, left_gap), (right, right_gap))):
            ids = self.ids[index]
            body = self.bodies[index]
            if goal is not None:
                dx = goal[0] - self.bases[index][0]
                dy = goal[1] - self.bases[index][1]
                quat = quat_from_euler_xyz(torch.zeros(1), torch.tensor([0.16]), torch.tensor([atan2(dy, dx)]))
                position = torch.as_tensor(goal, dtype=torch.float32).reshape(1, 3)
                link_goal = position - quat_apply(quat, self.pad_offset)
                controller = self.controllers[index]
                controller.set_command(torch.cat((link_goal, quat), dim=-1))
                current = joint_pos[:, ids]
                target = controller.compute(
                    self.robot.data.body_link_pos_w[:, body], self.robot.data.body_link_quat_w[:, body],
                    jacobians[:, body - 1][:, :, ids], current,
                )
                result[:, ids] = current + (target - current).clamp(-0.14, 0.14)
            result[:, self.fingers[index]] = torch.tensor([gap, -gap])
        low, high = self.robot.data.soft_joint_pos_limits.unbind(-1)
        return torch.maximum(low, torch.minimum(high, result))


def handle_top(angle):
    return np.array([-0.047, HINGE[1] - 0.047 * cos(angle) - 0.062 * sin(angle),
                     HINGE[2] - 0.047 * sin(angle) + 0.062 * cos(angle)])


def probe(out, robot_usd):
    """Run a lid-opening and a right-gripper pick probe before collection."""
    env = PackingScene(robot_usd)
    controller = WaypointController(env)
    env.reset(DYNAMICS["static"])
    for _ in range(30):
        env.step()
    print("home pad positions", controller.tip(0), controller.tip(1), flush=True)

    left_hover = handle_top(0) + [0, 0, 0.030]
    left_press = LEFT_HOLD_POSITION
    right_pick = np.asarray(STAGING[0])
    right_hover = right_pick + [0, 0, 0.048]
    right_lift = right_pick + [0, 0, 0.062]
    waypoints = [
        ("approach", left_hover, right_hover, 0.032, 180),
        ("press_and_pick", left_press, right_pick, 0.032, 160),
        ("grasp", left_press, right_pick, 0.010, 90),
        ("lift", left_press, right_lift, 0.010, 140),
        ("enter", left_press, (-0.047, 0.032, 0.131), 0.010, 160),
        ("place", left_press, (-0.047, -0.010, 0.045), 0.010, 130),
        ("release", left_press, (-0.047, -0.010, 0.045), 0.032, 80),
        ("return", left_hover, right_hover, 0.032, 150),
    ]
    trace = []
    for name, left, right, gap, steps in waypoints:
        for _ in range(steps):
            targets = controller.action(left, right, right_gap=gap)
            env.step(targets)
            obj = env.objects[0].data.root_link_pos_w[0].numpy().copy()
            trace.append([env.angle, *controller.tip(0), *controller.tip(1), *obj,
                          *env.labels(0)])
        print(name, trace[-1], env.robot.data.joint_pos[0].tolist(), flush=True)
    np.savez_compressed(out / "robot_waypoints.npz", trace=np.asarray(trace))
