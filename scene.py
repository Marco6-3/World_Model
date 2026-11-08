"""IsaacLab spring-lid packing scene and simulator-only labels."""

from dataclasses import dataclass
from math import atan2, cos, sin
from pathlib import Path

import numpy as np
import torch
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat


DT = 1 / 120
WIDTH = 0.24
DEPTH = 0.18
RIM = 0.070
LID_THICKNESS = 0.014
HINGE = (0.0, -DEPTH / 2, RIM + LID_THICKNESS / 2)
ENTRY = DEPTH / 2
FLOOR = 0.014
BLOCK_SIZE = (0.036, 0.048, 0.024)
TARGETS = ((-0.047, -0.010, 0.026), (0.047, -0.010, 0.026))
STAGING_HEIGHT = 0.075
STAGING = ((-0.047, 0.210, STAGING_HEIGHT + BLOCK_SIZE[2] / 2),
           (0.047, 0.210, STAGING_HEIGHT + BLOCK_SIZE[2] / 2))
MAX_ANGLE = 1.22
COLLISION_THRESHOLD = 0.004  # N s over one physics step
ROOT = "/World/Packing"
LID = f"{ROOT}/Lid"
OBJECTS = (f"{ROOT}/ItemA", f"{ROOT}/ItemB")
ROBOT = "/World/Robot"
HOME = {
    "waist": 0.0, "shoulder": -0.96, "elbow": 1.16, "forearm_roll": 0.0,
    "wrist_angle": -0.3, "wrist_rotate": 0.0, "left_finger": 0.035, "right_finger": -0.035,
}


@dataclass(frozen=True)
class Dynamics:
    stiffness: float  # N m / rad
    damping: float  # N m s / rad


DYNAMICS = {
    "static": Dynamics(0.0, 0.012),
    "slow": Dynamics(0.060, 0.035),
    "fast": Dynamics(0.62, 0.022),
}


def box(path, size, position, color, *, mass=None, gravity=True):
    """Spawn a static collider or dynamic cuboid with a small contact margin."""
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0),
        rigid_props=(
            sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=not gravity, solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            )
            if mass is not None else None
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass) if mass is not None else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
    )
    return cfg.func(path, cfg, translation=position)


def spawn_container(stage):
    UsdGeom.Xform.Define(stage, ROOT)
    shell = (0.22, 0.27, 0.35)
    wall = 0.008
    box(f"{ROOT}/Floor", (WIDTH, DEPTH, FLOOR - 0.005),
        (0, 0, (FLOOR + 0.005) / 2), shell)
    box(f"{ROOT}/Front", (WIDTH, wall, RIM - FLOOR),
        (0, ENTRY - wall / 2, (RIM + FLOOR) / 2), shell)
    box(f"{ROOT}/Back", (WIDTH, wall, RIM - FLOOR),
        (0, -ENTRY + wall / 2, (RIM + FLOOR) / 2), shell)
    for name, sign in (("West", -1), ("East", 1)):
        box(f"{ROOT}/{name}", (wall, DEPTH - 2 * wall, RIM - FLOOR),
            (sign * (WIDTH - wall) / 2, 0, (RIM + FLOOR) / 2), shell)

    # The rigid-body origin is the panel centre. The joint is offset to its rear
    # edge. Gravity is balanced so kappa=0 can genuinely hold its release pose.
    panel = box(LID, (WIDTH, DEPTH, LID_THICKNESS),
                (0, 0, HINGE[2]), (0.35, 0.40, 0.49), mass=0.16, gravity=False)
    PhysxSchema.PhysxContactReportAPI.Apply(panel).CreateThresholdAttr(0.0)
    # Keep the swept volume clear of the tabletop and the back rim.
    box(f"{LID}/RearLever", (0.044, 0.044, 0.014),
        (-0.047, -ENTRY - 0.047, 0.055), (0.47, 0.49, 0.53))
    box(f"{LID}/LeverBracket", (0.018, 0.060, 0.016),
        (-0.047, -ENTRY - 0.029, 0.006), shell)
    box(f"{LID}/LeverPost", (0.018, 0.018, 0.050),
        (-0.047, -ENTRY - 0.047, 0.029), shell)
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ROOT}/Hinge")
    joint.CreateBody1Rel().SetTargets([LID])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*HINGE))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, -DEPTH / 2, 0))
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(float(np.degrees(MAX_ANGLE)))


def envelope(pose):
    """Axis-aligned world envelope of the oriented workpiece."""
    rot = matrix_from_quat(torch.as_tensor(pose[3:7])).numpy()
    half_extent = np.abs(rot) @ (np.asarray(BLOCK_SIZE) / 2)
    return pose[:3] - half_extent, pose[:3] + half_extent


def condition_state(angle, pose, target, impulse):
    """Clearance (m), occupancy, release-pose progress, collision impulse (N s)."""
    lo, hi = envelope(pose)
    panel_edge_z = HINGE[2] + DEPTH * sin(angle) - LID_THICKNESS / 2 * cos(angle)
    clear = panel_edge_z - RIM - (hi[2] - lo[2])
    # The front wall has thickness. Include that slab and its vertical
    # approach zone, but exclude a workpiece fully inside or fully outside.
    intersects = (
        lo[1] < ENTRY + 0.004 and hi[1] > ENTRY - 0.008
        and lo[0] < WIDTH / 2 and hi[0] > -WIDTH / 2
        and hi[2] > RIM - 0.010 and lo[2] < RIM + DEPTH
    )
    progress = 1.0 - np.linalg.norm(pose[:3] - target) / 0.25
    return np.array([clear, float(intersects), np.clip(progress, 0, 1), impulse], dtype=np.float32)


def body_pose(position, angle=0.0):
    return torch.tensor([[*position, cos(angle / 2), sin(angle / 2), 0, 0]], dtype=torch.float32)


class PackingScene:
    """CPU physics scene shared by fixture calibration and robot experiments."""

    def __init__(self, robot_usd: Path | None = None, *, fixture=False):
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            device="cpu", dt=DT, render_interval=1,
            physx=sim_utils.PhysxCfg(enable_enhanced_determinism=True),
        ))
        stage = self.sim.stage
        self.fixture = fixture
        spawn_container(stage)
        self.lid = RigidObject(RigidObjectCfg(prim_path=LID))
        self.objects = []
        for index, path in enumerate(OBJECTS):
            color = ((0.58, 0.27, 0.20), (0.27, 0.53, 0.29))[index]
            box(path, BLOCK_SIZE, (0, 0.3 + index * 0.07, 0.1), color,
                mass=0.045, gravity=not fixture)
            self.objects.append(RigidObject(RigidObjectCfg(prim_path=path)))

        self.robot = None
        robot_filters = []
        if robot_usd is not None:
            for index, (x, y, _) in enumerate(STAGING):
                box(f"{ROOT}/Stand{index}", (0.063, 0.063, STAGING_HEIGHT - 0.005),
                    (x, y, (STAGING_HEIGHT + 0.005) / 2), (0.21, 0.24, 0.28))
            self.robot = Articulation(ArticulationCfg(
                prim_path=ROBOT,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(robot_usd),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    joint_pos={f".*_{name}": position for name, position in HOME.items()},
                ),
                actuators={"joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"], stiffness=400.0, damping=40.0,
                )},
            ))
            robot_filters = [
                str(prim.GetPath()) for prim in stage.GetPrimAtPath(ROBOT).GetChildren()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName().startswith("right_")
            ]
        else:
            box("/World/FixtureTable", (0.8, 1.2, 0.01), (0, 0, 0), (0.4, 0.4, 0.4))

        self.object_contact = ContactSensor(ContactSensorCfg(
            prim_path=LID, filter_prim_paths_expr=list(OBJECTS),
        ))
        self.robot_contact = ContactSensor(ContactSensorCfg(
            prim_path=LID, filter_prim_paths_expr=robot_filters,
        )) if robot_filters else None
        self.support_contact = ContactSensor(ContactSensorCfg(
            prim_path=LID,
            filter_prim_paths_expr=[f"{ROOT}/{name}" for name in ("Front", "Back", "West", "East")]
            + (["/World/FixtureTable"] if robot_usd is None else []),
        )) if fixture else None
        self.sim.reset()
        self.reset()

    def reset(self, dynamics=DYNAMICS["static"], angle=0.0, items=None):
        self.dynamics = dynamics
        centre = np.asarray(HINGE) + np.array([0, DEPTH / 2 * cos(angle), DEPTH / 2 * sin(angle)])
        self.lid.reset()
        self.lid.write_root_pose_to_sim(body_pose(centre, angle))
        self.lid.write_root_velocity_to_sim(torch.zeros(1, 6))
        if items is None:
            items = STAGING
        for obj, position in zip(self.objects, items, strict=True):
            obj.reset()
            obj.write_root_pose_to_sim(body_pose(position))
            obj.write_root_velocity_to_sim(torch.zeros(1, 6))
        if self.robot is not None:
            state = self.robot.data
            self.robot.reset()
            self.robot.write_joint_state_to_sim(state.default_joint_pos, state.default_joint_vel)
            self.robot.set_joint_position_target(state.default_joint_pos)
        self.object_contact.reset()
        if self.robot_contact is not None:
            self.robot_contact.reset()
        if self.support_contact is not None:
            self.support_contact.reset()

    @property
    def angle(self):
        q = self.lid.data.root_link_quat_w[0]
        w, x = map(float, q[:2])
        return atan2(2 * w * x, 1 - 2 * x * x)

    def step(self, targets=None, *, fixture_target=None):
        angle = self.angle
        omega = float(self.lid.data.root_com_ang_vel_w[0, 0])
        torque = -self.dynamics.stiffness * angle - self.dynamics.damping * omega
        self.lid.permanent_wrench_composer.set_forces_and_torques(
            torques=torch.tensor([[[torque, 0.0, 0.0]]]), is_global=True,
        )
        self.lid.write_data_to_sim()
        if fixture_target is not None:
            if not self.fixture:
                raise ValueError("Fixture actuation is only available in calibration mode")
            obj = self.objects[0]
            pos = obj.data.root_link_pos_w[0]
            vel = obj.data.root_com_lin_vel_w[0]
            force = (55 * (torch.as_tensor(fixture_target, dtype=pos.dtype) - pos) - 2.8 * vel).clamp(-3, 3)
            obj.permanent_wrench_composer.set_forces_and_torques(
                forces=force.reshape(1, 1, 3), is_global=True,
            )
            obj.write_data_to_sim()
        if self.robot is not None:
            if targets is not None:
                self.robot.set_joint_position_target(targets)
            self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.lid.update(DT)
        for obj in self.objects:
            obj.update(DT)
        if self.robot is not None:
            self.robot.update(DT)
        self.object_contact.update(DT)
        if self.robot_contact is not None:
            self.robot_contact.update(DT)
        if self.support_contact is not None:
            self.support_contact.update(DT)

    def contact_impulse(self, *, closing=False):
        """Lid contact with either item or with the right arm outside closure."""
        forces = self.object_contact.data.force_matrix_w[0, 0]
        impulse = float(forces.norm(dim=-1).sum()) * DT
        if self.robot_contact is not None and not closing:
            forces = self.robot_contact.data.force_matrix_w[0, 0]
            impulse += float(forces.norm(dim=-1).sum()) * DT
        return impulse

    def labels(self, active, *, closing=False, impulse=None):
        if impulse is None:
            impulse = self.contact_impulse(closing=closing)
        pose = self.objects[active].data.root_link_pose_w[0].numpy()
        return condition_state(self.angle, pose, np.asarray(TARGETS[active]), impulse)
