"""Physical Stage C scene. Privileged labels never enter the policy observation."""
from collections import deque
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, matrix_from_quat
from robot import WaypointController
from scene import box, body_pose, HOME, DT
from bag_packing import MODES, METHODS, dump

from bag_common import (ROOT, COLLAR_Z, LIP_HALF, OPEN, CLOSED, FLOOR, OCCLUDE_Z, SIZE, SLOTS,
                        STAGING, HANDLE_OFFSET, DYNAMICS, CONTROL_DT, OBS_DIM, ACTION_DIM, HISTORY,
                        HORIZON, THRESHOLD, history_indices, visible_contents, measure_trace,
                        effective_clearance, slot_pos, wallclock_lag_ticks, align_action_chunk)


def envelope(pose, size):
    rot = matrix_from_quat(torch.as_tensor(pose[3:7])).numpy()
    extent = np.abs(rot) @ (size/2)
    return pose[:3] - extent, pose[:3] + extent


class BagController(WaypointController):
    """Vertical right wrist keeps the forearm above the upright bag."""
    def action(self, left, right, left_gap=.008, right_gap=.032):
        jacobians = self.robot.root_physx_view.get_jacobians()
        q = self.robot.data.joint_pos
        result = q.clone()
        for index, (goal, gap) in enumerate(((left, left_gap), (right, right_gap))):
            dx, dy = goal[0] - self.bases[index][0], goal[1] - self.bases[index][1]
            pitch = .16 if index == 0 else np.pi*.4
            quat = quat_from_euler_xyz(torch.zeros(1), torch.tensor([pitch], dtype=torch.float32),
                                      torch.tensor([np.arctan2(dy, dx)], dtype=torch.float32))
            position = torch.as_tensor(goal, dtype=torch.float32)[None] - quat_apply(quat, self.pad_offset)
            controller, body, ids = self.controllers[index], self.bodies[index], self.ids[index]
            controller.set_command(torch.cat((position, quat), -1))
            current = q[:, ids]
            target = controller.compute(self.robot.data.body_link_pos_w[:, body],
                                        self.robot.data.body_link_quat_w[:, body],
                                        jacobians[:, body-1][:, :, ids], current)
            result[:, ids] = current + (target-current).clamp(-.12, .12)
            result[:, self.fingers[index]] = torch.tensor([gap, -gap])
        low, high = self.robot.data.soft_joint_pos_limits.unbind(-1)
        return torch.maximum(low, torch.minimum(high, result))


class BagScene:
    def __init__(self, robot_usd):
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=DT,
                          physx=sim_utils.PhysxCfg(enable_enhanced_determinism=True)))
        stage = self.sim.stage
        UsdGeom.Xform.Define(stage, ROOT)
        for name, x in (("Slot_Left", -.048), ("Slot_Right", .048)):
            prim = box(f"{ROOT}/{name}", (.090, .15, .009), (x, 0, .0095), (.16, .22, .28))
            prim.SetDisplayName(name.replace("_", "-"))
        # Opaque shell: rigid collision proxy, NOT a simulated textile mesh.
        for name, pos, size in (("Back", (0,-.087,.058), (.19,.008,.10)),
                                ("Front", (0,.087,.058), (.19,.008,.10)),
                                ("West", (-.098,0,.058), (.008,.18,.10)),
                                ("East", (.098,0,.058), (.008,.18,.10))):
            box(f"{ROOT}/{name}", size, pos, (.19,.34,.43))
        self.lips = []
        self.lip_paths = [f"{ROOT}/CollarRear", f"{ROOT}/CollarFront"]
        for sign, path in zip((-1, 1), self.lip_paths):
            prim = box(path, (.19, 2*LIP_HALF, .018), (0, sign*CLOSED, COLLAR_Z),
                       (.66,.48,.22), mass=.04, gravity=False)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            joint = UsdPhysics.PrismaticJoint.Define(stage, path + "_Slider")
            joint.CreateBody1Rel().SetTargets([path])
            joint.CreateAxisAttr("Y")
            joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, COLLAR_Z))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            joint.CreateLowerLimitAttr(float(-OPEN))
            joint.CreateUpperLimitAttr(float(OPEN))
            self.lips.append(RigidObject(RigidObjectCfg(prim_path=path)))
        box(self.lip_paths[0] + "/CollarHandle", (.033,.030,.018),
            tuple(HANDLE_OFFSET), (.78,.60,.26))
        self.objects = []
        self.item_paths = [f"{ROOT}/ItemA", f"{ROOT}/ItemB"]
        for i, path in enumerate(self.item_paths):
            prim = box(path, tuple(SIZE[i]), tuple(STAGING[i]), ((.76,.32,.23),(.30,.57,.63))[i], mass=.045)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            self.objects.append(RigidObject(RigidObjectCfg(prim_path=path)))
            box(f"{ROOT}/Stand{i}", (.062,.062,.070), (STAGING[i,0],STAGING[i,1],.04), (.24,.27,.30))
        self.robot = Articulation(ArticulationCfg(prim_path="/World/Robot",
              spawn=sim_utils.UsdFileCfg(usd_path=str(robot_usd),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)),
              init_state=ArticulationCfg.InitialStateCfg(joint_pos={f".*_{k}":v for k,v in HOME.items()}),
              actuators={"joints":ImplicitActuatorCfg(joint_names_expr=[".*"],stiffness=400.,damping=40.)}))
        right = [str(p.GetPath()) for p in stage.GetPrimAtPath("/World/Robot").GetChildren()
                 if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName().startswith("right_")]
        self.contacts = [ContactSensor(ContactSensorCfg(prim_path=path,
                         filter_prim_paths_expr=self.item_paths + right)) for path in self.lip_paths]
        self.inter_item = ContactSensor(ContactSensorCfg(prim_path=self.item_paths[0],
                                                       filter_prim_paths_expr=[self.item_paths[1]]))
        self.sim.reset()
        self.controller = BagController(self)

    def reset(self, mode, seed):
        self.mode = mode
        self.attached = False
        self.slip_remaining = 0
        self.pickups = STAGING.copy()
        self.pickups[:, :2] += np.random.default_rng(seed).uniform(-.003,.003,(2,2))
        for lip, sign in zip(self.lips, (-1,1)):
            lip.reset();lip.write_root_pose_to_sim(body_pose((0,sign*CLOSED,COLLAR_Z)))
            lip.write_root_velocity_to_sim(torch.zeros(1,6))
        for obj, pos in zip(self.objects,self.pickups):
            obj.reset();obj.write_root_pose_to_sim(body_pose(pos));obj.write_root_velocity_to_sim(torch.zeros(1,6))
        self.robot.reset()
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos,self.robot.data.default_joint_vel)
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos)
        for sensor in self.contacts+[self.inter_item]:sensor.reset()
        for _ in range(24):self.step(None,.032)

    @property
    def aperture(self):
        return float(self.lips[1].data.root_link_pos_w[0,1]-self.lips[0].data.root_link_pos_w[0,1]-2*LIP_HALF)

    @property
    def handle(self):
        return self.lips[0].data.root_link_pos_w[0].numpy()+HANDLE_OFFSET

    def slip(self):
        self.attached=False
        self.slip_remaining=36  # 0.3 s of failed handle grasp, same for every method.
        for lip,sign in zip(self.lips,(-1,1)):
            velocity=torch.zeros(1,6);velocity[0,1]=-sign*.12
            lip.write_root_velocity_to_sim(velocity)

    def step(self, targets, left_gap):
        slip_active=self.slip_remaining>0
        rear,front=[float(o.data.root_link_pos_w[0,1]) for o in self.lips]
        velocities=[float(o.data.root_com_lin_vel_w[0,1]) for o in self.lips]
        tip=self.controller.tip(0)
        distance=np.linalg.norm(tip-self.handle)
        if left_gap>.025 or distance>.065 or self.slip_remaining:self.attached=False
        elif distance<.028 and left_gap<.018:self.attached=True
        stiffness,damping=DYNAMICS[self.mode]
        if self.slip_remaining:stiffness,damping=8.,.10;self.slip_remaining-=1
        # Passive spring linkage mirrors the two sliders; only the rear handle is grasped.
        couple=-95.*(rear+front)-.20*sum(velocities)
        self.last_forces=[]
        for i,(lip,y,v,sign) in enumerate(zip(self.lips,(rear,front),velocities,(-1,1))):
            force=-stiffness*(y-sign*CLOSED)-damping*v+couple
            if i==0 and self.attached:
                force+=float(np.clip(160.*(tip[1]-self.handle[1])-1.8*v,-8.,8.))
            self.last_forces.append(float(force))
            lip.permanent_wrench_composer.set_forces_and_torques(forces=torch.tensor([[[0.,force,0.]]]),is_global=True)
            lip.write_data_to_sim()
        if targets is not None:
            if slip_active:
                targets=targets.clone()
                targets[:,self.controller.fingers[0]]=torch.tensor([.038,-.038])
            self.robot.set_joint_position_target(targets)
        self.robot.write_data_to_sim();self.sim.step(render=False)
        for body in self.lips+self.objects:body.update(DT)
        self.robot.update(DT)
        for sensor in self.contacts+[self.inter_item]:sensor.update(DT)

    def impulses(self):
        lip=sum(float(s.data.force_matrix_w.norm(dim=-1).sum())*DT for s in self.contacts)
        inter=float(self.inter_item.data.force_matrix_w.norm(dim=-1).sum())*DT
        return lip,inter


class BagTask:
    def __init__(self, asset):
        self.env=BagScene(asset)
        self.controller=self.env.controller

    def reset(self,mode,order,seed):
        self.env.reset(mode,seed)
        self.mode,self.order,self.seed=mode,order,seed
        self.sequence=(0,1)
        self.phase=0;self.active=self.sequence[0];self.tick=0
        self.trace=[];self.frames=[];self.events=[];self.learn_ranges=[]
        self.slipped=False;self.slip_tick=None;self.withdraw_tick=None;self.reopen_tick=None
        self.second_slot=None;self.last_obs=None
        self.last_action=np.r_[self.controller.tip(0),self.controller.tip(1),.032,.032].astype(np.float32)
        self.observations=[self.observe()]

    def poses(self):
        return np.stack([o.data.root_link_pose_w[0].numpy().copy() for o in self.env.objects])

    def target_slot(self, phase=None):
        p=self.phase if phase is None else int(phase)
        return int(self.order) if p==0 else 1-int(self.order)

    def slot_occupancy(self):
        poses=self.poses()
        return np.array([any(p[2]<OCCLUDE_Z and abs(p[0]-x)<.039 and abs(p[1])<.065 for p in poses)
                         for x in (-.048,.048)],dtype=np.float32)

    def observe(self):
        poses=self.poses()
        lips=[o.data for o in self.env.lips]
        values=[float(lips[0].root_link_pos_w[0,1]),float(lips[1].root_link_pos_w[0,1]),
                float(lips[0].root_com_lin_vel_w[0,1]),float(lips[1].root_com_lin_vel_w[0,1]),self.env.aperture]
        values.extend(self.controller.tip(0));values.extend(self.controller.tip(1))
        values.extend(visible_contents(poses,self.slot_occupancy()))
        values.extend(np.eye(2)[self.active]);values.extend(np.eye(3)[min(self.phase,2)])
        cue_visible=bool(self.phase==0 and poses[0,2]>=OCCLUDE_Z)
        values.extend(np.eye(2)[self.order] if cue_visible else np.zeros(2));values.append(float(cue_visible))
        values.extend(self.last_action)
        result=np.asarray(values,dtype=np.float32)
        assert result.shape==(OBS_DIM,)
        return result

    def history(self):
        return np.asarray(self.observations)[history_indices(len(self.observations)-1)]

    def label(self,impulse):
        pose=self.poses()[self.active];lo,hi=envelope(pose,SIZE[self.active])
        rear=float(self.env.lips[0].data.root_link_pos_w[0,1])+LIP_HALF
        front=float(self.env.lips[1].data.root_link_pos_w[0,1])-LIP_HALF
        tip=self.controller.tip(1)
        # Include an explicit conservative 12 mm gripper radius as well as the oriented item envelope.
        clear=min(lo[1]-rear,front-hi[1],tip[1]-.012-rear,front-tip[1]-.012,
                  .094-max(abs(lo[0]),abs(hi[0])))
        clear=effective_clearance(clear,pose[:3],tip,self.active)
        occupied=(lo[2]<COLLAR_Z+.045 and hi[2]>COLLAR_Z-.018 and abs(pose[1])<.10)
        progress=np.clip(1.-np.linalg.norm(pose[:3]-slot_pos(self.active,self.target_slot()))/.23,0,1)
        other=1-self.active
        if self.poses()[other,2]<OCCLUDE_Z and abs(pose[0]-self.poses()[other,0])<.04:progress*=.2
        return np.array([clear,float(occupied),progress,impulse],dtype=np.float32)

    def step(self,action,*,expert=False,learn=False,perturb=False):
        action=np.asarray(action,dtype=np.float32).copy()
        delta=(action[:6]-self.last_action[:6]).reshape(2,3)
        action[:6]=self.last_action[:6]+(delta*np.minimum(1,.0075/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-8))).ravel()
        action[6:]=np.clip(action[6:],.006,.038)
        pose=self.poses()[self.active]
        if perturb and not self.slipped and self.phase==0 and pose[2]-SIZE[self.active,2]/2<COLLAR_Z+.038 and abs(pose[1])<.045:
            self.env.slip();self.slipped=True;self.slip_tick=self.tick
            self.events.append(dict(t=self.tick,event="collar_slip"))
        maxima=np.zeros(2);peak_total=0.
        before=self.observations[-1].copy()
        # BagScene applies the physical finger fault. The policy command remains
        # intact in last_action and recorded observations (no fault flag leak).
        for _ in range(4):
            self.env.step(self.controller.action(action[:3],action[3:6],action[6],action[7]),action[6])
            impulses=self.env.impulses()
            maxima=np.maximum(maxima,impulses);peak_total=max(peak_total,sum(impulses))
        self.last_action=action;self.tick+=1
        safety=self.label(float(peak_total))
        poses=self.poses()
        if self.phase==1 and learn and self.second_slot is None and abs(poses[self.active,1])<.05 and poses[self.active,2]-SIZE[self.active,2]/2<COLLAR_Z+.005 and poses[self.active,2]+SIZE[self.active,2]/2>COLLAR_Z-.025:
            self.second_slot=int(poses[self.active,0]>0)
        if self.slipped and self.phase==0:
            if self.withdraw_tick is None and poses[self.active,2]-SIZE[self.active,2]/2>COLLAR_Z+.050:
                self.withdraw_tick=self.tick;self.events.append(dict(t=self.tick,event="withdraw"))
            if self.withdraw_tick is not None and self.reopen_tick is None and self.env.aperture>.095:
                self.reopen_tick=self.tick;self.events.append(dict(t=self.tick,event="reopen"))
        truth=np.r_[self.env.aperture, np.mean([float(l.data.root_com_lin_vel_w[0,1]) for l in self.env.lips]),poses[:,:3].ravel(),self.slot_occupancy()]
        self.trace.append(dict(obs=before,action=action,state=safety,truth=truth.astype(np.float32),
                               expert=bool(expert and peak_total<THRESHOLD),learn=bool(learn),stage=self.phase,
                               collar_impulse=float(maxima[0]),item_impulse=float(maxima[1]),
                               attached=self.env.attached,forces=np.asarray(self.env.last_forces),poses=poses.copy()))
        self.frames.append(np.r_[self.tick*CONTROL_DT,self.env.aperture,self.controller.tip(0),self.controller.tip(1),
                                 poses[:,:3].ravel(),safety,maxima].astype(np.float32))
        self.observations.append(self.observe())

    def do(self,name,left,right,lg,rg,steps,*,expert=False,learn=False,perturb=False):
        self.events.append(dict(t=self.tick,event=name))
        for _ in range(steps):self.step(np.r_[left,right,lg,rg],expert=expert,learn=learn,perturb=perturb)

    def spread(self,right,rg,*,learn=False):
        if self.env.attached and self.env.aperture > .105:
            self.do("hold_collar",self.hold,right,.008,rg,4,expert=learn,learn=learn)
            return
        handle=self.env.handle.copy()
        self.do("approach_handle",handle,right,.032,rg,35,expert=learn,learn=learn)
        self.do("grasp_handle",handle,right,.008,rg,18,expert=learn,learn=learn)
        target=np.array([-.048,-OPEN+HANDLE_OFFSET[1],COLLAR_Z+HANDLE_OFFSET[2]])
        self.do("spread_collar",target,right,.008,rg,35,expert=learn,learn=learn)

    @property
    def hold(self):return np.array([-.048,-OPEN+HANDLE_OFFSET[1],COLLAR_Z+HANDLE_OFFSET[2]])

    def pickup(self,item):
        pick=self.env.pickups[item]
        hover=pick+np.array([0,0,.110])
        self.do("approach_pick",self.hold,hover,.008,.032,30)
        self.do("lower_pick",self.hold,pick,.008,.032,24)
        self.do("grasp_item",self.hold,pick,.008,.009,18)
        self.do("lift_item",self.hold,hover,.008,.009,28)
        self.do("carry_high",self.hold,[pick[0],pick[1],.23],.008,.009,24)
        self.do("pre_entry",self.hold,[0.,0.,.23],.008,.009,48)

    def placed(self,item):
        p=self.poses()[item];lo,hi=envelope(p,SIZE[item])
        return bool(np.linalg.norm(p[:2]-slot_pos(item,self.target_slot(item))[:2])<.035 and lo[2]>=.009 and hi[2]<.085)

    def release_ready(self):
        p=self.poses()[self.active]
        return bool(abs(p[0])<.085 and abs(p[1])<.058 and p[2]<.071)

    def end_item(self):
        right=self.controller.tip(1).copy()
        self.do("release_item",self.last_action[:3],right,self.last_action[6],.032,18)
        self.do("withdraw_empty",self.last_action[:3],[right[0],0,.215],self.last_action[6],.032,24)
        self.do("return_empty",self.last_action[:3],[0,.22,.215],self.last_action[6],.032,24)

    def close(self):
        right=np.array([0,.22,.215])
        self.spread(right,.032)
        self.do("close_collar",[-.048,-CLOSED+HANDLE_OFFSET[1],COLLAR_Z+HANDLE_OFFSET[2]],right,.008,.032,30)
        self.do("release_handle",[-.048,-.18,.19],right,.032,.032,25)

    def expert_episode(self,mode,order,seed,scheme):
        self.reset(mode,order,seed)
        for phase,item in enumerate(self.sequence):
            self.phase=phase;self.active=item
            self.observations[-1]=self.observe()
            self.spread([0,.22,.215],.032)
            self.pickup(item)
            start=self.tick
            target=slot_pos(item,self.target_slot(phase));x=target[0]
            if scheme=="direct_insert":
                self.do("release_before_entry",self.hold,[x,0,.205],.032,.009,5,expert=True,learn=True)
            if scheme=="slip_withdraw_reopen" and phase==0:
                self.events.append(dict(t=self.tick,event="enter_slip"))
                for _ in range(30):
                    self.step(np.r_[self.hold,[x,0,.145],.008,.009],expert=True,learn=True,perturb=True)
                    if self.slipped:break
                self.do("withdraw",self.last_action[:3],[x,0.,.245],.038,.009,28,expert=True,learn=True)
                self.spread([x,0.,.245],.009,learn=True)
                self.do("resume_entry",self.hold,[x,0,.23],.008,.009,24,expert=True,learn=True)
            if scheme=="exploration":
                rng=np.random.default_rng(seed*7+phase)
                self.do("explore_release",self.hold,[x,0,.15],.032,.009,18,learn=True,perturb=True)
                x=-x if phase==1 else float(rng.choice([-.048,.048]))
            self.do("insert",self.hold,[x,0,target[2]+.030],
                    .032 if scheme in ("direct_insert","exploration") else .008,.009,50,
                    expert=scheme!="exploration",learn=True,perturb=scheme=="hold_through_slip")
            self.learn_ranges.append((start,self.tick))
            self.end_item()
        self.phase=2;self.close()
        return self.summary(scheme)

    def summary(self,method):
        impulse=max((float(r['state'][3]) for r in self.trace),default=0.)
        success=all(self.placed(i) for i in (0,1)) and self.env.aperture<.035
        measurement=measure_trace(np.asarray(self.frames),np.asarray([r["stage"] for r in self.trace]),
                                  np.asarray([r["learn"] for r in self.trace]),np.asarray([r["attached"] for r in self.trace]),
                                  self.order,self.slip_tick)
        self.second_slot=measurement["second_slot"]
        slot_correct=measurement["second_slot_correct"]
        recovered=measurement["withdraw_reopen"]
        minimum_aperture=float(np.asarray(self.frames)[self.slip_tick:self.slip_tick+30,1].min()) if self.slip_tick is not None else None
        return dict(protocol_version='stage_c2',perturbation_version='forced_finger_release_v2',mode=self.mode,order=self.order,seed=self.seed,method=method,success=bool(success),
                    placed=[self.placed(i) for i in (0,1)],final_aperture_m=self.env.aperture,
                    peak_impulse_Ns=impulse,collision=bool(impulse>=THRESHOLD),
                    second_slot=self.second_slot,second_slot_correct=bool(slot_correct),
                    slipped=self.slipped,slip_effective=bool(minimum_aperture is not None and minimum_aperture<.070),
                    min_aperture_after_slip_m=minimum_aperture,withdraw_reopen=bool(recovered),recovery_measurement=measurement,
                    safe_recovery=bool(recovered and success and impulse<THRESHOLD),
                    steps=self.tick,events=self.events,learn_ranges=self.learn_ranges,
                    final_item_positions=self.poses()[:,:3].tolist())

    def save(self,path,metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        arrays={k:np.asarray([r[k] for r in self.trace]) for k in self.trace[0]}
        arrays.update(frames=np.asarray(self.frames),metadata=json.dumps(metadata),observations=np.asarray(self.observations))
        np.savez_compressed(path,**arrays)


def collect(task,args):
    folder=args.out/'dataset';folder.mkdir(exist_ok=True)
    schemes=("direct_insert","spread_and_hold","slip_withdraw_reopen","exploration")
    for seed in args.seeds or range(1000,1008):
        for mode in args.modes:
            for order in (0,1):
                for scheme in schemes:
                    path=folder/f"{seed}_{mode}_{order}_{scheme}.npz"
                    if path.exists():continue
                    result=task.expert_episode(mode,order,seed,scheme)
                    task.save(path,result)
                    print({k:result[k] for k in ('seed','mode','order','method','success','collision','placed')},flush=True)


def evaluate(task,args):
    from bag_learning import load_models, decide
    from hashlib import sha256
    model_root=args.model_root or args.out
    models=load_models(model_root,args.device)
    wall_clock=bool(getattr(args,'wall_clock',False))
    use_cuda_graph=not bool(getattr(args,'no_cuda_graph',False))
    pert_arg=getattr(args,'perturbed','both')
    perturbed_modes=(False,True) if pert_arg=='both' else ((True,) if pert_arg=='true' else (False,))
    dummy_hist=np.zeros((HISTORY,OBS_DIM),dtype=np.float32)
    for m_warm in args.methods:
        decide(models,dummy_hist,m_warm,torch.Generator().manual_seed(0),use_cuda_graph=use_cuda_graph)
    if args.device=='cuda':torch.cuda.synchronize()
    seeds=args.seeds or [2000,2001,2002]
    journal=args.out/'episodes.jsonl'
    prior=[json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []
    done={(r['method'],r['mode'],r['order'],r['seed'],r['perturbed']) for r in prior}
    hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (model_root/'trained').glob('*.pt')}
    if any(r['checkpoint_hashes']!=hashes for r in prior):raise ValueError('Checkpoint changed; choose a new evaluation directory')
    source_hashes={name:sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                   for name in ('bag_scene.py','bag_learning.py','bag_common.py','mamba_wm.py')}
    if any(r.get('planner_source_hashes')!=source_hashes for r in prior):raise ValueError('Planner source changed; choose a new evaluation directory')
    if any(bool(r.get('wall_clock',False))!=wall_clock or bool(r.get('use_cuda_graph',True))!=use_cuda_graph for r in prior):
        raise ValueError('Rollout timing mode changed; choose a new evaluation directory')
    with journal.open('a',encoding='utf-8') as stream:
        for seed in seeds:
            for mode in args.modes:
                for order in (0,1):
                    for perturbed in perturbed_modes:
                        for method in args.methods:
                            key=(method,mode,order,seed,perturbed)
                            if key in done:continue
                            task.reset(mode,order,seed)
                            rng=torch.Generator().manual_seed(seed+12000)
                            latency=[];probes=[];planning=[];budgets=[];commits=[];interrupts=0
                            lag_ticks_all=[];chunk_offsets_all=[]
                            for phase,item in enumerate(task.sequence):
                                task.phase=phase;task.active=item;task.observations[-1]=task.observe()
                                task.spread([0,.22,.215],.032);task.pickup(item)
                                start=task.tick
                                inflight=[]
                                while task.tick-start < 216:
                                    if not wall_clock:
                                        if args.device=='cuda':torch.cuda.synchronize()
                                        begin=perf_counter()
                                        actions,diagnostic=decide(models,task.history(),method,rng,use_cuda_graph=use_cuda_graph)
                                        if args.device=='cuda':torch.cuda.synchronize()
                                        latency.append((perf_counter()-begin)*1000)
                                        planning.append(dict(t=task.tick,**diagnostic))
                                        if phase==1:probes.append(diagnostic)
                                        executed=0
                                        for action in actions[:min(6,216-(task.tick-start))]:
                                            previous_aperture=task.env.aperture
                                            task.step(action,learn=True,perturb=perturbed)
                                            executed+=1
                                            if task.release_ready():break
                                            if previous_aperture-task.env.aperture>.006:
                                                interrupts+=1
                                                break
                                        commits.append(executed)
                                        if task.release_ready():break
                                    else:
                                        t_obs=task.tick
                                        hist=task.history()
                                        if args.device=='cuda':torch.cuda.synchronize()
                                        begin=perf_counter()
                                        actions,diagnostic=decide(models,hist,method,rng,use_cuda_graph=use_cuda_graph)
                                        if args.device=='cuda':torch.cuda.synchronize()
                                        infer_ms=(perf_counter()-begin)*1000
                                        latency.append(infer_ms)
                                        dk=wallclock_lag_ticks(infer_ms,CONTROL_DT)
                                        lag_executed=0
                                        interrupted_during_infer=False
                                        while lag_executed<dk and task.tick-start<216:
                                            cmd=inflight.pop(0) if inflight else task.last_action
                                            previous_aperture=task.env.aperture
                                            task.step(cmd,learn=True,perturb=perturbed)
                                            lag_executed+=1
                                            if task.release_ready():break
                                            if previous_aperture-task.env.aperture>.006:
                                                interrupts+=1
                                                interrupted_during_infer=True
                                        t_apply=task.tick
                                        aligned,offset=align_action_chunk(actions,lag_executed)
                                        inflight=list(aligned)
                                        post_executed=0
                                        if not interrupted_during_infer and not task.release_ready():
                                            budget_left=216-(task.tick-start)
                                            for _ in range(min(max(0,6-lag_executed),budget_left,len(inflight))):
                                                cmd=inflight.pop(0)
                                                previous_aperture=task.env.aperture
                                                task.step(cmd,learn=True,perturb=perturbed)
                                                post_executed+=1
                                                if task.release_ready():break
                                                if previous_aperture-task.env.aperture>.006:
                                                    interrupts+=1
                                                    break
                                        lag_ticks_all.append(lag_executed)
                                        chunk_offsets_all.append(offset)
                                        planning.append(dict(t=t_obs,t_apply=t_apply,infer_ms=infer_ms,dk_ticks=dk,
                                                             lag_ticks=lag_executed,chunk_offset=offset,post_ticks=post_executed,
                                                             interrupted_during_infer=interrupted_during_infer,**diagnostic))
                                        if phase==1:probes.append(diagnostic)
                                        commits.append(lag_executed+post_executed)
                                        if task.release_ready():break
                                budgets.append(not task.release_ready())
                                task.learn_ranges.append((start,task.tick));task.end_item()
                            task.phase=2;task.close()
                            result=task.summary(method)
                            result.update(perturbed=perturbed,wall_clock=wall_clock,use_cuda_graph=use_cuda_graph,
                                          execution_context=args.execution_context,
                                          perturbation_version="forced_finger_release_v2" if perturbed else "none",
                                          decision_ms=latency,lag_ticks=lag_ticks_all,chunk_offsets=chunk_offsets_all,
                                          slot_predictions=probes,planning_diagnostics=planning,
                                          insertion_budget_exhausted=budgets,committed_steps=commits,
                                          aperture_interrupts=interrupts,checkpoint_hashes=hashes,
                                          planner_source_hashes=source_hashes,device=args.device,torch=str(torch.__version__))
                            name=f"{seed}_{mode}_{order}_{int(perturbed)}_{METHODS.index(method)}.npz"
                            result['trace_file']='traces/'+name
                            task.save(args.out/result['trace_file'],result)
                            stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush();done.add(key)
                            print(f"Bag {len(done)} {method} {mode}/{order}/{seed} perturb={perturbed} wall_clock={wall_clock}: success={result['success']} collision={result['collision']} slot={result['second_slot_correct']} p50_ms={np.median(latency):.2f}",flush=True)
    rows=[json.loads(l) for l in journal.read_text().splitlines()]
    dump(args.out/'results.json',dict(seeds=seeds,modes=list(args.modes),methods=list(args.methods),
                                    wall_clock=wall_clock,use_cuda_graph=use_cuda_graph,episodes=rows,
                                    physics='IsaacLab/PhysX rigid prismatic-collar proxy',control_dt=CONTROL_DT,
                                    collision_threshold_Ns=THRESHOLD,history_tokens=HISTORY,history_span_steps=444,
                                    identity_slot_shortcut=False,protocol_version='stage_c2'))
