"""State-based public Push-T and paired comparisons of flow planning methods."""

import os
import sys

# Keep Isaac Sim's already-loaded torch ABI intact.
cuda_site = r"C:\isaacsim_env\phase2_cuda"
if "torch" not in sys.modules and os.path.isdir(cuda_site):
    sys.path.insert(0, cuda_site)

import argparse
from collections import deque
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import gymnasium as gym
import gym_pusht  # Registers the official environment.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from inference import GPC_OPT, GPC_RANK, GPC_RANK_OPT, Search, flow_search
from learn import flow_loss, temporal_noise
from mamba_wm import UnifiedMambaWorldModel
from phase2_spring import dump


SOURCE = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
GOAL = np.array([256., 256., np.pi / 4])
HISTORY = 8
HORIZON = 16
COMMIT = 8
STATE_DIM = 12
ENV_ID = "gym_pusht/PushT-v0"
VERTICES = np.array([[-60, 0], [60, 0], [60, 30], [15, 30], [15, 120],
                     [-15, 120], [-15, 30], [-60, 30]], dtype=np.float32) / 256
METHODS = {
    "Behavior Cloning": Search(1), "GPC-RANK K=16": GPC_RANK, "GPC-RANK K=64": Search(64),
    "GPC-OPT": GPC_OPT, "GPC-RANK+OPT": GPC_RANK_OPT,
    "Early In-Flow": Search(8, tau_start=0.0), "Ours": Search(8, tau_start=0.7),
    **{f"tau={tau}": Search(8, tau_start=tau) for tau in (0.3, 0.5, 0.9, 1.0)},
}


def set_pusht_state(env, state):
    base = env.unwrapped
    base.block.angle = float(state[4])
    base.block.position = tuple(state[2:4])
    base.agent.position = tuple(state[:2])
    base.block.velocity = (0, 0)
    base.block.angular_velocity = 0
    base.agent.velocity = (0, 0)
    base.space.step(base.dt)
    return base.get_obs()


def make_env():
    from types import MethodType
    env = gym.make(ENV_ID, obs_type="state")
    env.unwrapped._set_state = MethodType(lambda base, state: set_pusht_state(base, state), env.unwrapped)
    return env


def alignment(x, y, sin, cos):
    """Mean squared corresponding-vertex registration error in normalized coordinates."""
    v = x.new_tensor(VERTICES)
    dx = x[..., None] + (cos[..., None] - 1) * v[:, 0] - sin[..., None] * v[:, 1]
    dy = y[..., None] + sin[..., None] * v[:, 0] + (cos[..., None] - 1) * v[:, 1]
    return (dx.square() + dy.square()).mean(-1)


def features(state, previous):
    delta = state - previous
    angle = state[..., 4] - GOAL[2]
    s, c = np.sin(angle), np.cos(angle)
    pos = (state[..., :4] - 256) / 256
    displacement = delta[..., :4] / 256
    omega = np.arctan2(np.sin(delta[..., 4]), np.cos(delta[..., 4]))
    # Express translation and orientation in the goal's reference frame for registration.
    gx = (pos[..., 2] + pos[..., 3]) / np.sqrt(2)
    gy = (pos[..., 3] - pos[..., 2]) / np.sqrt(2)
    u, v = VERTICES.T
    dx = gx[..., None] + (c[..., None] - 1) * u - s[..., None] * v
    dy = gy[..., None] + s[..., None] * u + (c[..., None] - 1) * v
    error = (dx**2 + dy**2).mean(-1)
    return np.concatenate((pos, s[..., None], c[..., None], displacement,
                           omega[..., None], error[..., None]), axis=-1).astype(np.float32)


def read_expert(archive):
    import zarr

    with ZipFile(archive) as source:
        roots = [name.rsplit("/", 2)[0] for name in source.namelist() if name.endswith("meta/.zgroup")]
    if len(roots) != 1:
        raise ValueError(f"Expected one Push-T replay group in {archive}")
    with zarr.ZipStore(str(archive), mode="r") as store:
        replay = zarr.open_group(store, path=roots[0], mode="r")
        states, actions = replay["data/state"][:], replay["data/action"][:]
        ends = replay["meta/episode_ends"][:]
    return [(states[lo:hi], actions[lo:hi]) for lo, hi in zip(np.r_[0, ends[:-1]], ends, strict=True)]


def shooting_action(env, state, rng):
    """Fallback expert proposal using geometry, then exact short Pymunk lookahead."""
    block = state[2:4]
    direction = GOAL[:2] - block
    direction /= max(np.linalg.norm(direction), 1)
    sides = rng.uniform(-np.pi, np.pi, 14)
    approaches = np.vstack((-direction, np.column_stack((np.cos(sides), np.sin(sides)))))
    best, best_cost = None, float("inf")
    for side in approaches:
        near = np.clip(block + 52 * side, 10, 502)
        far = np.clip(block - 30 * side, 10, 502)
        obs, _ = env.reset(seed=0, options={"reset_to_state": state})
        # Set the geometric pose directly to undo the legacy reset's CoM shift.
        env.unwrapped.block.position = tuple(state[2:4])
        controls = [near] * 4 + [far] * 8
        for action in controls:
            obs, _, _, _, _ = env.step(action)
        cost = float(features(obs, obs)[-1]) + 0.00001 * float(np.linalg.norm(near - state[:2]))
        if cost < best_cost:
            best, best_cost = controls, cost
    return best


def collect_axial_demo(env, seed, rng):
    """Produce a contact-aligned example when the public replay is unavailable."""
    axis = np.array([-np.sin(GOAL[2]), np.cos(GOAL[2])])
    side = rng.choice([-1, 1])
    block = GOAL[:2] + side * rng.uniform(18, 43) * axis
    rear = 148 if side > 0 else -33
    agent = block + rear * axis
    env.reset(seed=seed, options={"reset_to_state": [*agent, *block, GOAL[2]]})
    env.unwrapped.block.position = tuple(block)
    obs = env.unwrapped.get_obs()
    states, actions = [obs], []
    for t in range(1, 65):
        action = agent - side * (1.5 * t) * axis
        obs, _, done, _, _ = env.step(action)
        states.append(obs)
        actions.append(action)
        if done:
            return np.asarray(states), np.asarray(actions + [actions[-1]])
    return None


def fallback_expert(env, count=100):
    episodes = []
    search = make_env()
    try:
        for seed in range(count):
            rng = np.random.default_rng(4000 + seed)
            if seed % 3:
                demo = collect_axial_demo(env, 4000 + seed, rng)
                if demo is not None and len(demo[0]) > HORIZON:
                    episodes.append(demo)
                continue
            obs, _ = env.reset(seed=4000 + seed)
            states, actions = [], []
            for _ in range(20):
                for a in shooting_action(search, obs, rng):
                    states.append(obs)
                    actions.append(a)
                    obs, _, done, _, info = env.step(a)
                    if done:
                        break
                if done:
                    break
            if done and len(states) >= HORIZON:
                episodes.append((np.asarray(states + [obs]), np.asarray(actions + [actions[-1]])))
        if len(episodes) < 10:
            raise RuntimeError(f"Fallback generated only {len(episodes)} successful expert demonstrations")
    finally:
        search.close()
    return episodes


def random_exploration(env, count, seed_offset, expert):
    episodes = []
    contacts = 0
    for seed in range(seed_offset, seed_offset + count):
        rng = np.random.default_rng(seed)
        source, guide = expert[int(rng.integers(len(expert)))]
        options = None
        if seed % 2 == 0:
            options = {"reset_to_state": source[int(rng.integers(max(1, len(source) - 48)))]}
        obs, _ = env.reset(seed=seed, options=options)
        states = [obs]
        actions = []
        for step in range(64):
            if step % 4 == 0:
                angle = rng.uniform(-np.pi, np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                centre = obs[2:4] + 35 * np.array([-np.sin(obs[4]), np.cos(obs[4])])
                target = centre + rng.uniform(-32, 95) * direction
                if rng.random() < 0.3:
                    target = guide[int(rng.integers(len(guide)))] + rng.normal(0, 32, 2)
            a = np.clip(target + rng.normal(0, 6, 2), 0, 512)
            obs, _, _, _, info = env.step(a)
            contacts += info["n_contacts"] > 0
            states.append(obs)
            actions.append(a)
        episodes.append((np.asarray(states), np.asarray(actions + [actions[-1]])))
    return episodes, int(contacts)


def save_exploration(path, train, validation):
    train_states, train_actions = (np.asarray(values) for values in zip(*train, strict=True))
    val_states, val_actions = (np.asarray(values) for values in zip(*validation, strict=True))
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, train_states=train_states, train_actions=train_actions[:, :-1],
                            val_states=val_states, val_actions=val_actions[:, :-1])
    tmp.replace(path)
    displacement = np.linalg.norm(np.diff(train_states[..., 2:4], axis=1), axis=-1)
    angular = np.diff(train_states[..., 4], axis=1)
    rotation = np.abs(np.arctan2(np.sin(angular), np.cos(angular)))
    return dict(archive=path.name, sha256=sha256(path.read_bytes()).hexdigest(),
                train_object_motion_steps=int(((displacement >= 1.0) | (rotation >= 0.01)).sum()))


def dataset(episodes, stride):
    rows = []
    for states, actions in episodes:
        obs = features(states, np.concatenate((states[:1], states[:-1])))
        padded = np.concatenate((np.repeat(obs[:1], HISTORY - 1, 0), obs))
        for t in range(0, len(obs) - HORIZON, stride):
            rows.append((padded[t:t + HISTORY], (actions[t:t + HORIZON] - 256) / 256,
                         obs[t + 1:t + HORIZON + 1]))
    if not rows:
        raise ValueError("No complete Push-T windows")
    return tuple(torch.as_tensor(np.asarray(array, dtype=np.float32)) for array in zip(*rows, strict=True))


class FlowResidual(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 3, padding=1)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.condition = nn.Linear(width, 2 * width)

    def forward(self, x, condition):
        scale, shift = self.condition(condition).chunk(2, -1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h) * (1 + scale[..., None]) + shift[..., None]
        return x + self.conv2(F.silu(h))


class PushTFlow(nn.Module):
    """Conditional rectified flow: predict velocity directly without an anchor mean."""
    def __init__(self):
        super().__init__()
        self.horizon, self.channels = HORIZON, 2
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, -9.21, 64)))
        self.condition = nn.Sequential(nn.Linear(HISTORY * STATE_DIM + 128, 256),
                                       nn.SiLU(), nn.Linear(256, 256))
        self.input = nn.Conv1d(2, 256, 3, padding=1)
        self.blocks = nn.ModuleList([FlowResidual() for _ in range(4)])
        self.output = nn.Conv1d(256, 2, 3, padding=1)

    def forward(self, x, tau, obs, stage=None):
        angles = tau * self.frequencies[None] * 1000
        condition = self.condition(torch.cat((obs.flatten(1), angles.sin(), angles.cos()), -1))
        h = self.input(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h, condition)
        return self.output(F.silu(h)).transpose(1, 2)


class PushModels(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = PushTFlow()
        self.world = UnifiedMambaWorldModel(STATE_DIM, 2, STATE_DIM, modes=1, safety=False)
        self.register_buffer("mean", torch.zeros(STATE_DIM))
        self.register_buffer("std", torch.ones(STATE_DIM))

    def encode(self, obs):
        return ((obs - self.mean) / self.std).clamp(-8, 8)

    def future(self, belief, context, action, origin):
        z = self.world.latent(belief, context, action, torch.ones_like(context[:, :1]))
        return self.world.g_state(z) + origin[:, None]


def fit(args):
    out = args.out
    archive = out / "pusht.zip"
    fallback = False
    if not archive.exists():
        tmp = archive.with_suffix(".tmp")
        try:
            with urlopen(SOURCE, timeout=25) as response, tmp.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            tmp.replace(archive)
        except (URLError, TimeoutError) as error:
            print(f"Official archive unavailable: {error}. Running Pymunk expert collection.", flush=True)
            tmp.unlink(missing_ok=True)
            fallback = True
    env = make_env()
    if fallback:
        expert = fallback_expert(env)
        archive_hash = None
    else:
        expert = read_expert(archive)
        archive_hash = sha256(archive.read_bytes()).hexdigest()
    shuffled = np.random.default_rng(903).permutation(len(expert))
    n_val = max(2, len(expert) // 10)
    train_ids, val_ids = shuffled[n_val:].tolist(), shuffled[:n_val].tolist()
    train_expert, val_expert = [expert[i] for i in train_ids], [expert[i] for i in val_ids]
    exploration, contacts = random_exploration(env, 206, 2000, train_expert)
    validation_exploration, val_contacts = random_exploration(env, 32, 3000, val_expert)
    env.close()
    exploration_archive = save_exploration(out / "exploration.npz", exploration, validation_exploration)
    expert_tensors = dataset(train_expert, 1)
    explore_tensors = dataset(exploration, 2)
    validation = dataset(val_expert + validation_exploration, 4)
    metadata = dict(expert_source="Pymunk 几何与单打靶控制器" if fallback else SOURCE,
                    expert_archive_sha256=archive_hash, expert_episodes=len(expert),
                    train_expert_ids=train_ids, val_expert_ids=val_ids,
                    train_exploration_seeds=list(range(2000, 2206)), val_exploration_seeds=list(range(3000, 3032)),
                    training_exploration_contact_steps=contacts, validation_exploration_contact_steps=val_contacts,
                    exploration_archive=exploration_archive,
                    training_expert_windows=len(expert_tensors[0]), training_exploration_windows=len(explore_tensors[0]),
                    gym_pusht=version("gym-pusht"), pymunk=version("pymunk"), seed=903)
    dump(out / "dataset.json", metadata)
    torch.manual_seed(903)
    model = PushModels().to(args.device)
    expert_tensors = tuple(x.to(args.device) for x in expert_tensors)
    explore_tensors = tuple(x.to(args.device) for x in explore_tensors)
    validation = tuple(x.to(args.device) for x in validation)
    data = tuple(torch.cat((x, y)) for x, y in zip(expert_tensors, explore_tensors, strict=True))
    model.mean.copy_(data[0][:, -1].mean(0))
    model.std.copy_(data[0][:, -1].std(0).clamp_min(0.035))
    device = model.mean.device
    opt_flow = torch.optim.AdamW(model.flow.parameters(), lr=0.0001)
    opt_world = torch.optim.AdamW(model.world.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_flow, args.flow_steps)
    mask = torch.ones((256, HORIZON), dtype=torch.bool, device=device)
    state_weight = model.mean.new_tensor([1, 1, 4, 4, 4, 4, 0.5, 0.5, 1, 1, 1, 3])
    for update in range(1, max(args.flow_steps, args.world_steps) + 1):
        if update <= args.flow_steps:
            ix = torch.randint(len(expert_tensors[0]), (256,), device=device)
            obs, acts, _ = (x[ix] for x in expert_tensors)
            fm = flow_loss(model.flow, acts, model.encode(obs), acts.new_zeros((len(acts), 4)), 2, mask, lambda_smooth=0.0)
            opt_flow.zero_grad(set_to_none=True)
            fm.backward()
            nn.utils.clip_grad_norm_(model.flow.parameters(), 1.0)
            opt_flow.step()
            scheduler.step()
        if update <= args.world_steps:
            ix = torch.randint(len(data[0]), (128,), device=device)
            obs, acts, future = (x[ix] for x in data)
            encoded = model.encode(obs)
            b, _ = model.world.encode(encoded)
            pred = model.future(b, acts.new_zeros((len(acts), 4)), acts, encoded[:, -1])
            err = F.smooth_l1_loss(pred, (future - model.mean) / model.std, reduction="none")
            wm = (err * state_weight).mean()
            opt_world.zero_grad(set_to_none=True)
            wm.backward()
            nn.utils.clip_grad_norm_(model.world.parameters(), 5)
            opt_world.step()
        if update % 200 == 0 or update == 1:
            print(f"Push-T update {update}: flow={fm.item():.4f} world={wm.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        errors = []
        for obs, acts, future in zip(*(x.split(64) for x in validation), strict=True):
            encoded = model.encode(obs)
            b, _ = model.world.encode(encoded)
            pred = model.future(b, acts.new_zeros((len(acts), 4)), acts, encoded[:, -1])
            errors.append((pred * model.std + model.mean - future).abs())
        mae = torch.cat(errors).mean((0, 1)).tolist()
    metadata.update(flow_updates=args.flow_steps, world_updates=args.world_steps, device=str(device),
                    heldout_state_mae=mae, heldout_block_xy_mae_px=float(np.mean(mae[2:4]) * 256),
                    flow_batch_size=256, world_batch_size=128, heldout_windows=len(validation[0]), torch_version=str(torch.__version__))
    tmp = out / "model.tmp"
    torch.save(model.state_dict(), tmp)
    tmp.replace(out / "model.pt")
    dump(out / "training.json", metadata)


def decide(model, history, config, rng):
    raw = torch.as_tensor(np.asarray(history), device=model.mean.device)[None]
    obs = model.encode(raw)
    noise = temporal_noise((config.particles, HORIZON, 2), generator=rng).to(obs.device)
    context = obs.new_zeros((1, 4))
    belief = None
    calls = rollouts = 0

    def field(x, tau):
        return model.flow(x, x.new_full((len(x), 1), tau), obs.expand(len(x), -1, -1),
                          context.expand(len(x), -1))

    def risk(x):
        nonlocal belief, calls, rollouts
        if belief is None:
            with torch.no_grad():
                belief, _ = model.world.encode(obs)
        n = len(x)
        calls += 1
        rollouts += n
        z = model.future(belief.expand(n, -1), context.expand(n, -1), x,
                         obs[:, -1].expand(n, -1))
        state = z * model.std + model.mean
        sin, cos = F.normalize(state[..., 4:6], dim=-1, eps=1e-5).unbind(-1)
        dx, dy = state[..., 2:4].unbind(-1)
        err = alignment((dx + dy) / np.sqrt(2), (dy - dx) / np.sqrt(2), sin, cos)
        direction = F.normalize(-state[..., 2:4], dim=-1, eps=0.05)
        contact = state[..., 2:4] - 0.20 * direction
        attraction = (state[..., :2] - contact).square().sum(-1)
        return err[:, -4:].mean(-1) + 0.25 * err.mean(-1) + 0.05 * (attraction * (err / (err + 0.05))).mean(-1)

    selected = flow_search(noise, field, risk, lambda x: x.clamp(-1, 1), config, opt_lr=0.08, guidance=0.35)
    return (selected.cpu().numpy() * 256 + 256), calls, rollouts


def run_episode(env, model, seed, method, max_steps):
    config = METHODS[method]
    state, _ = env.reset(seed=seed)
    history = deque([features(state, state)] * HISTORY, maxlen=HISTORY)
    rng = torch.Generator().manual_seed(713_000 + seed)
    latency, batches, rollouts, executed, coverage = [], [], [], [], []
    start_coverage = env.unwrapped._get_coverage()
    success = False
    while len(executed) < max_steps and not success:
        start = perf_counter()
        actions, calls, count = decide(model, history, config, rng)
        latency.append((perf_counter() - start) * 1000)
        batches.append(calls)
        rollouts.append(count)
        for action in actions[:min(COMMIT, max_steps - len(executed))]:
            obs, _, success, truncated, info = env.step(action)
            history.append(features(obs, state))
            state = obs
            executed.append(action)
            coverage.append(float(info["coverage"]))
            if success or truncated:
                break
        if truncated:
            break
    jumps = np.linalg.norm(np.diff(executed, axis=0), axis=-1).tolist()
    def iou(value):
        return value / (2 - value)
    return dict(seed=seed, method=method, config=vars(config), final_iou=iou(coverage[-1]),
                max_iou=iou(max(coverage)), coverage=coverage[-1], success=bool(success),
                initial_iou=iou(start_coverage), final_registration_error=float(history[-1][-1]),
                actions=len(executed), jump_px=jumps, decision_ms=latency,
                wm_calls=batches, wm_rollouts=rollouts)


def evaluate(args):
    checkpoint = args.checkpoint or args.out / "model.pt"
    model = PushModels().to(args.device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    env = make_env()
    methods = args.methods or list(METHODS)
    path = args.out / "episodes.jsonl"
    prior = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    run = dict(device=str(model.mean.device), torch=str(torch.__version__), torch_threads=torch.get_num_threads(),
               horizon=HORIZON, history=HISTORY, commit=COMMIT, max_steps=args.max_steps, flow_steps=10,
               opt_lr=0.08, guidance=0.35, weights_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
               source_sha256={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("pusht_benchmark.py", "inference.py", "mamba_wm.py")})
    if any(row["run"] != run for row in prior):
        raise ValueError("Existing episode journal uses a different checkpoint or evaluation configuration")
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    done = {(row["method"], row["seed"]): row for row in prior}
    for method in methods:
        decide(model, [np.zeros(STATE_DIM, dtype=np.float32)] * HISTORY, METHODS[method],
               torch.Generator().manual_seed(1))
    with path.open("a", encoding="utf-8") as stream:
        for index, seed in enumerate(seeds):
            offset = index % len(methods)
            for method in methods[offset:] + methods[:offset]:
                key = method, seed
                if key not in done:
                    row = run_episode(env, model, seed, method, args.max_steps)
                    row["run"] = run
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    done[key] = row
            if (index + 1) % 5 == 0 or index == 0:
                print(f"Push-T: {index + 1}/{len(seeds)} seeds, {len(done)} episodes", flush=True)
    env.close()
    rows = [done[method, seed] for method in methods for seed in seeds]
    dump(args.out / "results.json", dict(config=run, seeds=seeds, methods=methods, episodes=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "evaluate", "all"))
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2/pusht"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--flow-steps", type=int, default=25000)
    parser.add_argument("--world-steps", type=int, default=15000)
    parser.add_argument("--first-seed", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300, choices=range(1, 301), metavar="1..300")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    if args.command in ("fit", "all"):
        fit(args)
    if args.command in ("evaluate", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
