"""Ten-step flow sampler and action-conditioned risk for Phase 1."""

from dataclasses import dataclass
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from control import limit_commands
from learn import load_model, temporal_noise

STEPS = 10
PARTICLES = 8
TAU_START = 0.7
SAFE_CLEARANCE = 0.012
TERMINAL_WEIGHT = 3200.0
TERMINAL_CLEARANCE = 0.090


@dataclass(frozen=True)
class Proposal:
    action: torch.Tensor
    requested: torch.Tensor
    predicted: torch.Tensor | None
    flow_batches: int
    flow_particles: int
    world_batches: int
    world_rollouts: int


def per_step_risk(states, physical_actions):
    clearance, occupancy, progress, impulse = states.unbind(-1)
    standby = physical_actions.new_tensor([-0.047, -0.182, 0.199])
    displacement = (physical_actions[..., :3] - standby).square().sum(-1)
    left_cost = (displacement / 0.05**2).clamp_max(4.0)
    return (180.0 * occupancy * F.relu(SAFE_CLEARANCE - clearance).square()
            + 5.0 * impulse - 0.14 * progress + 0.006 * left_cost)


def discounted_risk(states, physical_actions, *, terminal=True):
    values = per_step_risk(states, physical_actions)
    discount = 0.98 ** torch.arange(1, values.shape[-1] + 1, dtype=values.dtype, device=values.device)
    cost = (values * discount).sum(-1)
    if not terminal:
        return cost
    d_end = states[..., -1, 0]
    d_mid = states[..., -12, 0]
    closing_rate = F.relu((d_mid - d_end) / 12.0)
    rem_progress = 1.0 - states[..., -1, 2].clamp(0, 1)
    rem_steps = 48.0 * rem_progress
    d_projected = d_end - closing_rate * rem_steps
    return cost + TERMINAL_WEIGHT * F.relu(TERMINAL_CLEARANCE - d_projected).square()


def clipped_descent(grad, limit=3.0):
    norm = torch.linalg.vector_norm(grad.flatten(1), dim=1).view(-1, 1, 1)
    return grad * (limit / norm.clamp_min(limit))


def propose(models, belief, stage_id, method, rng, *, tau_start=TAU_START, exact=False):
    if method not in ("M1", "M2", "M3", "M4-a", "M4-b"):
        raise ValueError(f"Unknown learned method {method}")
    start_index = round(tau_start * STEPS)
    if not 0 <= start_index <= STEPS or abs(start_index / STEPS - tau_start) > 1e-7:
        raise ValueError("tau_start must be on the ten-step Euler grid")
    if method == "M3":
        start_index = STEPS
    particles = PARTICLES if method in ("M3", "M4-b") else 1
    world_enabled = method in ("M3", "M4-a", "M4-b")
    is_joint = method == "M2"
    flow = models.joint if is_joint else models.flow
    da = models.action_mean.numel()
    stage = F.one_hot(torch.tensor([stage_id]), 4).float()
    raw_obs = torch.as_tensor(belief, dtype=torch.float32).unsqueeze(0)
    obs = models.encode(raw_obs)
    previous = raw_obs[0, -1, -da:]
    posterior = models.xi_prob(obs)[0] if world_enabled else None
    x = temporal_noise((particles, flow.horizon, flow.channels), generator=rng)
    flow_batches = flow_particles = world_batches = world_rollouts = 0

    def field(current, tau):
        nonlocal flow_batches, flow_particles
        n = len(current)
        flow_batches += 1
        flow_particles += n
        return flow(current, current.new_full((n, 1), tau), obs.expand(n, -1, -1),
                    stage.expand(n, -1))

    def world_risk(clean):
        nonlocal world_batches, world_rollouts
        n = len(clean)
        physical = limit_commands(models.physical(clean), previous.expand(n, -1))
        action = models.actions(physical).repeat_interleave(3, dim=0)
        xi = torch.eye(3, dtype=clean.dtype).repeat(n, 1)
        pred = models.world(obs.expand(3 * n, -1, -1), stage.expand(3 * n, -1), action, xi)
        pred = pred.reshape(n, 3, flow.horizon, 4) * models.state_scale
        world_batches += 1
        world_rollouts += 3 * n
        costs = discounted_risk(pred, physical[:, None])
        return costs @ posterior

    selected = particles == 1
    for index in range(STEPS):
        tau = index / STEPS
        if not selected and index == start_index:
            with torch.no_grad():
                estimate = x + (1 - tau) * field(x, tau)
                best = int(world_risk(estimate).argmin())
                x = x[best:best + 1]
            selected = True
        if world_enabled and index >= start_index:
            with torch.enable_grad():
                current = x.detach().requires_grad_(True)
                velocity = field(current, tau)
                estimate = current + (1 - tau) * (velocity if exact else velocity.detach())
                grad, = torch.autograd.grad(world_risk(estimate).sum(), current)
            x = (current + (velocity - 0.35 * clipped_descent(grad)) / STEPS).detach()
        elif is_joint and index >= start_index:
            with torch.enable_grad():
                current_state = x[..., da:].detach().requires_grad_(True)
                physical = limit_commands(models.physical(x[..., :da]), previous.expand(len(x), -1))
                clearance, occupancy, progress, impulse = (current_state * models.state_scale).unbind(-1)
                direct = torch.stack((clearance, occupancy.clamp(0, 1), progress.clamp(0, 1),
                                      impulse.clamp_min(0)), dim=-1)
                grad, = torch.autograd.grad(discounted_risk(direct, physical).sum(), current_state)
            with torch.no_grad():
                velocity = field(x, tau)
                velocity[..., da:] -= 0.35 * clipped_descent(grad)
                x = x + velocity / STEPS
        else:
            with torch.no_grad():
                x = x + field(x, tau) / STEPS
    if not selected:
        with torch.no_grad():
            best = int(world_risk(x).argmin())
            x = x[best:best + 1]
    predicted = x[0, :, da:] * models.state_scale if is_joint else None
    requested = models.physical(x[0, :, :da])
    physical = limit_commands(requested, previous)
    return Proposal(physical, requested, predicted, flow_batches, flow_particles,
                    world_batches, world_rollouts)


def action_probe(model_path, raw, out, seeds):
    """Check held-out proposals at the scripted-to-learned handoff."""
    models = load_model(model_path)
    torch.set_num_threads(1)
    rows = []
    for regime in ("static", "slow", "fast"):
        for method in ("M1", "M4-b"):
            jumps, requested_jumps, full_chunk_jumps = [], [], []
            for seed in seeds:
                with np.load(raw / f"{seed:04d}_{regime}_hold.npz", allow_pickle=False) as data:
                    start = json.loads(data["metadata"].item())["branch_starts"][0]
                    obs = data["belief"][start]
                rng = torch.Generator().manual_seed(9811 + seed)
                for _ in range(16):
                    result = propose(models, obs, 1, method, rng)
                    full_chunk_jumps.extend(result.action[:, 3:6].diff(dim=0).norm(dim=-1).tolist())
                    jumps.extend(result.action[:8, 3:6].diff(dim=0).norm(dim=-1).tolist())
                    requested_jumps.extend(result.requested[:8, 3:6].diff(dim=0).norm(dim=-1).tolist())
            rows.append({
                "regime": regime, "method": method, "seeds": list(seeds), "samples": 16 * len(seeds),
                "consecutive_right_waypoint_jump_m": float(np.median(jumps)),
                "max_consecutive_right_waypoint_jump_m": max(full_chunk_jumps),
                "requested_consecutive_right_waypoint_jump_m": float(np.median(requested_jumps)),
            })
    (out / "action_probe.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(rows, flush=True)
    return all(row["max_consecutive_right_waypoint_jump_m"] < 0.008 for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check action continuity on held-out insertion starts")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if action_probe(args.model, args.raw, args.out, args.seeds) else 2)
