"""Grouped dataset construction and supervised Phase 1 training on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STATE_SCALE = (0.1, 1.0, 1.0, 0.05)
IMPULSE_THRESHOLD = 0.004
REGIMES = ("static", "slow", "fast")


def load_windows(raw, split, horizon, stride=2):
    fields = {name: [] for name in ("belief", "A1", "S1", "stage", "xi", "expert", "expert_mask", "group")}
    for file in sorted(raw.glob("*.npz")):
        with np.load(file, allow_pickle=False) as row:
            meta = json.loads(row["metadata"].item())
            seed = meta["seed"]
            if seed not in split:
                continue
            belief, actions, states, stages, expert = (
                row[name] for name in ("belief", "action", "next_state", "stage", "expert")
            )
        ranges = meta.get("insertion_ranges")
        if not ranges:
            raise ValueError(f"Missing insertion ranges in {file}")
        controlled = np.zeros(len(actions), dtype=bool)
        for start, end in ranges:
            if not (0 <= start < end <= len(actions) and np.all(np.isin(stages[start:end], (1, 2)))):
                raise ValueError(f"Invalid insertion range in {file}: {(start, end)}")
            controlled[start:end] = True
        if np.any(expert & ~controlled):
            raise ValueError(f"Expert samples outside the learned insertion phase in {file}")
        starts = np.concatenate([np.arange(start, end - horizon + 1, stride) for start, end in ranges])
        indices = starts[:, None] + np.arange(horizon)
        same_stage = (stages[indices] == stages[starts, None]).all(axis=1)
        starts, indices = starts[same_stage], indices[same_stage]
        if not len(starts):
            continue
        fields["belief"].append(belief[starts])
        fields["A1"].append(actions[indices])
        fields["S1"].append(states[indices])
        fields["stage"].append(stages[starts])
        fields["xi"].append(np.full(len(starts), REGIMES.index(meta["regime"])))
        masks = np.logical_and.accumulate(expert[indices], axis=1)
        minimum = 4 if meta["scheme"] == "direct" else horizon
        fields["expert"].append(masks.sum(axis=1) >= minimum)
        fields["expert_mask"].append(masks)
        fields["group"].append(np.full(len(starts), seed))
    if not fields["A1"]:
        raise ValueError(f"No complete within-stage windows for seeds {split}")
    return {name: np.concatenate(values) for name, values in fields.items()}


class Flow(nn.Module):
    def __init__(self, obs_dim, horizon, width, channels):
        super().__init__()
        self.horizon = horizon
        self.channels = channels
        self.anchors = 7
        self.condition = nn.Sequential(
            nn.Linear(obs_dim + 4, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
        )
        self.anchor_head = nn.Linear(width, self.anchors * channels)
        self.context = nn.Linear(width, 16)
        self.field = nn.Sequential(
            nn.Conv1d(2 * channels + 16 + 4, 48, 5, padding=2), nn.SiLU(),
            nn.Conv1d(48, 48, 5, padding=2), nn.SiLU(), nn.Conv1d(48, channels, 3, padding=1),
        )
        self.register_buffer("position", torch.linspace(0, 1, horizon).reshape(1, 1, horizon))

    def forward(self, x, tau, obs, stage):
        phase = tau * torch.pi
        cond = self.condition(torch.cat((obs.flatten(1), stage), dim=-1))
        anchors = self.anchor_head(cond).reshape(len(x), self.channels, self.anchors)
        centre = F.interpolate(anchors, size=self.horizon, mode="linear", align_corners=True)
        context = torch.cat((self.context(cond), tau, phase.sin(), phase.cos()), dim=-1)
        features = torch.cat((x.transpose(1, 2), centre, context[:, :, None].expand(-1, -1, self.horizon),
                              self.position.expand(len(x), -1, -1)), dim=1)
        endpoint = (centre + 0.35 * self.field(features)).transpose(1, 2)
        return (endpoint - x) / (1.12 - tau[:, :, None])


class World(nn.Module):
    def __init__(self, obs_dim, channels, width):
        super().__init__()
        self.initialize = nn.Sequential(nn.Linear(obs_dim + 4 + 3, width), nn.SiLU(), nn.Linear(width, width))
        self.action_rnn = nn.GRU(channels, width, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 5))
        self.register_buffer("event_log_weight", torch.zeros(()))

    def forward(self, obs, stage, action, xi, *, raw=False):
        start = torch.cat((obs.flatten(1), stage, xi), dim=-1)
        features, _ = self.action_rnn(action, self.initialize(start).unsqueeze(0))
        pred = self.readout(features)
        if raw:
            return pred
        gate = (pred[..., 4] - self.event_log_weight).sigmoid()
        return torch.stack((pred[..., 0], pred[..., 1].sigmoid(), pred[..., 2].sigmoid(),
                            gate * F.softplus(pred[..., 3])), dim=-1)


class Posterior(nn.Module):
    def __init__(self, history, width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(history * 2, width), nn.SiLU(), nn.Linear(width, 3))

    def forward(self, lid_history):
        return self.encoder(lid_history.flatten(1))


class Modules(nn.Module):
    def __init__(self, history, obs_dim, horizon, action_dim, width):
        super().__init__()
        self.flow = Flow(history * obs_dim, horizon, width, action_dim)
        self.joint = Flow(history * obs_dim, horizon, width, action_dim + 4)
        self.world = World(history * obs_dim, action_dim, width)
        self.posterior = Posterior(history, width)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("state_scale", torch.tensor(STATE_SCALE))
        self.register_buffer("action_low", torch.zeros(action_dim))
        self.register_buffer("action_high", torch.zeros(action_dim))

    def encode(self, obs):
        return ((obs - self.obs_mean) / self.obs_std).clamp(-8, 8)

    def actions(self, actions):
        return (actions - self.action_mean) / self.action_std

    def physical(self, actions):
        actions = actions * self.action_std + self.action_mean
        return torch.maximum(self.action_low, torch.minimum(self.action_high, actions))

    def xi_prob(self, normalized_obs):
        return self.posterior(normalized_obs[..., 32:34]).softmax(dim=-1)


def tensors(data):
    return {name: torch.from_numpy(value) for name, value in data.items()}


def normalized_batch(models, data, ix):
    obs = models.encode(data["belief"][ix])
    action = models.actions(data["A1"][ix])
    state = data["S1"][ix] / models.state_scale
    stage = F.one_hot(data["stage"][ix].long(), 4).float()
    xi = F.one_hot(data["xi"][ix].long(), 3).float()
    return obs, action, state, stage, xi


def temporal_noise(shape, *, generator=None, device=None, dtype=None):
    batch, horizon, channels = shape
    anchors = torch.randn((batch, channels, 7), generator=generator, device=device, dtype=dtype)
    smooth = F.interpolate(anchors, size=horizon, mode="linear", align_corners=True).transpose(1, 2)
    residual = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return 0.65 * smooth + 0.05 * residual


def flow_loss(model, clean, obs, stage, action_dim, mask, lambda_smooth=0.6):
    last = (mask.sum(dim=1) - 1).clamp_min(0)
    endpoint = clean[torch.arange(len(clean)), last]
    clean = torch.where(mask[..., None], clean, endpoint[:, None])
    noise = temporal_noise(clean.shape, device=clean.device, dtype=clean.dtype)
    tau = torch.rand((len(clean), 1))
    mixed = torch.lerp(noise, clean, tau[:, :, None])
    velocity = model(mixed, tau, obs, stage)
    pred_clean = mixed + (1 - tau[:, :, None]) * velocity
    predicted_change = pred_clean[..., :action_dim].diff(dim=1)
    target_change = clean[..., :action_dim].diff(dim=1)
    smooth_error = (predicted_change - target_change).square().mean(dim=-1)
    smoothness = (smooth_error * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp_min(1)
    error = (velocity - (clean - noise)).square().mean(dim=-1)
    return (error * mask).sum() / mask.sum() + lambda_smooth * smoothness


def world_loss(world, obs, action, state, stage, xi, event_weight):
    pred = world(obs, stage, action, xi, raw=True)
    positive = state[..., 3] >= IMPULSE_THRESHOLD / STATE_SCALE[3]
    collision = F.binary_cross_entropy_with_logits(pred[..., 4], positive.float(), pos_weight=event_weight)
    error = F.smooth_l1_loss(F.softplus(pred[..., 3]), state[..., 3], reduction="none")
    magnitude = (error * positive).sum() / positive.sum().clamp_min(1)
    clearance = F.smooth_l1_loss(pred[..., 0], state[..., 0])
    occupancy = F.binary_cross_entropy_with_logits(pred[..., 1], state[..., 1], pos_weight=torch.tensor(3.0))
    progress = F.smooth_l1_loss(pred[..., 2].sigmoid(), state[..., 2])
    return 3 * clearance + occupancy + 2 * progress + collision + 0.5 * magnitude


@torch.no_grad()
def metrics(models, data, *, marginalize=False):
    all_pred, probs = [], []
    for indices in torch.arange(len(data["A1"])).split(128):
        obs, action, _, stage, xi = normalized_batch(models, data, indices)
        probability = models.xi_prob(obs)
        probs.append(probability)
        if marginalize:
            outcomes = models.world(obs.repeat_interleave(3, dim=0), stage.repeat_interleave(3, dim=0),
                                    action.repeat_interleave(3, dim=0), torch.eye(3).repeat(len(obs), 1))
            expected = (outcomes.reshape(len(obs), 3, *outcomes.shape[1:]) * probability[:, :, None, None]).sum(1)
        else:
            expected = models.world(obs, stage, action, xi)
        all_pred.append(expected * models.state_scale)
    pred, truth = torch.cat(all_pred), data["S1"]
    event = truth[..., 3] >= IMPULSE_THRESHOLD
    positive = pred[..., 3] >= IMPULSE_THRESHOLD
    tp = int((positive & event).sum())
    occ = truth[..., 1] > 0.5
    return {
        "windows": len(truth), "expert_windows": int(data["expert"].sum()),
        "clearance_mae_m": float((pred[..., 0] - truth[..., 0]).abs().mean()),
        "occupied_clearance_mae_m": float((pred[..., 0] - truth[..., 0]).abs()[occ].mean()) if occ.any() else None,
        "impulse_mae_Ns": float((pred[..., 3] - truth[..., 3]).abs().mean()),
        "collision_positive_steps": int(event.sum()),
        "collision_precision": tp / int(positive.sum()) if positive.any() else None,
        "collision_recall": tp / int(event.sum()) if event.any() else None,
        "latent_accuracy": float((torch.cat(probs).argmax(-1) == data["xi"]).float().mean()),
    }


@torch.no_grad()
def paired_metrics(models, raw, seeds):
    clearance_errors, differences, history_errors = [], [], []
    horizon = models.flow.horizon
    for seed in sorted(seeds):
        for regime_index, regime in enumerate(REGIMES):
            pair = []
            for scheme in ("direct", "hold"):
                file = raw / f"{seed:04d}_{regime}_{scheme}.npz"
                if not file.is_file():
                    break
                with np.load(file, allow_pickle=False) as episode:
                    start = json.loads(episode["metadata"].item())["branch_starts"][0]
                    stop = start + horizon
                    pair.append((episode["belief"][start], episode["action"][start:stop],
                                 episode["next_state"][start:stop]))
            if len(pair) != 2:
                continue
            belief = torch.as_tensor(pair[0][0]).repeat(2, 1, 1)
            actions = torch.as_tensor(np.stack([branch[1] for branch in pair]))
            truth = np.stack([branch[2] for branch in pair])
            stage = torch.eye(4)[1].repeat(2, 1)
            xi = torch.eye(3)[regime_index].repeat(2, 1)
            pred = (models.world(models.encode(belief), stage, models.actions(actions), xi)
                    * models.state_scale).numpy()
            delta_truth = truth[0] - truth[1]
            delta_pred = pred[0] - pred[1]
            clearance_errors.append(float(np.abs(delta_truth[:, 0] - delta_pred[:, 0]).mean()))
            history_errors.append(float(np.abs(pair[0][0] - pair[1][0]).max()))
            gap = truth[0, :, 3].max() - truth[1, :, 3].max()
            if abs(gap) >= IMPULSE_THRESHOLD:
                differences.append(bool(gap * (pred[0, :, 3].max() - pred[1, :, 3].max()) > 0))
    return {
        "matched_action_pairs": len(clearance_errors),
        "max_prefix_feature_difference": max(history_errors, default=None),
        "paired_clearance_effect_mae_m": float(np.mean(clearance_errors)) if clearance_errors else None,
        "informative_collision_pairs": len(differences),
        "correct_collision_order": sum(differences),
    }


def fit(raw, out, train_seeds, val_seeds, test_seeds, *, horizon=24, steps=850, seed=124):
    groups = [set(s) for s in (train_seeds, val_seeds, test_seeds)]
    if len(set.union(*groups)) != sum(map(len, groups)):
        raise ValueError("A rollout seed cannot appear in multiple splits")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    data = {}
    for name, selected in zip(("train", "val", "test"), groups, strict=True):
        windows = load_windows(raw, selected, horizon)
        if set(windows["group"]) != selected:
            raise ValueError(f"Missing one or more requested {name} groups")
        np.savez_compressed(out / f"{name}_windows.npz", **windows)
        data[name] = tensors(windows)
    train = data["train"]
    history, obs_dim = train["belief"].shape[-2:]
    action_dim = train["A1"].shape[-1]
    config = dict(history=history, obs_dim=obs_dim, horizon=horizon, action_dim=action_dim, width=96)
    models = Modules(**config)
    obs = train["belief"].reshape(-1, obs_dim)
    action = train["A1"].reshape(-1, action_dim)
    models.obs_mean.copy_(obs.mean(0))
    models.obs_std.copy_(obs.std(0).clamp_min(0.015))
    models.action_mean.copy_(action.mean(0))
    models.action_std.copy_(action.std(0).clamp_min(0.006))
    models.action_low.copy_(action.min(0).values)
    models.action_high.copy_(action.max(0).values)
    mask = train["expert"].bool().nonzero().flatten()
    handoff = (train["belief"][:, -1, -8] == 0).nonzero().flatten()
    expert_handoff = handoff[train["expert"][handoff]]
    if not len(mask):
        raise ValueError("No valid expert action windows")
    events = train["S1"][..., 3] >= IMPULSE_THRESHOLD
    if not events.any():
        raise ValueError("Training trajectories lack actual lid collision events")
    event_weight = ((events.numel() - events.sum()) / events.sum()).sqrt().clamp(1, 12)
    models.world.event_log_weight.copy_(event_weight.log())
    optimizer = torch.optim.AdamW(models.parameters(), lr=0.001, weight_decay=0.0001)
    batch_size = 64
    curve = []
    for step in range(1, steps + 1):
        obs, act, state, stage, xi = normalized_batch(models, train, torch.randint(len(events), (batch_size,)))
        wm = world_loss(models.world, obs, act, state, stage, xi, event_weight)
        latent = F.cross_entropy(models.posterior(obs[..., 32:34]), xi.argmax(-1))
        starts = handoff[torch.randint(len(handoff), (batch_size,))]
        handoff_obs = models.encode(train["belief"][starts])
        handoff_loss = F.cross_entropy(models.posterior(handoff_obs[..., 32:34]), train["xi"][starts].long())
        latent = 0.5 * (latent + handoff_loss)
        ix = torch.cat((mask[torch.randint(len(mask), (batch_size // 2,))],
                        expert_handoff[torch.randint(len(expert_handoff), (batch_size // 2,))]))
        obs, act, state, stage, _ = normalized_batch(models, train, ix)
        expert_mask = train["expert_mask"][ix]
        fm = flow_loss(models.flow, act, obs, stage, action_dim, expert_mask)
        joint = flow_loss(models.joint, torch.cat((act, state), dim=-1), obs, stage, action_dim, expert_mask)
        optimizer.zero_grad()
        (wm + fm + joint + 0.4 * latent).backward()
        nn.utils.clip_grad_norm_(models.parameters(), 5.0)
        optimizer.step()
        if step % 100 == 0 or step in (1, steps):
            row = dict(step=step, flow=float(fm.detach()), joint=float(joint.detach()),
                       world=float(wm.detach()), latent=float(latent.detach()))
            curve.append(row)
            print(row, flush=True)
    models.eval()
    report = {name: metrics(models, values) for name, values in data.items()}
    report["val"]["paired"] = paired_metrics(models, raw, groups[1])
    report["test"]["paired"] = paired_metrics(models, raw, groups[2])
    report["posterior_marginalized"] = {name: metrics(models, data[name], marginalize=True)
                                       for name in ("val", "test")}
    report.update(training_seed=seed, updates=steps, group_seeds=[sorted(x) for x in groups], curve=curve)
    torch.save({"config": config, "weights": models.state_dict()}, out / "model.pt")
    (out / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "curve"}, flush=True)


def load_model(file):
    ckpt = torch.load(file, map_location="cpu", weights_only=True)
    models = Modules(**ckpt["config"])
    models.load_state_dict(ckpt["weights"])
    return models.eval().requires_grad_(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, nargs="+", required=True)
    parser.add_argument("--val", type=int, nargs="+", required=True)
    parser.add_argument("--test", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=850)
    args = parser.parse_args()
    fit(args.raw, args.out, args.train, args.val, args.test, steps=args.steps)
