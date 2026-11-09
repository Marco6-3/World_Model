"""Train Phase 2 packing dynamics and evaluate held-out geometry and occlusion."""

import os
import sys

# Keep Isaac Sim's already-loaded torch ABI intact.
cuda_site = r"C:\isaacsim_env\phase2_cuda"
if "torch" not in sys.modules and os.path.isdir(cuda_site):
    sys.path.insert(0, cuda_site)

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learn import IMPULSE_THRESHOLD, REGIMES, load_model, normalized_batch, safety_loss, tensors
from mamba_wm import MambaPast, UnifiedMambaWorldModel


HISTORY = 24
HORIZON = 24
GEOMETRY = (32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 49, 50, 51)
SPLITS = (list(range(100, 106)), [106, 107], [108, 109])
DIMENSIONS = (128, 4, 32, 256)


class SafetyStateWorld(UnifiedMambaWorldModel):
    """An explicit four-coordinate Markov recurrence in safety state space."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim, len(GEOMETRY), d_z=4, safety=False)
        del self.future
        self.transition = nn.Sequential(nn.Linear(4 + action_dim, 128), nn.SiLU(), nn.Linear(128, 4))
        self.event = nn.Linear(1, 1)

    def latent(self, belief, context, action, xi):
        z = self.initialize(torch.cat((belief, xi, context), dim=-1))
        outputs = []
        for command in action.unbind(1):
            z = z + 0.15 * self.transition(torch.cat((z, command), dim=-1))
            outputs.append(z)
        return torch.stack(outputs, 1)

    def decode_safety(self, z, *, raw=False):
        pred = torch.cat((z, self.event(z[..., 3:4])), dim=-1)
        return pred if raw else self.safety_values(pred)


def configure(models, d_z):
    models.phase2 = True
    obs_dim, action_dim = models.obs_mean.numel(), models.action_mean.numel()
    models.flow_history = models.flow.condition[0].in_features // obs_dim
    models.world = (SafetyStateWorld(obs_dim, action_dim) if d_z == 4 else
                    UnifiedMambaWorldModel(obs_dim, action_dim, len(GEOMETRY), d_z=d_z))


def load_rollouts(raw, seeds):
    windows, memory, pairs = [], [], []
    for file in sorted(raw.glob("*.npz")):
        if int(file.name.split("_")[0]) not in seeds:
            continue
        with np.load(file, allow_pickle=False) as row:
            meta = json.loads(row["metadata"].item())
            obs = row["belief"][:, -1]
            action, state, stages = row["action"], row["next_state"], row["stage"]
        regime = REGIMES.index(meta["regime"])
        padded = np.concatenate((np.repeat(obs[:1], HISTORY - 1, axis=0), obs))
        for begin, end in meta["insertion_ranges"]:
            for i in range(begin, min(end, len(obs) - 1) - HORIZON + 1, 2):
                stop = i + HORIZON
                future = obs[i + 1:stop + 1]
                hist = padded[i:i + HISTORY]
                mask = np.ones((HORIZON, len(GEOMETRY)), dtype=bool)
                mask[:, 8:11] = future[:, 43:44] > 0.5
                mask[:, 11:14] = future[:, 52:53] > 0.5
                entry = dict(belief=hist, A1=action[i:stop], S1=state[i:stop],
                             stage=stages[i], xi=regime, geo=future[:, GEOMETRY],
                             geo_mask=mask, handoff=i == begin)
                windows.append(entry)
                if i == begin:
                    # Six measured frames through release, then an artificial
                    # observation blackout. Mask angle too to remove velocity leakage.
                    masked = np.zeros((HISTORY, 3), dtype=np.float32)
                    masked[:6, :2] = hist[-6:, 32:34]
                    masked[:6, 2] = 1
                    memory.append(dict(obs=masked, action=entry["A1"], target=future[:, 32:34],
                                       regime=regime, seed=meta["seed"]))
                    if begin == meta["branch_starts"][0] and meta["scheme"] in ("direct", "hold"):
                        pairs.append(dict(**entry, seed=meta["seed"], scheme=meta["scheme"],
                                          episode_peak_impulse=float(state[:, 3].max()),
                                          episode_success=bool(meta["success"])))
    if {row["seed"] for row in memory} != set(seeds):
        raise ValueError(f"Missing packing episodes in split {seeds}")
    return stack(windows), stack(memory), pairs


def stack(rows):
    return tensors({key: np.asarray([row[key] for row in rows]) for key in rows[0]})


def predict(model, obs, action, stage):
    world = model.world
    belief, logits = world.encode(obs)
    prob = logits.softmax(-1)
    n = len(action)
    modes = torch.eye(3, device=obs.device).repeat(n, 1)
    z = world.latent(belief.repeat_interleave(3, 0), stage.repeat_interleave(3, 0),
                     action.repeat_interleave(3, 0), modes)
    def marginal(values):
        return (values.reshape(n, 3, *values.shape[1:]) * prob[:, :, None, None]).sum(1)
    return marginal(world.decode_safety(z)) * model.state_scale, marginal(world.g_state(z)), prob


@torch.no_grad()
def dynamics_metrics(model, data, geo_mean, geo_std):
    states, geometry, posterior = [], [], []
    for ix in torch.arange(len(data["S1"]), device=geo_mean.device).split(32):
        obs, action, _, stage, _ = normalized_batch(model, data, ix)
        safety, geo, prob = predict(model, obs, action, stage)
        states.append(safety)
        geometry.append(geo * geo_std + geo_mean)
        posterior.append(prob)
    truth = data["S1"]
    pred = torch.cat(states)
    err = (torch.cat(geometry) - data["geo"]).abs()
    contacts = truth[..., 3] >= IMPULSE_THRESHOLD
    missed = contacts & (pred[..., 3] < IMPULSE_THRESHOLD)
    false_alarms = ~contacts & (pred[..., 3] >= IMPULSE_THRESHOLD)
    return {
        "windows": len(truth), "clearance_mae_m": float((pred[..., 0] - truth[..., 0]).abs().mean()),
        "lid_angle_mae_rad": float(err[..., 0].mean()), "lid_speed_mae_rad_s": float(err[..., 1].mean()),
        "visible_object_xyz_mae_m": float(err[..., 8:][data["geo_mask"][..., 8:]].mean()),
        "impulse_mae_Ns": float((pred[..., 3] - truth[..., 3]).abs().mean()),
        "collision_steps": int(contacts.sum()), "missed_steps": int(missed.sum()),
        "noncollision_steps": int((~contacts).sum()), "false_alarm_steps": int(false_alarms.sum()),
        "posterior_accuracy": float((torch.cat(posterior).argmax(-1) == data["xi"]).float().mean()),
    }


@torch.no_grad()
def offline_ranking(model, pairs):
    from inference import discounted_risk

    rows = []
    for start in range(0, len(pairs), 2):
        duo = pairs[start:start + 2]
        assert len(duo) == 2 and duo[0]["seed"] == duo[1]["seed"] and duo[0]["xi"] == duo[1]["xi"]
        assert [row["scheme"] for row in duo] == ["direct", "hold"]
        device = model.action_mean.device
        prefix_difference = float(np.max(np.abs(duo[0]["belief"] - duo[1]["belief"])))
        obs = model.encode(torch.as_tensor(duo[0]["belief"], device=device)[None].expand(2, -1, -1))
        actions = torch.as_tensor(np.stack([row["A1"] for row in duo]), device=device)
        stage = F.one_hot(torch.tensor([row["stage"] for row in duo], dtype=torch.long, device=device), 4).float()
        b, logits = model.world.encode(obs)
        z = model.world.latent(b.repeat_interleave(3, 0), stage.repeat_interleave(3, 0),
                               model.actions(actions).repeat_interleave(3, 0),
                               torch.eye(3, device=device).repeat(2, 1))
        pred = model.world.decode_safety(z).reshape(2, 3, HORIZON, 4) * model.state_scale
        opts = dict(safe_clearance=0.018, terminal_clearance=0.090)
        cost = (discounted_risk(pred, actions[:, None], **opts) * logits.softmax(-1)).sum(-1).tolist()
        labels = torch.as_tensor(np.stack([row["S1"] for row in duo]), device=device)
        observed = discounted_risk(labels, actions, **opts).tolist()
        rows.append(dict(seed=duo[0]["seed"], regime=REGIMES[duo[0]["xi"]], predicted_cost=cost,
                         observed_cost=observed, predicted_choice=duo[int(np.argmin(cost))]["scheme"],
                         observed_choice=duo[int(np.argmin(observed))]["scheme"],
                         peak_impulse=labels[..., 3].amax(-1).tolist(),
                         episode_peak_impulse=[row["episode_peak_impulse"] for row in duo],
                         episode_success=[row["episode_success"] for row in duo],
                         max_prefix_feature_difference=prefix_difference,
                         expected_safe_choice="direct" if duo[0]["xi"] == 0 else "hold",
                         safe_ranking_pass=(int(np.argmin(cost)) == (0 if duo[0]["xi"] == 0 else 1)
                                            and duo[0 if duo[0]["xi"] == 0 else 1]["episode_peak_impulse"] < 0.004
                                            and duo[0 if duo[0]["xi"] == 0 else 1]["episode_success"])))
    return rows


def dump(path, result):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def fit_world(raw, policy, out, steps, device, dimensions):
    splits = [load_rollouts(raw, seeds) for seeds in SPLITS]
    datasets = [{key: value.to(device) for key, value in data.items()} for data, _, _ in splits]
    train, val, test = datasets
    geo_mean = train["geo"].flatten(0, 1).mean(0)
    geo_std = train["geo"].flatten(0, 1).std(0).clamp_min(0.025)
    contacts = train["S1"][..., 3] >= IMPULSE_THRESHOLD
    weight = ((contacts.numel() - contacts.sum()) / contacts.sum()).sqrt().clamp(1, 12)
    handoffs = train["handoff"].nonzero().flatten()
    config = torch.load(policy, map_location="cpu", weights_only=True)["config"]
    report_path = out / "dimensions.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for d_z in dimensions:
        torch.manual_seed(214)
        model = load_model(policy)
        configure(model, d_z)
        model.to(device).train()
        model.world.event_log_weight.copy_(weight.log())
        optimizer = torch.optim.AdamW(model.world.parameters(), lr=0.001, weight_decay=0.0001)
        for step in range(1, steps + 1):
            ix = torch.cat((torch.randint(len(contacts), (24,), device=device),
                            handoffs[torch.randint(len(handoffs), (8,), device=device)]))
            obs, action, state, stage, xi = normalized_batch(model, train, ix)
            b, logits = model.world.encode(obs)
            z = model.world.latent(b, stage, action, xi)
            safety = safety_loss(model.world.decode_safety(z, raw=True), state, weight)
            error = F.smooth_l1_loss(model.world.g_state(z), (train["geo"][ix] - geo_mean) / geo_std,
                                     reduction="none")
            valid = train["geo_mask"][ix]
            loss = safety + 1.5 * error[valid].mean() + 0.4 * F.cross_entropy(logits, xi.argmax(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.world.parameters(), 5.0)
            optimizer.step()
            if step in (1, steps) or step % 100 == 0:
                print(f"Dz={d_z} update={step}/{steps} loss={loss.item():.4f}", flush=True)
        model.eval()
        occupied = val["S1"][..., 0][val["S1"][..., 1] > 0.5]
        model.warning_clearance = float(occupied.quantile(0.75).clamp(0.012, 0.055))
        phase2 = dict(d_z=d_z, warning_clearance=model.warning_clearance)
        checkpoint = out / f"world_{d_z}.pt"
        tmp = checkpoint.with_suffix(".tmp")
        torch.save(dict(config=config, weights=model.state_dict(), phase2=phase2), tmp)
        tmp.replace(checkpoint)
        report[str(d_z)] = dict(**phase2, updates=steps, seed=214, split_seeds=SPLITS,
                                warning_occupied_fraction=float((occupied < model.warning_clearance).float().mean()),
                                parameters=sum(p.numel() for p in model.world.parameters()),
                                val=dynamics_metrics(model, val, geo_mean, geo_std),
                                test=dynamics_metrics(model, test, geo_mean, geo_std),
                                offline_test=offline_ranking(model, splits[2][2]))
        dump(report_path, report)


class MemoryProbe(nn.Module):
    def __init__(self, name, action_dim):
        super().__init__()
        self.name = name
        self.window = 4 if name == "Short-Window MLP" else 24
        if name == "Mamba-Past":
            self.encoder = MambaPast(3, 128, 3)
        elif name == "GRU":
            self.encoder = nn.GRU(3, 128, batch_first=True)
        else:
            self.encoder = nn.Sequential(nn.Flatten(1), nn.Linear(3 * self.window, 128), nn.SiLU(),
                                         nn.Linear(128, 128), nn.SiLU())
        self.posterior = nn.Linear(128, 3) if name != "Mamba-Past" else None
        self.action = nn.Linear(action_dim, 32)
        self.trajectory = nn.Sequential(nn.Linear(161, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, obs, action):
        if self.name == "Mamba-Past":
            b, logits = self.encoder(obs)
        else:
            if self.name == "GRU":
                b = self.encoder(obs)[1][0]
            else:
                b = self.encoder(obs[:, -self.window:])
            logits = self.posterior(b)
        time = torch.linspace(0, 1, action.shape[1], device=obs.device)[None, :, None]
        context = torch.cat((b[:, None].expand(-1, action.shape[1], -1), self.action(action),
                             time.expand(len(obs), -1, -1)), dim=-1)
        return logits, self.trajectory(context)


def fit_memory(raw, policy, out, device):
    samples = [load_rollouts(raw, seeds)[1] for seeds in SPLITS]
    train, val, test = [{k: v.to(device) for k, v in data.items()} for data in samples]
    model = load_model(policy).to(device)
    report = {}
    for name in ("Short-Window MLP", "Full-Window MLP", "GRU", "Mamba-Past"):
        rows = []
        for seed in (214, 215, 216):
            torch.manual_seed(seed)
            probe = MemoryProbe(name, model.action_mean.numel()).to(device)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=0.002)
            scale = train["obs"].new_tensor([1.0, 4.0])
            for _ in range(180):
                ix = torch.randint(len(train["obs"]), (32,), device=device)
                obs = train["obs"][ix].clone()
                obs[:, :6, :2] += 0.025 * torch.randn_like(obs[:, :6, :2]) * scale
                logits, trajectory = probe(obs, model.actions(train["action"][ix]))
                loss = F.cross_entropy(logits, train["regime"][ix].long()) + F.smooth_l1_loss(
                    trajectory, train["target"][ix] / scale)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            probe.eval()
            with torch.no_grad():
                row = dict(seed=seed, updates=180, parameters=sum(p.numel() for p in probe.parameters()))
                for split, data in (("val", val), ("test", test)):
                    logits, pred = probe(data["obs"], model.actions(data["action"]))
                    mae = (pred * scale - data["target"]).abs().mean((0, 1))
                    row[split] = dict(samples=len(pred), correct=int((logits.argmax(-1) == data["regime"]).sum()),
                                      angle_mae_rad=float(mae[0]), speed_mae_rad_s=float(mae[1]))
            rows.append(row)
        report[name] = rows
        print(name, [row["test"] for row in rows], flush=True)
        dump(out / "memory.json", report)


def probe_planner_ranking(raw, out, device="cpu"):
    """Compare handoff proposals against the better of two logged action prefixes."""
    from inference import TAU_START, propose

    model = load_model(out / "world_128.pt").to(device)
    _, _, pairs = load_rollouts(raw, SPLITS[-1])
    references = offline_ranking(model, pairs)
    methods = {"BC": "M1", "Joint-FM": "M2", "GPC-RANK": "GPC-RANK",
               "GPC-OPT": "GPC-OPT", "GPC-RANK+OPT": "GPC-RANK+OPT", "Ours": "M4-b"}
    rows = []
    for index, reference in enumerate(references):
        duo = pairs[2 * index:2 * index + 2]
        belief = duo[0]["belief"]
        stage = int(duo[0]["stage"])
        target = torch.as_tensor(np.stack([row["A1"][:8, :6] for row in duo]))
        for seed in (19, 83, 156, 201):
            for label, method in methods.items():
                tau = 1.0 if method in ("M1", "GPC-RANK", "GPC-OPT", "GPC-RANK+OPT") else TAU_START
                proposal = propose(model, belief, stage, method, torch.Generator().manual_seed(seed),
                                   tau_start=tau, exact=True)
                delta = (proposal.action[:8, :6] - target).reshape(2, 8, 2, 3)
                error = delta.norm(dim=-1).mean(dim=(1, 2)).tolist()
                candidate = ("direct", "hold")[int(np.argmin(error))]
                rows.append(dict(seed=reference["seed"], regime=reference["regime"], proposal_seed=seed,
                                 method=label, matched_candidate=candidate,
                                 reference_candidate=reference["expected_safe_choice"],
                                 mean_distance_m=error, match=candidate == reference["expected_safe_choice"]))
    dump(out / "planner_ranking.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "memory", "ranking"))
    parser.add_argument("--raw", type=Path, default=Path("outputs/final_v4/dataset"))
    parser.add_argument("--policy", type=Path, default=Path("outputs/final_v4/trained/model.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2/spring"))
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--dimensions", type=int, choices=DIMENSIONS, nargs="+", default=DIMENSIONS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "fit":
        fit_world(args.raw, args.policy, args.out, args.steps, args.device, args.dimensions)
    elif args.command == "memory":
        fit_memory(args.raw, args.policy, args.out, args.device)
    else:
        probe_planner_ranking(args.raw, args.out, args.device)


if __name__ == "__main__":
    main()
