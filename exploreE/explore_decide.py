"""exploreE: copy of bag_learning.decide with ONE switch, ``recovery_prior``.

recovery_prior=True  -> byte-identical behaviour to bag_learning.decide (verified by check_equivalence.py)
recovery_prior=False -> removes the hand-written emergency withdraw/re-spread noise offsets
                        (bag_learning.py L295-301). Everything else (models, slot-side exploration
                        offsets, risk, guidance schedule, projection) is unchanged.

Method strings: "<original method>" or "<original method>|noprior".
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from bag_learning import temporal_noise, HORIZON, Search, flow_search, safety_risk  # noqa: F401


def split(method):
    base, _, tag = method.partition('|')
    return base, tag != 'noprior'


def decide(models, history, method, rng):
    method, recovery_prior = split(method)
    kind = 'short' if method in ('BC H4', 'Joint-FM', 'Ours Short-MLP H4') else 'four' if method == 'Ours 4D-WM' else 'full'
    model = models[kind]; joint = method == 'Joint-FM'
    configs = {'BC H4': Search(1), 'Joint-FM': Search(1, tau_start=.7), 'GPC-RANK': Search(16),
               'GPC-OPT': Search(1, 20), 'Ours Short-MLP H4': Search(8, tau_start=.7),
               'Ours 4D-WM': Search(8, tau_start=.7), 'Ours tau=0.7': Search(8, tau_start=.7),
               'Ours tau=0.9': Search(8, tau_start=.9)}
    config = configs[method]
    raw = torch.as_tensor(history, dtype=torch.float32, device=model.mean.device)[None]
    obs = model.encode(raw); stage = raw[:, -1, 25:28]
    flow = model.joint if joint else model.flow
    noise = temporal_noise((config.particles, HORIZON, flow.channels), generator=rng).to(obs.device)
    with torch.no_grad():
        belief, logits = model.world.encode(obs); prob = logits.softmax(-1)[0]
        slots = model.world.slots(belief).sigmoid()[0]
        hint = (1 - slots)[None] if int(stage.argmax()) == 1 else raw[:, -1, 28:30]
        hint = hint / hint.sum(-1, keepdim=True).clamp_min(1e-6)
    emergency = bool(raw[0, -1, 4] < .085 or raw[0, -1, 4] - raw[0, -2, 4] < -.006)
    if config.particles > 1 and int(stage.argmax()) == 1:
        noise[:config.particles // 2, :, 3] -= .35
        noise[config.particles // 2:, :, 3] += .35
    if recovery_prior and emergency and config.particles >= 8:
        noise[:2, :, 5] += 1.2
        noise[:2, :, 4] -= .4
        noise[:2, :, 6] += .5
        noise[2:4, :, 1] -= .5
        noise[2:4, :, 6] -= .5

    def field(x, t): return flow(x, x.new_full((len(x), 1), t), obs.expand(len(x), -1, -1), hint.expand(len(x), -1))

    def project(x):
        action = model.physical(x[..., :8])
        low = action.new_tensor([-.09, -.23, .08, -.09, -.06, .04, .006, .006])
        high = action.new_tensor([.02, -.04, .24, .09, .25, .25, .038, .038])
        action = torch.maximum(low, torch.minimum(high, action))
        previous = raw[:, -1, -8:-2].expand(len(x), -1); steps = []
        for a in action.unbind(1):
            delta = (a[:, :6] - previous).reshape(-1, 2, 3)
            previous = previous + (delta * (.0075 / delta.norm(dim=-1, keepdim=True).clamp_min(.0075))).flatten(1)
            steps.append(torch.cat((previous, a[:, 6:]), -1))
        normalized = model.actions(torch.stack(steps, 1))
        return torch.cat((normalized, x[..., 8:]), -1) if joint else normalized

    def risk(x):
        nonlocal belief, prob, slots
        physical = model.physical(x[..., :8])
        if joint: return safety_risk(x[..., 8:] * model.ss, physical)
        n = len(x)
        z = model.world.latent(belief.expand(n * 3, -1), stage.expand(n * 3, -1),
                               x.repeat_interleave(3, 0), torch.eye(3, device=x.device).repeat(n, 1))
        safety = model.world.decode_safety(z) * model.ss
        base = safety_risk(safety, physical.repeat_interleave(3, 0)).reshape(n, 3) @ prob
        pos = physical[..., 3]
        occupied_overlap = slots[0] * torch.exp(-((pos + .048) / .025).square()) + slots[1] * torch.exp(-((pos - .048) / .025).square())
        target = .048 * (slots[0] - slots[1]) if int(stage.argmax()) == 1 else (.048 if raw[0, -1, 29] > .5 else -.048)
        return base + 35 * (pos - target).square().mean(-1) + 4 * occupied_overlap[:, -8:].mean(-1)

    remaining = max(1, 10 - round(config.tau_start * 10))
    guidance = .5 * 10 / remaining
    if emergency and config.tau_start < 1: guidance = max(guidance, 1.2 if config.tau_start == .7 else 2.5)
    result = flow_search(noise, field, risk, project, config, opt_lr=.08, guidance=guidance)
    prediction = int(slots.argmax()) if slots is not None else None
    return model.physical(result[..., :8]).cpu().numpy(), dict(predicted_occupied_slot=prediction,
                                                              recovery_prior=recovery_prior, emergency=emergency)
