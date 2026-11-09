"""Native PyTorch selective SSMs and task-independent past-to-future dynamics."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 2 * self.inner)
        self.conv = nn.Conv1d(self.inner, self.inner, d_conv, groups=self.inner, padding=d_conv - 1)
        self.select = nn.Linear(self.inner, self.inner + 2 * d_state, bias=False)
        delta = torch.exp(torch.empty(self.inner).uniform_(math.log(0.001), math.log(0.1)))
        self.delta_bias = nn.Parameter(delta + torch.log(-torch.expm1(-delta)))
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _transition(self, x, decay, drive, read, state, a):
        # Phase 2 specifies exp(delta*A) for state decay and delta*B
        # for the approximate ZOH input term.
        delta = F.softplus(decay + self.delta_bias)
        state = (delta[..., None] * a).exp() * state + (delta * x)[..., None] * drive[..., None, :]
        return state, (state * read[..., None, :]).sum(-1) + self.D * x

    def forward(self, u):
        x, gate = self.in_proj(u).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :u.size(1)].transpose(1, 2))
        decay, drive, read = self.select(x).split((self.inner, self.d_state, self.d_state), dim=-1)
        state = x.new_zeros((len(x), self.inner, self.d_state))
        a = -self.A_log.exp()
        outputs = []
        for xt, dt, bt, ct in zip(*(v.unbind(1) for v in (x, decay, drive, read)), strict=True):
            state, y = self._transition(xt, dt, bt, ct, state, a)
            outputs.append(y)
        return self.norm(u + self.out_proj(torch.stack(outputs, 1) * F.silu(gate)))

    def start(self, u):
        conv = u.new_zeros((len(u), self.inner, self.d_conv - 1))
        state = u.new_zeros((len(u), self.inner, self.d_state))
        return conv, state, -self.A_log.exp()

    def step(self, u, cache):
        conv, state, a = cache
        x, gate = self.in_proj(u).chunk(2, dim=-1)
        window = torch.cat((conv, x.unsqueeze(-1)), dim=-1)
        x = F.silu((window * self.conv.weight[:, 0]).sum(-1) + self.conv.bias)
        decay, drive, read = self.select(x).split((self.inner, self.d_state, self.d_state), dim=-1)
        state, y = self._transition(x, decay, drive, read, state, a)
        return self.norm(u + self.out_proj(y * F.silu(gate))), (window[..., 1:], state, a)


class MambaPast(nn.Module):
    def __init__(self, obs_dim, d_z, modes):
        super().__init__()
        self.embed = nn.Linear(obs_dim, d_z)
        self.scan = MambaBlock(d_z)
        self.posterior = nn.Linear(d_z, modes)

    def forward(self, obs):
        belief = self.scan(self.embed(obs))[:, -1]
        return belief, self.posterior(belief)


class MambaFuture(nn.Module):
    def __init__(self, d_z, action_dim):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, d_z)
        self.feedback = nn.Linear(d_z, d_z, bias=False)
        self.scan = MambaBlock(d_z)

    def forward(self, z, action):
        cache = self.scan.start(z)
        _, cache = self.scan.step(z, cache)
        context = z
        outputs = []
        for token in self.action_embed(action).unbind(1):
            z, cache = self.scan.step(token + context + self.feedback(z), cache)
            outputs.append(z)
        return torch.stack(outputs, dim=1)


class UnifiedMambaWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, state_dim, *, d_z=128, belief_dim=128, modes=3,
                 context_dim=4, safety=True):
        super().__init__()
        self.past = MambaPast(obs_dim, belief_dim, modes)
        self.initialize = nn.Linear(belief_dim + modes + context_dim, d_z)
        self.future = MambaFuture(d_z, action_dim)
        self.g_state = nn.Sequential(nn.Linear(d_z, d_z), nn.SiLU(), nn.Linear(d_z, state_dim))
        self.g_safe = nn.Sequential(nn.Linear(d_z, d_z), nn.SiLU(), nn.Linear(d_z, 5)) if safety else None
        self.register_buffer("event_log_weight", torch.zeros(()))

    def encode(self, obs):
        return self.past(obs)

    def latent(self, belief, context, action, xi):
        initial = self.initialize(torch.cat((belief, xi, context), dim=-1)).tanh()
        return self.future(initial, action)

    def decode_safety(self, z, *, raw=False):
        pred = self.g_safe(z)
        if raw:
            return pred
        return self.safety_values(pred)

    def safety_values(self, pred):
        probability = (pred[..., 4] - self.event_log_weight).sigmoid()
        return torch.stack((pred[..., 0], pred[..., 1].sigmoid(), pred[..., 2].sigmoid(),
                            probability * F.softplus(pred[..., 3])), dim=-1)

    def forward(self, obs, context, action, xi, *, raw=False):
        belief, _ = self.encode(obs)
        return self.decode_safety(self.latent(belief, context, action, xi), raw=raw)
