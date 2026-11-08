"""Shared command and observation conventions for insertion rollouts."""

import torch


ARM_STEP_M = 0.0075
RELEASE_STEPS = 24
LEFT_HOLD_POSITION = (-0.047, -0.181, 0.060)


def limit_commands(actions, previous):
    """Project each requested arm position into the ball around the last command."""
    arms = previous[..., :6].reshape(*previous.shape[:-1], 2, 3)
    commands = []
    for wanted in actions[..., :6].unflatten(-1, (2, 3)).unbind(-3):
        delta = wanted - arms
        ratio = ARM_STEP_M / delta.norm(dim=-1, keepdim=True).clamp_min(ARM_STEP_M)
        arms = arms + delta * ratio
        commands.append(arms.flatten(-2))
    return torch.cat((torch.stack(commands, dim=-2), actions[..., 6:]), dim=-1)
