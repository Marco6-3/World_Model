"""Regression checks for reset geometry, terminal risk, and flow gradients."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pusht_benchmark as push
import torch
from control import limit_commands, ARM_STEP_M
from inference import discounted_risk, flow_search, Search, propose_phase2


class FixTests(unittest.TestCase):
    def test_pusht_reset_preserves_geometric_pose(self):
        env = push.make_env()
        try:
            for angle in (0, np.pi / 2, np.pi, -np.pi / 2):
                state = np.array([50., 50., 256., 256., angle])
                observed, _ = env.reset(seed=11, options={"reset_to_state": state})
                np.testing.assert_allclose(observed[:4], state[:4], atol=1e-5)
                self.assertAlmostEqual(float(np.arctan2(np.sin(observed[4] - angle),
                                                       np.cos(observed[4] - angle))), 0.0, places=5)
        finally:
            env.close()

    def test_terminal_warning_is_distinct_from_step_threshold(self):
        states = torch.zeros(1, 24, 4)
        states[..., 0] = 0.065
        states[..., 1] = 1
        actions = torch.zeros(1, 24, 7)
        base = discounted_risk(states, actions, terminal=False)
        risk = discounted_risk(states, actions)
        self.assertAlmostEqual(float(risk - base), 3200 * (0.090 - 0.065)**2, places=5)
        states[..., 0] = 0.1
        torch.testing.assert_close(discounted_risk(states, actions),
                                   discounted_risk(states, actions, terminal=False))

    def test_flow_guidance_survives_output_projection(self):
        noise = torch.full((1, 24, 7), 0.5)
        field = lambda x, tau: x * 0
        project = lambda x: limit_commands(x, torch.zeros(len(x), 7))
        seen = []

        def risk(x):
            seen.append(float(x.detach().abs().max()))
            return x.square().sum((1, 2))

        guided = flow_search(noise.clone(), field, risk, project, Search(1, tau_start=0.7))
        self.assertGreater(min(seen), ARM_STEP_M)
        self.assertLess(float(guided[-1, -1]), 0.5)
        delta = torch.diff(torch.cat((torch.zeros(1, 7), guided)), dim=0)
        self.assertLessEqual(float(delta[:, :6].reshape(-1, 2, 3).norm(dim=-1).max()), ARM_STEP_M + 1e-6)

    def test_pusht_velocity_has_action_gradient(self):
        torch.manual_seed(7)
        model = push.PushTFlow()
        action = torch.randn(2, 16, 2, requires_grad=True)
        velocity = model(action, torch.full((2, 1), 0.7), torch.randn(2, 8, 12))
        grad, = torch.autograd.grad(velocity.square().mean(), action)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 1e-6)

    def test_phase2_risk_receives_unprojected_action(self):
        class World:
            def encode(self, obs):
                return torch.zeros(1, 2), torch.zeros(1, 3)

            def latent(self, belief, stage, action, xi):
                self.action = action
                return action

            def decode_safety(self, action):
                value = action[..., 0]
                return torch.stack((0.1 - 0.02 * value, value * 0 + 1,
                                    value * 0 + 0.2, value * 0), -1)

        world = World()
        model = SimpleNamespace(action_mean=torch.zeros(7), state_scale=torch.ones(4), world=world,
                                flow=SimpleNamespace(horizon=24, channels=7), flow_history=2,
                                encode=lambda x: x, physical=lambda x: x, actions=lambda x: x)

        def inspect_search(noise, field, risk, project, search):
            action = torch.full((1, 24, 7), 2.0, requires_grad=True)
            cost = risk(action)
            torch.testing.assert_close(world.action, action.repeat_interleave(3, 0))
            gradient, = torch.autograd.grad(cost.sum(), action)
            self.assertGreater(float(gradient[..., 0].abs().sum()), 0.1)
            return project(action.detach())[0]

        with patch("inference.flow_search", inspect_search):
            propose_phase2(model, np.zeros((24, 7), dtype=np.float32), 1,
                           "M4-b", torch.Generator().manual_seed(1), 0.7)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
