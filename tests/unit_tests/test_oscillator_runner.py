"""Paper observation noise has bounded support and honors physical feature scales."""

from types import SimpleNamespace

import pytest
import torch

from learning.runners import OscillatorRunner, get_runner_class


def _runner(num_envs=4096):
    runner = OscillatorRunner.__new__(OscillatorRunner)
    runner.device = "cpu"
    values = {
        "joint_position": torch.zeros(num_envs, 2),
        "phase": torch.zeros(num_envs, 4),
        "angular_velocity": torch.zeros(num_envs, 3),
    }
    runner.env = SimpleNamespace(
        num_envs=num_envs,
        scales={
            "joint_position": torch.tensor([2.0, 4.0]),
            "angular_velocity": 2.0,
        },
        get_states=lambda names: torch.cat([values[name] for name in names], dim=1),
    )
    return runner


def test_oscillator_runner_is_registered():
    assert get_runner_class("OscillatorRunner") is OscillatorRunner


def test_uniform_noise_support_and_variance_with_feature_and_global_scales():
    runner = _runner(num_envs=20000)
    torch.manual_seed(914)
    noise = runner.get_noise(
        ["joint_position", "phase", "angular_velocity"],
        {
            "scale": 2.0,
            "joint_position": [0.2, 0.8],
            "angular_velocity": 0.3,
        },
    )

    assert noise.shape == (20000, 9)
    torch.testing.assert_close(noise[:, 2:6], torch.zeros(20000, 4))
    amplitude = torch.tensor([0.2, 0.4, 0.3, 0.3, 0.3])
    noisy_features = torch.cat([noise[:, :2], noise[:, 6:]], dim=1)
    assert torch.all(noisy_features.abs() <= amplitude)
    normalized = noisy_features / amplitude
    assert normalized.mean().item() == pytest.approx(0.0, abs=0.01)
    assert normalized.square().mean().item() == pytest.approx(1 / 3, abs=0.01)


def test_zero_global_scale_disables_all_noise():
    runner = _runner()
    noise = runner.get_noise(
        ["joint_position", "angular_velocity"],
        {"scale": 0.0, "joint_position": 1.0, "angular_velocity": 1.0},
    )
    assert torch.count_nonzero(noise).item() == 0


def test_undeclared_features_are_clean_even_with_global_noise():
    runner = _runner()
    noise = runner.get_noise(["phase"], {"scale": 100.0})
    assert torch.count_nonzero(noise).item() == 0


def test_terminal_penalty_is_unscaled_and_regular_rewards_are_time_integrated():
    runner = _runner(num_envs=3)
    runner.env.dt = 0.01
    runner.env.terminated = torch.tensor([False, True, False])
    runner.critic_cfg = {
        "reward": {
            "termination_weight": {"termination": 0.15},
            "weights": {"tracking": 4.0},
        }
    }
    runner.reward_functions = {
        "termination": lambda: -torch.ones(3),
        "tracking": lambda: torch.ones(3),
    }
    rewards = runner.initialize_rewards_dict(1)

    runner.update_rewards_dict(rewards, 0)

    torch.testing.assert_close(
        rewards["termination"][0], torch.tensor([0.0, -0.15, 0.0])
    )
    torch.testing.assert_close(rewards["tracking"][0], torch.tensor([0.04, 0.0, 0.04]))
