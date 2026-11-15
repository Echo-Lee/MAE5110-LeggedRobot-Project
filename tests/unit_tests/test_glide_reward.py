"""Glide reward: zero point, friction gate, direction, load, and command gating."""

from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2_lowfric import Go2LowFric
from gym.envs.go2.go2_lowfric_config import Go2LowFricCfg, Go2LowFricRunnerCfg

FEET = torch.tensor([1, 2, 3, 4])
RADIUS = 0.022


def _state(
    mu=0.1,
    body_v=(1.0, 0.0),
    foot_v=(1.0, 0.0),
    foot_w=(0.0, 0.0, 0.0),
    load=50.0,
    command=(1.0, 0.0, 0.0),
):
    """One environment, five bodies (base + four feet), every foot alike."""
    lin = torch.zeros(1, 5, 3)
    lin[0, 1:, :2] = torch.tensor(foot_v)
    ang = torch.zeros(1, 5, 3)
    ang[0, 1:] = torch.tensor(foot_w)
    root = torch.zeros(1, 13)
    root[0, 7:9] = torch.tensor(body_v)
    forces = torch.zeros(1, 5, 3)
    forces[0, 1:, 2] = load
    settings = Go2LowFricCfg.reward_settings
    return SimpleNamespace(
        cfg=SimpleNamespace(
            reward_settings=SimpleNamespace(
                glide_friction_threshold=settings.glide_friction_threshold,
                glide_friction_width=settings.glide_friction_width,
                glide_min_command=settings.glide_min_command,
                foot_radius=RADIUS,
            )
        ),
        _backend=SimpleNamespace(contact_friction=torch.tensor([mu])),
        feet_indices=FEET,
        _rigid_body_lin_vel=lin,
        _rigid_body_ang_vel=ang,
        root_states=root,
        contact_forces=forces,
        commands=torch.tensor([command]),
    )


def _glide(**kwargs):
    return Go2LowFric._reward_glide(_state(**kwargs)).item()


def _gate(mu):
    settings = Go2LowFricCfg.reward_settings
    return torch.sigmoid(
        torch.tensor(
            (settings.glide_friction_threshold - mu) / settings.glide_friction_width
        )
    ).item()


def test_loaded_feet_sliding_with_the_body_on_ice_earn_the_gate():
    assert _glide(mu=0.1) == pytest.approx(_gate(0.1))
    assert _gate(0.1) > 0.98


def test_planted_feet_earn_nothing():
    assert _glide(foot_v=(0.0, 0.0)) == 0.0


def test_ground_friction_switches_the_reward_off():
    assert _glide(mu=1.0) < 1e-5
    assert _glide(mu=0.6) < 0.003


def test_feet_sliding_faster_than_the_body_are_capped_at_one():
    assert _glide(foot_v=(3.0, 0.0)) == pytest.approx(_gate(0.1))


def test_half_speed_slide_earns_half():
    assert _glide(foot_v=(0.5, 0.0)) == pytest.approx(0.5 * _gate(0.1))


def test_backward_and_sideways_slip_earn_nothing():
    assert _glide(foot_v=(-1.0, 0.0)) == 0.0
    assert _glide(foot_v=(0.0, 1.0)) == 0.0


def test_unloaded_swing_feet_earn_nothing():
    assert _glide(load=0.0) == 0.0


def test_rolling_foot_is_not_gliding():
    # Center moves forward but the contact point is still: pure rolling.
    spin = 1.0 / RADIUS
    assert _glide(foot_v=(1.0, 0.0), foot_w=(0.0, spin, 0.0)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_stand_still_command_earns_nothing():
    assert _glide(command=(0.0, 0.0, 0.0)) == 0.0
    assert _glide(command=(0.0, 0.0, 1.0)) == 0.0  # turning in place


def test_shuffling_under_a_slow_body_is_scaled_down():
    # Body barely moving, feet sliding along with it at the same small speed.
    slow = _glide(body_v=(0.06, 0.0), foot_v=(0.06, 0.0))
    assert slow == pytest.approx(0.2 * _gate(0.1))


def test_misreported_friction_observation_does_not_change_the_gate():
    state = _state(mu=0.1)
    state.contact_friction = torch.tensor([[1.0]])  # what the policy is told
    assert Go2LowFric._reward_glide(state).item() == pytest.approx(_gate(0.1))


def test_glide_weight_is_only_in_go2_lowfric():
    from gym.envs.go2.go2_config import Go2RunnerCfg

    assert Go2LowFricRunnerCfg().critic.reward.weights.glide == pytest.approx(1.0)
    assert not hasattr(Go2RunnerCfg().critic.reward.weights, "glide")


def test_glide_runs_on_a_real_go2_lowfric_step():
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2LowFricCfg()
    cfg.env.num_envs = 4
    cfg.seed = 2
    task_registry.convert_frequencies_to_params(cfg, Go2LowFricRunnerCfg())
    env = Go2LowFric(cfg, "cpu", True, select_backend(cfg, "cpu", "mujoco"))
    try:
        env.commands[:] = torch.tensor([1.0, 0.0, 0.0])
        env.step()
        glide = env._reward_glide()
        assert glide.shape == (4,)
        assert torch.isfinite(glide).all()
        assert ((glide >= 0) & (glide <= 1)).all()
    finally:
        env._backend.close()
