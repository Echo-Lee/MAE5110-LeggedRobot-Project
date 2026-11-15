"""Go2LowFric: privileged friction observation and episodic bimodal friction."""

import pytest
import torch

from gym.envs.go2.go2_lowfric_config import (
    GROUND_FRICTION,
    ICE_FRICTION,
    Go2LowFricCfg,
    Go2LowFricRunnerCfg,
)


@pytest.fixture
def env():
    from gym.envs.go2.go2_lowfric import Go2LowFric
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2LowFricCfg()
    runner_cfg = Go2LowFricRunnerCfg()
    cfg.env.num_envs = 8
    cfg.seed = 3
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, "cpu", "mujoco")
    task = Go2LowFric(cfg, "cpu", True, backend)
    yield task
    task._backend.close()


def _in_bins(values):
    ice = (ICE_FRICTION[0] <= values) & (values <= ICE_FRICTION[1])
    ground = (GROUND_FRICTION[0] <= values) & (values <= GROUND_FRICTION[1])
    return ice | ground


def test_friction_is_an_actor_and_critic_observation():
    runner_cfg = Go2LowFricRunnerCfg()
    assert "contact_friction" in runner_cfg.actor.obs
    assert "contact_friction" in runner_cfg.critic.obs


def test_friction_observation_is_live_view_of_backend(env):
    obs = env.get_state("contact_friction")
    assert obs.shape == (8, 1)
    assert _in_bins(obs[:, 0]).all()
    torch.testing.assert_close(obs[:, 0], env._backend.contact_friction)

    mask = torch.zeros(8, dtype=torch.bool)
    mask[[2, 5]] = True
    before = env.contact_friction.clone()
    env._reset_idx(mask)

    after = env.get_state("contact_friction")
    torch.testing.assert_close(after[:, 0], env._backend.contact_friction)
    assert torch.equal(after[~mask], before[~mask])
    assert not torch.equal(after[mask], before[mask])
    assert _in_bins(after[:, 0]).all()


def test_actor_observation_includes_friction_and_steps(env):
    runner_cfg = Go2LowFricRunnerCfg()
    actor_obs = env.get_states(runner_cfg.actor.obs)
    assert actor_obs.shape[0] == 8
    torch.testing.assert_close(actor_obs[:, -1], env._backend.contact_friction)
    env.step()
    assert torch.isfinite(env.get_states(runner_cfg.critic.obs)).all()


# --- run-up-and-slide episodes ---------------------------------------------------


def test_slide_profile_holds_then_decays_at_sliding_deceleration():
    from gym.envs.go2.go2_lowfric import slide_speed_profile
    from gym.utils.locomotion_metrics import GRAVITY_M_S2

    t = torch.tensor([0.0, 3.9, 4.0, 5.0, 100.0])
    speed = slide_speed_profile(t, 1.2, 0.1, runup_s=4.0)
    decel = 0.1 * GRAVITY_M_S2
    torch.testing.assert_close(speed, torch.tensor([1.2, 1.2, 1.2, 1.2 - decel, 0.0]))
    # Elementwise over envs, and the scale multiplies the deceleration.
    both = slide_speed_profile(
        torch.tensor([5.0, 5.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.05, 0.05]),
        4.0,
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        both, torch.tensor([1.0 - 0.05 * GRAVITY_M_S2, 1.0 - 0.1 * GRAVITY_M_S2])
    )


def _set_friction(env, values):
    env._backend.set_contact_friction(torch.arange(env.num_envs), torch.tensor(values))


def test_slide_episodes_are_only_sampled_on_ice_for_resetting_envs(env):
    env.cfg.commands.slide_probability = 1.0
    _set_friction(env, [0.05, 0.1, 0.2, 0.25, 0.6, 1.0, 0.1, 0.1])
    env.slide_active[:] = False
    mask = torch.tensor([True] * 6 + [False, False])
    env._sample_slide_episodes(mask)
    expected = torch.tensor([True, True, True, False, False, False, False, False])
    assert torch.equal(env.slide_active, expected)
    low, high = env.cfg.commands.slide_speed_range
    assert ((env.slide_speed[:3] >= low) & (env.slide_speed[:3] <= high)).all()


def test_zero_probability_never_slides(env):
    env.cfg.commands.slide_probability = 0.0
    _set_friction(env, [0.05] * 8)
    env._sample_slide_episodes(torch.ones(8, dtype=torch.bool))
    assert not env.slide_active.any()


def test_slide_envs_follow_the_profile_through_command_resamples(env):
    from gym.envs.go2.go2_lowfric import slide_speed_profile

    settings = env.cfg.commands
    settings.slide_probability = 1.0
    # Keep the friction set here: a reset would otherwise redraw it.
    env.domain_randomizer._resample_contact_friction = False
    _set_friction(env, [0.1] * 4 + [1.0] * 4)
    env._reset_idx(torch.ones(8, dtype=torch.bool))
    assert torch.equal(env.slide_active, torch.tensor([True] * 4 + [False] * 4))
    run_up = env.slide_speed.clone()
    # Past the 3 s command resample and into the slide phase.
    for _ in range(round(5.0 / env.dt)):
        env.step()
    expected = slide_speed_profile(
        env.episode_length_buf * env.dt, run_up, 0.1, settings.slide_runup_time
    )
    torch.testing.assert_close(env.commands[:4, 0], expected[:4])
    assert (env.commands[:4, 1:] == 0).all()
    assert (env.commands[:4, 0] < run_up[:4]).all()  # decaying, not held


def test_config_snapshots_without_slide_settings_never_slide(env):
    from types import SimpleNamespace

    # The base sampler reads ranges parsed at construction, so a commands
    # config with only resampling_time stands in for a run saved before
    # slide episodes.
    env.cfg.commands = SimpleNamespace(resampling_time=env.cfg.commands.resampling_time)
    env.domain_randomizer._resample_contact_friction = False
    _set_friction(env, [0.05] * 8)
    env.slide_active[:] = False
    env._reset_idx(torch.ones(8, dtype=torch.bool))
    env.step()
    assert not env.slide_active.any()


def test_default_slide_settings():
    commands = Go2LowFricCfg().commands
    assert commands.slide_probability == pytest.approx(0.3)
    assert commands.slide_max_friction == pytest.approx(ICE_FRICTION[1])
    assert commands.slide_runup_time == pytest.approx(4.0)
