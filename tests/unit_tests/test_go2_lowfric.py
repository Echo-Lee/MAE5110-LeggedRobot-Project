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
    assert runner_cfg.actor.obs[-2:] == ["contact_friction", "slide_phase"]
    torch.testing.assert_close(actor_obs[:, -2], env._backend.contact_friction)
    torch.testing.assert_close(actor_obs[:, -1], env.slide_phase[:, 0])
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


# --- run 8: slide ends after a short hold; slide-phase observation ---------------


def test_slide_phase_profile_covers_slide_and_hold_only():
    from gym.envs.go2.go2_lowfric import slide_phase_profile
    from gym.utils.locomotion_metrics import GRAVITY_M_S2

    stop = 4.0 + 1.0 / (0.1 * GRAVITY_M_S2)  # v0 = 1 m/s on mu = 0.1
    t = torch.tensor([0.0, 3.99, 4.0, stop, stop + 0.49, stop + 0.51, 9.0])
    phase = slide_phase_profile(t, 1.0, 0.1, runup_s=4.0, hold_s=0.5)
    assert phase.tolist() == [False, False, True, True, True, False, False]


def _start_slides(env, friction=0.1, speed=1.0):
    env.cfg.commands.slide_probability = 1.0
    env.domain_randomizer._resample_contact_friction = False
    _set_friction(env, [friction] * 4 + [1.0] * 4)
    env._reset_idx(torch.ones(8, dtype=torch.bool))
    env.slide_speed[:] = speed


def _step_to(env, seconds):
    while env.episode_length_buf[0] * env.dt < seconds - 1e-9:
        env.step()


def test_phase_is_off_during_run_up_and_on_during_the_slide(env):
    _start_slides(env)
    _step_to(env, 3.0)
    assert (env.slide_phase == 0).all()
    _step_to(env, 4.5)  # sliding: v0 = 1 decays at ~0.98 m/s^2 from 4 s
    assert (env.slide_phase[:4] == 1).all()
    assert (env.slide_phase[4:] == 0).all()  # ground envs never slide
    assert env.slide_active[:4].all()


def test_slide_ends_after_the_hold_and_random_commands_resume(env):
    _start_slides(env)
    _step_to(env, 6.0)  # stop ~5.02 s, hold until ~5.52 s
    assert not env.slide_active.any()
    assert (env.slide_phase == 0).all()
    # Normal sampling resumed: not every former slide env is commanded zero.
    assert not (env.commands[:4] == 0).all()


def test_run_7_snapshots_without_a_hold_keep_holding_zero(env):
    from types import SimpleNamespace

    settings = env.cfg.commands
    env.cfg.commands = SimpleNamespace(
        **{
            key: getattr(settings, key)
            for key in (
                "resampling_time",
                "slide_probability",
                "slide_max_friction",
                "slide_speed_range",
                "slide_runup_time",
                "slide_decel_scale",
            )
        }
    )
    env.cfg.commands.slide_probability = 1.0
    env.domain_randomizer._resample_contact_friction = False
    _set_friction(env, [0.1] * 8)
    env._reset_idx(torch.ones(8, dtype=torch.bool))
    env.slide_speed[:] = 1.0
    _step_to(env, 6.5)  # past the stop and any 0.5 s hold
    assert env.slide_active.all()
    assert (env.commands == 0).all()


def test_slide_still_penalizes_joint_motion_only_in_the_slide_phase():
    from types import SimpleNamespace

    from gym.envs.go2.go2_lowfric import Go2LowFric

    scale = torch.full((12,), 2.0)
    state = SimpleNamespace(
        dof_vel=torch.full((2, 12), 4.0),
        scales={"dof_vel": scale},
        slide_phase=torch.tensor([[1.0], [0.0]]),
    )
    torch.testing.assert_close(
        Go2LowFric._reward_slide_still(state), torch.tensor([-4.0, 0.0])
    )
    assert Go2LowFricRunnerCfg().critic.reward.weights.slide_still == 0.0


def test_play_slide_schedule_sets_the_phase_for_phase_aware_policies():
    from types import SimpleNamespace

    from scripts.play import command_schedule

    fake = SimpleNamespace(
        commands=torch.zeros(2, 3),
        num_envs=2,
        device="cpu",
        dt=0.01,
        _backend=SimpleNamespace(contact_friction=torch.tensor([0.1, 0.1])),
        slide_phase=torch.zeros(2, 1),
    )
    args = SimpleNamespace(
        task="go2_lowfric", command=None, slide=1.0, slide_runup=4.0, slide_hold=0.5
    )
    schedule = command_schedule(fake, args)
    command, phase = schedule(100)  # 1 s: run-up
    assert command[:, 0].tolist() == [1.0, 1.0] and phase.tolist() == [0.0, 0.0]
    command, phase = schedule(450)  # 4.5 s: sliding
    assert (command[:, 0] < 1.0).all() and phase.tolist() == [1.0, 1.0]
