"""go2osc_push_glide_robust_est: the robust sliding task with the estimator."""

import warnings

import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2 import go2osc_config, go2osc_robust_est_config as configs
from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from gym.envs.go2.go2osc_robust_est import Go2OscPushGlideRobustEst
from gym.utils.task_registry import task_registry
from learning.modules.history_estimator import targets_from_env

NUM_ENVS = 4


def _make(edit=None):
    cfg = configs.Go2OscPushGlideRobustEstCfg()
    runner_cfg = configs.Go2OscPushGlideRobustEstRunnerCfg()
    cfg.seed = runner_cfg.seed = 11
    cfg.env.num_envs = NUM_ENVS
    cfg.push_robots.toggle = False
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.state_estimator.hidden_dims = [16]
    cfg.state_estimator.samples_per_step = 4
    cfg.state_estimator.update_interval = 2
    cfg.state_estimator.gradient_steps = 1
    if edit:
        edit(cfg)
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = MuJocoCPUBackend()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # untrained-estimator warning
        env = Go2OscPushGlideRobustEst(cfg, "cpu", True, backend)
    return env, runner_cfg, backend


@pytest.fixture
def made():
    env, runner_cfg, backend = _make()
    try:
        yield env, runner_cfg
    finally:
        backend.close()


def _step(env, steps=1):
    for _ in range(steps):
        env.step()


def _made_with(edit):
    env, _, backend = _make(edit)
    return env, backend


def test_task_is_registered_on_the_robust_ground():
    import gym.envs  # noqa: F401  (registers tasks)

    name = "go2osc_push_glide_robust_est"
    assert task_registry.get_task_class(name) is Go2OscPushGlideRobustEst
    assert issubclass(Go2OscPushGlideRobustEst, Go2OscPushGlideEst)
    cfg = configs.Go2OscPushGlideRobustEstCfg
    robust = go2osc_config.Go2OscPushGlideRobustCfg
    # Same sliding stops, fast slides and ground changes as the robust task.
    assert cfg.commands.sliding_stop_probability == (
        robust.commands.sliding_stop_probability
    )
    assert cfg.fast_slides.probability == robust.fast_slides.probability
    assert cfg.friction_changes.rate == robust.friction_changes.rate
    # The whole-env switch would overwrite per-foot friction.
    assert getattr(cfg, "friction_switch", None) is None


def test_actor_swaps_privileged_inputs_in_place_for_deployable_ones():
    robust = go2osc_config.Go2OscPushGlideRobustRunnerCfg.actor.obs
    est = configs.Go2OscPushGlideRobustEstRunnerCfg.actor.obs
    assert len(est) == len(robust)
    for old, new in zip(robust, est):
        assert new == configs.DEPLOYABLE_REPLACEMENTS.get(old, old)
    for privileged in configs.DEPLOYABLE_REPLACEMENTS:
        assert privileged not in est
    noise = configs.Go2OscPushGlideRobustEstRunnerCfg.actor.noise
    assert noise.foot_traction_est == noise.foot_traction_obs


def test_robust_checkpoints_fit_the_actor(made):
    """Each replacement has the size and scale of what it replaces."""
    env, runner_cfg = made
    robust = go2osc_config.Go2OscPushGlideRobustRunnerCfg.actor.obs
    for old, new in configs.DEPLOYABLE_REPLACEMENTS.items():
        assert env.get_states([new]).shape == env.get_states([old]).shape
        assert env.scales.get(new, 1.0) == env.scales.get(old, 1.0)
    width = env.get_states(list(runner_cfg.actor.obs)).shape[1]
    assert width == env.get_states(list(robust)).shape[1]


def test_speed_clock_brakes_with_mu_hat(made):
    env, _ = made
    assert torch.equal(env.friction_estimate_obs, env.contact_friction_hat)
    _step(env, 3)
    assert torch.equal(env.friction_estimate_obs, env.contact_friction_hat)
    env.reset()
    assert torch.equal(env.friction_estimate_obs, env.contact_friction_hat)


def test_estimator_label_is_the_mean_friction_under_the_feet():
    def always_split(cfg):
        cfg.friction_changes.split_probability = 1.0

    env, backend = _made_with(always_split)
    try:
        env._draw_ground_friction(torch.ones(env.num_envs, dtype=torch.bool))
        feet = env.foot_friction_obs
        assert not torch.allclose(feet, feet[:, :1].expand_as(feet))  # split
        label = targets_from_env(env)[:, 0]
        torch.testing.assert_close(label, feet.mean(dim=1))
    finally:
        backend.close()


def test_ground_redraw_restarts_the_settling_clock(made):
    env, _ = made
    _step(env, 5)
    assert (env.time_since_friction_change > 0).all()
    redrawn = torch.tensor([True, False, True, False])
    env._draw_ground_friction(redrawn)
    assert (env.time_since_friction_change[redrawn] == 0).all()
    assert (env.time_since_friction_change[~redrawn] > 0).all()


def test_scheduled_switch_sets_all_four_feet():
    def no_redraws(cfg):
        cfg.friction_changes.rate = 0.0  # only the scheduled switch
        cfg.friction_changes.split_probability = 1.0

    env, backend = _made_with(no_redraws)
    try:
        env.schedule_friction_switch(torch.tensor([1]), 0.0, 0.15)
        _step(env)
        torch.testing.assert_close(
            env.foot_friction_obs[1], torch.full((4,), 0.15), atol=1e-6, rtol=0
        )
        assert env._backend.contact_friction[1].item() == pytest.approx(0.15)
        assert env.contact_friction_obs[1, 0].item() == pytest.approx(0.15)
        assert env.time_since_friction_change[1].item() == 0.0
        assert torch.isinf(env.friction_switch_time[1])
    finally:
        backend.close()


def test_fast_slides_and_turning_stops_run_with_the_estimator():
    def every_episode_slides(cfg):
        cfg.fast_slides.probability = 1.0

    env, _, backend = _make(every_episode_slides)
    try:
        speed = env.root_states[:, 7:9].norm(dim=1)
        low, high = env.cfg.fast_slides.speed_range
        assert ((speed > low - 0.05) & (speed < high + 0.05)).all()
        # A stop with a turn rate: zero planar command, yaw command kept.
        assert (env.commands[:, :2] == 0).all()
        assert (env._sliding_stop_steps > 0).all()
        friction = env.foot_friction_obs
        f_low, f_high = env.cfg.fast_slides.friction_range
        assert ((friction >= f_low) & (friction <= f_high)).all()
        _step(env, 10)
        for name in ("contact_friction_hat", "base_lin_vel_hat", "foot_traction_est"):
            assert torch.isfinite(getattr(env, name)).all()
    finally:
        backend.close()
