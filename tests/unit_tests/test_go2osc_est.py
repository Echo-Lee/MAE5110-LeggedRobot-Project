"""go2osc_push_glide_est: accelerometer, friction switches and in-env estimator."""

import warnings

import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2 import go2osc_config, go2osc_est_config as configs
from gym.envs.go2.go2osc import Go2OscPushGlide
from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from gym.utils.task_registry import task_registry
from learning.modules.history_estimator import FRICTION_CLIP, HistoryStateEstimator

NUM_ENVS = 4


def _make(edit=None):
    cfg = configs.Go2OscPushGlideEstCfg()
    runner_cfg = configs.Go2OscPushGlideEstRunnerCfg()
    cfg.seed = runner_cfg.seed = 11
    cfg.env.num_envs = NUM_ENVS
    cfg.push_robots.toggle = False
    cfg.init_state.reset_mode = "reset_to_basic"
    # Small, fast estimator updates.
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
        env = Go2OscPushGlideEst(cfg, "cpu", True, backend)
    return env, runner_cfg, backend


@pytest.fixture
def env():
    task, _, backend = _make()
    try:
        yield task
    finally:
        backend.close()


def test_task_is_registered_and_keeps_the_team_policy_inputs():
    import gym.envs  # noqa: F401  (registers tasks)

    assert task_registry.get_task_class("go2osc_push_glide_est") is Go2OscPushGlideEst
    assert issubclass(Go2OscPushGlideEst, Go2OscPushGlide)
    base = go2osc_config.Go2OscPushGlideRunnerCfg
    est = configs.Go2OscPushGlideEstRunnerCfg
    # Step 1: a go2osc_push_glide checkpoint loads unchanged.
    assert est.actor.obs == base.actor.obs
    assert est.critic.obs == base.critic.obs


def test_estimator_reads_only_deployable_signals():
    features = configs.Go2OscPushGlideEstCfg.state_estimator.features
    assert "base_lin_acc" in features
    for privileged in (
        "base_lin_vel",
        "base_height",
        "foot_traction_obs",
        "contact_friction_obs",
    ):
        assert privileged not in features


def test_actor_and_critic_inputs_have_the_team_sizes(env):
    runner_cfg = configs.Go2OscPushGlideEstRunnerCfg
    assert env.get_states(runner_cfg.actor.obs).shape == (NUM_ENVS, 61)
    assert env.get_states(runner_cfg.critic.obs).shape == (NUM_ENVS, 70)


def test_accelerometer_reads_free_fall_then_one_g_standing(env):
    # A reset sets the at-rest reading (+1 g along body z).
    env._reset_idx(torch.ones(NUM_ENVS, dtype=torch.bool))
    at_rest = env.get_state("base_lin_acc")
    expected = torch.tensor([0.0, 0.0, 1.0]).expand_as(at_rest)
    torch.testing.assert_close(at_rest, expected, atol=1e-5, rtol=0)
    # The robot spawns above the ground: in free fall an IMU reads ~0 g.
    env.step()
    assert env.get_state("base_lin_acc").norm(dim=1).max().item() < 0.05
    # Once it stands, the reading averages ~1 g up.
    readings = []
    for _ in range(150):
        env.step()
        assert torch.isfinite(env.base_lin_acc).all()
        readings.append(env.get_state("base_lin_acc")[:, 2].mean().item())
    assert sum(readings[50:]) / 100 == pytest.approx(1.0, abs=0.1)


def test_accelerometer_is_the_velocity_difference_over_one_physics_step(env):
    env.step()
    before = env.root_states[:, 7:10].clone()
    env._step_backend()
    env._post_physics_step()
    after = env.root_states[:, 7:10]
    accel = (after - before) / env._physics_dt
    accel[:, 2] += 9.81
    from gym.utils.torch_quat import quat_rotate_inverse

    torch.testing.assert_close(
        env.base_lin_acc, quat_rotate_inverse(env.base_quat, accel)
    )


def test_scheduled_switch_changes_physics_and_critic_friction(env):
    ids = torch.tensor([0, 2])
    env.schedule_friction_switch(ids, 0.0, 0.9)
    env.step()
    friction = env._backend.contact_friction
    torch.testing.assert_close(friction[ids], torch.full((2,), 0.9))
    torch.testing.assert_close(env.contact_friction_obs[ids, 0], torch.full((2,), 0.9))
    assert torch.all(env.time_since_friction_change[ids] == 0.0)
    assert torch.isinf(env.friction_switch_time[ids]).all()


def test_sampled_switches_jump_to_another_bin_inside_the_time_range():
    def always_switch(cfg):
        cfg.friction_switch.probability = 1.0

    env, _, backend = _make(always_switch)
    try:
        everyone = torch.ones(NUM_ENVS, dtype=torch.bool)
        bounds = torch.tensor(configs.FRICTION_BINS)
        for _ in range(5):
            env._reset_idx(everyone)
            low, high = configs.Go2OscPushGlideEstCfg.friction_switch.time_range
            assert torch.all(env.friction_switch_time >= low)
            assert torch.all(env.friction_switch_time <= high)
            current = env._backend.contact_friction
            now = (current.unsqueeze(1) > bounds[:, 1]).sum(dim=1).clamp(max=2)
            then = (
                (env.friction_switch_target.unsqueeze(1) > bounds[:, 1])
                .sum(dim=1)
                .clamp(max=2)
            )
            assert torch.all(now != then)
    finally:
        backend.close()


def test_switches_are_off_without_probability():
    def no_switch(cfg):
        cfg.friction_switch.probability = 0.0

    env, _, backend = _make(no_switch)
    try:
        assert torch.isinf(env.friction_switch_time).all()
        before = env._backend.contact_friction.clone()
        for _ in range(3):
            env.step()
        torch.testing.assert_close(env._backend.contact_friction, before)
    finally:
        backend.close()


def test_estimates_are_published_with_their_shapes_and_clip(env):
    env.step()
    mu_hat = env.get_state("contact_friction_hat")
    v_hat = env.get_state("base_lin_vel_hat")
    assert mu_hat.shape == (NUM_ENVS, 1) and v_hat.shape == (NUM_ENVS, 3)
    torch.testing.assert_close(
        env.contact_friction_hat[:, 0], env.estimator.mu_hat.clamp(*FRICTION_CLIP)
    )
    torch.testing.assert_close(env.base_lin_vel_hat, env.estimator.v_hat)


def _weights(env):
    return [p.detach().clone() for p in env.estimator.model.parameters()]


def test_estimator_trains_only_while_learning(env):
    before = _weights(env)
    for _ in range(4):
        env.step()
    assert env.estimator.updates == 0
    env.set_learning(True)
    with torch.inference_mode():  # as in the PPO rollout
        for _ in range(4):
            env.step()
    assert env.estimator.updates > 0
    after = _weights(env)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    env.set_learning(False)
    frozen = env.estimator.updates
    with torch.inference_mode():
        for _ in range(4):
            env.step()
    assert env.estimator.updates == frozen


def test_checkpoint_state_restores_the_same_estimator(env):
    env.set_learning(True)
    for _ in range(4):
        env.step()
    state = env.checkpoint_state()
    other, _, backend = _make()
    try:
        other.load_checkpoint_state(state)
        for a, b in zip(_weights(env), _weights(other)):
            torch.testing.assert_close(a, b)
        assert other.estimator_features == env.estimator_features
    finally:
        backend.close()


def test_estimator_file_without_the_accelerometer_is_widened(env, tmp_path):
    legacy = [f for f in env.estimator_features if f != "base_lin_acc"]
    model = HistoryStateEstimator(
        features=legacy,
        feature_dims=env._feature_dims(legacy),
        frames=configs.Go2OscPushGlideEstCfg.state_estimator.frames,
        stride=configs.Go2OscPushGlideEstCfg.state_estimator.stride,
        hidden_dims=[16],
    )
    path = tmp_path / "estimator.pt"
    model.save(path)

    def from_file(cfg):
        cfg.state_estimator.init_path = str(path)

    widened, _, backend = _make(from_file)
    try:
        assert widened.estimator_features == list(
            configs.Go2OscPushGlideEstCfg.state_estimator.features
        )
        assert widened.estimator.unnormalized_inputs is not None
    finally:
        backend.close()


def test_missing_estimator_file_fails_loudly(tmp_path):
    def missing(cfg):
        cfg.state_estimator.init_path = str(tmp_path / "nope.pt")

    with pytest.raises(FileNotFoundError):
        _make(missing)
