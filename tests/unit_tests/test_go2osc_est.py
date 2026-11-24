"""go2osc_push_glide_est: accelerometer, friction switches and in-env estimator."""

import warnings

import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2 import go2osc_deploy_config as configs
from gym.envs.go2.go2osc_est import JACOBIAN_FEATURES
from gym.envs.go2.go2osc_deploy import Go2OscPushGlideDeploy
from gym.utils.task_registry import task_registry
from learning.modules.history_estimator import FRICTION_CLIP

NUM_ENVS = 4


def _make(edit=None):
    cfg = configs.Go2OscPushGlideDeployCfg()
    runner_cfg = configs.Go2OscPushGlideDeployRunnerCfg()
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
        env = Go2OscPushGlideDeploy(cfg, "cpu", True, backend)
    return env, runner_cfg, backend


@pytest.fixture
def env():
    task, _, backend = _make()
    try:
        yield task
    finally:
        backend.close()


def test_estimator_reads_only_deployable_signals():
    features = configs.Go2OscPushGlideDeployCfg.state_estimator.features
    assert "base_lin_acc" in features
    for privileged in (
        "base_lin_vel",
        "base_height",
        "foot_traction_obs",
        "contact_friction_obs",
    ):
        assert privileged not in features


def test_actor_and_critic_have_current_observation_layouts(env):
    runner_cfg = configs.Go2OscPushGlideDeployRunnerCfg
    assert env.get_states(runner_cfg.actor.obs).shape == (NUM_ENVS, 58)
    assert env.get_states(runner_cfg.critic.obs).shape == (NUM_ENVS, 75)


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


def test_untrained_estimator_takes_its_input_normalization_from_data(env):
    model = env.estimator.model
    assert env.estimator.unnormalized_inputs.all()
    assert torch.all(model.input_std == 1.0)
    env.set_learning(True)
    for _ in range(4):
        env.step()
    assert env.estimator.updates == 2
    assert env.estimator.unnormalized_inputs.all()
    assert env.estimator._normalization_batches == 2
    assert not torch.all(model.input_std == 1.0)
    assert torch.all(model.input_std >= 1e-3)


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


def test_jacobian_force_inverts_the_joint_torques_it_would_cause(env):
    for _ in range(3):
        env.step()
    jacobian, _ = env._leg_jacobians()
    n = env.num_envs
    weight = (env._backend.link_mass.sum(dim=1) * 9.81).view(n, 1, 1)
    known = torch.tensor([0.05, -0.03, 0.25]).expand(n, 4, 3)  # body weights
    # Static leg: tau_motor = -J^T F_ground.
    env.torques = -(jacobian.transpose(-1, -2) @ (known * weight).unsqueeze(-1)).view(
        n, 12
    )
    env._refresh_leg_kinematics()
    torch.testing.assert_close(
        env.foot_force_est.view(n, 4, 3), known, atol=2e-3, rtol=1e-2
    )
    torch.testing.assert_close(
        env.foot_traction_est.view(n, 4, 2),
        known[..., :2] / known[..., 2:3],
        atol=1e-2,
        rtol=1e-2,
    )


def test_unloaded_feet_report_zero_traction(env):
    env.torques = torch.zeros_like(env.torques)
    env._refresh_leg_kinematics()
    assert torch.all(env.foot_force_est.abs() < 1e-6)
    assert torch.all(env.foot_traction_est == 0.0)


def test_foot_velocity_relative_to_base_matches_the_simulator(env):
    from gym.utils.torch_quat import quat_rotate_inverse

    for step in range(6):
        env.dof_pos_target[:] = 0.3 * torch.sin(torch.tensor(float(step)))
        env.step()
    velocity, _ = env._foot_sliding_velocities()  # world, foot point
    relative = velocity - env.root_states[:, None, 7:10]
    n = env.num_envs
    expected = quat_rotate_inverse(
        env.base_quat.repeat_interleave(4, dim=0), relative.reshape(-1, 3)
    ).view(n, 12)
    torch.testing.assert_close(env.foot_vel_rel_base, expected, atol=1e-4, rtol=1e-3)


def test_estimator_inputs_are_clean_without_sensor_noise(env):
    env.sensor_noise_enabled = False
    torch.testing.assert_close(
        env._estimator_inputs(), env.get_states(env.estimator_features)
    )


def test_sensor_noise_hits_only_the_noisy_sensors_at_their_scale(env):
    torch.manual_seed(0)
    clean = env.get_states(env.estimator_features)
    noisy = torch.stack([env._estimator_inputs() for _ in range(400)])
    diff = noisy - clean
    dims = dict(zip(env.estimator_features, env._feature_dims(env.estimator_features)))
    offset = 0
    noise = configs.Go2OscPushGlideDeployCfg.state_estimator.sensor_noise.std
    for name in env.estimator_features:
        part = diff[..., offset : offset + dims[name]]
        if name in noise or name in JACOBIAN_FEATURES:  # via torques, dof_vel
            assert part.std().item() > 0
        else:
            assert torch.all(part == 0), name
        offset += dims[name]
    # Accelerometer: std 0.5 m/s^2 in units of g, plus a constant bias.
    acc = diff[..., env._acc_slice]
    expected = noise["base_lin_acc"] / 9.81
    assert acc.std(dim=0).mean().item() == pytest.approx(expected, rel=0.15)
    torch.testing.assert_close(
        acc.mean(dim=0), env.accelerometer_bias, atol=4 * expected / 20, rtol=0
    )


def test_jacobian_features_carry_the_same_noise_as_the_torques_and_velocities():
    jacobian = ["foot_force_est", "foot_traction_est", "foot_vel_rel_base"]

    def edit(cfg):
        cfg.state_estimator.features = [
            n for n in cfg.state_estimator.features if n not in jacobian
        ] + jacobian

    task, _, backend = _make(edit)
    try:
        task.step()
        torch.manual_seed(0)
        inputs = task._estimator_inputs()
        part = {n: inputs[:, task._feature_slices[n]] for n in task.estimator_features}
        scale = {
            n: torch.as_tensor(task.scales.get(n, 1.0), dtype=torch.float)
            for n in ("torques", "dof_vel")
        }
        # Recomputed from the noisy torques and velocities the estimator sees.
        force, ratio, velocity = task._leg_features(
            part["torques"] * scale["torques"], part["dof_vel"] * scale["dof_vel"]
        )
        for name, value in zip(jacobian, (force, ratio, velocity)):
            torch.testing.assert_close(part[name], value, atol=1e-5, rtol=1e-4)
        # ... and not the clean values.
        assert not torch.allclose(part["foot_force_est"], task.foot_force_est)
        assert not torch.allclose(part["foot_vel_rel_base"], task.foot_vel_rel_base)
        task.sensor_noise_enabled = False
        torch.testing.assert_close(
            task._estimator_inputs(), task.get_states(task.estimator_features)
        )
    finally:
        backend.close()


def test_accelerometer_bias_is_constant_per_episode_and_resampled_on_reset(env):
    limit = configs.Go2OscPushGlideDeployCfg.state_estimator.sensor_noise
    bound = limit.accelerometer_bias / 9.81
    before = env.accelerometer_bias.clone()
    assert torch.all(before.abs() <= bound)
    for _ in range(3):
        env.step()
    torch.testing.assert_close(env.accelerometer_bias, before)
    mask = torch.tensor([True, False, True, False])
    env._reset_idx(mask)
    assert not torch.allclose(env.accelerometer_bias[mask], before[mask])
    torch.testing.assert_close(env.accelerometer_bias[~mask], before[~mask])


def test_domain_randomization_off_builds_a_fixed_friction_task():
    from gym.envs.base.domain_randomization import apply_domain_randomization_override

    def fixed(cfg):
        cfg.friction_switch.probability = 1.0
        apply_domain_randomization_override(cfg, "off")

    task, _, backend = _make(fixed)
    try:
        assert task.cfg.friction_switch.probability > 0  # config untouched
        assert task._friction_switch_settings() is None  # ... but no switching
        for _ in range(3):
            task.step()
        assert torch.isinf(task.friction_switch_time).all()
    finally:
        backend.close()
