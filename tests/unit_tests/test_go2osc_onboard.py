"""Current task observations, shared signals, estimator and curriculum."""

import warnings
from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2 import go2osc_deploy_config as configs
from gym.envs.go2.go2osc_deploy import Go2OscPushGlideDeploy
from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from gym.envs.go2.go2osc_onboard import govern_command, push_signal

from gym.utils.task_registry import task_registry

NUM_ENVS = 4
PRIVILEGED = (
    "base_lin_vel",
    "base_height",
    "foot_traction_obs",
    "friction_estimate_obs",
    "contact_friction_obs",
    "foot_friction_obs",
    "foot_grf_obs",
    "foot_grf_angle",
)


def _make(edit=None, clean=True):
    cfg = configs.Go2OscPushGlideDeployCfg()
    runner_cfg = configs.Go2OscPushGlideDeployRunnerCfg()
    cfg.seed = runner_cfg.seed = 5
    cfg.env.num_envs = NUM_ENVS
    cfg.push_robots.toggle = False
    cfg.env.episode_length_s = 100.0
    cfg.commands.resampling_time = 100.0
    cfg.commands.sliding_stop_probability = 0.0
    cfg.fast_slides.probability = 0.0
    cfg.state_estimator.hidden_dims = [16]
    cfg.state_estimator.samples_per_step = 4
    cfg.state_estimator.update_interval = 2
    cfg.state_estimator.gradient_steps = 1
    if clean:
        # No noise, delay or variation: both sides must then agree exactly.
        apply_domain_randomization_override(cfg, "off")
        cfg.state_estimator.sensor_noise.enabled = False
        cfg.osc.omega_noise_std = 0.0
        cfg.osc.coupling_noise_std = 0.0
        cfg.osc.process_noise_std = 0.0
        cfg.init_state.root_vel_range = [
            [0.0, 0.0] for _ in cfg.init_state.root_vel_range
        ]
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


# ── Configuration ──────────────────────────────────────────────────────────


def test_task_is_registered_on_the_estimator_environment():
    import gym.envs  # noqa: F401  (registers tasks)

    task = task_registry.get_task_class("go2osc_push_glide_deploy")
    assert task is Go2OscPushGlideDeploy
    assert issubclass(Go2OscPushGlideDeploy, Go2OscPushGlideEst)


def test_actor_and_estimator_read_nothing_the_robot_lacks():
    actor = configs.Go2OscPushGlideDeployRunnerCfg.actor.obs
    features = configs.Go2OscPushGlideDeployCfg.state_estimator.features
    for name in PRIVILEGED:
        assert name not in actor
        assert name not in features
    assert actor == [
        "base_ang_vel",
        "projected_gravity",
        "governed_commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "oscillator_obs",
        "contact_friction_hat",
        "base_lin_vel_hat",
        "clock_speed_obs",
    ]
    # The critic is training-only and keeps the simulator's true values.
    critic = configs.Go2OscPushGlideDeployRunnerCfg.critic.obs
    assert "base_lin_vel" in critic and "foot_friction_obs" in critic


def test_actor_has_58_inputs(env):
    actor = configs.Go2OscPushGlideDeployRunnerCfg.actor.obs
    assert env.get_states(actor).shape == (NUM_ENVS, 58)


def test_actor_order_and_scaling_use_the_shared_definition(monkeypatch):
    from gym.envs.go2.go2osc_onboard import ACTOR_OBSERVATIONS, ObservationScaling

    assert configs.Go2OscPushGlideDeployRunnerCfg.actor.obs is ACTOR_OBSERVATIONS
    scales = configs.Go2OscPushGlideDeployCfg.scaling
    assert issubclass(scales, ObservationScaling)
    for name in ACTOR_OBSERVATIONS:
        assert getattr(scales, name, 1.0) == getattr(ObservationScaling, name, 1.0)
    sentinel = [0.4] * 12
    monkeypatch.setattr(ObservationScaling, "dof_pos_target", sentinel)
    assert scales.dof_pos_target is sentinel


def test_rejects_an_unknown_foot_load_source():
    def edit(cfg):
        cfg.sensing.foot_load_source = "foot_sensor"

    with pytest.raises(ValueError, match="foot_load_source"):
        _make(edit)


# ── Joint limits ───────────────────────────────────────────────────────────


def test_joint_targets_stay_a_margin_inside_the_limits(env):
    actuated = env.actuated_dof_indices
    limits = env.dof_pos_limits.index_select(0, actuated)
    default = env.default_dof_pos.index_select(1, actuated)
    margin = env.cfg.control.joint_target_margin
    assert margin == pytest.approx(0.2)
    for sign in (1.0, -1.0):
        env.dof_pos_target[:] = sign * 100.0
        env._pre_decimation_step()
        target = default + env.dof_pos_target
        assert (target <= limits[:, 1] - margin + 1e-6).all()
        assert (target >= limits[:, 0] + margin - 1e-6).all()
        # The delayed copy the motors receive is the clamped one.
        applied = default + env._joint_position_target()
        assert (applied <= limits[:, 1] - margin + 1e-6).all()
        assert (applied >= limits[:, 0] + margin - 1e-6).all()
    # The calf can no longer be driven onto its upper stop.
    calf = default[0, 2] + env.dof_pos_target[0, 2]
    assert calf.item() == pytest.approx(limits[2, 0].item() + margin, abs=1e-5)


def test_without_a_margin_targets_are_not_limited_by_the_joints():
    def edit(cfg):
        cfg.control.joint_target_margin = None

    env, _, backend = _make(edit)
    try:
        assert env._joint_target_bounds() is None
        env.dof_pos_target[:] = 100.0
        env._pre_decimation_step()
        bound = env.cfg.control.action_limit * torch.as_tensor(
            env.scales["dof_pos_target"]
        )
        torch.testing.assert_close(env.dof_pos_target[0], bound.expand(12))
    finally:
        backend.close()


def test_joint_target_clamp_closes_over_the_training_ramp(env):
    ramp = env.cfg.control.joint_target_ramp
    margin = env.cfg.control.joint_target_margin
    assert 0 < ramp.start < ramp.end <= env.cfg.state_estimator.handover.start
    actuated = env.actuated_dof_indices
    limits = env.dof_pos_limits.index_select(0, actuated)
    default = env.default_dof_pos.index_select(1, actuated)
    middle = (ramp.start + ramp.end) // 2
    share = (middle - ramp.start) / (ramp.end - ramp.start)
    for iteration, expected in (
        (0, -ramp.slack),
        (ramp.start, -ramp.slack),
        (middle, margin - (1.0 - share) * (ramp.slack + margin)),
        (ramp.end, margin),
        (ramp.end + 1000, margin),
    ):
        env.set_iteration(iteration)
        assert env._joint_target_margin() == pytest.approx(expected)
        env.dof_pos_target[:] = 100.0
        env._pre_decimation_step()
        # The calf's upper stop binds before the action limit does.
        calf = default[0, 2] + env.dof_pos_target[0, 2]
        assert calf.item() == pytest.approx(limits[2, 1].item() - expected, abs=1e-5)


def test_a_fresh_environment_has_the_closed_clamp(env):
    # The robot's case: the exported policy carries this clamp.
    assert env.joint_target_progress == 1.0
    assert env._joint_target_margin() == pytest.approx(0.2)


def test_without_a_ramp_the_clamp_is_closed_from_the_start():
    def edit(cfg):
        cfg.control.joint_target_ramp = None

    env, _, backend = _make(edit)
    try:
        env.set_iteration(0)
        assert env._joint_target_margin() == pytest.approx(0.2)
    finally:
        backend.close()


def test_rejects_a_joint_target_ramp_that_ends_before_it_starts():
    def edit(cfg):
        cfg.control.joint_target_ramp.start = 10
        cfg.control.joint_target_ramp.end = 5

    with pytest.raises(ValueError, match="joint_target_ramp"):
        _make(edit)


def test_rejects_a_negative_joint_target_margin():
    def edit(cfg):
        cfg.control.joint_target_margin = -0.1

    with pytest.raises(ValueError, match="joint_target_margin"):
        _make(edit)


def test_joint_angles_near_a_limit_are_penalised(env):
    weights = configs.Go2OscPushGlideDeployRunnerCfg.critic.reward.weights
    assert weights.joint_limit_margin > 0.0
    margin = env.cfg.reward_settings.joint_limit_margin
    assert margin == pytest.approx(0.1)
    actuated = env.actuated_dof_indices
    limits = env.dof_pos_limits.index_select(0, actuated)
    calf = actuated[2]
    # The default posture is clear of every limit.
    env.dof_pos[:] = env.default_dof_pos
    assert env._reward_joint_limit_margin().abs().max() == 0.0
    # Free until the margin, 1/2 halfway in, 1 on the stop, 2 a margin beyond.
    for offset, cost in ((-1.5, 0.0), (-0.5, 0.5), (0.0, 1.0), (1.0, 2.0)):
        env.dof_pos[:] = env.default_dof_pos
        env.dof_pos[0, calf] = limits[2, 1] + offset * margin
        reward = env._reward_joint_limit_margin()
        assert reward[0].item() == pytest.approx(-cost, abs=1e-5)
        assert reward[1:].abs().max() == 0.0
    # The lower limit counts the same way.
    env.dof_pos[:] = env.default_dof_pos
    env.dof_pos[0, calf] = limits[2, 0]
    assert env._reward_joint_limit_margin()[0].item() == pytest.approx(-1.0, abs=1e-5)


# ── Speed target and estimator ─────────────────────────────────────────────


def test_speed_target_uses_the_estimated_friction(env):
    env.commands[:] = torch.tensor([2.0, 0.0, 0.0])
    env.clock_speed_obs.zero_()
    env.friction_estimate_obs.fill_(1.0)  # the robust task's input: unused
    env.contact_friction_hat = torch.tensor([[0.05], [0.1], [0.5], [1.5]])
    env._advance_clock_speed()
    gain = env.cfg.reward_settings.speed_reference_gain
    expected = (gain * env.contact_friction_hat[:, 0] * 9.81).clamp(
        max=env.cfg.reward_settings.speed_reference_max_gain
    ) * env.dt
    torch.testing.assert_close(env.clock_speed_obs[:, 0], expected)


def test_an_episode_starts_the_speed_target_at_the_body_speed(env):
    # Not at v-hat: at a reset the estimator has seen one frame.
    everyone = torch.ones(NUM_ENVS, dtype=torch.bool)
    env._reset_idx(everyone)
    torch.testing.assert_close(env.clock_speed_obs, env.base_lin_vel[:, :1])


def test_govern_command_limits_the_lead_over_the_forward_speed():
    command = torch.tensor(
        [[2.0, 0.3, 0.5], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.5], [1.0, 0.0, 0.0]]
    )
    speed = torch.tensor([0.0, 0.0, 1.8, 1.8])
    governed = govern_command(command, speed, 0.5)
    # From rest a fast command is handed over a margin at a time, either way.
    # Stopping and slowing down are never limited.
    assert governed[:, 0].tolist() == pytest.approx([0.5, -0.5, 0.0, 1.0])
    torch.testing.assert_close(governed[:, 1:], command[:, 1:])
    # Moving against the command, the lead counts from zero.
    reverse = govern_command(command[:2], torch.tensor([-1.0, 1.0]), 0.5)
    assert reverse[:, 0].tolist() == pytest.approx([0.5, -0.5])
    torch.testing.assert_close(govern_command(command, speed, None), command)


def test_actor_and_speed_target_get_the_governed_command(env):
    margin = env.cfg.command_governor.margin
    assert margin > env.cfg.osc.stop_threshold
    env.commands[:] = torch.tensor([2.0, 0.0, 0.0])
    env.base_lin_vel_hat = torch.zeros(NUM_ENVS, 3)
    env.contact_friction_hat = torch.ones(NUM_ENVS, 1)
    env.clock_speed_obs.zero_()
    for _ in range(100):
        env._advance_clock_speed()
    assert env.governed_commands[:, 0].tolist() == pytest.approx([margin] * NUM_ENVS)
    # v_t stops at the governed command, not at the command.
    assert env.clock_speed_obs[:, 0].tolist() == pytest.approx([margin] * NUM_ENVS)
    column = env.get_states(["governed_commands"])
    assert column[:, 0].tolist() == pytest.approx(
        [margin / env.scales["governed_commands"][0].item()] * NUM_ENVS
    )
    # The estimator keeps the command as given: a governed input would feed
    # v-hat back into itself.
    assert "commands" in env.estimator_features
    assert "governed_commands" not in env.estimator_features


def test_estimator_learns_the_mean_friction_under_the_feet(env):
    env.foot_friction_obs.copy_(
        torch.tensor([[0.1, 0.3, 0.1, 0.3]]).repeat(NUM_ENVS, 1)
    )
    targets = env._estimator_targets()
    torch.testing.assert_close(targets[:, 0], torch.full((NUM_ENVS,), 0.2))
    torch.testing.assert_close(targets[:, 1:], env.base_lin_vel)


def test_estimator_trains_beside_the_policy_and_is_checkpointed(env):
    env.set_learning(True)
    for _ in range(6):
        env.step()
    assert env.estimator.updates >= 2
    state = env.checkpoint_state()
    assert "state_estimator" in state and "estimator_config" in state


def test_estimator_input_statistics_average_before_they_freeze():
    def edit(cfg):
        cfg.state_estimator.normalization_updates = 3

    env, _, backend = _make(edit)
    try:
        env.set_learning(True)
        assert env.estimator.unnormalized_inputs is not None
        means = []
        while env.estimator.updates < 5:
            before = env.estimator.updates
            env.step()
            if env.estimator.updates > before:
                means.append(env.estimator.model.input_mean.clone())
        assert not torch.equal(means[0], means[1])
        assert not torch.equal(means[1], means[2])
        torch.testing.assert_close(means[2], means[3])  # frozen after 3
        torch.testing.assert_close(means[3], means[4])
        assert env.estimator.unnormalized_inputs is None
    finally:
        backend.close()


# ── Clocks ─────────────────────────────────────────────────────────────────


def test_clocks_use_a_push_signal_sampled_once_per_control_step(env):
    env.commands[:] = torch.tensor([1.0, 0.0, 0.0])
    for _ in range(20):
        env.step()
    held = env._push_signal.clone()
    torch.testing.assert_close(env._compute_push_signal(), held)
    env._post_physics_step()  # a physics step does not resample it
    torch.testing.assert_close(env._compute_push_signal(), held)
    assert (held >= 0.0).all() and (held <= 1.0).all()
    assert held.sum() > 0.0


def test_push_signal_gates_load_by_the_stroke_direction():
    load = torch.full((1, 4), 0.25)
    stroke = torch.zeros(1, 4, 3)
    stroke[0, 0, 0] = -1.0  # pushing back against forward travel
    stroke[0, 1, 0] = 1.0  # swinging forward
    command = torch.tensor([[1.0, 0.0, 0.0]])
    speed = torch.tensor([[1.0]])
    moving = torch.tensor([[True]])
    signal = push_signal(load, stroke, speed, command, moving, 0.1)
    assert signal[0, 0] > 0.249 and signal[0, 1] < 0.001
    torch.testing.assert_close(signal[0, 2:], torch.full((2,), 0.125))
    # Standing or turning in place: no travel direction, no gate.
    still = push_signal(load, stroke, speed * 0, command * 0, ~moving, 0.1)
    torch.testing.assert_close(still, load)


# ── Robot-side arithmetic against the simulator ────────────────────────────


# ── The deploy file, end to end ────────────────────────────────────────────


QGYM_TO_UNITREE = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


# ── A clamped target must stay learnable ───────────────────────────────────


def test_target_excess_records_what_the_clamp_removed(env):
    scale = torch.as_tensor(env.scales["dof_pos_target"]).expand(12)
    low, high = env._joint_target_bounds()
    inside = 0.5 * (low[0] + high[0])
    env.dof_pos_target[:] = inside
    env._pre_decimation_step()
    torch.testing.assert_close(env.dof_pos_target_excess, torch.zeros(NUM_ENVS, 12))
    assert env._reward_target_saturation().abs().max() == 0.0
    # Three action units beyond the calf's upper bound, on one robot.
    env.dof_pos_target[:] = inside
    env.dof_pos_target[0, 2] = high[0, 2] + 3.0 * scale[2]
    env._pre_decimation_step()
    assert env.dof_pos_target[0, 2].item() == pytest.approx(high[0, 2].item())
    assert env.dof_pos_target_excess[0, 2].item() == pytest.approx(3.0, abs=1e-4)
    reward = env._reward_target_saturation()
    assert reward[0].item() == pytest.approx(-2.5, abs=1e-4)  # |e| - 1/2
    assert reward[1:].abs().max() == 0.0


def test_saturation_penalty_grows_with_the_distance_beyond_the_clamp(env):
    scale = torch.as_tensor(env.scales["dof_pos_target"]).expand(12)
    _, high = env._joint_target_bounds()
    costs = []
    for beyond in (0.5, 2.0, 80.0):
        env.dof_pos_target[:] = 0.0
        env.dof_pos_target[:, 2] = high[0, 2] + beyond * scale[2]
        env._pre_decimation_step()
        costs.append(-env._reward_target_saturation()[0].item())
    assert costs[0] == pytest.approx(0.125, abs=1e-4)  # e^2 / 2
    assert costs[0] < costs[1] < costs[2]
    # Without it every one of these outputs would cost the same: nothing.
    assert costs[2] == pytest.approx(79.5, abs=1e-2)


def test_deploy_task_penalises_saturated_targets():
    weights = configs.Go2OscPushGlideDeployRunnerCfg.critic.reward.weights
    assert weights.target_saturation == pytest.approx(1.0)
    assert weights.standing_saturation == pytest.approx(4.0)


def test_standing_saturation_counts_only_while_the_clocks_stand():
    # No environment: only the excess and the standing mask enter.
    standing = torch.tensor([[True], [True], [False], [False]])
    robot = SimpleNamespace(
        dof_pos_target_excess=torch.ones(4, 12),
        _slow_command_mask=lambda: standing,
    )
    everywhere = Go2OscPushGlideDeploy._reward_target_saturation(robot)
    stood = Go2OscPushGlideDeploy._reward_standing_saturation(robot)
    assert (everywhere < 0).all()
    torch.testing.assert_close(stood[:2], everywhere[:2])
    assert stood[2:].abs().max() == 0.0


# ── Hand-over from the simulator's stand-ins to the estimates ──────────────


def test_a_fresh_environment_reads_only_the_estimates(env):
    # The robot's case, and what every parity test above relies on.
    assert env.estimate_share == 1.0


def test_handover_schedule_follows_the_training_iteration(env):
    handover = env.cfg.state_estimator.handover
    assert 0 < handover.start < handover.end
    middle = (handover.start + handover.end) // 2
    for iteration, share in (
        (0, 0.0),
        (handover.start, 0.0),
        (middle, (middle - handover.start) / (handover.end - handover.start)),
        (handover.end, 1.0),
        (handover.end + 1000, 1.0),
    ):
        env.set_iteration(iteration)
        assert env.estimate_share == pytest.approx(share)


def test_before_the_handover_the_actor_reads_the_stand_ins(env):
    env.set_iteration(0)
    noise = env.cfg.state_estimator.handover.velocity_noise
    for _ in range(3):
        env.step()
    torch.testing.assert_close(
        env.contact_friction_hat, env.friction_estimate_obs.clamp(0.02, 1.5)
    )
    error = (env.base_lin_vel_hat - env.base_lin_vel).abs()
    assert error.max() <= noise + 1e-6
    assert error.max() > 0.0
    # The speed target follows the same friction.
    torch.testing.assert_close(env._clock_friction(), env.contact_friction_hat[:, 0])
    # The estimator itself is untouched: it still publishes its own output.
    assert not torch.equal(env.estimator.v_hat, env.base_lin_vel_hat)


def test_halfway_through_the_handover_the_inputs_are_blended(env):
    handover = env.cfg.state_estimator.handover
    env.set_iteration(0)
    env.step()
    env.friction_estimate_obs.fill_(0.2)
    env.estimator.mu_hat = torch.full((NUM_ENVS,), 0.6)
    env.set_iteration((handover.start + handover.end) // 2)
    torch.testing.assert_close(env.contact_friction_hat, torch.full((NUM_ENVS, 1), 0.4))
    env.set_iteration(handover.end)
    torch.testing.assert_close(env.contact_friction_hat, torch.full((NUM_ENVS, 1), 0.6))
    torch.testing.assert_close(env.base_lin_vel_hat, env.estimator.v_hat)


def test_without_a_handover_the_estimates_are_read_from_the_start():
    def edit(cfg):
        cfg.state_estimator.handover = None

    env, _, backend = _make(edit)
    try:
        env.set_iteration(0)
        assert env.estimate_share == 1.0
    finally:
        backend.close()


def test_rejects_a_handover_that_ends_before_it_starts():
    def edit(cfg):
        cfg.state_estimator.handover.start = 10
        cfg.state_estimator.handover.end = 5

    with pytest.raises(ValueError, match="handover"):
        _make(edit)


def test_runner_tells_the_environment_the_checkpoint_iteration(tmp_path):
    env, runner_cfg, backend = _make()
    try:
        runner_cfg.runner.device = "cpu"
        runner_cfg.logging.enable_local_saving = False
        runner_cfg.log_dir = str(tmp_path)
        runner = task_registry.make_alg_runner(env, runner_cfg)
        handover = env.cfg.state_estimator.handover
        for iteration, share in ((0, 0.0), (handover.end, 1.0)):
            runner.it = iteration
            runner.save()
            env.estimate_share = 0.5
            runner.load(str(tmp_path / f"model_{iteration}.pt"))
            assert env.estimate_share == share
    finally:
        backend.close()
