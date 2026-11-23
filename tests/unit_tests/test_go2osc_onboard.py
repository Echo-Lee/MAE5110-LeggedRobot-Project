"""go2osc_push_glide_deploy: the robot-side code reproduces the simulator.

The policy's inputs come from oscillators, a speed target and a state
estimator. Training computes them in the environment; the robot computes
them in go2osc_onboard.py through go2_deploy/custom_implementation/
go2osc_push_glide.py. These tests step both side by side from the same
sensor readings and require the same numbers.
"""

import importlib.util
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2 import go2osc_deploy_config as configs
from gym.envs.go2.go2osc_deploy import Go2OscPushGlideDeploy, onboard_settings
from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from gym.envs.go2.go2osc_onboard import (
    ONBOARD_OBSERVATIONS,
    STANDING_PHASE,
    Go2OscOnboard,
    advance_clock_speed,
    clock_schedule,
    govern_command,
    leg_geometry,
    leg_kinematics,
    push_signal,
)
from gym.utils.task_registry import task_registry

NUM_ENVS = 4
REPOSITORY = Path(__file__).resolve().parents[2]
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


def _sensors(env):
    actuated = env.actuated_dof_indices
    return {
        "dof_pos": env.dof_pos.index_select(1, actuated).clone(),
        "dof_vel": env.dof_vel.index_select(1, actuated).clone(),
        "torques": env.torques.clone(),
        "base_ang_vel": env.base_ang_vel.clone(),
        "projected_gravity": env.projected_gravity.clone(),
        "base_lin_acc": env.base_lin_acc.clone(),
        "dof_pos_target": env.dof_pos_target.clone(),
    }


def _onboard(env):
    model = env.estimator.model

    def estimator(window):
        return model(window) * model.target_scale

    return Go2OscOnboard(onboard_settings(env), estimator, num_robots=env.num_envs)


def _phase_error(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs().max().item()


def _command_schedule(step):
    """Commands [N, 3] over a run: walk, turn, sidestep, then a carved stop."""
    table = torch.tensor(
        [
            [1.5, 0.0, 0.0],
            [0.8, 0.0, 0.4],
            [0.0, 0.3, 0.0],
            [-0.8, 0.0, 0.0],
        ]
    )
    if step >= 80:
        table = torch.tensor(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -0.5],
            ]
        )
    return table


def _drive(env, steps, after_step):
    """Step the environment with moving joint targets and scheduled commands.

    Commands change where the simulator changes them: inside the step,
    before the speed target advances. ``after_step(step, command)`` runs
    once the step is complete. Stops at the first fall.
    """
    schedule = {"command": _command_schedule(0)}
    count_down = env._count_down_sliding_stops

    def change_command():
        count_down()
        env.commands[:] = schedule["command"]

    env._count_down_sliding_stops = change_command
    compared = 0
    for step in range(steps):
        schedule["command"] = _command_schedule(step + 1)
        wave = 0.25 * math.sin(0.3 * step)
        pattern = torch.tensor([0.0, 1.0, -1.5] * 4) * wave
        pattern = pattern * torch.tensor([1.0, -1.0, -1.0, 1.0]).repeat_interleave(3)
        env.dof_pos_target[:] = pattern
        env.step()
        if (env.terminated | env.timed_out).any():
            break
        after_step(step, schedule["command"])
        compared += 1
    return compared


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


def test_clock_schedule_stands_only_without_any_command():
    settings = onboard_settings_for_tests()
    speed = torch.tensor([[0.0], [0.0], [0.0], [1.0], [3.5]])
    command = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.0, 0.2, 0.0],
            [1.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
        ]
    )
    omega, coupling, offset, moving, translating = clock_schedule(
        speed, command, settings["osc"]
    )
    assert moving[:, 0].tolist() == [False, True, True, True, True]
    assert translating[:, 0].tolist() == [False, False, True, True, True]
    torch.testing.assert_close(omega[:, 0], torch.tensor([1.0, 1.5, 1.5, 2.5, 4.0]))
    torch.testing.assert_close(coupling[:, 0], torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(offset[:, 0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


def onboard_settings_for_tests():
    osc = configs.Go2OscPushGlideDeployCfg.osc
    rewards = configs.Go2OscPushGlideDeployCfg.reward_settings
    return {
        "osc": {
            "stop_threshold": osc.stop_threshold,
            "sideways_stop_threshold": osc.sideways_stop_threshold,
            "turning_stop_threshold": osc.turning_stop_threshold,
            "omega_stop": osc.omega_stop,
            "omega_offset": osc.omega_offset,
            "omega_max": osc.omega_max,
            "coupling_stop": osc.coupling_stop,
            "coupling_moving": osc.coupling_moving,
            "stance_offset": osc.stance_offset,
        },
        "speed_target": {
            "min_friction": osc.clock_speed_min_friction,
            "gain": rewards.speed_reference_gain,
            "max_gain": rewards.speed_reference_max_gain,
            "brake": rewards.speed_reference_brake,
            "max_brake": rewards.speed_reference_max_brake,
            "carve": rewards.speed_reference_carve,
            "carve_min_brake": rewards.speed_reference_carve_min_brake,
        },
    }


def test_speed_target_rates_follow_friction_and_carve():
    settings = onboard_settings_for_tests()["speed_target"]
    dt = 0.01
    friction = torch.tensor([0.1])
    # Speeding up: 0.5 mu g.
    up = advance_clock_speed(
        torch.tensor([[1.0]]), torch.tensor([[2.0, 0.0, 0.0]]), friction, dt, settings
    )
    assert up.item() == pytest.approx(1.0 + 0.5 * 0.1 * 9.81 * dt)
    # A straight stop: 0.9 mu g.
    straight = advance_clock_speed(
        torch.tensor([[1.0]]), torch.tensor([[0.0, 0.0, 0.0]]), friction, dt, settings
    )
    assert straight.item() == pytest.approx(1.0 - 0.9 * 0.1 * 9.81 * dt)
    # A carved stop: the turn takes its share, at least half still brakes.
    carved = advance_clock_speed(
        torch.tensor([[1.0]]), torch.tensor([[0.0, 0.0, 0.5]]), friction, dt, settings
    )
    budget = 0.9 * 0.1 * 9.81
    brake = (budget**2 - (1.0 * 0.5) ** 2) ** 0.5
    assert carved.item() == pytest.approx(1.0 - brake * dt)
    fast = advance_clock_speed(
        torch.tensor([[2.5]]), torch.tensor([[0.0, 0.0, 0.5]]), friction, dt, settings
    )
    assert fast.item() == pytest.approx(2.5 - 0.5 * budget * dt)


# ── Robot-side arithmetic against the simulator ────────────────────────────


def test_leg_kinematics_match_the_simulated_bodies(env):
    geometry = leg_geometry()
    for _ in range(3):
        env.dof_pos_target[:] = 0.6 * (torch.rand(NUM_ENVS, 12) - 0.5)
        for _ in range(15):
            env.step()
        env._refresh_leg_kinematics()
        jacobian, lever = leg_kinematics(
            env.dof_pos.index_select(1, env.actuated_dof_indices), geometry
        )
        torch.testing.assert_close(lever, env._leg_lever, atol=2e-5, rtol=0)
        torch.testing.assert_close(jacobian, env._leg_jacobian, atol=2e-5, rtol=0)


def test_onboard_reset_matches_an_environment_reset(env):
    everyone = torch.ones(NUM_ENVS, dtype=torch.bool)
    env.commands[:] = _command_schedule(0)
    env._reset_idx(everyone)
    onboard = _onboard(env)
    observations = onboard.reset(
        _sensors(env),
        env.commands,
        phase=env.oscillators,
        clock_speed=env.clock_speed_obs,
    )
    for name in ONBOARD_OBSERVATIONS:
        torch.testing.assert_close(
            observations[name], getattr(env, name), atol=1e-5, rtol=0
        )
    torch.testing.assert_close(onboard.push, env._push_signal, atol=1e-5, rtol=0)


def test_onboard_follows_the_environment_step_by_step(env):
    everyone = torch.ones(NUM_ENVS, dtype=torch.bool)
    env.commands[:] = _command_schedule(0)
    env._reset_idx(everyone)
    # Begin on the moving clocks, whatever friction the untrained estimator
    # reports: an episode that starts already at speed.
    env.clock_speed_obs.fill_(0.6)
    env._refresh_onboard_push_signal(everyone)
    onboard = _onboard(env)
    onboard.reset(
        _sensors(env),
        env.commands,
        phase=env.oscillators,
        clock_speed=env.clock_speed_obs,
    )
    worst = {"phase": 0.0, "speed": 0.0, "friction": 0.0, "velocity": 0.0, "push": 0.0}
    worst["command"] = 0.0
    fastest = {"speed": 0.0}

    def compare(step, command):
        observations = onboard.step(_sensors(env), command)

        def gap(a, b):
            return (a - b).abs().max().item()

        worst["phase"] = max(
            worst["phase"], _phase_error(onboard.phase, env.oscillators)
        )
        worst["speed"] = max(
            worst["speed"], gap(observations["clock_speed_obs"], env.clock_speed_obs)
        )
        worst["friction"] = max(
            worst["friction"],
            gap(observations["contact_friction_hat"], env.contact_friction_hat),
        )
        worst["velocity"] = max(
            worst["velocity"],
            gap(observations["base_lin_vel_hat"], env.base_lin_vel_hat),
        )
        worst["push"] = max(worst["push"], gap(onboard.push, env._push_signal))
        worst["command"] = max(
            worst["command"],
            gap(observations["governed_commands"], env.governed_commands),
        )
        fastest["speed"] = max(fastest["speed"], env.clock_speed_obs.abs().max().item())

    compared = _drive(env, 160, compare)
    assert compared >= 120, f"only {compared} steps before a fall"
    assert worst["command"] < 1e-3, worst
    assert worst["phase"] < 1e-3, worst
    assert worst["speed"] < 1e-4, worst
    assert worst["friction"] < 1e-3, worst
    assert worst["velocity"] < 1e-3, worst
    assert worst["push"] < 1e-3, worst
    # The run exercised the moving clocks and a speed target that moved.
    assert fastest["speed"] > 0.1


def test_onboard_starts_on_the_standing_clocks():
    settings = {
        "leg_geometry": leg_geometry(),
        "dt": 0.01,
        "substeps": 5,
        "weight": 150.0,
        "default_dof_pos": [0.0, 0.66, -1.36] * 4,
        "friction_clip": [0.02, 1.5],
        "load_time_constant": 0.02,
        "estimator": {
            "features": ["dof_pos_obs"],
            "frames": 2,
            "stride": 1,
            "scales": {},
        },
        **onboard_settings_for_tests(),
    }
    settings["osc"].update(push_gate_speed=0.1, directional_min_load=0.05)
    onboard = Go2OscOnboard(settings, lambda window: torch.zeros(1, 4))
    torch.testing.assert_close(onboard.phase, torch.full((1, 4), STANDING_PHASE))
    sensors = {
        "dof_pos": torch.tensor([[0.0, 0.66, -1.36] * 4]),
        "dof_vel": torch.zeros(1, 12),
        # A quarter of the weight on each foot through the knee torque.
        "torques": torch.tensor([[0.0, 0.0, 5.0] * 4]),
        "base_ang_vel": torch.zeros(1, 3),
        "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]),
        "base_lin_acc": torch.tensor([[0.0, 0.0, 9.81]]),
        "dof_pos_target": torch.zeros(1, 12),
    }
    command = torch.zeros(1, 3)
    onboard.reset(sensors, command)
    assert onboard.clock_speed.abs().max() == 0.0
    for _ in range(300):
        onboard.step(sensors, command)
    # Standing with loaded feet, the clocks settle where cos(phase) = 0 on
    # the stance side, as the simulator's standing clocks do.
    assert onboard.load.min() > 0.05
    assert onboard.phase.cos().abs().max() < 0.4
    assert (onboard.phase.sin() < 0).all()


# ── The deploy file, end to end ────────────────────────────────────────────


def _load_deploy_module():
    path = REPOSITORY / "go2_deploy/custom_implementation/go2osc_push_glide.py"
    spec = importlib.util.spec_from_file_location("go2osc_push_glide_deploy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QGYM_TO_UNITREE = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def _lowstate(env, row):
    """What the Go2 would report in the environment's present state."""
    actuated = env.actuated_dof_indices
    q = env.dof_pos.index_select(1, actuated)[row][QGYM_TO_UNITREE]
    dq = env.dof_vel.index_select(1, actuated)[row][QGYM_TO_UNITREE]
    tau = env.torques[row][QGYM_TO_UNITREE]
    x, y, z, w = env.base_quat[row].tolist()
    return SimpleNamespace(
        motor_state=[
            SimpleNamespace(q=q[i].item(), dq=dq[i].item(), tau_est=tau[i].item())
            for i in range(12)
        ],
        imu_state=SimpleNamespace(
            quaternion=[w, x, y, z],
            gyroscope=env.base_ang_vel[row].tolist(),
            accelerometer=env.base_lin_acc[row].tolist(),
        ),
    )


@pytest.fixture
def bundle(tmp_path):
    """A bundle exported from a freshly built environment and runner."""
    import gym.envs  # noqa: F401
    from scripts.export_go2osc_bundle import export

    env, runner_cfg, backend = _make()
    try:
        runner_cfg.runner.device = "cpu"
        runner_cfg.logging.enable_local_saving = False
        runner_cfg.log_dir = None
        runner = task_registry.make_alg_runner(env, runner_cfg)
        runner.switch_to_eval()
        checkpoint = tmp_path / "model_0.pt"
        torch.save({"placeholder": True}, checkpoint)
        export(env, runner, tmp_path / "bundle", checkpoint)
        yield env, runner, tmp_path / "bundle"
    finally:
        backend.close()


def test_bundle_holds_policy_estimator_settings_and_hashes(bundle):
    _, _, directory = bundle
    names = {path.name for path in directory.iterdir()}
    assert {
        "policy.pt",
        "estimator.pt",
        "go2osc_onboard.json",
        "deploy_meta.json",
        "SHA256SUMS",
    } <= names
    sums = (directory / "SHA256SUMS").read_text()
    for name in ("policy.pt", "estimator.pt", "checkpoint model_0.pt"):
        assert name in sums


def test_exported_policy_cannot_command_a_joint_past_its_limits(bundle):
    from go2_deploy.policy_bundle import load_bundle

    env, _, directory = bundle
    policy, cfg = load_bundle(str(directory))
    assert cfg.kp == pytest.approx(20.0) and cfg.kd == pytest.approx(0.5)
    assert cfg.exp_moving_avg is False
    scale = torch.tensor(cfg.DeployScaling.dof_pos_target)
    limits = env.dof_pos_limits.index_select(0, env.actuated_dof_indices)
    margin = env.cfg.control.joint_target_margin
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for magnitude in (1.0, 30.0, 1000.0):
            observation = magnitude * torch.randn(cfg.obs_dim(), generator=generator)
            target = policy(observation) * scale + cfg.default_dof_pos
            assert (target <= limits[:, 1] - margin + 1e-5).all()
            assert (target >= limits[:, 0] + margin - 1e-5).all()


def test_deploy_file_reproduces_the_training_observation(bundle, monkeypatch, capsys):
    from go2_deploy.policy_bundle import load_bundle
    from go2_deploy.utility import deploy_utility

    env, runner, directory = bundle
    module = _load_deploy_module()
    monkeypatch.setenv(module.BUNDLE_VARIABLE, str(directory))
    monkeypatch.setattr("sys.argv", ["deploy"])
    implementation = module.Go2OscPushGlide()
    policy, cfg = load_bundle(str(directory), implementation)
    row = 0
    everyone = torch.ones(NUM_ENVS, dtype=torch.bool)
    env.commands[:] = _command_schedule(0)
    env._reset_idx(everyone)
    # A reset draws a random command; use the scheduled one on both sides.
    env.commands[:] = _command_schedule(0)
    # The robot starts on standing clocks; give the simulator the same.
    env.oscillators[:] = STANDING_PHASE
    env._refresh_oscillator_observations()
    env.estimator.reset(everyone, env._estimator_inputs())
    env._publish_estimate()
    env._advance_clock_speed(everyone)
    # The robot's controller starts at rest: v_t = 0.
    env.clock_speed_obs.zero_()
    env._refresh_onboard_push_signal(everyone)

    limits = cfg.command_limits
    remote = SimpleNamespace(lin_vel_x=0.0, lin_vel_y=0.0, yaw_vel=0.0)
    controller = SimpleNamespace(
        cfg=cfg,
        obs_vec_size=cfg.obs_dim(),
        remote_controller=remote,
        last_action=torch.zeros(2, 12),
        last_command=torch.zeros(3),
        last_lowstate_msg=None,
        _state=SimpleNamespace(name="RECOVERY"),
    )
    for key, value in implementation.get_custom_buffers_dict().items():
        setattr(controller, key, value)

    def tick(command):
        remote.lin_vel_x, remote.lin_vel_y, remote.yaw_vel = command[row].tolist()
        controller.last_lowstate_msg = _lowstate(env, row)
        controller.last_action[0] = env.dof_pos_target[row]
        implementation.update_custom_buffers(controller, 0.0)
        return deploy_utility.lowstate_to_obs(controller, controller.last_lowstate_msg)

    assert max(limits.values()) == pytest.approx(2.0)
    # Before the policy takes over the implementation only restarts itself.
    observation = tick(env.commands)
    controller._state = SimpleNamespace(name="CUSTOM_CTRL")
    # At takeover it reports the on-board signals for a check by eye.
    implementation.update_custom_buffers(controller, 10.0)
    report = capsys.readouterr().out
    assert "go2osc check | foot load" in report and "accelerometer" in report
    controller.go2osc_running = False
    observation = tick(env.commands)
    expected = runner.get_obs(runner.actor_cfg["obs"])[row]
    torch.testing.assert_close(observation, expected, atol=2e-4, rtol=0)
    worst = {"observation": 0.0, "action": 0.0}

    def compare(step, command):
        observation = tick(command)
        expected = runner.get_obs(runner.actor_cfg["obs"])[row]
        worst["observation"] = max(
            worst["observation"], (observation - expected).abs().max().item()
        )
        with torch.no_grad():
            action = policy(observation)
            reference = runner.alg.actor.act_inference(expected.unsqueeze(0))[0]
        low, high = (
            torch.tensor(bound)
            / torch.tensor(controller.go2osc_settings["action_scale"])
            for bound in controller.go2osc_settings["target_offset_bounds"]
        )
        reference = torch.maximum(torch.minimum(reference, high), low)
        worst["action"] = max(worst["action"], (action - reference).abs().max().item())

    compared = _drive(env, 160, compare)
    assert compared >= 120, f"only {compared} steps before a fall"
    assert worst["observation"] < 2e-3, worst
    assert worst["action"] < 2e-2, worst


def test_deploy_file_clips_commands_and_zeroes_a_released_stick(bundle, monkeypatch):
    _, _, directory = bundle
    module = _load_deploy_module()
    monkeypatch.setenv(module.BUNDLE_VARIABLE, str(directory))
    monkeypatch.setattr("sys.argv", ["deploy"])
    settings = module.Go2OscPushGlide().get_custom_buffers_dict()["go2osc_settings"]
    controller = SimpleNamespace(
        cfg=SimpleNamespace(
            command_limits={"lin_vel_x": 2.0, "lin_vel_y": 0.3, "yaw_vel": 0.5}
        ),
        go2osc_settings=settings,
        remote_controller=SimpleNamespace(lin_vel_x=-2.0, lin_vel_y=0.3, yaw_vel=-0.5),
    )
    # Backward was trained to 1 m/s only.
    torch.testing.assert_close(
        module.read_command(controller), torch.tensor([-1.0, 0.3, -0.5])
    )
    controller.remote_controller = SimpleNamespace(
        lin_vel_x=0.05, lin_vel_y=-0.01, yaw_vel=0.4
    )
    torch.testing.assert_close(
        module.read_command(controller), torch.tensor([0.0, 0.0, 0.4])
    )


def test_deploy_file_needs_a_bundle_and_refuses_action_smoothing(bundle, monkeypatch):
    _, _, directory = bundle
    module = _load_deploy_module()
    monkeypatch.delenv(module.BUNDLE_VARIABLE, raising=False)
    monkeypatch.setattr("sys.argv", ["deploy", "eth0"])
    with pytest.raises(RuntimeError, match="policy bundle"):
        module.Go2OscPushGlide().get_custom_buffers_dict()
    monkeypatch.setattr("sys.argv", ["deploy", "eth0", "--bundle", str(directory)])
    assert module.bundle_directory() == directory
    settings = module.load_onboard(directory)[1]
    smoothing = SimpleNamespace(ctrl_freq=100, exp_moving_avg=True)
    with pytest.raises(RuntimeError, match="action smoothing"):
        module.check_runner_settings(smoothing, settings)
    slow = SimpleNamespace(ctrl_freq=50, exp_moving_avg=False)
    with pytest.raises(RuntimeError, match="Hz"):
        module.check_runner_settings(slow, settings)
    # A tampered estimator file is refused.
    (directory / "estimator.pt").write_bytes(b"not the estimator")
    with pytest.raises(ValueError, match="does not match"):
        module.load_onboard(directory)


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


def test_export_refuses_a_checkpoint_from_before_the_handover(bundle, tmp_path):
    from scripts.export_go2osc_bundle import export

    env, runner, _ = bundle
    env.set_iteration(0)
    with pytest.raises(ValueError, match="hand-over"):
        export(env, runner, tmp_path / "early", tmp_path / "model_0.pt")


def test_export_refuses_a_checkpoint_from_before_the_clamp_closed(bundle, tmp_path):
    from scripts.export_go2osc_bundle import export

    env, runner, _ = bundle
    env.estimate_share = 1.0
    env.joint_target_progress = 0.5
    with pytest.raises(ValueError, match="clamp"):
        export(env, runner, tmp_path / "early", tmp_path / "model_0.pt")


def test_robust_policy_inputs_map_onto_the_deploy_actor(env):
    from scripts.convert_go2osc_robust_to_deploy import (
        convert,
        input_columns,
    )

    source = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "oscillator_obs",
        "foot_traction_obs",
        "friction_estimate_obs",
        "base_lin_vel",
        "clock_speed_obs",
    ]
    target = list(configs.Go2OscPushGlideDeployRunnerCfg.actor.obs)
    columns, width, dropped = input_columns(env, source, target)
    assert (width, len(columns)) == (66, 58)
    assert dropped == ["foot_traction_obs"]
    # The estimates take the columns of the signals they replace.
    assert columns[-5:].tolist() == [61, 62, 63, 64, 65]
    assert columns[:53].tolist() == list(range(53))
    weight = torch.arange(66.0).repeat(3, 1)
    state = {
        "actor_state_dict": {"NN.0.weight": weight, "NN.2.bias": torch.zeros(66, 4)},
        "optimizer_state_dict": {"state": {0: {"exp_avg": weight.clone()}}, "x": 1},
    }
    actor, optimizer = convert(state, columns, width)
    torch.testing.assert_close(actor["NN.0.weight"][0], columns.float())
    assert actor["NN.2.bias"].shape == (66, 4)  # not laid out over the inputs
    assert optimizer["state"][0]["exp_avg"].shape == (3, 58)
