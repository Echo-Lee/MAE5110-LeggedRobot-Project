"""Robust push/glide task: sensing errors, delay, changing ground, sliding stops."""

import math
from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2 import go2osc_config as configs
from gym.envs.go2.go2osc import Go2OscPushGlide
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.modules import Actor, Critic
from learning.runners.oscillator_runner import OscillatorRunner
from tests.unit_tests.test_go2osc_push_glide import (
    WEIGHT,
    _env,
    _make_cpu_env,
    _silence_oscillator_noise,
)

PHYSICS_DT = 0.002
CONTROL_DT = 0.01


def _varied_env(num_envs=1, *, sensing=None, friction_changes=None, delay=None):
    """Hand-built environment with the optional settings and their buffers."""
    env = _env(num_envs)
    env.dt = CONTROL_DT
    env.cfg.control.decimation = 5
    env.cfg.control.actuation_delay_range = delay
    env.cfg.seed = 5
    env.num_actuators = 2
    env.dof_pos_target = torch.zeros(num_envs, 2)
    env._physical_reset_mask = torch.ones(num_envs, dtype=torch.bool)
    if sensing is not None:
        env.cfg.sensing = SimpleNamespace(**sensing)
    if friction_changes is not None:
        env.cfg.friction_changes = SimpleNamespace(**friction_changes)
    env._init_variation_buffers()
    return env


def test_travelled_speed_charges_energy_per_metre_made_along_the_command():
    env = _env(7)
    env.cfg.reward_settings.cost_of_transport_speed = "travelled"
    env.actuated_dof_indices = torch.tensor([0, 2])
    env.dof_vel = torch.tensor([[2.0, 9.0, -3.0]]).repeat(7, 1)
    env.torques = torch.tensor([[5.0, 4.0]]).repeat(7, 1)
    power = 5.0 * 2.0 + 4.0 * 3.0
    env.commands = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # on the command
            [1.0, 0.0, 0.0],  # slower, drifting sideways
            [1.0, 0.0, 0.0],  # faster than told
            [-1.0, 0.0, 0.0],  # backward
            [1.0, 0.0, 0.0],  # going the wrong way
            [0.0, 0.0, 0.5],  # turning in place
            [0.6, 0.8, 0.0],  # diagonal
        ]
    )
    env.base_lin_vel = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.6, 0.3, 0.0],
            [1.4, 0.0, 0.0],
            [-0.8, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.54, 0.72, 0.0],
        ]
    )
    speed = torch.tensor([1.0, 0.6, 1.0, 0.8, 0.5, 0.5, 0.9])
    torch.testing.assert_close(
        env._reward_cost_of_transport(), -power / (WEIGHT * speed)
    )
    # The commanded-speed charge ignores how fast the body really goes.
    env.cfg.reward_settings.cost_of_transport_speed = "command"
    commanded = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])
    torch.testing.assert_close(
        env._reward_cost_of_transport(), -power / (WEIGHT * commanded)
    )


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("reward_settings", "cost_of_transport_speed", "distance"),
        ("control", "actuation_delay_range", [0.02, 0.0]),
        ("control", "actuation_delay_range", [-0.01, 0.02]),
        ("control", "actuation_delay_range", 0.02),
        ("sensing", "friction_estimate_gain_range", [1.1, 1.4]),
        ("sensing", "foot_load_gain_range", [0.0, 1.2]),
        ("sensing", "foot_traction_gain_range", [0.7, float("inf")]),
        ("sensing", "friction_estimate_noise_std", -0.1),
        ("sensing", "friction_estimate_time_constant", float("nan")),
        ("sensing", "foot_load_noise_std", True),
        ("sensing", "foot_force_time_constant", -0.02),
        ("friction_changes", "rate", -1.0),
        ("friction_changes", "split_probability", 1.5),
        ("commands", "sliding_stop_probability", 1.5),
        ("commands", "sliding_stop_min_speed", -0.5),
        ("commands", "sliding_stop_duration", 0.0),
        ("reward_settings", "speed_reference_gain", 0.0),
        ("reward_settings", "speed_reference_brake", float("nan")),
        ("reward_settings", "speed_reference_max_gain", -1.0),
        ("reward_settings", "speed_reference_max_brake", None),
        ("reward_settings", "yaw_overshoot_tolerance", -0.1),
        ("fast_slides", "probability", 1.2),
        ("fast_slides", "friction_range", [0.3, 0.1]),
        ("fast_slides", "friction_range", [0.0, 0.3]),
        ("fast_slides", "speed_range", [2.5, 1.5]),
        ("falls", "max_tilt", 0.0),
        ("falls", "max_tilt", float("inf")),
        ("commands", "braking_range", [0.6, 0.25]),
        ("commands", "braking_range", [0.0, 0.5]),
        ("osc", "clock_speed_min_friction", 0.0),
        ("reward_settings", "standing_lift_weight", -1.0),
        ("reward_settings", "speed_reference_carve_min_brake", 1.5),
    ],
)
def test_invalid_variation_settings_fail_before_simulation(section, name, value):
    cfg = configs.Go2OscPushGlideRobustCfg()
    Go2OscPushGlide._validate_push_glide_cfg(cfg)
    setattr(getattr(cfg, section), name, value)
    with pytest.raises(ValueError, match=name):
        Go2OscPushGlide._validate_push_glide_cfg(cfg)


def test_carving_requires_the_speed_reference_that_updates_its_yaw_target():
    cfg = configs.Go2OscPushGlideRobustCfg()
    cfg.osc.clock_follows_speed_target = False
    cfg.reward_settings.speed_reference = False
    with pytest.raises(ValueError, match="speed_reference_carve needs speed_reference"):
        Go2OscPushGlide._validate_push_glide_cfg(cfg)
    cfg.reward_settings.speed_reference_carve = False
    Go2OscPushGlide._validate_push_glide_cfg(cfg)


def test_motors_follow_the_target_issued_one_delay_earlier():
    delays = torch.tensor([0, 3, 5, 7, 10])
    env = _varied_env(5, delay=[0.0, 0.02])
    assert env._target_history.shape == (5, 3, 2)
    env._actuation_delay_steps.copy_(delays)
    for control_step in range(6):
        env.dof_pos_target = torch.full((5, 2), float(control_step + 1))
        env._record_joint_targets()
        for physics_step in range(5):
            issued = (control_step * 5 + physics_step - delays).div(
                5, rounding_mode="floor"
            )
            # Before the first target arrives the motors hold the reset pose.
            expected = (issued + 1).clamp(min=0).float()
            applied = env._joint_position_target()
            torch.testing.assert_close(applied, expected[:, None].expand(5, 2))
            env._post_compute_torques()
    # A reset holds the reset pose through the whole delay and redraws it.
    env.dof_pos_target = torch.full((5, 2), -4.0)
    mask = torch.tensor([True, False, True, False, True])
    env._resample_variation(mask)
    assert torch.all(env._target_history[mask] == -4.0)
    assert torch.all(env._target_history[~mask, 0] == 6.0)
    assert torch.equal(env._actuation_delay_steps[~mask], delays[~mask])
    assert torch.all(
        (env._actuation_delay_steps >= 0) & (env._actuation_delay_steps <= 10)
    )


def test_without_a_delay_range_the_motors_follow_the_current_target():
    env = _varied_env(2)
    assert env._target_history is None
    env.dof_pos_target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    env._record_joint_targets()
    assert env._joint_position_target() is env.dof_pos_target


def test_friction_estimate_is_the_lagged_scaled_noisy_mean_under_the_feet():
    env = _varied_env(3)
    everyone = torch.ones(3, dtype=torch.bool)
    env.foot_friction_obs.copy_(
        torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.4, 0.6, 0.8], [1.0, 1.0, 1.0, 1.0]])
    )
    truth = torch.tensor([[0.1], [0.5], [1.0]])
    cached = env.friction_estimate_obs
    # Nothing configured: the estimate is the truth, with no delay.
    env._refresh_friction_estimate()
    assert env.friction_estimate_obs is cached
    torch.testing.assert_close(cached, truth)
    # A gain fixed for the episode.
    env._friction_estimate_gain.copy_(torch.tensor([[0.7], [1.0], [1.4]]))
    env._refresh_friction_estimate()
    torch.testing.assert_close(cached, truth * torch.tensor([[0.7], [1.0], [1.4]]))
    # A first-order lag: ice appears under robot 2 and the estimate follows.
    env._friction_estimate_gain.fill_(1.0)
    env.cfg.sensing = SimpleNamespace(friction_estimate_time_constant=0.5)
    env._refresh_friction_estimate(everyone)
    env.foot_friction_obs[2] = 0.1
    for _ in range(50):
        env._refresh_friction_estimate()
    remaining = (1 - CONTROL_DT / (0.5 + CONTROL_DT)) ** 50
    assert 0.36 < remaining < 0.38
    torch.testing.assert_close(cached[2, 0], torch.tensor(0.1 + 0.9 * remaining))
    torch.testing.assert_close(cached[:2], truth[:2])
    # A reset starts selected robots from the ground they stand on.
    env._refresh_friction_estimate(torch.tensor([False, False, True]))
    torch.testing.assert_close(cached[2, 0], torch.tensor(0.1))
    # Noise at every step, never below zero.
    env.cfg.sensing = SimpleNamespace(friction_estimate_noise_std=0.05)
    samples = []
    for _ in range(400):
        env._refresh_friction_estimate()
        samples.append(cached.clone())
    samples = torch.stack(samples)
    assert samples.min() >= 0.0
    assert abs(samples[:, 1].std().item() - 0.05) < 0.01
    assert abs(samples[:, 1].mean().item() - 0.5) < 0.01
    assert samples[:, 0].mean() > 0.1  # clipped at zero from below


def test_clocks_receive_the_sensed_foot_load_and_rewards_keep_the_true_one():
    env = _varied_env(2, sensing={"foot_load_gain_range": [0.8, 1.2]})
    _silence_oscillator_noise(env)
    assert env._foot_load_is_sensed()
    env.commands[:] = torch.tensor([0.0, 0.0, 0.0])  # plain load, no gate
    gain = torch.tensor([[0.8, 1.0, 1.2, 1.1], [1.0, 1.0, 1.0, 1.0]])
    env._foot_load_gain.copy_(gain)
    env._refresh_sensed_foot_load(reset_mask=torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(env._compute_push_signal(), 0.25 * gain)
    # One oscillator step integrates Equation (4) with the sensed load.
    env.osc_coupling.fill_(4.0)
    env.osc_offset.fill_(1.0)
    env.oscillators[:] = torch.tensor([0.3, 1.7, 3.1, 5.9])
    feedback = 4.0 * (torch.cos(env.oscillators) + 1.0)
    env._step_oscillators(PHYSICS_DT)
    rate = env.oscillators_vel / (2 * math.pi)
    torch.testing.assert_close(rate, 2.5 - 0.25 * gain * feedback)
    # The push reward is paid on the true load.
    env.oscillators[:] = 1.5 * math.pi
    torch.testing.assert_close(env._reward_push_load(), torch.full((2,), 1.0))


def test_sensed_foot_load_lags_and_stays_within_zero_and_one_body_weight():
    env = _varied_env(1, sensing={"foot_force_time_constant": 0.02})
    assert env._foot_load_is_sensed()
    env._refresh_sensed_foot_load(reset_mask=torch.ones(1, dtype=torch.bool))
    torch.testing.assert_close(env._sensed_foot_load, torch.full((1, 4), 0.25))
    # The load on every foot doubles; ten physics steps cover one time constant.
    env.contact_forces[:, env.feet_indices, 2] = 0.5 * WEIGHT
    for _ in range(10):
        env._refresh_sensed_foot_load(PHYSICS_DT)
    remaining = (1 - PHYSICS_DT / (0.02 + PHYSICS_DT)) ** 10
    torch.testing.assert_close(
        env._sensed_foot_load, torch.full((1, 4), 0.5 - 0.25 * remaining)
    )
    # Noise cannot make a load negative or exceed one body weight.
    noisy = _varied_env(1, sensing={"foot_load_noise_std": 0.5})
    values = []
    for _ in range(200):
        noisy._refresh_sensed_foot_load(PHYSICS_DT)
        values.append(noisy._sensed_foot_load.clone())
    values = torch.cat(values)
    assert values.min() == 0.0 and values.max() == 1.0
    assert 0.2 < values.mean() < 0.4


def test_traction_input_passes_through_a_gain_per_foot_and_a_lag():
    env = _varied_env(
        2,
        sensing={
            "foot_traction_gain_range": [0.7, 1.3],
            "foot_force_time_constant": 0.02,
        },
    )
    everyone = torch.ones(2, dtype=torch.bool)
    env.contact_forces[:, env.feet_indices, 0] = 0.05 * WEIGHT  # ratio 0.2 forward
    env.contact_forces[:, env.feet_indices, 1] = -0.025 * WEIGHT  # 0.1 to the right
    gain = torch.tensor([[0.7, 1.0, 1.3, 1.1], [1.0, 1.0, 1.0, 1.0]])
    env._foot_traction_gain.copy_(gain)
    cached = env.foot_traction_obs
    env._refresh_traction_observations(everyone)
    assert env.foot_traction_obs is cached
    truth = torch.tensor([0.2, -0.1]).repeat(2, 4)
    scale = gain.repeat_interleave(2, dim=1)
    torch.testing.assert_close(cached, truth * scale)
    # The forward ratio doubles; the input follows with the lag.
    env.contact_forces[:, env.feet_indices, 0] = 0.1 * WEIGHT
    for _ in range(10):
        env._refresh_traction_observations()
    remaining = (1 - PHYSICS_DT / (0.02 + PHYSICS_DT)) ** 10
    expected = truth.clone()
    expected[:, 0::2] = 0.4 - 0.2 * remaining
    torch.testing.assert_close(cached, expected * scale)
    # A reset of one robot starts its filter from the current ratio.
    env._refresh_traction_observations(torch.tensor([True, False]))
    torch.testing.assert_close(cached[0, 0::2], 0.4 * gain[0])
    torch.testing.assert_close(cached[1], (expected * scale)[1])


def test_ground_friction_draws_share_one_value_or_split_between_feet():
    calls = []
    env = _varied_env(4000, friction_changes={"rate": 0.2, "split_probability": 0.3})
    env._backend.set_contact_friction_by_body = lambda *args: calls.append(args)
    assert env._ground_friction_varies()
    before = env.foot_friction_obs.clone()
    mask = torch.arange(4000) % 2 == 0
    env._draw_ground_friction(mask)
    ids, bodies, values = calls[0]
    assert torch.equal(ids, mask.nonzero().flatten())
    assert torch.equal(bodies, env.feet_indices)
    torch.testing.assert_close(values, env.foot_friction_obs[mask], rtol=0, atol=0)
    torch.testing.assert_close(env.foot_friction_obs[~mask], before[~mask])
    assert torch.all((values >= 0.1) & (values <= 1.0))
    split = (values != values[:, :1]).any(dim=1)
    assert abs(split.float().mean().item() - 0.3) < 0.04
    assert values.min() < 0.15 and values.max() > 0.95
    # Within an episode, changes arrive at the configured mean rate.
    calls.clear()
    changed = 0
    for _ in range(100):
        env._change_ground_friction()
        if calls:
            changed += calls.pop()[0].numel()
    assert abs(changed - 4000 * 100 * CONTROL_DT * 0.2) < 120
    # No range to draw from, or nothing configured: the ground stays as set.
    env._ground_friction_range = None
    assert not env._ground_friction_varies()
    assert not _varied_env(2)._ground_friction_varies()
    idle = _varied_env(2, friction_changes={"rate": 0.0, "split_probability": 0.0})
    assert not idle._ground_friction_varies()


def test_runner_noise_reaches_the_signals_named_in_the_robust_config():
    runner_cfg = class_to_dict(configs.Go2OscPushGlideRobustRunnerCfg())
    cfg = configs.Go2OscPushGlideRobustCfg()
    sizes = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "commands": 3,
        "dof_pos_obs": 12,
        "dof_vel": 12,
        "dof_pos_target": 12,
        "oscillator_obs": 8,
        "foot_traction_obs": 8,
        "friction_estimate_obs": 1,
        "base_lin_vel": 3,
        "clock_speed_obs": 1,
    }
    names = runner_cfg["actor"]["obs"]
    assert names == list(sizes)
    stub = SimpleNamespace(
        device="cpu",
        get_obs_size=lambda selected: sum(sizes[name] for name in selected),
        env=SimpleNamespace(num_envs=4000, scales=class_to_dict(cfg.scaling, "cpu")),
    )
    noise = OscillatorRunner.get_noise(stub, names, runner_cfg["actor"]["noise"])
    assert noise.shape == (4000, 66)
    widths = noise.abs().max(dim=0).values
    offsets = dict(zip(names, torch.tensor([0] + list(sizes.values())).cumsum(0)))
    expected = {
        "base_ang_vel": torch.tensor([0.2 / 0.3, 0.2 / 0.3, 0.2 / 0.1]),
        "projected_gravity": torch.full((3,), 0.05),
        "dof_pos_obs": 0.01 / torch.tensor([0.2, 0.3, 0.3] * 4),
        "dof_vel": 0.5 / torch.tensor([2.0, 2.0, 4.0] * 4),
        "foot_traction_obs": torch.full((8,), 0.1),
        "base_lin_vel": torch.full((3,), 0.1) / stub.env.scales["base_lin_vel"],
    }
    for name, size in sizes.items():
        column = widths[offsets[name] : offsets[name] + size]
        if name in expected:
            assert torch.all(column <= expected[name] + 1e-6), name
            assert torch.all(column >= 0.98 * expected[name]), name
        else:
            assert torch.count_nonzero(column) == 0, name


def test_robust_variant_changes_only_the_declared_settings():
    baseline = class_to_dict(configs.Go2OscPushGlideCotCfg())
    variant = class_to_dict(configs.Go2OscPushGlideRobustCfg())
    assert variant["commands"]["forward_uniform_range"] == [-1.0, 2.0]
    variant["commands"]["forward_uniform_range"] = [0.0, 1.5]
    assert variant["commands"]["ranges"]["lin_vel_x"] == [-1.0, 2.0]
    variant["commands"]["ranges"]["lin_vel_x"] = [0.0, 1.5]
    assert variant["commands"].pop("sliding_stop_probability") == 0.5
    assert variant["commands"].pop("sliding_stop_min_speed") == 0.5
    assert variant["commands"].pop("sliding_stop_duration") == 1.5
    assert variant["commands"].pop("braking_range") == [0.9, 0.9]
    # A stop ends before the next scheduled command change.
    assert variant["commands"]["resampling_time"] == 3.0
    assert baseline["env"]["episode_length_s"] == 4.0
    assert variant["env"]["episode_length_s"] == 6.0
    variant["env"]["episode_length_s"] = 4.0
    assert baseline["init_state"]["root_vel_range"][0] == [-0.5, 2.0]
    assert variant["init_state"]["root_vel_range"][0] == [-1.5, 2.5]
    variant["init_state"]["root_vel_range"][0] = [-0.5, 2.0]
    assert variant["control"].pop("actuation_delay_range") == [0.0, 0.02]
    assert baseline["push_robots"]["interval_s"] == 10
    assert baseline["push_robots"]["max_push_vel_xy"] == 0.2
    assert variant["push_robots"]["interval_s"] == 5
    assert variant["push_robots"]["max_push_vel_xy"] == 0.5
    variant["push_robots"].update(interval_s=10, max_push_vel_xy=0.2)
    randomization = variant["domain_randomization"]
    assert baseline["domain_randomization"]["startup"]["link_mass_scale_range"] is None
    assert randomization["startup"]["link_mass_scale_range"] == [0.9, 1.1]
    randomization["startup"]["link_mass_scale_range"] = None
    assert baseline["domain_randomization"]["startup"]["contact_friction_range"] == [
        0.1,
        1.0,
    ]
    assert randomization["startup"]["contact_friction_range"] == [0.05, 1.0]
    randomization["startup"]["contact_friction_range"] = [0.1, 1.0]
    assert variant["osc"].pop("clock_follows_speed_target") is True
    assert variant["osc"].pop("clock_speed_min_friction") == 0.02
    assert baseline["domain_randomization"]["episode"]["scale_ranges"] == {}
    assert randomization["episode"]["scale_ranges"] == {
        "p_gains": [0.8, 1.2],
        "d_gains": [0.8, 1.2],
    }
    randomization["episode"]["scale_ranges"] = {}
    assert variant.pop("friction_changes") == {"rate": 0.2, "split_probability": 0.3}
    assert variant.pop("sensing") == {
        "friction_estimate_gain_range": [0.7, 1.4],
        "friction_estimate_noise_std": 0.05,
        "friction_estimate_time_constant": 0.5,
        "foot_load_gain_range": [0.8, 1.2],
        "foot_load_noise_std": 0.05,
        "foot_traction_gain_range": [0.7, 1.3],
        "foot_force_time_constant": 0.02,
    }
    assert variant["scaling"].pop("friction_estimate_obs") == 1.0
    assert variant["scaling"].pop("foot_friction_obs") == 1.0
    assert variant["scaling"].pop("clock_speed_obs") == 3.0
    assert variant["reward_settings"].pop("cost_of_transport_speed") == "travelled"
    assert variant["reward_settings"].pop("speed_reference") is True
    assert variant["reward_settings"].pop("speed_reference_gain") == 0.5
    assert variant["reward_settings"].pop("speed_reference_brake") == 0.9
    assert variant["reward_settings"].pop("speed_reference_max_gain") == 2.0
    assert variant["reward_settings"].pop("speed_reference_max_brake") == 5.0
    assert variant["reward_settings"].pop("yaw_overshoot_tolerance") == 0.25
    assert variant["reward_settings"].pop("speed_reference_carve") is True
    assert variant["reward_settings"].pop("speed_reference_carve_min_brake") == 0.5
    assert variant["reward_settings"].pop("standing_turn_weight") == 0.5
    assert variant["reward_settings"].pop("standing_lift_weight") == 0.25
    assert variant["reward_settings"].pop("standing_lift_load") == 0.05
    assert variant.pop("fast_slides") == {
        "probability": 0.2,
        "friction_range": [0.05, 0.3],
        "speed_range": [1.5, 2.5],
    }
    assert variant.pop("falls") == {"max_tilt": 60.0}
    assert variant == baseline

    baseline = class_to_dict(configs.Go2OscPushGlideCotRunnerCfg())
    variant = class_to_dict(configs.Go2OscPushGlideRobustRunnerCfg())
    assert variant["runner"]["experiment_name"] == "go2osc_push_glide_robust"
    variant["runner"]["experiment_name"] = baseline["runner"]["experiment_name"]
    assert variant["actor"]["obs"] == baseline["actor"]["obs"] + [
        "friction_estimate_obs",
        "base_lin_vel",
        "clock_speed_obs",
    ]
    variant["actor"]["obs"] = baseline["actor"]["obs"]
    assert "contact_friction_obs" not in variant["actor"]["obs"]
    assert variant["critic"]["obs"] == baseline["critic"]["obs"] + [
        "foot_friction_obs",
        "clock_speed_obs",
    ]
    variant["critic"]["obs"] = baseline["critic"]["obs"]
    noise = variant["actor"]["noise"]
    assert baseline["actor"]["noise"]["base_ang_vel"] == 0.01
    assert baseline["actor"]["noise"]["dof_pos_obs"] == 0.01
    assert baseline["actor"]["noise"]["dof_vel"] == 0.005
    assert noise["base_ang_vel"] == 0.2 and noise["dof_vel"] == 0.5
    noise.update(base_ang_vel=0.01, dof_vel=0.005)
    assert noise.pop("projected_gravity") == 0.05
    assert noise.pop("foot_traction_obs") == 0.1
    assert noise.pop("base_lin_vel") == 0.1
    weights = variant["critic"]["reward"]["weights"]
    assert baseline["critic"]["reward"]["weights"]["cost_of_transport"] == 1.0
    assert weights["cost_of_transport"] == 3.0
    weights["cost_of_transport"] = 1.0
    assert baseline["critic"]["reward"]["weights"]["tracking_ang_vel"] == 2.0
    assert weights["tracking_ang_vel"] == 4.0
    weights["tracking_ang_vel"] = 2.0
    assert weights.pop("yaw_overshoot") == 1.0
    assert weights.pop("standing") == 2.0
    termination = variant["critic"]["reward"]["termination_weight"]
    assert baseline["critic"]["reward"]["termination_weight"]["termination"] == 0.15
    assert termination["termination"] == 5.0
    termination["termination"] = 0.15
    assert variant == baseline
    assert task_registry.get_task_class("go2osc_push_glide_robust") is Go2OscPushGlide


@pytest.mark.parametrize("control_dt", [0.01, 0.02])
def test_robust_fall_penalty_is_applied_once_without_time_scaling(control_dt):
    runner = OscillatorRunner.__new__(OscillatorRunner)
    runner.device = "cpu"
    runner.env = _env(3)
    runner.env.dt = control_dt
    runner.env.terminated = torch.tensor([False, True, False])
    runner.critic_cfg = class_to_dict(configs.Go2OscPushGlideRobustRunnerCfg())[
        "critic"
    ]
    # Isolate the actual task's terminal reward from ordinary reward terms.
    runner.critic_cfg["reward"]["weights"] = {}
    runner.reward_functions = {"termination": runner.env._reward_termination}
    rewards = runner.initialize_rewards_dict(2)

    runner.update_rewards_dict(rewards, 0)
    runner.env.terminated[:] = False
    runner.update_rewards_dict(rewards, 1)

    torch.testing.assert_close(
        rewards["termination"],
        torch.tensor([[0.0, -5.0, 0.0], [0.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize("mode", ["off", "friction-only", "pd-only", "mass-only"])
def test_randomization_override_also_switches_off_task_level_variation(mode):
    cfg = configs.Go2OscPushGlideRobustCfg()
    apply_domain_randomization_override(cfg, mode)
    assert cfg.control.actuation_delay_range is None
    assert class_to_dict(cfg.sensing) == {
        "friction_estimate_gain_range": None,
        "friction_estimate_noise_std": 0.0,
        "friction_estimate_time_constant": 0.0,
        "foot_load_gain_range": None,
        "foot_load_noise_std": 0.0,
        "foot_traction_gain_range": None,
        "foot_force_time_constant": 0.0,
    }
    ground = class_to_dict(cfg.friction_changes)
    if mode == "friction-only":
        assert ground == {"rate": 0.2, "split_probability": 0.3}
    else:
        assert ground == {"rate": 0.0, "split_probability": 0.0}
    Go2OscPushGlide._validate_push_glide_cfg(cfg)
    # The task's own configuration is untouched by "config".
    cfg = configs.Go2OscPushGlideRobustCfg()
    apply_domain_randomization_override(cfg, "config")
    assert cfg.control.actuation_delay_range == [0.0, 0.02]
    assert cfg.sensing.foot_load_gain_range == [0.8, 1.2]


def test_native_robust_task_with_variation_off_matches_ideal_sensing():
    cfg = configs.Go2OscPushGlideRobustCfg()
    runner_cfg = configs.Go2OscPushGlideRobustRunnerCfg()
    cfg.seed = runner_cfg.seed = 23
    cfg.env.num_envs = 3
    cfg.push_robots.toggle = False
    apply_domain_randomization_override(cfg, "off")
    cfg.domain_randomization.startup.contact_friction_range = [0.3, 0.3]
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = MuJocoCPUBackend()
    try:
        env = Go2OscPushGlide(cfg, "cpu", True, backend)
        assert env._target_history is None
        assert not env._foot_load_is_sensed() and not env._ground_friction_varies()
        generator = torch.Generator().manual_seed(4)
        for _ in range(20):
            action = 0.5 * torch.randn(3, 12, generator=generator)
            env.set_states(runner_cfg.actor.actions, action)
            env.step()
        torch.testing.assert_close(env.friction_estimate_obs, torch.full((3, 1), 0.3))
        torch.testing.assert_close(env.foot_friction_obs, torch.full((3, 4), 0.3))
        torch.testing.assert_close(env.p_gains, torch.full((3, 12), 20.0))
        assert env._joint_position_target() is env.dof_pos_target
        assert env.get_states(runner_cfg.actor.obs).shape == (3, 66)
    finally:
        backend.close()


def _make_robust_env(num_envs=4, configure=None):
    cfg = configs.Go2OscPushGlideRobustCfg()
    runner_cfg = configs.Go2OscPushGlideRobustRunnerCfg()
    cfg.seed = runner_cfg.seed = 23
    cfg.env.num_envs = num_envs
    cfg.push_robots.toggle = False
    if configure is not None:
        configure(cfg)
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = MuJocoCPUBackend()
    return Go2OscPushGlide(cfg, "cpu", True, backend), runner_cfg, backend


def _foot_contact_friction(backend, env_id, feet):
    """Sliding friction MuJoCo used in each foot-ground contact of one robot."""
    geoms, columns = backend._geoms_by_body(feet)
    data = backend._datas[env_id]
    found = {}
    for pair, friction in zip(data.contact.geom, data.contact.friction):
        for geom in pair:
            if geom in geoms and 0 in pair:
                found[int(columns[list(geoms).index(geom)])] = float(friction[0])
    return found


def _assert_per_body_friction(backend, feet, read_geom_friction):
    """Shared by the CPU and Warp tests: values land on the right geoms."""
    geoms, columns = backend._geoms_by_body(feet)
    assert len(geoms) == 4 and list(columns) == [0, 1, 2, 3]
    uniform = backend.contact_friction.clone()
    table = torch.tensor([[0.2, 0.4, 0.6, 0.8]], device=backend.device)
    backend.set_contact_friction_by_body(torch.tensor([1]), feet, table)
    torch.testing.assert_close(backend.contact_friction[1], table.mean())
    torch.testing.assert_close(backend.contact_friction[0], uniform[0])
    applied = read_geom_friction(1)
    torch.testing.assert_close(applied[torch.as_tensor(geoms)].cpu(), table[0].cpu())
    others = torch.ones(applied.numel(), dtype=torch.bool)
    others[torch.as_tensor(geoms)] = False
    # The ground and every other geom take the smallest value.
    assert torch.all(applied[others] == 0.2)
    assert torch.all(read_geom_friction(0) == uniform[0])
    # One coefficient per environment restores a uniform ground.
    backend.set_contact_friction(torch.tensor([1]), torch.tensor([0.3]))
    assert torch.all(read_geom_friction(1) == 0.3)
    torch.testing.assert_close(
        backend.contact_friction[1], torch.tensor(0.3, device=backend.device)
    )
    with pytest.raises(ValueError, match="per environment and body"):
        backend.set_contact_friction_by_body(torch.tensor([1]), feet, table[:, :3])
    with pytest.raises(ValueError, match="finite and >= 0"):
        backend.set_contact_friction_by_body(torch.tensor([1]), feet, -table)


def test_cpu_backend_gives_each_foot_its_own_friction_against_the_ground():
    env, _, backend = _make_cpu_env(
        num_envs=2, gains=(100.0, 3.0), friction_range=[0.1, 1.0]
    )
    try:
        feet = env.feet_indices

        def read(env_id):
            backend._activate_domain(env_id)
            return torch.tensor(backend._mjm.geom_friction[:, 0], dtype=torch.float)

        uniform = backend.contact_friction.clone()
        table = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
        backend.set_contact_friction_by_body(torch.tensor([1]), feet, table)
        for _ in range(100):
            env.commands.zero_()
            env.step()
        assert not env.terminated.any()
        # MuJoCo mixes by the larger coefficient: each standing foot meets
        # the ground at its own value, and the other robot is untouched.
        varied = _foot_contact_friction(backend, 1, feet)
        assert varied == pytest.approx({0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8})
        plain = _foot_contact_friction(backend, 0, feet)
        assert plain == pytest.approx(dict.fromkeys(range(4), uniform[0].item()))
        torch.testing.assert_close(
            env.contact_friction_obs[:, 0], backend.contact_friction
        )
        backend.set_contact_friction(torch.tensor([1]), uniform[1:])
        _assert_per_body_friction(backend, feet, read)
    finally:
        backend.close()


def test_cpu_backend_per_foot_friction_reaches_private_models_too():
    # Link-mass randomization gives every robot its own model.
    env, _, backend = _make_robust_env(num_envs=2)
    try:
        assert backend._models is not None

        def read(env_id):
            return torch.tensor(
                backend._models[env_id].geom_friction[:, 0], dtype=torch.float
            )

        backend.set_contact_friction(torch.arange(2), torch.tensor([0.5, 0.9]))
        _assert_per_body_friction(backend, env.feet_indices, read)
    finally:
        backend.close()


@pytest.mark.warp
def test_mujoco_warp_gives_each_foot_its_own_friction_against_the_ground():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    cfg = configs.Go2OscPushGlideCfg()
    runner_cfg = configs.Go2OscPushGlideRunnerCfg()
    cfg.seed = runner_cfg.seed = 23
    cfg.env.num_envs = 2
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = MuJocoWarpBackend()
    try:
        env = Go2OscPushGlide(cfg, "cuda:0", True, backend)
        _assert_per_body_friction(
            backend,
            env.feet_indices,
            lambda env_id: backend._geom_friction_t[env_id, :, 0].clone(),
        )
        env.step()
        assert torch.isfinite(env.root_states).all()
    finally:
        backend.close()


def test_native_robust_task_observes_steps_and_varies_what_it_declares():
    env, runner_cfg, backend = _make_robust_env(num_envs=6)
    try:
        actor_obs, critic_obs = runner_cfg.actor.obs, runner_cfg.critic.obs
        assert env.get_states(actor_obs).shape == (6, 66)
        assert env.get_states(critic_obs).shape == (6, 75)
        # Per-episode draws stay inside their configured ranges and differ.
        assert torch.all(
            (env._actuation_delay_steps >= 0) & (env._actuation_delay_steps <= 10)
        )
        for gain, (low, high) in (
            (env._friction_estimate_gain, (0.7, 1.4)),
            (env._foot_load_gain, (0.8, 1.2)),
            (env._foot_traction_gain, (0.7, 1.3)),
            (env.p_gains / 20.0, (0.8, 1.2)),
            (env.d_gains / 0.5, (0.8, 1.2)),
        ):
            assert torch.all((gain >= low) & (gain <= high))
            assert gain.unique().numel() > 4
        assert backend.link_mass.sum(dim=1).unique().numel() == 6
        friction = env.foot_friction_obs
        assert torch.all((friction >= 0.05) & (friction <= 1.0))
        torch.testing.assert_close(backend.contact_friction, friction.mean(dim=1))
        cached = env.friction_estimate_obs
        clock = env.clock_speed_obs
        generator = torch.Generator().manual_seed(3)
        for _ in range(40):
            action = 0.5 * torch.randn(6, 12, generator=generator)
            env.set_states(runner_cfg.actor.actions, action)
            env.step()
            actor = env.get_states(actor_obs)
            critic = env.get_states(critic_obs)
            assert torch.isfinite(actor).all() and torch.isfinite(critic).all()
            assert env.friction_estimate_obs is cached
            assert env.clock_speed_obs is clock
            torch.testing.assert_close(actor[:, -5:-4], cached)
            torch.testing.assert_close(
                actor[:, -4:-1], env.base_lin_vel / env.scales["base_lin_vel"]
            )
            torch.testing.assert_close(actor[:, -1:], clock / 3.0)
            torch.testing.assert_close(critic[:, -1:], clock / 3.0)
            torch.testing.assert_close(critic[:, -5:-1], env.foot_friction_obs)
            torch.testing.assert_close(critic[:, -6], env.foot_friction_obs.mean(dim=1))
            for name in ("cost_of_transport", "push_load", "glide_drag", "tilt_rate"):
                assert torch.isfinite(getattr(env, f"_reward_{name}")()).all(), name
        actor = Actor(66, 12, **class_to_dict(runner_cfg.actor)).eval()
        critic = Critic(75, **class_to_dict(runner_cfg.critic)).eval()
        with torch.inference_mode():
            actions = actor(env.get_states(actor_obs))
            values = critic(env.get_states(critic_obs))
        assert actions.shape == (6, 12) and values.shape == (6,)
        assert torch.isfinite(actions).all() and torch.isfinite(values).all()
    finally:
        backend.close()


def test_native_ground_is_redrawn_on_a_fresh_episode_not_on_a_continued_one():
    env, _, backend = _make_robust_env(num_envs=6)
    try:
        everyone = torch.ones(6, dtype=torch.bool)
        env.cfg.friction_changes.rate = 0.0
        before = env.foot_friction_obs.clone()
        # A timeout that keeps the physical state keeps the ground as well.
        env.cfg.osc.timeout_physical_reset_probability = 0.0
        env.timed_out[:] = True
        env.terminated[:] = False
        env._reset_idx(everyone)
        assert not env._physical_reset_mask.any()
        torch.testing.assert_close(env.foot_friction_obs, before, rtol=0, atol=0)
        # A fall starts a new episode on new ground.
        env.terminated[:] = torch.tensor([True, True, True, False, False, False])
        env._reset_idx(everyone)
        changed = (env.foot_friction_obs != before).any(dim=1)
        assert torch.equal(changed, env.terminated)
        torch.testing.assert_close(
            backend.contact_friction, env.foot_friction_obs.mean(dim=1)
        )
        torch.testing.assert_close(
            env.contact_friction_obs[:, 0], backend.contact_friction
        )
        # The estimate restarts from the new ground, times its new gain.
        env.cfg.sensing.friction_estimate_noise_std = 0.0
        env._refresh_friction_estimate(everyone)
        torch.testing.assert_close(
            env.friction_estimate_obs,
            env._friction_estimate_gain
            * env.foot_friction_obs.mean(dim=1, keepdim=True),
        )
    finally:
        backend.close()


def test_native_actuation_delay_holds_the_previous_target_in_the_motors():
    env, runner_cfg, backend = _make_robust_env(num_envs=3)
    try:
        env._actuation_delay_steps.copy_(torch.tensor([0, 3, 5]))
        applied = []
        compute = env._compute_torques

        def record():
            applied.append(env._joint_position_target().clone())
            return compute()

        env._compute_torques = record
        scale = env.scales["dof_pos_target"]
        first = torch.full((3, 12), 0.5)
        second = torch.full((3, 12), -0.25)
        env.set_states(runner_cfg.actor.actions, first)
        env.step()
        applied.clear()
        env.set_states(runner_cfg.actor.actions, second)
        env.step()
        assert len(applied) == 5
        for physics_step, target in enumerate(applied):
            torch.testing.assert_close(target[0], second[0] * scale)
            late = first if physics_step < 3 else second
            torch.testing.assert_close(target[1], late[1] * scale)
            torch.testing.assert_close(target[2], first[2] * scale)
    finally:
        backend.close()


def test_native_commands_cover_backward_walking():
    env, _, backend = _make_robust_env(num_envs=128)
    try:
        everyone = torch.ones(128, dtype=torch.bool)
        samples = []
        for _ in range(60):
            env._resample_commands(everyone)
            samples.append(env.commands[:, 0].clone())
        forward = torch.cat(samples)
        assert torch.all((forward >= -1.0) & (forward <= 2.0))
        assert forward.min() < -0.95 and forward.max() > 1.95
        moving = forward[forward != 0]
        # Uniform over -1.0 to 2.0: a third of moving commands go backward.
        assert abs((moving < 0).float().mean().item() - 1.0 / 3.0) < 0.03
        # Backward commands above the threshold run the moving clocks once the
        # clock speed target has followed them.
        env.commands[:] = torch.tensor([-0.6, 0.0, 0.0])
        env.clock_speed_obs[:] = -0.6
        assert not env._slow_command_mask().any()
        env.compute_osc_slope()
        assert torch.all(env.osc_omega > 1.2)
        # Walking backward at the reference speed earns the full tracking reward.
        env._speed_reference[:] = env._commanded_world_velocity()
        env.root_states[:, 7:9] = env._speed_reference
        torch.testing.assert_close(env._reward_tracking_lin_vel(), torch.ones(128))
    finally:
        backend.close()


def _stopping_env(num_envs, probability=1.0):
    """Hand-built environment with sliding stops configured."""
    env = _varied_env(num_envs)
    env.cfg.commands.sliding_stop_probability = probability
    env.cfg.commands.sliding_stop_min_speed = 0.5
    env.cfg.commands.sliding_stop_duration = 1.5
    env.command_ranges = {"yaw_vel": 0.5}
    return env


def test_sliding_stop_replaces_the_new_command_of_moving_robots_only():
    env = _stopping_env(6)
    env.root_states[:, 7:9] = torch.tensor(
        [
            [1.0, 0.0],  # forward
            [0.4, 0.0],  # too slow to slide
            [-0.8, 0.0],  # backward
            [0.0, 0.6],  # sideways
            [0.4, 0.4],  # diagonal: 0.57 m/s
            [2.0, 0.0],  # fast, but its command is not being redrawn
        ]
    )
    redrawn = torch.tensor([True, True, True, True, True, False])
    stopped = torch.tensor([True, False, True, True, True, False])
    env.commands = torch.tensor([1.0, 0.2, 0.3]).repeat(6, 1)
    env._sliding_stop_steps.fill_(7)
    env._command_sliding_stops(redrawn)
    assert torch.count_nonzero(env.commands[stopped, :2]) == 0
    turn = env.commands[stopped, 2]
    assert torch.all(turn.abs() <= 0.5) and turn.unique().numel() == 4
    torch.testing.assert_close(
        env.commands[~stopped], torch.tensor([1.0, 0.2, 0.3]).repeat(2, 1)
    )
    # 1.5 s of 0.01 s control steps; a redrawn command clears an older count.
    assert env._sliding_stop_steps.tolist() == [150, 0, 150, 150, 150, 7]


def test_sliding_stops_take_the_configured_share_with_turn_rates_over_the_range():
    env = _stopping_env(20000, probability=0.5)
    env.root_states[:, 7] = 1.0
    env.commands = torch.tensor([1.0, 0.0, 0.0]).repeat(20000, 1)
    env._command_sliding_stops(torch.ones(20000, dtype=torch.bool))
    stopped = env.commands[:, 0] == 0
    assert abs(stopped.float().mean().item() - 0.5) < 0.02
    assert torch.equal(stopped, env._sliding_stop_steps == 150)
    turn = env.commands[stopped, 2]
    assert turn.min() < -0.49 and turn.max() > 0.49 and abs(turn.mean()) < 0.01
    # One stop in five has a turn rate inside the standing threshold.
    assert abs((turn.abs() <= 0.1).float().mean().item() - 0.2) < 0.02


@pytest.mark.parametrize("held", [False, True])
def test_sliding_stops_are_off_without_a_probability_or_with_held_commands(held):
    env = _stopping_env(5, probability=1.0 if held else 0.0)
    if held:
        # Play and evaluation hold one command for the whole episode.
        env.cfg.commands.resampling_time = env.cfg.env.episode_length_s
    env.root_states[:, 7] = 1.5
    env.commands = torch.tensor([1.0, 0.2, 0.3]).repeat(5, 1)
    env._sliding_stop_steps.fill_(2)
    stream = env._variation_rng.get_state().clone()
    env._command_sliding_stops(torch.ones(5, dtype=torch.bool))
    env._count_down_sliding_stops()
    torch.testing.assert_close(env.commands, torch.tensor([1.0, 0.2, 0.3]).repeat(5, 1))
    assert env._sliding_stop_steps.tolist() == [2] * 5
    assert env._sliding_stop_ended is None
    assert torch.equal(env._variation_rng.get_state(), stream)


def test_sliding_stop_count_marks_the_step_on_which_a_stop_ends():
    env = _stopping_env(3)
    env._sliding_stop_steps.copy_(torch.tensor([3, 1, 0]))
    marks = []
    for _ in range(3):
        env._count_down_sliding_stops()
        marks.append(env._sliding_stop_ended.tolist())
    assert marks == [[False, True, False], [False] * 3, [True, False, False]]
    assert env._sliding_stop_steps.tolist() == [0, 0, 0]


def test_native_sliding_stop_holds_for_its_time_then_a_new_command_is_drawn():
    def configure(cfg):
        cfg.init_state.reset_mode = "reset_to_basic"
        cfg.commands.sliding_stop_probability = 1.0
        cfg.fast_slides.probability = 0.0

    env, runner_cfg, backend = _make_robust_env(num_envs=16, configure=configure)
    try:
        everyone = torch.ones(16, dtype=torch.bool)
        # Robots that start at rest are not told to stop.
        assert torch.count_nonzero(env._sliding_stop_steps) == 0
        env.root_states[:, 7] = 1.0
        env._resample_commands(everyone)
        assert torch.count_nonzero(env.commands[:, :2]) == 0
        assert env.commands[:, 2].unique().numel() == 16
        assert torch.all(env._sliding_stop_steps == 150)
        held = env.commands.clone()
        for _ in range(149):
            env.set_states(runner_cfg.actor.actions, torch.zeros(16, 12))
            env.step()
            assert torch.equal(env.commands, held)
        env.set_states(runner_cfg.actor.actions, torch.zeros(16, 12))
        env.step()
        # Standing still again, they get ordinary commands and no second stop.
        assert torch.all((env.commands != held).any(dim=1))
        assert torch.count_nonzero(env.commands[:, 0]) > 4
        assert torch.count_nonzero(env._sliding_stop_steps) == 0
    finally:
        backend.close()


def _reference_env(friction):
    """Hand-built environment with the speed reference on; one robot per friction."""
    env = _varied_env(len(friction))
    settings = env.cfg.reward_settings
    settings.speed_reference = True
    settings.speed_reference_gain = 0.5
    settings.speed_reference_brake = 0.9
    settings.speed_reference_max_gain = 2.0
    settings.speed_reference_max_brake = 5.0
    settings.velocity_tracking_tolerance = 0.5
    env.foot_friction_obs = torch.tensor(friction).reshape(-1, 1).repeat(1, 4)
    env.root_states[:, 6] = 1.0  # level, heading along world x
    env.commands = torch.zeros(len(friction), 3)
    return env


def test_speed_reference_speeds_up_at_half_of_friction_up_to_the_cap():
    env = _reference_env([0.1, 0.2, 1.0])
    env.commands[:, 0] = 2.0
    for _ in range(50):
        env._advance_speed_reference()
    # 0.5 s at 0.5 * friction * g, and at the 2.0 m/s^2 cap on grippy ground.
    torch.testing.assert_close(
        env._speed_reference[:, 0],
        torch.tensor([0.24525, 0.4905, 1.0]),
        rtol=0,
        atol=1e-4,
    )
    assert torch.count_nonzero(env._speed_reference[:, 1]) == 0
    for _ in range(150):
        env._advance_speed_reference()
    torch.testing.assert_close(
        env._speed_reference[:, 0], torch.tensor([0.981, 1.962, 2.0]), rtol=0, atol=1e-4
    )


def test_speed_reference_brakes_at_nine_tenths_of_friction_and_stops_at_the_command():
    env = _reference_env([0.1, 0.2, 1.0])
    env._speed_reference[:, 0] = 2.0
    for _ in range(50):
        env._advance_speed_reference()
    # 0.5 s at 0.9 * friction * g; the 5.0 m/s^2 cap ends the third at rest.
    torch.testing.assert_close(
        env._speed_reference[:, 0],
        torch.tensor([1.55855, 1.1171, 0.0]),
        rtol=0,
        atol=1e-4,
    )


def test_speed_reference_is_kept_in_the_world_frame():
    env = _reference_env([1.0, 1.0])
    half = 0.5**0.5
    env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, half, half])  # heading +y
    env.commands[0, 0] = 1.0
    # The second robot reverses: it brakes to rest, then speeds up backward.
    env._speed_reference[1, 0] = 1.0
    env.commands[1, 0] = -1.0
    for _ in range(50):
        env._advance_speed_reference()
    torch.testing.assert_close(
        env._speed_reference,
        torch.tensor([[0.0, 1.0], [-0.6, 0.0]]),
        rtol=0,
        atol=1e-4,
    )


def test_speed_reference_restarts_at_the_speed_the_body_has():
    env = _reference_env([0.5, 0.5])
    env.root_states[:, 7:9] = torch.tensor([[1.2, -0.3], [2.0, 0.4]])
    env._speed_reference[:] = 9.0
    env._advance_speed_reference(torch.tensor([True, False]))
    torch.testing.assert_close(
        env._speed_reference, torch.tensor([[1.2, -0.3], [9.0, 9.0]])
    )


def test_speed_reward_follows_the_reference_whatever_the_heading():
    env = _reference_env([0.1, 0.1])
    env._speed_reference[:, 0] = 1.0
    env.root_states[:, 7] = 0.5
    half = 0.5**0.5
    env.root_states[1, 3:7] = torch.tensor([0.0, 0.0, half, half])
    # One tolerance away: exp(-1), for the turned body too.
    torch.testing.assert_close(
        env._reward_tracking_lin_vel(), torch.full((2,), math.exp(-1.0))
    )


def test_speed_reference_is_off_unless_configured():
    env = _varied_env(3)
    env.commands = torch.ones(3, 3)
    env._advance_speed_reference()
    env._advance_speed_reference(torch.ones(3, dtype=torch.bool))
    assert not env._speed_reference_enabled()
    assert torch.count_nonzero(env._speed_reference) == 0
    assert not configs.Go2OscPushGlideCotCfg().reward_settings.__dict__.get(
        "speed_reference", False
    )


def test_native_speed_reference_starts_at_body_speed_and_brakes_with_friction():
    env, runner_cfg, backend = _make_robust_env(num_envs=16)
    try:
        torch.testing.assert_close(
            env._speed_reference, env.root_states[:, 7:9], rtol=0, atol=0.2
        )
        env._advance_speed_reference(torch.ones(16, dtype=torch.bool))
        env.commands.zero_()
        before = env._speed_reference.norm(dim=1)
        friction = env.foot_friction_obs.mean(dim=1)
        env._advance_speed_reference()
        braking = env._braking_coefficient
        assert torch.all(braking == 0.9)
        rate = (braking * friction * 9.81).clamp(max=5.0)
        expected = (before - rate * env.dt).clamp(min=0.0)
        torch.testing.assert_close(
            env._speed_reference.norm(dim=1), expected, rtol=0, atol=1e-5
        )
        assert torch.all(env._reward_tracking_lin_vel() <= 1.0)
    finally:
        backend.close()


def test_native_fast_slides_start_fast_on_low_friction_told_to_stop():
    def configure(cfg):
        cfg.init_state.reset_mode = "reset_to_basic"
        cfg.fast_slides.probability = 1.0
        cfg.friction_changes.split_probability = 1.0

    env, _, backend = _make_robust_env(num_envs=8, configure=configure)
    try:
        # Every robot is placed fresh after a fall, so every robot starts sliding.
        env.terminated[:] = True
        env._reset_idx(torch.ones(8, dtype=torch.bool))
        speed = env.root_states[:, 7:9].norm(dim=1)
        assert torch.all((speed >= 1.5) & (speed <= 2.5))
        assert speed.unique().numel() == 8
        # Along its heading: the body frame sees it moving straight forward.
        torch.testing.assert_close(env.base_lin_vel[:, 0], speed, rtol=0, atol=1e-5)
        assert torch.all(env.base_lin_vel[:, 1].abs() < 1e-5)
        friction = env.foot_friction_obs
        assert torch.all((friction >= 0.05) & (friction <= 0.3))
        # One value under every foot, even where draws would split the feet.
        assert torch.all(friction == friction[:, :1])
        torch.testing.assert_close(backend.contact_friction, friction[:, 0])
        assert torch.all(_foot_contact_friction_values(backend, env) == friction)
        # A sliding stop with a turn, held for its time.
        assert torch.count_nonzero(env.commands[:, :2]) == 0
        assert torch.all(env.commands[:, 2].abs() <= env.command_ranges["yaw_vel"])
        assert torch.all(env._sliding_stop_steps == 150)
        # Speed tracking starts from the body's speed and brakes from there.
        torch.testing.assert_close(env._speed_reference, env.root_states[:, 7:9])
        torch.testing.assert_close(
            env.friction_estimate_obs / env._friction_estimate_gain,
            friction[:, :1],
            rtol=0,
            atol=0.25,
        )
    finally:
        backend.close()


def _foot_contact_friction_values(backend, env):
    """Sliding friction of each robot's foot geoms in the model it simulates."""
    geoms, columns = backend._geoms_by_body(env.feet_indices)
    values = torch.zeros(env.num_envs, 4)
    for i in range(env.num_envs):
        friction = backend._model_for_env(i).geom_friction
        for geom, column in zip(geoms, columns):
            values[i, int(column)] = float(friction[int(geom), 0])
    return values


@pytest.mark.parametrize("ordinary_friction", [0.8, 0.9])
def test_fast_slide_ground_restores_only_on_the_next_physical_reset(ordinary_friction):
    def configure(cfg):
        cfg.init_state.reset_mode = "reset_to_basic"
        cfg.domain_randomization.startup.contact_friction_range = [
            ordinary_friction,
            ordinary_friction,
        ]
        cfg.friction_changes.rate = 0.0
        cfg.friction_changes.split_probability = 0.0
        cfg.fast_slides.probability = 1.0
        cfg.fast_slides.friction_range = [0.1, 0.1]

    env, _, backend = _make_robust_env(num_envs=4, configure=configure)
    try:
        slide_ground = torch.full((4, 4), 0.1)
        torch.testing.assert_close(
            _foot_contact_friction_values(backend, env), slide_ground
        )
        env.cfg.fast_slides.probability = 0.0
        everyone = torch.ones(4, dtype=torch.bool)

        # Logical timeout resets continue moving on the slide's ground.
        env.cfg.osc.timeout_physical_reset_probability = 0.0
        env.timed_out[:] = True
        env.terminated[:] = False
        env._reset_idx(everyone)
        assert not env._physical_reset_mask.any()
        torch.testing.assert_close(
            _foot_contact_friction_values(backend, env), slide_ground
        )

        # A fresh ordinary episode restores only the selected environments.
        fresh = torch.tensor([True, False, False, True])
        env.timed_out[:] = False
        env.terminated[:] = fresh
        env._reset_idx(fresh)
        expected = slide_ground.clone()
        expected[fresh] = ordinary_friction
        torch.testing.assert_close(
            _foot_contact_friction_values(backend, env), expected
        )
        torch.testing.assert_close(env.foot_friction_obs, expected)
        torch.testing.assert_close(backend.contact_friction, expected[:, 0])

        # Subsequent fresh resets do not reapply an obsolete slide override.
        env.terminated[:] = True
        env._reset_idx(everyone)
        torch.testing.assert_close(
            _foot_contact_friction_values(backend, env),
            torch.full((4, 4), ordinary_friction),
        )
    finally:
        backend.close()


@pytest.mark.parametrize("change", ["probability", "held"])
def test_fast_slides_are_off_without_a_probability_or_with_held_commands(change):
    def configure(cfg):
        cfg.init_state.reset_mode = "reset_to_basic"
        cfg.fast_slides.probability = 0.0 if change == "probability" else 1.0
        if change == "held":
            cfg.commands.resampling_time = cfg.env.episode_length_s

    env, _, backend = _make_robust_env(num_envs=4, configure=configure)
    try:
        assert torch.all(env.root_states[:, 7:9].abs() < 1e-9)
        assert torch.count_nonzero(env._sliding_stop_steps) == 0
    finally:
        backend.close()


def test_native_fast_slides_wait_for_a_fresh_episode():
    def configure(cfg):
        cfg.init_state.reset_mode = "reset_to_basic"
        cfg.fast_slides.probability = 1.0

    env, _, backend = _make_robust_env(num_envs=4, configure=configure)
    try:
        everyone = torch.ones(4, dtype=torch.bool)
        env.root_states[:, 7:9] = 0.3
        env.cfg.osc.timeout_physical_reset_probability = 0.0
        env.timed_out[:] = True
        env.terminated[:] = False
        env._reset_idx(everyone)
        # A continued timeout keeps its own motion.
        assert torch.all(env.root_states[:, 7:9] == 0.3)
    finally:
        backend.close()


def test_tilt_past_the_fall_limit_ends_the_episode():
    env, _, backend = _make_robust_env(num_envs=4)
    try:
        angles = torch.deg2rad(torch.tensor([0.0, 55.0, 65.0, 120.0]))
        env.projected_gravity[:] = torch.stack(
            (angles.sin(), torch.zeros(4), -angles.cos()), dim=1
        )
        env.terminated[:] = False
        env._check_terminations_and_timeouts()
        assert env.terminated.tolist() == [False, False, True, True]
        env.cfg.falls.max_tilt = None
        env.terminated[:] = False
        env._check_terminations_and_timeouts()
        assert not env.terminated.any()
    finally:
        backend.close()


def test_yaw_overshoot_charges_turning_faster_than_commanded():
    env = _varied_env(5)
    env.cfg.reward_settings.yaw_overshoot_tolerance = 0.25
    env.commands = torch.tensor(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    env.base_ang_vel = torch.zeros(5, 3)
    env.base_ang_vel[:, 2] = torch.tensor([0.6, 1.0, 3.0, -2.0, -0.2])
    expected = -torch.tensor([0.0, 0.25, 2.25, 1.25, 0.0]).square()
    torch.testing.assert_close(env._reward_yaw_overshoot(), expected)


def _quiet_clocks(cfg):
    cfg.osc.omega_noise_std = 0.0
    cfg.osc.coupling_noise_std = 0.0
    cfg.osc.process_noise_std = 0.0


def test_native_clock_speed_target_builds_at_the_friction_limit():
    env, _, backend = _make_robust_env(num_envs=4, configure=_quiet_clocks)
    try:
        g, dt = 9.81, env.dt
        env.friction_estimate_obs[:] = torch.tensor([[0.1], [0.1], [0.5], [0.01]])
        env._braking_coefficient[:] = 0.3
        env.clock_speed_obs[:] = torch.tensor([[0.0], [1.5], [0.0], [0.0]])
        env.commands[:] = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        env._advance_clock_speed()
        expected = torch.tensor(
            [
                0.5 * 0.1 * g * dt,  # speeding up at 0.5 mu g
                1.5 - 0.3 * 0.1 * g * dt,  # slowing at the drawn braking rate
                min(0.5 * 0.5 * g, 2.0) * dt,  # speed-up cap 2 m/s^2
                0.5 * 0.02 * g * dt,  # estimate below the floor
            ]
        )
        torch.testing.assert_close(env.clock_speed_obs[:, 0], expected)
        # It stops at the command instead of overshooting.
        env.friction_estimate_obs[:] = 0.1
        env.clock_speed_obs[:] = 1.999
        env.commands[:, 0] = 2.0
        env._advance_clock_speed()
        torch.testing.assert_close(env.clock_speed_obs[:, 0], torch.full((4,), 2.0))
        # Reversing slows through zero at the braking rate first.
        env.clock_speed_obs[:] = 0.5
        env.commands[:, 0] = -1.0
        env._advance_clock_speed()
        torch.testing.assert_close(
            env.clock_speed_obs[:, 0], torch.full((4,), 0.5 - 0.3 * 0.1 * g * dt)
        )
    finally:
        backend.close()


def test_native_clocks_follow_the_speed_target_not_the_command():
    env, _, backend = _make_robust_env(num_envs=4, configure=_quiet_clocks)
    try:
        env.commands[:] = torch.tensor(
            [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]
        )
        env.clock_speed_obs[:] = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        env.compute_osc_slope()
        # A fast command from rest still stands until the target moves; a
        # sliding robot told to stop keeps moving clocks at 1.5 + |v_t| Hz;
        # a turn in place moves at the base frequency.
        assert env._slow_command_mask()[:, 0].tolist() == [True, False, False, False]
        torch.testing.assert_close(
            env.osc_omega[:, 0], torch.tensor([1.0, 2.5, 2.5, 1.5])
        )
        assert env._moving()[:, 0].tolist() == [False, True, True, True]
    finally:
        backend.close()


def test_native_clock_speed_target_starts_at_the_body_speed():
    def configure(cfg):
        cfg.init_state.reset_mode = "reset_to_basic"
        cfg.fast_slides.probability = 1.0

    env, _, backend = _make_robust_env(num_envs=6, configure=configure)
    try:
        env.terminated[:] = True
        env._reset_idx(torch.ones(6, dtype=torch.bool))
        torch.testing.assert_close(env.clock_speed_obs, env.base_lin_vel[:, :1])
        assert torch.all(env.clock_speed_obs >= 1.5)
    finally:
        backend.close()


def test_native_braking_is_fixed_by_default_and_drawn_from_a_range():
    env, _, backend = _make_robust_env(num_envs=256)
    try:
        everyone = torch.ones(256, dtype=torch.bool)
        env._resample_commands(everyone)
        assert torch.all(env._braking_coefficient == 0.9)
        # A range is drawn with every command.
        env.cfg.commands.braking_range = [0.25, 0.6]
        env._resample_commands(everyone)
        first = env._braking_coefficient.clone()
        assert torch.all((first >= 0.25) & (first <= 0.6))
        assert first.unique().numel() > 200
        some = torch.arange(256) < 128
        env._resample_commands(some)
        assert torch.all(env._braking_coefficient[128:] == first[128:])
        assert torch.all(env._braking_coefficient[:128] != first[:128])
    finally:
        backend.close()


def test_native_standing_term_charges_motion_only_while_the_clocks_stand():
    env, _, backend = _make_robust_env(num_envs=3, configure=_quiet_clocks)
    try:
        env.commands.zero_()
        env.clock_speed_obs[:] = torch.tensor([[0.0], [0.0], [1.0]])
        env.root_states[:, 7:9] = torch.tensor([[0.2, 0.0], [0.0, 0.0], [1.0, 0.0]])
        env.base_ang_vel[:, 2] = torch.tensor([0.0, 0.4, 0.0])
        weight = env._backend.link_mass.sum(dim=1, keepdim=True) * 9.81
        loads = torch.tensor([[0.25] * 4, [0.0, 0.5, 0.5, 0.0], [0.25] * 4])
        forces = torch.zeros_like(env.contact_forces)
        forces[:, env.feet_indices, 2] = loads * weight
        env.contact_forces[:] = forces
        expected = -torch.tensor([0.04, 0.5 * 0.16 + 0.25 * 2, 0.0])
        torch.testing.assert_close(env._reward_standing(), expected)
    finally:
        backend.close()


def _carve_env(speed, turn, friction=0.1, forward=0.0):
    """Robust env with a world-frame reference along x and a stop command."""
    env, _, backend = _make_robust_env(num_envs=1, configure=_quiet_clocks)
    env.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    env.foot_friction_obs[:] = friction
    env._speed_reference[:] = torch.tensor([[speed, 0.0]])
    env.commands[:] = torch.tensor([[forward, 0.0, turn]])
    return env, backend


def test_native_carve_shares_the_braking_budget_with_the_turn():
    g, dt = 9.81, 0.01
    budget = 0.9 * 0.1 * g
    env, backend = _carve_env(speed=1.0, turn=0.5)
    try:
        env._advance_speed_reference()
        brake = (budget**2 - (1.0 * 0.5) ** 2) ** 0.5
        angle = 0.5 * dt
        expected = (1.0 - brake * dt) * torch.tensor([math.cos(angle), math.sin(angle)])
        torch.testing.assert_close(env._speed_reference[0], expected)
        torch.testing.assert_close(env._yaw_target(), torch.tensor([0.5]))
    finally:
        backend.close()


def test_native_carve_caps_the_turn_so_that_half_the_budget_still_brakes():
    g, dt = 9.81, 0.01
    budget = 0.9 * 0.1 * g
    env, backend = _carve_env(speed=2.0, turn=-0.5)
    try:
        env._advance_speed_reference()
        cap = budget * 0.75**0.5 / 2.0
        torch.testing.assert_close(env._yaw_target(), torch.tensor([-cap]))
        angle = -cap * dt
        expected = (2.0 - 0.5 * budget * dt) * torch.tensor(
            [math.cos(angle), math.sin(angle)]
        )
        torch.testing.assert_close(env._speed_reference[0], expected)
    finally:
        backend.close()


@pytest.mark.parametrize("forward,turn", [(0.0, 0.0), (1.0, 0.5)])
def test_native_no_carve_without_a_turn_or_while_walking(forward, turn):
    g, dt = 9.81, 0.01
    env, backend = _carve_env(speed=1.5, turn=turn, forward=forward)
    try:
        env._advance_speed_reference()
        # Straight stop: the whole budget brakes; a walking command is
        # followed as before. Either way the turn target is the command.
        if forward == 0.0:
            expected = torch.tensor([1.5 - 0.9 * 0.1 * g * dt, 0.0])
            torch.testing.assert_close(env._speed_reference[0], expected)
        torch.testing.assert_close(env._yaw_target(), torch.tensor([turn]))
    finally:
        backend.close()


def test_native_clock_speed_target_slows_less_while_carving():
    g, dt = 9.81, 0.01
    env, backend = _carve_env(speed=1.0, turn=0.5)
    try:
        env.friction_estimate_obs[:] = 0.1
        env.clock_speed_obs[:] = 1.0
        env._advance_clock_speed()
        budget = 0.9 * 0.1 * g
        brake = (budget**2 - 0.5**2) ** 0.5
        torch.testing.assert_close(
            env.clock_speed_obs[:, 0], torch.tensor([1.0 - brake * dt])
        )
    finally:
        backend.close()


def test_native_turn_rewards_follow_the_carved_turn():
    env, backend = _carve_env(speed=2.0, turn=0.5)
    try:
        env._advance_speed_reference()
        target = env._yaw_target()
        env.base_ang_vel[:, 2] = target
        torch.testing.assert_close(env._reward_tracking_ang_vel(), torch.ones(1))
        env.base_ang_vel[:, 2] = target + 0.35
        torch.testing.assert_close(
            env._reward_yaw_overshoot(), -torch.tensor([0.1]).square()
        )
    finally:
        backend.close()
