"""Push/glide phase signals: gated load feedback, drag cost and traction ratios."""

import math
from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2osc import Go2OscDirectional, Go2OscPushGlide
from gym.envs.go2 import go2osc_config as configs
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from gym.utils.torch_quat import quat_from_euler_xyz
from learning.modules import Actor, Critic

WEIGHT = 98.1
GATE_SPEED = 0.1


def _env(num_envs=1):
    env = Go2OscPushGlide.__new__(Go2OscPushGlide)
    env.cfg = configs.Go2OscPushGlideCfg()
    env.num_envs = num_envs
    env.device = "cpu"
    env.commands = torch.tensor([1.0, 0.0, 0.0]).repeat(num_envs, 1)
    env.root_states = torch.zeros(num_envs, 13)
    env.root_states[:, 6] = 1.0
    env.base_quat = env.root_states[:, 3:7]
    env.base_ang_vel = torch.zeros(num_envs, 3)
    env.oscillators = torch.full((num_envs, 4), 1.5 * math.pi)
    env.oscillators_vel = torch.zeros_like(env.oscillators)
    env.oscillator_obs = torch.zeros(num_envs, 8)
    env.foot_traction_obs = torch.zeros(num_envs, 8)
    env.contact_friction_obs = torch.zeros(num_envs, 1)
    env.grf = torch.zeros_like(env.oscillators)
    env.osc_omega = torch.full((num_envs, 1), 2.5)
    env.osc_coupling = torch.ones_like(env.osc_omega)
    env.osc_offset = torch.zeros_like(env.osc_omega)
    env.contact_forces = torch.zeros(num_envs, 8, 3)
    env.feet_indices = torch.tensor([5, 1, 6, 3])
    env._backend = SimpleNamespace(
        link_mass=torch.full((num_envs, 2), 5.0),
        contact_friction=torch.ones(num_envs),
    )
    env._rigid_body_pos = torch.zeros(num_envs, 8, 3)
    env._rigid_body_quat = torch.zeros(num_envs, 8, 4)
    env._rigid_body_quat[..., 3] = 1.0
    env._rigid_body_lin_vel = torch.zeros(num_envs, 8, 3)
    env._rigid_body_ang_vel = torch.zeros(num_envs, 8, 3)
    env._foot_collision_offsets = torch.zeros(num_envs, 4, 3)
    env._foot_collision_radii = torch.full((4,), 0.02)
    env.contact_forces[:, env.feet_indices, 2] = 0.25 * WEIGHT
    return env


def _set_foot_velocities(env, velocities):
    env._rigid_body_lin_vel[:, env.feet_indices] = torch.tensor(
        velocities, dtype=torch.float
    )


def _gate(relative_speed):
    return 1 / (1 + math.exp(relative_speed / GATE_SPEED))


def test_gate_follows_foot_motion_relative_to_base_along_the_command():
    env = _env()
    env.root_states[:, 7] = 1.0
    # Planted, carried with the base, swinging forward, and moving sideways.
    _set_foot_velocities(env, [[0, 0, 0], [1, 0, 0], [1.2, 0, 0], [1, 3, 0.5]])
    gate = env._compute_push_gate()
    expected = torch.tensor([[_gate(-1.0), 0.5, _gate(0.2), 0.5]])
    torch.testing.assert_close(gate, expected)
    assert gate[0, 0] > 0.999 and gate[0, 2] < 0.13


def test_gate_reverses_for_backward_commands_and_follows_body_heading():
    env = _env(3)
    env.root_states[:, 7] = 1.0
    _set_foot_velocities(env, [[0.8, 0, 0]] * 4)
    env.commands[1] = torch.tensor([-1.0, 0.0, 0.0])
    # A quarter turn makes world -x the body's left: a left command sees it.
    env.root_states[2, 3:7] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    env.commands[2] = torch.tensor([0.6, 0.8, 0.0])
    gate = env._compute_push_gate()
    # Relative velocity is 0.2 m/s toward world -x. Body axes after the turn:
    # forward = world +y, left = world -x, so the command projects 0.8 * 0.2.
    expected = torch.tensor([[_gate(-0.2)], [_gate(0.2)], [_gate(0.16)]])
    torch.testing.assert_close(gate, expected.expand(-1, 4))


def test_rigid_base_rotation_and_translation_leave_the_gate_neutral():
    env = _env()
    env.root_states[0, :3] = torch.tensor([2.0, -1.0, 0.3])
    linear = torch.tensor([0.7, -0.4, 0.2])
    angular = torch.tensor([0.5, -1.5, 2.0])
    env.root_states[0, 7:10] = linear
    env.root_states[0, 10:13] = angular
    levers = torch.tensor(
        [[0.2, 0.15, -0.3], [0.2, -0.15, -0.3], [-0.2, 0.15, -0.3], [-0.2, -0.15, -0.3]]
    )
    offset = torch.tensor([0.0, 0.0, -0.02])
    env._foot_collision_offsets[:] = offset
    # Body origins sit so that sphere centres land on the chosen levers.
    env._rigid_body_pos[0, env.feet_indices] = env.root_states[0, :3] + levers - offset
    env._rigid_body_ang_vel[0, env.feet_indices] = angular
    origin_velocity = linear + torch.cross(
        angular.expand(4, 3), levers - offset, dim=-1
    )
    env._rigid_body_lin_vel[0, env.feet_indices] = origin_velocity
    assert origin_velocity.norm(dim=-1).min() > 0.5
    relative = env._foot_velocity_relative_to_base()
    torch.testing.assert_close(relative, torch.zeros(1, 4, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(env._compute_push_gate(), torch.full((1, 4), 0.5))


def _silence_oscillator_noise(env):
    for name in ("omega_noise_std", "coupling_noise_std", "process_noise_std"):
        setattr(env.cfg.osc, name, 0.0)


@pytest.mark.parametrize(
    ("command", "standing", "gated"),
    [
        ([0.0, 0.0, 0.0], True, False),
        ([0.2, 0.1, 0.1], True, False),
        ([-0.2, -0.1, -0.1], True, False),
        ([0.0, 0.0, 0.5], False, False),
        ([0.0, 0.0, -0.11], False, False),
        ([0.0, 0.3, 0.0], False, True),
        ([0.1, -0.11, 0.0], False, True),
        ([0.21, 0.0, 0.0], False, True),
        ([1.0, 0.3, 0.5], False, True),
    ],
)
def test_any_command_selects_moving_clocks_and_only_travel_enables_the_gate(
    command, standing, gated
):
    env = _env()
    env.commands[:] = torch.tensor(command)
    assert env._slow_command_mask().tolist() == [[standing]]
    assert env._moving().tolist() == [[not standing]]
    assert env._translating().tolist() == [[gated]]


def test_turn_in_place_runs_the_moving_clocks_on_ungated_load():
    env = _env(2)
    _silence_oscillator_noise(env)
    env.commands[:] = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    env.compute_osc_slope()
    # Equation (5) moving set at zero forward speed, against the standing set.
    torch.testing.assert_close(env.osc_omega, torch.tensor([[1.5], [1.0]]))
    torch.testing.assert_close(env.osc_coupling, torch.tensor([[1.0], [4.0]]))
    torch.testing.assert_close(env.osc_offset, torch.tensor([[0.0], [1.0]]))
    # Foot motion must not gate the load when there is no travel direction.
    env.root_states[:, 7] = 1.0
    _set_foot_velocities(env, [[0, 0, 0], [1, 0, 0], [3, 0, 0], [-2, 1, 0]])
    torch.testing.assert_close(env._compute_push_signal(), torch.full((2, 4), 0.25))
    env.oscillators[:] = torch.tensor([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi])
    turning = 1.5 - 0.25 * torch.tensor([1.0, 0.0, -1.0, 0.0])
    standing = 1.0 - 4.0 * 0.25 * torch.tensor([2.0, 1.0, 0.0, 1.0])
    drift = env._compute_oscillator_drift()
    torch.testing.assert_close(
        drift, torch.stack((turning, standing)), atol=1e-6, rtol=0
    )
    # The glide half charges drag for the turn and load for the stand.
    env.contact_forces[:, env.feet_indices, 0] = 0.05 * WEIGHT
    env.oscillators.fill_(0.5 * math.pi)
    torch.testing.assert_close(env._reward_glide_drag(), torch.tensor([-0.2, -1.0]))


def test_frequency_scale_reaches_turn_in_place_clocks():
    env = _env(2)
    env.cfg = configs.Go2OscPushGlideF050Cfg()
    _silence_oscillator_noise(env)
    env.commands[:] = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    env.compute_osc_slope()
    torch.testing.assert_close(env.osc_omega, torch.tensor([[0.75], [1.0]]))


def test_sidestep_gates_load_along_the_sideways_command():
    env = _env()
    env.commands[:] = torch.tensor([0.0, 0.3, 0.0])
    env.root_states[:, 8] = 0.3
    # Planted, carried with the base, swinging with it, and moving forward only.
    _set_foot_velocities(env, [[0, 0, 0], [0, 0.3, 0], [0, 0.5, 0], [2, 0.3, 0]])
    expected = 0.25 * torch.tensor([[_gate(-0.3), 0.5, _gate(0.2), 0.5]])
    torch.testing.assert_close(env._compute_push_signal(), expected)


def test_missing_steering_thresholds_keep_the_forward_only_rule():
    env = _env(3)
    env.cfg.osc.sideways_stop_threshold = None
    env.cfg.osc.turning_stop_threshold = None
    env.commands[:] = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.3, 0.0], [0.3, 0.0, 0.0]])
    assert env._slow_command_mask().flatten().tolist() == [True, True, False]
    assert env._moving().flatten().tolist() == [False, False, True]
    assert env._translating().flatten().tolist() == [False, False, True]
    _silence_oscillator_noise(env)
    env.compute_osc_slope()
    torch.testing.assert_close(env.osc_omega, torch.tensor([[1.0], [1.0], [1.8]]))


def test_tilt_rate_penalty_is_squared_roll_and_pitch_rate_only():
    env = _env(3)
    env.base_ang_vel = torch.tensor(
        [[0.3, -0.4, 5.0], [0.0, 0.0, 2.0], [1.0, 1.5, 0.0]]
    )
    expected = torch.tensor([-0.25, 0.0, -3.25])
    torch.testing.assert_close(env._reward_tilt_rate(), expected)
    # Thrashing costs no more than the cap.
    env.base_ang_vel[2] = torch.tensor([3.0, -4.0, 0.0])
    assert env.cfg.reward_settings.tilt_rate_cap == 4.0
    torch.testing.assert_close(env._reward_tilt_rate()[2], torch.tensor(-4.0))


@pytest.mark.parametrize(
    "command", [[0, 0, 0], [0.2, 0, 0], [-0.2, 0.1, 0.1], [0.1, -0.1, -0.1]]
)
def test_slow_branch_feeds_back_plain_normal_load(command):
    env = _env()
    env.commands[:] = torch.tensor(command, dtype=torch.float)
    env.root_states[:, 7] = 1.0
    _set_foot_velocities(env, [[0, 0, 0], [1, 0, 0], [3, 0, 0], [-2, 1, 0]])
    torch.testing.assert_close(env._compute_push_signal(), torch.full((1, 4), 0.25))


def test_moving_signal_is_normal_load_times_gate_and_ignores_tangential_force():
    env = _env()
    env.root_states[:, 7] = 1.0
    _set_foot_velocities(env, [[0, 0, 0], [1, 0, 0], [1.2, 0, 0], [0.9, 0, 0]])
    expected = 0.25 * torch.tensor([[_gate(-1.0), 0.5, _gate(0.2), _gate(-0.1)]])
    torch.testing.assert_close(env._compute_push_signal(), expected)
    env.contact_forces[:, env.feet_indices, 0] = 0.6 * WEIGHT
    env.contact_forces[:, env.feet_indices, 1] = -0.3 * WEIGHT
    torch.testing.assert_close(env._compute_push_signal(), expected)


def test_loads_are_clamped_to_one_body_weight_and_reject_pulling_contacts():
    env = _env()
    env.contact_forces[:, env.feet_indices] = (
        torch.tensor(
            [[0.0, 0.0, 3.0], [3.0, 4.0, 0.5], [0.3, 0.4, -0.2], [0.0, 0.0, 0.0]]
        )
        * WEIGHT
    )
    normal, tangent = env._foot_loads()
    torch.testing.assert_close(normal, torch.tensor([[1.0, 0.5, 0.0, 0.0]]))
    torch.testing.assert_close(tangent, torch.tensor([[0.0, 1.0, 0.5, 0.0]]))


def test_loads_use_live_per_environment_weight():
    env = _env(2)
    env._backend.link_mass[1].mul_(2)
    normal, _ = env._foot_loads()
    torch.testing.assert_close(normal, torch.tensor([[0.25] * 4, [0.125] * 4]))


def test_standing_fixed_point_stays_in_mid_stance_whatever_the_feet_do():
    # Equation (4) with omega=1, sigma=4, xi=1 and a quarter of body weight per
    # foot has its stable fixed point at 3*pi/2. Halving the signal with a
    # neutral gate would move it to the marginal point at 2*pi.
    env = _env(3)
    env.commands.zero_()
    env.osc_omega.fill_(1.0)
    env.osc_coupling.fill_(4.0)
    env.osc_offset.fill_(1.0)
    _set_foot_velocities(env, [[0, 0, 0], [0.4, 0, 0], [-0.4, 0, 0], [0, 0.4, 0]])
    env.oscillators[:] = torch.tensor([[1.5 * math.pi - 0.2], [1.5 * math.pi], [5.0]])
    drift = env._compute_oscillator_drift()
    torch.testing.assert_close(drift[1], torch.zeros(4), atol=1e-6, rtol=0)
    assert torch.all(drift[0] > 0.1) and torch.all(drift[2] < -0.1)


def test_moving_drift_is_equation_four_with_the_gated_load():
    env = _env()
    env.root_states[:, 7] = 1.0
    env.osc_offset.fill_(0.5)
    env.osc_coupling.fill_(2.0)
    _set_foot_velocities(env, [[0, 0, 0], [1, 0, 0], [0, 0, 0], [2, 0, 0]])
    env.oscillators[:] = torch.tensor([[0.0, math.pi, 1.5 * math.pi, 0.0]])
    signal = [0.25 * _gate(-1.0), 0.125, 0.25 * _gate(-1.0), 0.25 * _gate(1.0)]
    expected = torch.tensor(
        [
            [
                2.5 - 2.0 * signal[0] * 1.5,
                2.5 - 2.0 * signal[1] * -0.5,
                2.5 - 2.0 * signal[2] * 0.5,
                2.5 - 2.0 * signal[3] * 1.5,
            ]
        ]
    )
    torch.testing.assert_close(env._compute_oscillator_drift(), expected)
    # A loaded foot swinging forward leaves its clock essentially free.
    assert abs(expected[0, 3].item() - 2.5) < 1e-4


def test_drift_ignores_cached_force_magnitude_and_integrates_into_the_phase():
    env = _env()
    env.cfg.osc.process_noise_std = 0.0
    env.root_states[:, 7] = 1.0
    env.oscillators[:] = torch.tensor([[0.1, 2.0, 4.0, 6.0]])
    before = env.oscillators.clone()
    drift = env._compute_oscillator_drift()
    env.grf.fill_(123.0)
    torch.testing.assert_close(env._compute_oscillator_drift(), drift, rtol=0, atol=0)
    cached = env.oscillator_obs
    env._step_oscillators(0.002)
    expected = (before + 0.002 * 2 * math.pi * drift).remainder(2 * math.pi)
    torch.testing.assert_close(env.oscillators, expected)
    torch.testing.assert_close(env.oscillators_vel, 2 * math.pi * drift)
    assert env.oscillator_obs is cached
    torch.testing.assert_close(cached[:, :4], env.oscillators.cos())


def test_zero_coupling_and_unloaded_feet_reduce_to_a_clock():
    env = _env(2)
    env.osc_coupling[0] = 0.0
    env.contact_forces[1].zero_()
    env.oscillators[:] = torch.tensor([0.3, 1.9, 3.3, 5.1])
    torch.testing.assert_close(
        env._compute_oscillator_drift(), torch.full((2, 4), 2.5), rtol=0, atol=0
    )


def _trot_and_burst(env):
    """One cycle sampled across environments; both gaits carry one body weight.

    The trot holds 0.5 body weights on each foot through its push half. The
    burst puts 1 body weight on every foot for a quarter cycle at mid-push.
    """
    count = env.num_envs
    cycle = torch.arange(count) * (2 * math.pi / count)
    env.root_states[:, 7] = 1.0
    trot_phase = torch.stack((cycle, cycle + math.pi, cycle + math.pi, cycle), dim=1)
    trot_load = 0.5 * (trot_phase.sin() < 0)
    burst_phase = cycle.unsqueeze(1).expand(-1, 4)
    burst_load = 1.0 * ((burst_phase - 1.5 * math.pi).abs() < math.pi / 4)
    scores = []
    for phase, load in ((trot_phase, trot_load), (burst_phase, burst_load)):
        assert abs(load.sum(dim=1).mean().item() - 1.0) < 0.01
        env.oscillators.copy_(phase.remainder(2 * math.pi))
        env.contact_forces[:, env.feet_indices, 2] = load * WEIGHT
        scores.append(env._reward_push_load().mean().item())
    return scores


def test_capped_flat_push_reward_prefers_steady_support_to_a_burst():
    env = _env(720)
    trot, burst = _trot_and_burst(env)
    assert trot > burst + 0.25
    # Flat-topped window over the push half; the capped burst earns 4 * 0.5 / 4.
    assert abs(trot - 0.837) < 0.01 and abs(burst - 0.5) < 0.01
    env.cfg.reward_settings.push_load_cap = 1.0
    env.cfg.reward_settings.push_window_saturation = 1.0
    trot, burst = _trot_and_burst(env)
    # The plain sine pays the burst 0.90 against 2 / pi for the trot.
    assert burst > trot + 0.2
    assert abs(trot - 2 / math.pi) < 0.01 and abs(burst - 0.9) < 0.01


def test_push_credit_stops_at_the_cap_while_feedback_keeps_the_full_load():
    env = _env()
    env.root_states[:, 7] = 1.0
    env.oscillators.fill_(1.5 * math.pi)
    env.contact_forces[:, env.feet_indices, 2] = (
        torch.tensor([0.2, 0.5, 0.8, 1.6]) * WEIGHT
    )
    gate = _gate(-1.0)
    torch.testing.assert_close(env._reward_push_load(), torch.tensor([1.7 * gate]))
    signal = torch.tensor([[0.2, 0.5, 0.8, 1.0]]) * gate
    torch.testing.assert_close(env._compute_push_signal(), signal)
    drift = env._compute_oscillator_drift().clone()
    env.cfg.reward_settings.push_load_cap = 0.3
    torch.testing.assert_close(env._reward_push_load(), torch.tensor([1.1 * gate]))
    torch.testing.assert_close(env._compute_oscillator_drift(), drift, rtol=0, atol=0)


def test_push_window_is_flat_between_thirty_degree_ramps():
    env = _env(2)
    env.root_states[:, 7] = 1.0
    degrees = torch.tensor([[180.0, 195, 210, 240], [270, 330, 345, 90]])
    env.oscillators[:] = degrees * math.pi / 180
    ramp = math.sin(math.radians(15)) / 0.5
    weights = torch.tensor([[0.0, ramp, 1.0, 1.0], [1.0, 1.0, ramp, 0.0]])
    expected = (weights * 0.25 * _gate(-1.0)).sum(dim=1)
    torch.testing.assert_close(env._reward_push_load(), expected, atol=1e-6, rtol=1e-5)


def test_unit_cap_and_saturation_give_the_plain_sine_weighted_push_reward():
    env = _env()
    env.cfg.reward_settings.push_load_cap = 1.0
    env.cfg.reward_settings.push_window_saturation = 1.0
    env.root_states[:, 7] = 1.0
    _set_foot_velocities(env, [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]])
    env.oscillators[:] = torch.tensor(
        [[1.5 * math.pi, 0.5 * math.pi, 7 * math.pi / 6, 1.5 * math.pi]]
    )
    stroking = 0.25 * _gate(-1.0)
    expected = 1.0 * stroking + 0.0 + 0.5 * stroking + 1.0 * 0.125
    torch.testing.assert_close(env._reward_push_load(), torch.tensor([expected]))
    # A configuration saved before the fields existed scores the same.
    env.cfg.reward_settings = SimpleNamespace()
    torch.testing.assert_close(env._reward_push_load(), torch.tensor([expected]))


def test_glide_cost_scales_with_the_friction_a_dragged_foot_meets():
    # Same quarter-weight load on a sliding foot: Coulomb drag is mu * normal.
    env = _env(3)
    env.oscillators.fill_(0.5 * math.pi)
    for index, friction in enumerate((0.1, 1.0)):
        env.contact_forces[index, env.feet_indices, 0] = -friction * 0.25 * WEIGHT
    env.contact_forces[2, :, 2] = 0.0
    reward = env._reward_glide_drag()
    torch.testing.assert_close(reward, torch.tensor([-0.1, -1.0, 0.0]))
    # Thrust in the glide half costs the same as drag, in any direction.
    env.contact_forces[0, env.feet_indices, 0] = 0.0
    env.contact_forces[0, env.feet_indices, 1] = 0.025 * WEIGHT
    torch.testing.assert_close(env._reward_glide_drag()[0], torch.tensor(-0.1))


def test_glide_cost_applies_only_in_the_glide_half_of_the_phase():
    env = _env()
    env.contact_forces[:, env.feet_indices, 0] = 0.2 * WEIGHT
    env.oscillators[:] = torch.tensor(
        [[0.5 * math.pi, math.pi / 6, 1.5 * math.pi, math.pi]]
    )
    torch.testing.assert_close(
        env._reward_glide_drag(), torch.tensor([-(0.2 + 0.1)]), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("command", [[0, 0, 0], [0.2, 0, 0], [-0.1, 0.1, 0.1]])
def test_slow_branch_glide_cost_is_the_original_normal_load_penalty(command):
    env = _env()
    env.commands[:] = torch.tensor(command, dtype=torch.float)
    env.contact_forces[:, env.feet_indices, 0] = 0.05 * WEIGHT
    env.oscillators[:] = torch.tensor(
        [[0.5 * math.pi, math.pi / 6, 1.5 * math.pi, 0.0]]
    )
    torch.testing.assert_close(
        env._reward_glide_drag(), torch.tensor([-(0.25 + 0.125)]), atol=1e-6, rtol=1e-6
    )


def test_traction_observation_is_signed_tangential_over_normal_in_heading_frame():
    env = _env(2)
    env.contact_forces.fill_(999.0)
    env.contact_forces[:, env.feet_indices] = (
        torch.tensor(
            [[0.05, 0.0, 0.25], [-0.1, 0.0, 0.5], [0.0, 0.025, 0.25], [0.3, -0.3, 0.3]]
        )
        * WEIGHT
    )
    # Roll and pitch must not leak into the level heading frame.
    env.root_states[1, 3:7] = quat_from_euler_xyz(
        torch.tensor(0.3), torch.tensor(-0.2), torch.tensor(math.pi / 2)
    )
    cached = env.foot_traction_obs
    env._refresh_traction_observations()
    assert env.foot_traction_obs is cached
    level = torch.tensor([0.2, 0.0, -0.2, 0.0, 0.0, 0.1, 1.0, -1.0])
    # After a quarter turn: forward = world +y, left = world -x.
    turned = torch.tensor([0.0, -0.2, 0.0, 0.2, 0.1, 0.0, -1.0, -1.0])
    torch.testing.assert_close(cached, torch.stack((level, turned)), atol=1e-6, rtol=0)


def test_traction_observation_is_zero_without_load_and_bounded():
    env = _env()
    env.contact_forces[:, env.feet_indices] = (
        torch.tensor(
            [[0.2, 0.0, 0.04], [0.2, 0.0, 0.0], [0.2, 0.1, -0.3], [-1.0, 0.9, 0.1]]
        )
        * WEIGHT
    )
    env._refresh_traction_observations()
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 2.0]])
    torch.testing.assert_close(env.foot_traction_obs, expected)
    assert torch.isfinite(env.foot_traction_obs).all()


def test_selected_traction_refresh_preserves_other_environments():
    env = _env(3)
    env.foot_traction_obs.fill_(7.0)
    env.contact_forces[:, env.feet_indices, 0] = 0.05 * WEIGHT
    mask = torch.tensor([True, False, True])
    env._refresh_traction_observations(mask)
    expected = torch.tensor([0.2, 0.0] * 4)
    torch.testing.assert_close(env.foot_traction_obs[mask], expected.expand(2, -1))
    torch.testing.assert_close(env.foot_traction_obs[1], torch.full((8,), 7.0))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("push_gate_speed", 0.0),
        ("push_gate_speed", -0.1),
        ("push_gate_speed", float("nan")),
        ("push_gate_speed", float("inf")),
        ("traction_ratio_limit", 0.0),
        ("traction_ratio_limit", float("nan")),
        ("directional_gain", 0.3),
        ("directional_gain", 1.0),
    ],
)
def test_invalid_push_glide_configuration_fails_before_simulation(field, value):
    cfg = configs.Go2OscPushGlideCfg()
    setattr(cfg.osc, field, value)
    with pytest.raises(ValueError, match=field):
        Go2OscPushGlide._validate_push_glide_cfg(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("push_load_cap", 0.0),
        ("push_load_cap", 1.5),
        ("push_load_cap", float("nan")),
        ("push_window_saturation", 0.0),
        ("push_window_saturation", -0.5),
        ("push_window_saturation", 2.0),
        ("push_window_saturation", float("inf")),
    ],
)
def test_invalid_push_reward_shape_fails_before_simulation(field, value):
    cfg = configs.Go2OscPushGlideCfg()
    setattr(cfg.reward_settings, field, value)
    with pytest.raises(ValueError, match=field):
        Go2OscPushGlide._validate_push_glide_cfg(cfg)


@pytest.mark.parametrize("cap", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_tilt_rate_cap_fails_before_simulation(cap):
    cfg = configs.Go2OscPushGlideCfg()
    cfg.reward_settings.tilt_rate_cap = cap
    with pytest.raises(ValueError, match="tilt_rate_cap"):
        Go2OscPushGlide._validate_push_glide_cfg(cfg)


@pytest.mark.parametrize(
    ("sideways", "turning"),
    [(None, 0.1), (0.1, None), (-0.1, 0.1), (0.1, float("nan")), (True, 0.1)],
)
def test_invalid_steering_thresholds_fail_before_simulation(sideways, turning):
    cfg = configs.Go2OscPushGlideCfg()
    cfg.osc.sideways_stop_threshold = sideways
    cfg.osc.turning_stop_threshold = turning
    with pytest.raises(ValueError, match="sideways_stop_threshold"):
        Go2OscPushGlide._validate_push_glide_cfg(cfg)
    cfg.osc.sideways_stop_threshold = cfg.osc.turning_stop_threshold = None
    Go2OscPushGlide._validate_push_glide_cfg(cfg)


def test_push_glide_variant_changes_only_declared_signals_inputs_and_rewards():
    baseline = class_to_dict(configs.Go2OscORC111GRFDirectionalCfg())
    variant = class_to_dict(configs.Go2OscPushGlideCfg())
    assert baseline["osc"]["directional_gain"] == 1.0
    assert variant["osc"]["directional_gain"] == 0.0
    variant["osc"]["directional_gain"] = 1.0
    assert variant["osc"].pop("push_gate_speed") == GATE_SPEED
    assert variant["osc"].pop("traction_ratio_limit") == 2.0
    assert baseline["osc"]["stop_threshold"] == 0.5
    assert variant["osc"]["stop_threshold"] == 0.2
    variant["osc"]["stop_threshold"] = 0.5
    assert variant["osc"].pop("sideways_stop_threshold") == 0.1
    assert variant["osc"].pop("turning_stop_threshold") == 0.1
    assert variant["reward_settings"].pop("tilt_rate_cap") == 4.0
    assert variant["control"].pop("action_limit") == 10.0
    assert variant["scaling"].pop("foot_traction_obs") == 1.0
    assert variant["scaling"].pop("contact_friction_obs") == 1.0
    assert variant["commands"]["forward_uniform_range"] == [0.0, 1.5]
    variant["commands"]["forward_uniform_range"] = None
    assert variant["commands"]["include_standing"] is True
    for key, value in {
        "lin_vel_x": [0.0, 1.5],
        "lin_vel_y": 0.3,
        "yaw_vel": 0.5,
    }.items():
        assert variant["commands"]["ranges"][key] == value
        variant["commands"]["ranges"][key] = baseline["commands"]["ranges"][key]
    assert variant["reward_settings"].pop("push_load_cap") == 0.5
    assert variant["reward_settings"].pop("push_window_saturation") == 0.5
    assert baseline["reward_settings"]["yaw_tracking_tolerance"] is None
    assert variant["reward_settings"]["yaw_tracking_tolerance"] == 0.25
    variant["reward_settings"]["yaw_tracking_tolerance"] = None
    assert variant == baseline

    baseline = class_to_dict(configs.Go2OscORC111GRFDirectionalRunnerCfg())
    variant = class_to_dict(configs.Go2OscPushGlideRunnerCfg())
    assert variant["runner"]["experiment_name"] == "go2osc_push_glide"
    variant["runner"]["experiment_name"] = baseline["runner"]["experiment_name"]
    added = {
        "actor": ["foot_traction_obs"],
        "critic": ["foot_traction_obs", "contact_friction_obs"],
    }
    for role in ("actor", "critic"):
        assert baseline[role]["obs"][-1] == "foot_grf_angle"
        assert variant[role]["obs"] == baseline[role]["obs"][:-1] + added[role]
        variant[role]["obs"] = baseline[role]["obs"]
    weights = variant["critic"]["reward"]["weights"]
    assert weights.pop("push_load") == 5.0 and weights.pop("glide_drag") == 5.0
    assert weights["swing_grf"] == 0.0 and weights["stance_grf"] == 0.0
    weights["swing_grf"] = weights["stance_grf"] = 5.0
    assert baseline["critic"]["reward"]["weights"]["lin_vel_z"] == 0.0
    assert weights["lin_vel_z"] == 2.0
    weights["lin_vel_z"] = 0.0
    assert weights.pop("tilt_rate") == 1.0
    assert variant == baseline


@pytest.mark.parametrize(("suffix", "scale"), [("F075", 0.75), ("F050", 0.5)])
def test_frequency_variants_change_only_the_moving_scale_and_experiment(suffix, scale):
    baseline = class_to_dict(configs.Go2OscPushGlideCfg())
    variant = class_to_dict(getattr(configs, f"Go2OscPushGlide{suffix}Cfg")())
    assert baseline["osc"]["moving_frequency_scale"] == 1.0
    assert variant["osc"]["moving_frequency_scale"] == scale
    variant["osc"]["moving_frequency_scale"] = 1.0
    assert variant == baseline
    baseline = class_to_dict(configs.Go2OscPushGlideRunnerCfg())
    variant = class_to_dict(getattr(configs, f"Go2OscPushGlide{suffix}RunnerCfg")())
    name = f"go2osc_push_glide_{suffix.lower()}"
    assert variant["runner"]["experiment_name"] == name
    variant["runner"]["experiment_name"] = baseline["runner"]["experiment_name"]
    assert variant == baseline
    assert task_registry.get_task_class(name) is Go2OscPushGlide


def test_command_ranges_stay_within_the_grip_of_the_lowest_friction():
    # Turning at forward speed v and yaw rate w needs sideways acceleration
    # v * w, and ground of friction mu supplies at most mu * g.
    cfg = configs.Go2OscPushGlideCfg()
    lowest_friction = cfg.domain_randomization.startup.contact_friction_range[0]
    fastest = cfg.commands.forward_uniform_range[1]
    assert fastest * cfg.commands.ranges.yaw_vel <= 0.8 * lowest_friction * 9.81
    assert 0 < cfg.commands.ranges.lin_vel_y <= 0.25 * fastest
    assert cfg.commands.include_standing is True


def test_sampled_commands_cover_slow_walking_standing_and_bounded_steering():
    env, _, backend = _make_cpu_env(num_envs=128)
    try:
        everyone = torch.ones(128, dtype=torch.bool)
        samples = []
        for _ in range(80):
            env._resample_commands(everyone)
            samples.append(env.commands.clone())
        forward, sideways, turning = torch.cat(samples).unbind(dim=1)
        halted = forward == 0
        assert torch.all((forward >= 0.0) & (forward <= 1.5))
        assert torch.all(sideways.abs() <= 0.3) and torch.all(turning.abs() <= 0.5)
        assert sideways.abs().max() > 0.28 and turning.abs().max() > 0.47
        # The forward command is only ever dropped together with the sideways one.
        assert torch.count_nonzero(sideways[halted]) == 0
        threshold = env.cfg.osc.stop_threshold
        assert threshold == 0.2
        moving = forward.abs() > threshold
        steering = (sideways != 0) | (turning != 0)
        # Dropout probabilities 0.2, 0.2 and 0.1 keep 72% of uniform 0-1.5 draws.
        expected = {
            "standing": (halted & ~steering, 0.136),
            "turning in place": (halted & steering, 0.144),
            "below the stop threshold": (~halted & ~moving, 0.72 * 0.2 / 1.5),
            "moving": (moving, 0.72 * 1.3 / 1.5),
            "formerly untrained 0.2-0.5 m/s": (moving & (forward < 0.5), 0.144),
        }
        # Standing clocks need all three commands small: every all-zero sample,
        # a fifth of the turn-only ones and a few slow, barely steered ones.
        idle = ~moving & (sideways.abs() <= 0.1) & (turning.abs() <= 0.1)
        expected["standing clocks"] = (idle, 0.189)
        expected["turning in place on moving clocks"] = (halted & ~idle, 0.144 * 0.8)
        assert torch.equal(env._slow_command_mask()[:, 0], idle[-128:])
        for name, (selected, share) in expected.items():
            assert abs(selected.float().mean().item() - share) < 0.025, name
    finally:
        backend.close()


def _make_cpu_env(num_envs=3, height=None, gains=None, friction_range=None):
    cfg = configs.Go2OscPushGlideCfg()
    runner_cfg = configs.Go2OscPushGlideRunnerCfg()
    cfg.seed = runner_cfg.seed = 23
    cfg.env.num_envs = num_envs
    cfg.push_robots.toggle = False
    cfg.domain_randomization.startup.contact_friction_range = friction_range
    cfg.init_state.reset_mode = "reset_to_basic"
    if height is not None:
        cfg.init_state.pos = [0.0, 0.0, height]
    if gains is not None:
        cfg.control.stiffness = dict.fromkeys(("hip", "thigh", "calf"), gains[0])
        cfg.control.damping = dict.fromkeys(("hip", "thigh", "calf"), gains[1])
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = MuJocoCPUBackend()
    return Go2OscPushGlide(cfg, "cpu", True, backend), runner_cfg, backend


@pytest.fixture
def cpu_env():
    env, runner_cfg, backend = _make_cpu_env()
    try:
        yield env, runner_cfg
    finally:
        backend.close()


def test_new_task_observes_traction_ratios_and_steps_with_finite_rewards(cpu_env):
    env, runner_cfg = cpu_env
    assert task_registry.get_task_class("go2osc_push_glide") is Go2OscPushGlide
    assert issubclass(Go2OscPushGlide, Go2OscDirectional)
    assert env.get_states(runner_cfg.actor.obs).shape == (3, 61)
    assert env.get_states(runner_cfg.critic.obs).shape == (3, 70)
    assert "contact_friction_obs" not in runner_cfg.actor.obs
    cached = env.foot_traction_obs
    phase_before = env.oscillators.clone()
    for _ in range(5):
        env.step()
        assert env.foot_traction_obs is cached
        assert torch.isfinite(env.get_states(runner_cfg.actor.obs)).all()
        torch.testing.assert_close(env.get_states(runner_cfg.actor.obs)[:, -8:], cached)
        push, glide = env._reward_push_load(), env._reward_glide_drag()
        assert push.shape == glide.shape == (3,)
        assert torch.isfinite(push).all() and torch.isfinite(glide).all()
        assert torch.all(push >= 0) and torch.all(glide <= 0)
    assert not torch.allclose(env.oscillators, phase_before)
    before = cached.clone()
    mask = torch.tensor([True, False, True])
    env._reset_idx(mask)
    assert env.foot_traction_obs is cached
    torch.testing.assert_close(cached[~mask], before[~mask], rtol=0, atol=0)
    actor = Actor(61, 12, **class_to_dict(runner_cfg.actor)).eval()
    critic = Critic(70, **class_to_dict(runner_cfg.critic)).eval()
    with torch.inference_mode():
        actions = actor(env.get_states(runner_cfg.actor.obs))
        values = critic(env.get_states(runner_cfg.critic.obs))
    assert actions.shape == (3, 12) and values.shape == (3,)
    assert torch.isfinite(actions).all() and torch.isfinite(values).all()


def test_friction_observation_copies_applied_values_into_its_cached_buffer():
    env = _env(3)
    cached = env.contact_friction_obs
    env._backend.contact_friction = torch.tensor([0.1, 0.55, 1.0])
    env._refresh_friction_observation()
    assert env.contact_friction_obs is cached
    torch.testing.assert_close(cached, torch.tensor([[0.1], [0.55], [1.0]]))
    env._backend.contact_friction = torch.tensor([0.3, 0.3, 0.3])
    env._refresh_friction_observation(torch.tensor([True, False, True]))
    torch.testing.assert_close(cached, torch.tensor([[0.3], [0.55], [0.3]]))


def test_critic_alone_observes_the_friction_the_physics_applies():
    env, runner_cfg, backend = _make_cpu_env(num_envs=4, friction_range=[0.1, 1.0])
    try:
        cached = env.contact_friction_obs
        applied = backend.contact_friction.clone()
        assert torch.all((applied >= 0.1) & (applied <= 1.0))
        assert applied.unique().numel() == 4
        torch.testing.assert_close(cached[:, 0], applied, rtol=0, atol=0)
        critic = env.get_states(runner_cfg.critic.obs)
        actor = env.get_states(runner_cfg.actor.obs)
        torch.testing.assert_close(critic[:, -1], applied, rtol=0, atol=0)
        torch.testing.assert_close(critic[:, -9:-1], actor[:, -8:], rtol=0, atol=0)
        assert critic.shape[1] - actor.shape[1] == 9
        # A later change reaches selected robots on reset and all on the next step.
        changed = torch.tensor([0.2, 0.4, 0.6, 0.8])
        backend.set_contact_friction(torch.arange(4), changed)
        mask = torch.tensor([True, False, False, True])
        env._reset_idx(mask)
        assert env.contact_friction_obs is cached
        torch.testing.assert_close(cached[mask, 0], changed[mask], rtol=0, atol=0)
        torch.testing.assert_close(cached[~mask, 0], applied[~mask], rtol=0, atol=0)
        env.step()
        torch.testing.assert_close(cached[:, 0], changed, rtol=0, atol=0)
        assert env.contact_friction_obs is cached
    finally:
        backend.close()


def test_native_relative_foot_velocity_depends_on_joints_not_base_motion():
    env, _, backend = _make_cpu_env(num_envs=2, height=3.0)
    try:
        everyone = torch.ones(2, dtype=torch.bool)
        orientation = torch.tensor([0.2, -0.3, 0.4, math.sqrt(0.71)])
        joint_speed = torch.linspace(-1.5, 1.5, env.dof_vel.shape[1])

        def publish(base_linear, base_angular, joint_velocity):
            env.root_states.zero_()
            env.root_states[:, 2] = 3.0
            env.root_states[:, 3:7] = orientation
            env.root_states[:, 7:10] = torch.tensor(base_linear)
            env.root_states[:, 10:13] = torch.tensor(base_angular)
            env.dof_vel[:] = joint_velocity
            backend.reset_state(everyone)
            world, _ = env._foot_sliding_velocities()
            return env._foot_velocity_relative_to_base().clone(), world.clone()

        # Rigid motion: feet move fast in the world, not at all in the base.
        relative, world = publish([0.7, -0.4, 0.2], [1.1, -0.6, 1.4], 0.0)
        assert world.norm(dim=-1).min() > 0.3
        torch.testing.assert_close(
            relative, torch.zeros_like(relative), atol=2e-5, rtol=0
        )
        # Joint motion: the base twist must not change what the legs produce.
        still, world_still = publish([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], joint_speed)
        moving, world_moving = publish([0.7, -0.4, 0.2], [1.1, -0.6, 1.4], joint_speed)
        assert still.norm(dim=-1).min() > 0.05
        torch.testing.assert_close(still, world_still, atol=2e-5, rtol=0)
        torch.testing.assert_close(moving, still, atol=5e-5, rtol=0)
        assert (world_moving - world_still).norm(dim=-1).min() > 0.3
    finally:
        backend.close()


def test_native_turn_command_keeps_the_clocks_running_on_a_standing_robot():
    env, _, backend = _make_cpu_env(num_envs=2, gains=(100.0, 3.0))
    try:
        _silence_oscillator_noise(env)
        command = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
        for _ in range(400):
            env.commands[:] = command
            env.step()
        assert not env.terminated.any()
        normal, _ = env._foot_loads()
        assert torch.all(normal > 0.15)
        rate = env.oscillators_vel / (2 * math.pi)
        # Moving clocks: 1.5 Hz, shifted by at most the load each foot carries.
        assert torch.all((rate[0] - 1.5).abs() <= normal[0] + 1e-4)
        assert torch.all(rate[0] > 1.0)
        # Standing clocks: locked.
        assert rate[1].abs().max() < 0.01
    finally:
        backend.close()


def test_native_standing_robot_phase_locks_at_the_predicted_stance_point():
    # Zero actions hold the home posture only with firmer gains than training
    # uses; with the task's low gains the calves share the load.
    env, _, backend = _make_cpu_env(num_envs=2, gains=(100.0, 3.0))
    try:
        for name in ("omega_noise_std", "coupling_noise_std", "process_noise_std"):
            setattr(env.cfg.osc, name, 0.0)
        env.oscillators[:] = torch.tensor([[0.3, 1.7, 3.1, 5.9], [2.4, 4.0, 5.2, 0.9]])
        for _ in range(400):
            env.commands.zero_()
            env.step()
        assert not env.terminated.any()
        normal, _ = env._foot_loads()
        torch.testing.assert_close(normal.sum(dim=1), torch.ones(2), atol=0.01, rtol=0)
        assert torch.all(normal > 0.15)
        # Stable root of omega - sigma * N * (cos(phi) + xi) with [1, 4, 1].
        predicted = 2 * math.pi - torch.acos(1 / (4 * normal) - 1)
        torch.testing.assert_close(env.oscillators, predicted, atol=0.02, rtol=0)
        assert (env.oscillators - 1.5 * math.pi).abs().max() < 0.3
        assert env.oscillators_vel.abs().max() < 0.05
    finally:
        backend.close()
