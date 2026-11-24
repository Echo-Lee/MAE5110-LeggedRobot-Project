"""Uniform velocity commands preserve legacy sampling and the paper parameter rule."""

import math
from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2 import Go2
from gym.envs.go2.go2osc import Go2Osc, Go2OscDirectional
from gym.envs.go2 import go2osc_config as configs
from gym.utils.helpers import class_to_dict
from gym.utils.sampling import masked_update
from gym.utils.task_registry import task_registry


def _sampler(num_envs=8, cls=Go2Osc):
    env = cls.__new__(cls)
    env.cfg = configs.Go2OscORC111GRFDirectionalCfg()
    env.num_envs = num_envs
    env.device = "cpu"
    env.commands = torch.full((num_envs, 3), 123.0)
    env.command_ranges = {"lin_vel_x": [-2.0, 3.0], "lin_vel_y": 1.0, "yaw_vel": 3.0}
    return env


def _legacy_resample(env, mask):
    """Frozen pre-option sampler, including draw order and standing dropouts."""
    Go2._resample_commands(env, mask)
    centers = torch.tensor(env.cfg.commands.forward_centers, device=env.device)
    indices = torch.randint(len(centers), (env.num_envs,), device=env.device)
    forward = centers[indices] + env.cfg.commands.forward_noise_std * torch.randn(
        env.num_envs, device=env.device
    )
    masked_update(env.commands[:, 0], forward, mask)
    if env.cfg.commands.include_standing:
        for selected in (slice(1, None), slice(None, 2), slice(None)):
            probability = 0.1 if selected == slice(None) else 0.2
            drop = mask & (torch.rand(env.num_envs, device=env.device) < probability)
            env.commands[:, selected].masked_fill_(drop.unsqueeze(1), 0.0)


def test_default_uniform_range_is_disabled_and_resampling_remains_three_seconds():
    cfg = configs.Go2OscCfg()
    assert cfg.commands.forward_uniform_range is None
    assert cfg.commands.forward_boundary_probability == 0.0
    assert cfg.commands.resampling_time == 3.0


@pytest.mark.parametrize(
    "invalid",
    [
        1,
        True,
        "0.5,1.5",
        [],
        [0.5],
        [0, 1, 2],
        [1, 1],
        [2, 1],
        [False, 1],
        [0, True],
        ["0", 1],
        [0, None],
        [math.nan, 1],
        [0, math.inf],
        [-math.inf, 0],
        [[0], [1]],
    ],
)
@pytest.mark.parametrize("cls", [Go2Osc, Go2OscDirectional])
def test_invalid_uniform_ranges_fail_before_backend_setup(invalid, cls):
    cfg = configs.Go2OscORC111GRFDirectionalCfg()
    cfg.commands.forward_uniform_range = invalid
    with pytest.raises(ValueError, match="forward_uniform_range"):
        cls(cfg, "cpu", True, backend=None)


@pytest.mark.parametrize("bounds", [None, [0.5, 1.5], (-1, 1), [0, 2]])
def test_valid_uniform_ranges(bounds):
    cfg = configs.Go2OscCfg()
    cfg.commands.forward_uniform_range = bounds
    Go2Osc._validate_forward_uniform_range(cfg)


@pytest.mark.parametrize("invalid", [None, True, -0.1, 1.1, math.nan, math.inf, "0.2"])
def test_invalid_boundary_probability_fails_before_physics(invalid):
    cfg = configs.Go2OscCfg()
    cfg.commands.forward_uniform_range = [0.5, 1.5]
    cfg.commands.forward_boundary_probability = invalid
    with pytest.raises(ValueError, match="forward_boundary_probability"):
        Go2Osc(cfg, "cpu", True, backend=None)


def test_nonzero_endpoint_mixture_requires_uniform_bounds():
    cfg = configs.Go2OscCfg()
    cfg.commands.forward_boundary_probability = 0.2
    with pytest.raises(ValueError, match="requires forward_uniform_range"):
        Go2Osc(cfg, "cpu", True, backend=None)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("standing", [False, True])
def test_disabled_or_missing_option_preserves_sampler_and_rng_exactly(
    missing, standing
):
    env = _sampler(64)
    env.cfg.commands.include_standing = standing
    if missing:
        values = class_to_dict(env.cfg.commands)
        values.pop("forward_uniform_range")
        values.pop("forward_boundary_probability")
        env.cfg.commands = SimpleNamespace(**values)
    reference = _sampler(64)
    reference.cfg = env.cfg
    mask = torch.arange(64) % 3 == 0
    torch.manual_seed(29)
    initial = torch.random.get_rng_state()
    for selected in (mask, ~mask, torch.zeros_like(mask), mask):
        _legacy_resample(reference, selected)
    final = torch.random.get_rng_state()
    torch.random.set_rng_state(initial)
    for selected in (mask, ~mask, torch.zeros_like(mask), mask):
        env._resample_commands(selected)
    assert torch.equal(env.commands, reference.commands)
    assert torch.equal(torch.random.get_rng_state(), final)


def test_uniform_sampler_has_continuous_bounded_distribution_and_ignores_centers():
    env = _sampler(16384)
    env.cfg.commands.forward_uniform_range = [0.5, 1.5]
    env.cfg.commands.forward_centers = [999.0]
    env.cfg.commands.forward_noise_std = 1e9
    env.cfg.commands.include_standing = False
    torch.manual_seed(17)
    env._resample_commands(torch.ones(env.num_envs, dtype=torch.bool))
    forward = env.commands[:, 0]
    assert torch.all((forward >= 0.5) & (forward < 1.5))
    assert forward.unique().numel() > 10000
    assert forward.mean().item() == pytest.approx(1.0, abs=0.01)
    assert forward.std().item() == pytest.approx(1 / math.sqrt(12), abs=0.01)


@pytest.mark.parametrize("probability", [0.2, 1.0])
def test_boundary_mixture_assigns_half_its_probability_to_each_exact_endpoint(
    probability,
):
    env = _sampler(16384)
    env.cfg.commands.forward_uniform_range = [0.5, 1.5]
    env.cfg.commands.forward_boundary_probability = probability
    env.cfg.commands.include_standing = False
    mask = torch.arange(env.num_envs) % 2 == 0
    torch.manual_seed(31)
    env._resample_commands(mask)
    assert torch.equal(env.commands[~mask], torch.full((8192, 3), 123.0))
    forward = env.commands[mask, 0]
    assert torch.all((forward >= 0.5) & (forward <= 1.5))
    assert (forward == 0.5).float().mean().item() == pytest.approx(
        probability / 2, abs=0.015
    )
    assert (forward == 1.5).float().mean().item() == pytest.approx(
        probability / 2, abs=0.015
    )
    if probability == 1.0:
        assert torch.all((forward == 0.5) | (forward == 1.5))
    else:
        assert (
            (forward > 0.5) & (forward < 1.5)
        ).float().mean().item() == pytest.approx(0.8, abs=0.015)


def test_uniform_sampler_changes_only_selected_rows_and_keeps_parent_yaw_lateral():
    env = _sampler(64)
    env.cfg.commands.forward_uniform_range = [0.5, 1.5]
    env.cfg.commands.include_standing = False
    reference = _sampler(64)
    mask = torch.arange(64) % 2 == 0
    torch.manual_seed(37)
    initial = torch.random.get_rng_state()
    Go2._resample_commands(reference, mask)
    torch.random.set_rng_state(initial)
    env._resample_commands(mask)
    assert torch.equal(env.commands[~mask], torch.full((32, 3), 123.0))
    assert torch.equal(env.commands[:, 1:], reference.commands[:, 1:])
    assert torch.all((env.commands[mask, 0] >= 0.5) & (env.commands[mask, 0] < 1.5))
    previous = env.commands.clone()
    env._resample_commands(torch.zeros_like(mask))
    assert torch.equal(env.commands, previous)


def test_standing_dropout_is_still_explicitly_available_with_uniform_commands():
    env = _sampler(4096)
    env.cfg.commands.forward_uniform_range = [0.5, 1.5]
    env.cfg.commands.include_standing = True
    torch.manual_seed(41)
    env._resample_commands(torch.ones(env.num_envs, dtype=torch.bool))
    assert (env.commands[:, 0] == 0).any()
    assert (env.commands[:, 0] >= 0.5).any()
    assert torch.all((env.commands[:, 0] == 0) | (env.commands[:, 0] >= 0.5))


class _LegacyDirectional(Go2OscDirectional):
    _resample_commands = _legacy_resample


def _make_env(cls=Go2OscDirectional, *, uniform=False, missing=False, tuned=False):
    cfg = configs.Go2OscORC111GRFDirectionalCfg()
    runner = configs.Go2OscORC111GRFDirectionalRunnerCfg()
    cfg.seed = runner.seed = 47
    cfg.env.num_envs = 3
    cfg.push_robots.toggle = False
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.osc.directional_feedback_mode = "fractional_stance"
    cfg.osc.moving_frequency_scale = 1.0
    cfg.osc.coupling_moving = 1.0
    cfg.osc.omega_noise_std = cfg.osc.coupling_noise_std = 0.0
    cfg.osc.process_noise_std = 0.0
    if tuned:
        cfg.osc.moving_frequency_scale = 0.5
        cfg.osc.coupling_moving = 0.5
        cfg.osc.stop_threshold = 0.25
    if uniform:
        cfg.commands.forward_uniform_range = [0.5, 1.5]
        cfg.commands.include_standing = False
    if missing:
        values = class_to_dict(cfg.commands)
        values.pop("forward_uniform_range")
        values.pop("forward_boundary_probability")
        cfg.commands = SimpleNamespace(**values)
    # Accelerated test scheduling only; production default stays three seconds.
    cfg.commands.resampling_time = 0.03
    task_registry.convert_frequencies_to_params(cfg, runner)
    return cls(cfg, "cpu", True, MuJocoCPUBackend()), runner


@pytest.mark.parametrize("missing", [False, True])
def test_legacy_option_preserves_cpu_trajectory_observations_and_rng(missing):
    actions = torch.sin(torch.arange(12).float()).unsqueeze(0).repeat(3, 1) * 0.05
    snapshots = []
    for cls in (_LegacyDirectional, Go2OscDirectional):
        torch.manual_seed(47)
        env, runner = _make_env(cls, missing=missing)
        rows = []
        try:
            for frame in range(12):
                env.dof_pos_target.copy_(actions)
                if frame == 5:
                    env._reset_idx(torch.tensor([False, True, False]))
                env.step()
                rows.append(
                    tuple(
                        value.clone()
                        for value in (
                            env.root_states,
                            env.dof_pos,
                            env.dof_vel,
                            env.contact_forces,
                            env.oscillators,
                            env.oscillators_vel,
                            env.commands,
                            env.get_states(runner.actor.obs),
                            env.get_states(runner.critic.obs),
                        )
                    )
                )
            snapshots.append((rows, torch.random.get_rng_state()))
        finally:
            env._backend.close()
    for candidate, baseline in zip(snapshots[1][0], snapshots[0][0], strict=True):
        for actual, expected in zip(candidate, baseline, strict=True):
            assert torch.equal(actual, expected)
    assert torch.equal(snapshots[1][1], snapshots[0][1])


@pytest.mark.parametrize("tuned", [False, True])
def test_each_control_step_uses_current_command_rule_and_keeps_fractional_angles(tuned):
    torch.manual_seed(47)
    env, runner = _make_env(uniform=True, tuned=tuned)
    try:
        previous_commands = env.commands.clone()
        changed_frames = 0
        for _ in range(12):
            before = env.commands[:, :1].clone()
            env.step()
            slow = before.abs() <= env.cfg.osc.stop_threshold
            expected_omega = torch.where(
                slow,
                torch.ones_like(before),
                (1.5 + before.abs()).clamp(max=4) * env.cfg.osc.moving_frequency_scale,
            )
            assert torch.equal(env.osc_omega, expected_omega)
            assert torch.equal(
                env.osc_coupling, torch.where(slow, 4.0, env.cfg.osc.coupling_moving)
            )
            assert torch.equal(env.osc_offset, slow.float())
            assert env.cfg.osc.directional_feedback_mode == "fractional_stance"
            assert env.get_states(runner.actor.obs).shape == (3, 57)
            assert env.get_states(runner.critic.obs).shape == (3, 65)
            assert torch.all((env.commands[:, 0] >= 0.5) & (env.commands[:, 0] < 1.5))
            changed_frames += not torch.equal(previous_commands, env.commands)
            previous_commands = env.commands.clone()
        assert changed_frames == 4
        threshold = env.cfg.osc.stop_threshold
        env.commands[:, 0] = torch.tensor([threshold, threshold + 0.0001, 1.5])
        env._pre_decimation_step()
        expected_omega = torch.tensor(
            [
                1.0,
                (1.5 + threshold + 0.0001) * env.cfg.osc.moving_frequency_scale,
                3.0 * env.cfg.osc.moving_frequency_scale,
            ]
        )
        torch.testing.assert_close(env.osc_omega[:, 0], expected_omega)
        assert torch.equal(
            env.osc_coupling[:, 0],
            torch.tensor(
                [4.0, env.cfg.osc.coupling_moving, env.cfg.osc.coupling_moving]
            ),
        )
        assert torch.equal(env.osc_offset[:, 0], torch.tensor([1.0, 0.0, 0.0]))
        # A loaded midstance still gets the selected fractional angle slowdown.
        env.commands[:, 0] = 1.0
        env.compute_osc_slope()
        env.oscillators.fill_(3 * math.pi / 2)
        env.contact_forces.zero_()
        weight = env._backend.link_mass.sum(dim=1, keepdim=True) * 9.81
        env.contact_forces[:, env.feet_indices, 2] = weight / 4
        env.grf.copy_(env._compute_grf())
        baseline = Go2Osc._compute_oscillator_drift(env)
        expected = baseline * env.cfg.osc.directional_min_progress
        torch.testing.assert_close(env._compute_oscillator_drift(), expected)
    finally:
        env._backend.close()
