from types import SimpleNamespace

import torch

from gym.envs.go2.go2_dcpg import Go2DCPG
from gym.envs.go2.go2_dcpg_config import Go2DCPGCfg, Go2DCPGRunnerCfg


def _oscillator_cfg(*, orc_toggle="111", process_noise_std=0.0):
    return SimpleNamespace(
        orc_toggle=orc_toggle,
        init_with_common_offset=True,
        stop_threshold=0.5,
        omega_stop=1.0,
        coupling_stop=4.0,
        offset_stop=1.0,
        omega_intercept=1.5,
        omega_max=4.0,
        coupling_moving=1.0,
        omega_min=0.1,
        omega_noise_std=0.0,
        coupling_noise_std=0.0,
        process_noise_std=process_noise_std,
    )


def _task(*, orc_toggle="111", process_noise_std=0.0):
    task = Go2DCPG.__new__(Go2DCPG)
    task.device = "cpu"
    task.num_envs = 2
    task.cfg = SimpleNamespace(
        oscillator=_oscillator_cfg(
            orc_toggle=orc_toggle, process_noise_std=process_noise_std
        )
    )
    task.commands = torch.zeros(2, 3)
    task.oscillators = torch.zeros(2, 4)
    task.oscillator_obs = torch.zeros(2, 8)
    task.oscillators_vel = torch.zeros(2, 4)
    task.normalized_grf = torch.zeros(2, 4)
    task.osc_omega = torch.zeros(2, 1)
    task.osc_coupling = torch.zeros(2, 1)
    task.osc_offset = torch.zeros(2, 1)
    task.feet_indices = torch.tensor([0, 1, 2, 3])
    task.contact_forces = torch.zeros(2, 4, 3)
    task._backend = SimpleNamespace(link_mass=torch.tensor([[10.0], [20.0]]))
    return task


def test_schedule_matches_paper_stopping_and_moving_parameters():
    task = _task()
    task.commands[:, 0] = torch.tensor([0.5, 2.0])

    task._update_oscillator_parameters()

    torch.testing.assert_close(task.osc_omega[:, 0], torch.tensor([1.0, 3.5]))
    torch.testing.assert_close(task.osc_coupling[:, 0], torch.tensor([4.0, 1.0]))
    torch.testing.assert_close(task.osc_offset[:, 0], torch.tensor([1.0, 0.0]))


def test_orc_110_disables_all_coupling_including_noise():
    task = _task(orc_toggle="110")
    task.cfg.oscillator.coupling_noise_std = 10.0
    task.commands[:, 0] = torch.tensor([0.0, 2.0])

    task._update_oscillator_parameters()

    torch.testing.assert_close(task.osc_coupling, torch.zeros_like(task.osc_coupling))


def test_grf_uses_vector_norm_robot_weight_and_clamp():
    task = _task()
    task.contact_forces[0, 0] = torch.tensor([0.0, 0.0, 24.525])
    task.contact_forces[0, 1] = torch.tensor([0.0, 0.0, 196.2])
    task.contact_forces[1, 0] = torch.tensor([58.86, 0.0, 0.0])

    grf = task._compute_normalized_grf()

    torch.testing.assert_close(grf[0], torch.tensor([0.25, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(grf[1], torch.tensor([0.3, 0.0, 0.0, 0.0]))


def test_oscillator_dynamics_and_substep_timing():
    task = _task()
    task.oscillators[0, :2] = torch.tensor([0.0, torch.pi])
    task.osc_omega[:] = 2.0
    task.osc_coupling[:] = 1.0
    task.normalized_grf[0, :2] = 0.5
    task._compute_normalized_grf = lambda: task.normalized_grf

    task._step_oscillators(0.002)

    torch.testing.assert_close(
        task.oscillators_vel[0, :2], 2.0 * torch.pi * torch.tensor([1.5, 2.5])
    )
    torch.testing.assert_close(
        task.oscillators[0, :2],
        0.002 * 2.0 * torch.pi * torch.tensor([1.5, 2.5])
        + torch.tensor([0.0, torch.pi]),
    )

    task.oscillators.zero_()
    task.osc_omega[:] = 1.0
    task.osc_coupling.zero_()
    task.normalized_grf.zero_()
    for _ in range(5):
        task._step_oscillators(0.002)
    torch.testing.assert_close(task.oscillators, torch.full((2, 4), 0.02 * torch.pi))


def test_phase_reset_is_uniform_shape_correct_and_has_no_fixed_offsets():
    torch.manual_seed(7)
    task = _task()
    phases = task._sample_reset_phases(512)

    assert phases.shape == (512, 4)
    assert torch.all((phases >= 0.0) & (phases < 2.0 * torch.pi))
    assert torch.unique(phases[:, 0] - phases[:, 1]).numel() > 500


def test_phase_observation_shape_and_official_cos_sin_order():
    task = _task()
    task.oscillators[0] = torch.tensor([0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2])

    task._update_oscillator_observation()

    assert task.oscillator_obs.shape == (2, 8)
    torch.testing.assert_close(
        task.oscillator_obs[0],
        torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0]),
        atol=1e-6,
        rtol=0.0,
    )


def test_phase_contact_rewards_have_paper_signs_and_sum_identity():
    task = _task()
    task.oscillators[0] = torch.tensor([torch.pi / 2, 3 * torch.pi / 2, 0.0, torch.pi])
    task.normalized_grf[0] = torch.tensor([0.2, 0.3, 0.4, 0.5])

    swing = task._reward_swing_grf()
    stance = task._reward_stance_grf()
    expected = -torch.sum(task.normalized_grf * torch.sin(task.oscillators), dim=1)

    torch.testing.assert_close(swing[0], torch.tensor(-0.2))
    torch.testing.assert_close(stance[0], torch.tensor(0.3))
    torch.testing.assert_close(swing + stance, expected)


def test_dcpg_observation_dimensions_and_reward_weights():
    env_cfg = Go2DCPGCfg()
    runner_cfg = Go2DCPGRunnerCfg()

    assert env_cfg.control.ctrl_frequency == 100
    assert env_cfg.control.desired_sim_frequency == 500
    assert runner_cfg.actor.obs[-1] == "oscillator_obs"
    assert runner_cfg.critic.obs[-2:] == ["oscillator_obs", "oscillators_vel"]
    assert runner_cfg.critic.reward.weights.swing_grf == 5.0
    assert runner_cfg.critic.reward.weights.stance_grf == 5.0
    assert runner_cfg.critic.reward.weights.energy == 0.0
    assert len(runner_cfg.actor.obs) == 7
    assert len(runner_cfg.critic.obs) == 10
