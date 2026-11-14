from types import SimpleNamespace

import torch

from gym.envs.go2.go2 import Go2


def _task(*, energy_weight=0.0, sigma_energy=200.0, min_velocity=0.1):
    task = Go2.__new__(Go2)
    task.cfg = SimpleNamespace(
        reward_settings=SimpleNamespace(
            energy_weight=energy_weight,
            sigma_energy=sigma_energy,
            min_velocity=min_velocity,
        )
    )
    task.dt = 0.01
    task._gross_mechanical_work = torch.zeros(2)
    task._forward_velocity_integral = torch.zeros(2)
    task._joint_velocity_squared_integral = torch.zeros(2)
    task.gross_mechanical_power = torch.zeros(2)
    task.energy_reward = torch.zeros(2)
    task.weighted_energy_reward = torch.zeros(2)
    task.actual_forward_velocity = torch.zeros(2)
    task.velocity_tracking_error = torch.zeros(2)
    task.joint_velocity_rms = torch.zeros(2)
    task.commands = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    return task


def test_energy_reward_decreases_with_gross_power_at_fixed_velocity():
    task = _task(energy_weight=2.0, sigma_energy=10.0)
    task._gross_mechanical_work[:] = torch.tensor([0.1, 0.2])
    task._forward_velocity_integral[:] = torch.tensor([1.0, 1.0]) * task.dt
    task._joint_velocity_squared_integral[:] = torch.tensor([4.0, 9.0]) * task.dt

    task._update_energy_diagnostics()

    torch.testing.assert_close(task.gross_mechanical_power, torch.tensor([10.0, 20.0]))
    assert task.energy_reward[0] > task.energy_reward[1]
    torch.testing.assert_close(task.weighted_energy_reward, 2.0 * task.energy_reward)
    torch.testing.assert_close(task.joint_velocity_rms, torch.tensor([2.0, 3.0]))


def test_zero_energy_weight_preserves_zero_reward_contribution():
    task = _task(energy_weight=0.0)
    task._gross_mechanical_work[:] = torch.tensor([0.1, 0.1])
    task._forward_velocity_integral[:] = torch.tensor([0.0, -1.0]) * task.dt
    task._joint_velocity_squared_integral[:] = task.dt

    task._update_energy_diagnostics()

    assert torch.all(task.energy_reward > 0.0)
    torch.testing.assert_close(task._reward_energy(), torch.zeros(2))
    torch.testing.assert_close(task.actual_forward_velocity, torch.tensor([0.0, -1.0]))
    torch.testing.assert_close(task.velocity_tracking_error, torch.tensor([1.0, 2.0]))


def test_diagnostics_expose_per_environment_energy_metrics():
    task = _task()
    diagnostics = task.get_training_diagnostics()

    assert set(diagnostics) == {
        "energy/gross_mechanical_power_w",
        "energy/reward",
        "energy/weighted_reward",
        "energy/actual_forward_velocity_m_s",
        "energy/velocity_tracking_error_m_s",
        "energy/joint_velocity_rms_rad_s",
    }
    assert all(value.shape == (2,) for value in diagnostics.values())
