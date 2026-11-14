"""Mechanical-power penalty: zero point, sign, units, and actuated routing."""

from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.legged_robot import LeggedRobot


def _power(torques, dof_vel, actuated=None):
    if actuated is None:
        actuated = torch.arange(torques.shape[1])
    task = SimpleNamespace(
        torques=torques, dof_vel=dof_vel, actuated_dof_indices=actuated
    )
    return LeggedRobot._reward_power(task)


def test_static_load_costs_no_power():
    # Holding a leg up or bearing weight with motionless joints does no work.
    torques = torch.full((2, 12), 15.0)
    assert torch.equal(_power(torques, torch.zeros(2, 12)), torch.zeros(2))


def test_power_is_negative_sum_of_absolute_joint_power_in_watts():
    torques = torch.tensor([[2.0, -3.0, 0.0], [1.0, 1.0, 1.0]])
    dof_vel = torch.tensor([[4.0, 5.0, 9.0], [-1.0, -1.0, -1.0]])
    # 8 W positive work + 15 W braking work count equally; 3 x 1 W.
    torch.testing.assert_close(_power(torques, dof_vel), torch.tensor([-23.0, -3.0]))


def test_power_uses_actuated_joint_velocities_only():
    torques = torch.tensor([[2.0, 3.0]])
    # Full-DOF velocities with a passive joint at index 1.
    dof_vel = torch.tensor([[1.0, 100.0, 2.0]])
    result = _power(torques, dof_vel, actuated=torch.tensor([0, 2]))
    torch.testing.assert_close(result, torch.tensor([-8.0]))


def test_go2_lowfric_enables_energy_terms_and_go2_does_not():
    from gym.envs.go2.go2_config import Go2RunnerCfg
    from gym.envs.go2.go2_lowfric_config import Go2LowFricRunnerCfg

    lowfric = Go2LowFricRunnerCfg().critic.reward.weights
    assert lowfric.power == pytest.approx(1.0e-3)
    assert lowfric.torques == pytest.approx(1.5e-3)
    go2 = Go2RunnerCfg().critic.reward.weights
    assert not hasattr(go2, "power")
    assert go2.torques == pytest.approx(5.0e-6)


def test_power_reward_runs_on_a_real_go2_lowfric_step():
    from gym.envs.go2.go2_lowfric import Go2LowFric
    from gym.envs.go2.go2_lowfric_config import Go2LowFricCfg, Go2LowFricRunnerCfg
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2LowFricCfg()
    cfg.env.num_envs = 4
    cfg.seed = 5
    task_registry.convert_frequencies_to_params(cfg, Go2LowFricRunnerCfg())
    env = Go2LowFric(cfg, "cpu", True, select_backend(cfg, "cpu", "mujoco"))
    try:
        env.dof_pos_target[:] = 0.3  # move away from home so joints do work
        env.step()
        power = env._reward_power()
        assert power.shape == (4,)
        assert torch.isfinite(power).all()
        assert (power < 0).all()
    finally:
        env._backend.close()
