import torch

from gym.envs.base.legged_robot import LeggedRobot
# from learning.utils.logger.SaveStates import (
#     init_env_log_buffers,
# )


class Go2(LeggedRobot):
    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)

    def _init_buffers(self):
        super()._init_buffers()
        settings = self.cfg.reward_settings
        if settings.sigma_energy <= 0.0:
            raise ValueError("cfg.reward_settings.sigma_energy must be positive")
        if settings.min_velocity <= 0.0:
            raise ValueError("cfg.reward_settings.min_velocity must be positive")

        self._energy_dof_vel_before = torch.zeros(
            self.num_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self._gross_mechanical_work = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._forward_velocity_integral = torch.zeros_like(self._gross_mechanical_work)
        self._joint_velocity_squared_integral = torch.zeros_like(
            self._gross_mechanical_work
        )

        # Public per-environment diagnostics, averaged over the control step.
        self.gross_mechanical_power = torch.zeros_like(self._gross_mechanical_work)
        self.energy_reward = torch.zeros_like(self._gross_mechanical_work)
        self.weighted_energy_reward = torch.zeros_like(self._gross_mechanical_work)
        self.actual_forward_velocity = torch.zeros_like(self._gross_mechanical_work)
        self.velocity_tracking_error = torch.zeros_like(self._gross_mechanical_work)
        self.joint_velocity_rms = torch.zeros_like(self._gross_mechanical_work)

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self._gross_mechanical_work.zero_()
        self._forward_velocity_integral.zero_()
        self._joint_velocity_squared_integral.zero_()

    def _pre_compute_torques(self):
        super()._pre_compute_torques()
        self._energy_dof_vel_before.copy_(
            self.dof_vel.index_select(1, self.actuated_dof_indices)
        )

    def _post_physics_step(self):
        super()._post_physics_step()
        substep_dt = self.dt / self.cfg.control.decimation
        joint_velocity = self.dof_vel.index_select(1, self.actuated_dof_indices)
        midpoint_velocity = 0.5 * (self._energy_dof_vel_before + joint_velocity)
        gross_power = (self.torques * midpoint_velocity).abs().sum(dim=1)
        self._gross_mechanical_work.add_(gross_power * substep_dt)
        self._forward_velocity_integral.add_(self.base_lin_vel[:, 0] * substep_dt)
        self._joint_velocity_squared_integral.add_(
            joint_velocity.square().mean(dim=1) * substep_dt
        )

    def _post_decimation_step(self):
        self._update_energy_diagnostics()
        super()._post_decimation_step()

    def _update_energy_diagnostics(self):
        self.gross_mechanical_power.copy_(self._gross_mechanical_work / self.dt)
        self.actual_forward_velocity.copy_(self._forward_velocity_integral / self.dt)
        self.joint_velocity_rms.copy_(
            torch.sqrt(self._joint_velocity_squared_integral / self.dt)
        )
        velocity_scale = torch.clamp(
            self.actual_forward_velocity.abs(),
            min=self.cfg.reward_settings.min_velocity,
        )
        self.energy_reward.copy_(
            torch.exp(
                -self.gross_mechanical_power
                / (self.cfg.reward_settings.sigma_energy * velocity_scale)
            )
        )
        self.weighted_energy_reward.copy_(
            self.cfg.reward_settings.energy_weight * self.energy_reward
        )
        self.velocity_tracking_error.copy_(
            (self.commands[:, 0] - self.actual_forward_velocity).abs()
        )

    def _reward_energy(self):
        """Weighted gross-power regularization; runner weight stays at one."""
        return self.weighted_energy_reward

    def get_training_diagnostics(self):
        """Instantaneous Go2-only metrics consumed by the episodic logger."""
        return {
            "energy/gross_mechanical_power_w": self.gross_mechanical_power,
            "energy/reward": self.energy_reward,
            "energy/weighted_reward": self.weighted_energy_reward,
            "energy/actual_forward_velocity_m_s": self.actual_forward_velocity,
            "energy/velocity_tracking_error_m_s": self.velocity_tracking_error,
            "energy/joint_velocity_rms_rad_s": self.joint_velocity_rms,
        }

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity with squared exp"""
        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        error = self._sqrdexp(self.base_ang_vel[:, :2] / self.scales["base_ang_vel"])
        return torch.sum(error, dim=1)

    def _reward_orientation(self):
        """Penalize non-flat base orientation"""
        error = (
            torch.square(self.projected_gravity[:, :2])
            / self.cfg.reward_settings.tracking_sigma
        )
        return torch.sum(torch.exp(-error), dim=1)

    def _reward_min_base_height(self):
        """Squared exponential saturating at base_height target"""
        error = self.base_height - self.cfg.reward_settings.base_height_target
        error /= self.scales["base_height"]
        error = torch.clamp(error, max=0, min=None).flatten()
        return self._sqrdexp(error)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error / self.cfg.reward_settings.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2]) / 2.5
        )
        return self._sqrdexp(ang_vel_error)

    def _reward_dof_vel(self):
        """Penalize dof velocities"""
        return torch.sum(self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1)

    def _reward_dof_near_home(self):
        return torch.sum(
            self._sqrdexp(
                (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]
            ),
            dim=1,
        )
