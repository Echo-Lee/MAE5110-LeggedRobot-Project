"""Decentralized phase-oscillator Go2 task from Zhang et al. (ICRA 2024)."""

import torch

from gym.envs.go2.go2 import Go2


class Go2DCPG(Go2):
    """Vanilla Go2 augmented with four uncoupled, GRF-feedback oscillators.

    The oscillators affect observations and phase-contact rewards only.  They do
    not generate joint references or otherwise alter the Go2 PD control path.
    """

    def _init_buffers(self):
        super()._init_buffers()
        osc_cfg = self.cfg.oscillator
        if osc_cfg.orc_toggle not in ("110", "111"):
            raise ValueError("cfg.oscillator.orc_toggle must be '110' or '111'")
        if osc_cfg.stop_threshold < 0.0:
            raise ValueError("cfg.oscillator.stop_threshold must be nonnegative")
        if osc_cfg.omega_noise_std < 0.0 or osc_cfg.coupling_noise_std < 0.0:
            raise ValueError(
                "oscillator parameter noise standard deviations must be nonnegative"
            )
        if osc_cfg.process_noise_std < 0.0:
            raise ValueError("cfg.oscillator.process_noise_std must be nonnegative")

        self.oscillators = torch.zeros(self.num_envs, 4, device=self.device)
        self.oscillator_obs = torch.zeros(self.num_envs, 8, device=self.device)
        self.oscillators_vel = torch.zeros_like(self.oscillators)
        self.normalized_grf = torch.zeros_like(self.oscillators)
        self.osc_omega = torch.zeros(self.num_envs, 1, device=self.device)
        self.osc_coupling = torch.zeros_like(self.osc_omega)
        self.osc_offset = torch.zeros_like(self.osc_omega)
        self._update_oscillator_observation()

    def _reset_system(self, reset_mask):
        super()._reset_system(reset_mask)
        count = int(reset_mask.sum().item())
        if count == 0:
            return

        # This is the official implementation's independent uniform reset plus
        # its optional common phase shift.  It imposes no inter-leg offset.
        self.oscillators[reset_mask] = self._sample_reset_phases(count)
        self.oscillators_vel[reset_mask] = 0.0
        self.normalized_grf[reset_mask] = 0.0
        self._update_oscillator_observation()

    def _sample_reset_phases(self, count):
        phase = torch.rand(count, 4, device=self.device) * (2.0 * torch.pi)
        if self.cfg.oscillator.init_with_common_offset:
            phase += torch.rand(count, 1, device=self.device) * (2.0 * torch.pi)
        return torch.remainder(phase, 2.0 * torch.pi)

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        # The official parameter noise is sampled once per 100 Hz control step.
        self._update_oscillator_parameters()

    def _post_physics_step(self):
        super()._post_physics_step()
        # LeggedRobot calls this immediately after backend.step().  Every
        # backend has refreshed contact_forces by then, so this consumes the
        # completed substep's GRF and integrates over the simulation dt.
        self._step_oscillators(self.dt / self.cfg.control.decimation)

    def _update_oscillator_parameters(self):
        cfg = self.cfg.oscillator
        speed = self.commands[:, 0:1].abs()
        stopped = speed <= cfg.stop_threshold

        moving_omega = torch.clamp(
            speed + cfg.omega_intercept,
            min=0.0,
            max=cfg.omega_max,
        )
        omega = torch.where(
            stopped,
            torch.full_like(speed, cfg.omega_stop),
            moving_omega,
        )
        coupling = torch.where(
            stopped,
            torch.full_like(speed, cfg.coupling_stop),
            torch.full_like(speed, cfg.coupling_moving),
        )
        self.osc_offset.copy_(
            torch.where(
                stopped,
                torch.full_like(speed, cfg.offset_stop),
                torch.zeros_like(speed),
            )
        )
        self.osc_omega.copy_(
            (omega + torch.randn_like(omega) * cfg.omega_noise_std).clamp_min(
                cfg.omega_min
            )
        )

        # ORC(110) must not retain any coupling perturbation.  Apply this after
        # sampling so its coefficient is identically zero for every environment.
        if cfg.orc_toggle == "110":
            self.osc_coupling.zero_()
        else:
            self.osc_coupling.copy_(
                (
                    coupling + torch.randn_like(coupling) * cfg.coupling_noise_std
                ).clamp_min(0.0)
            )

    def _compute_normalized_grf(self):
        foot_forces = self.contact_forces.index_select(1, self.feet_indices)
        force_magnitude = torch.linalg.vector_norm(foot_forces, dim=-1)
        robot_weight = 9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)
        return (force_magnitude / robot_weight).clamp_(0.0, 1.0)

    def _step_oscillators(self, sim_dt):
        self.normalized_grf.copy_(self._compute_normalized_grf())
        local_feedback = self.osc_coupling * (
            torch.cos(self.oscillators) + self.osc_offset
        )
        cycles_per_second = self.osc_omega - self.normalized_grf * local_feedback
        # Unlike parameter noise, official process noise is sampled per physics
        # substep.  Keep it separate so changing decimation remains explicit.
        cycles_per_second += (
            torch.randn_like(cycles_per_second) * self.cfg.oscillator.process_noise_std
        )
        self.oscillators_vel.copy_(2.0 * torch.pi * cycles_per_second)
        self.oscillators.add_(self.oscillators_vel * sim_dt).remainder_(2.0 * torch.pi)
        self._update_oscillator_observation()

    def _update_oscillator_observation(self):
        # Preserve the official [cos(phi_1..4), sin(phi_1..4)] ordering.
        self.oscillator_obs[:, :4] = torch.cos(self.oscillators)
        self.oscillator_obs[:, 4:] = torch.sin(self.oscillators)

    def _reward_swing_grf(self):
        return -torch.sum(
            torch.relu(torch.sin(self.oscillators)) * self.normalized_grf,
            dim=1,
        )

    def _reward_stance_grf(self):
        return torch.sum(
            torch.relu(-torch.sin(self.oscillators)) * self.normalized_grf,
            dim=1,
        )

    def get_reward_diagnostics(self, reward_weights):
        """Return unscaled-by-dt weighted reward terms for episodic logging."""
        return {
            "dcpg/swing_grf_weighted": (
                reward_weights["swing_grf"] * self._reward_swing_grf()
            ),
            "dcpg/stance_grf_weighted": (
                reward_weights["stance_grf"] * self._reward_stance_grf()
            ),
            "dcpg/tracking_lin_vel_weighted": (
                reward_weights["tracking_lin_vel"] * self._reward_tracking_lin_vel()
            ),
            "dcpg/tracking_ang_vel_weighted": (
                reward_weights["tracking_ang_vel"] * self._reward_tracking_ang_vel()
            ),
        }
