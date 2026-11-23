"""Go2 joint-position control with decentralized GRF-driven phase oscillators."""

import math
import xml.etree.ElementTree as ET

import torch

from gym import GYM_ROOT_DIR
from gym.envs.base.domain_randomization import contact_friction_range
from gym.envs.go2.go2 import Go2
from gym.envs.go2.go2osc_joint_targets import (
    joint_limit_margin_reward,
    joint_target_bounds,
    keep_joint_targets,
    limit_joint_targets,
    standing_saturation_reward,
    target_saturation_reward,
)
from gym.envs.go2.go2osc_speed_target import Go2OscSpeedTargetMixin
from gym.utils.sampling import masked_update
from gym.utils.torch_quat import quat_apply


class Go2Osc(Go2):
    """One oscillator per canonical Go2 foot: FL, FR, RL, RR."""

    def __init__(self, cfg, device, headless, backend):
        self._validate_forward_uniform_range(cfg)
        probability = cfg.osc.timeout_physical_reset_probability
        if not 0.0 <= probability <= 1.0:
            raise ValueError("timeout_physical_reset_probability must lie in [0, 1]")
        self._explicit_reset = False
        super().__init__(cfg, device, headless, backend)

    @staticmethod
    def _validate_forward_uniform_range(cfg):
        bounds = getattr(cfg.commands, "forward_uniform_range", None)
        probability = getattr(cfg.commands, "forward_boundary_probability", 0.0)
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError(
                "forward_boundary_probability must be finite and in [0, 1]"
            )
        if bounds is None:
            if probability > 0.0:
                raise ValueError(
                    "forward_boundary_probability requires forward_uniform_range"
                )
            return
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in bounds
            )
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(
                "forward_uniform_range must be None or two finite, increasing bounds"
            )

    def reset(self):
        """A public reset starts a fresh physical episode, regardless of flags."""
        self._explicit_reset = True
        try:
            super().reset()
        finally:
            self._explicit_reset = False

    def _reset_system(self, reset_mask):
        # Only pure timeout resets may retain physical state. Logical resets
        # (phase, command, history, counters) still cover the full request mask.
        if self._explicit_reset:
            physical_mask = reset_mask
        else:
            restart = torch.rand(self.num_envs, device=self.device) < (
                self.cfg.osc.timeout_physical_reset_probability
            )
            pure_timeout = self.timed_out & ~self.terminated
            physical_mask = reset_mask & (~pure_timeout | restart)
        self._physical_reset_mask.copy_(physical_mask)
        super()._reset_system(physical_mask)

    def _init_buffers(self):
        super()._init_buffers()
        self._physical_reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.oscillators = torch.zeros(self.num_envs, 4, device=self.device)
        self.oscillators_vel = torch.zeros_like(self.oscillators)
        self.oscillator_obs = torch.zeros(self.num_envs, 8, device=self.device)
        self.grf = torch.zeros_like(self.oscillators)
        self.osc_omega = torch.zeros(self.num_envs, 1, device=self.device)
        self.osc_coupling = torch.zeros_like(self.osc_omega)
        self.osc_offset = torch.zeros_like(self.osc_omega)
        self.hip_dof_indices = torch.tensor(
            self.robot_layout.dof_group_indices("hip_abduction"),
            dtype=torch.long,
            device=self.device,
        )
        self._refresh_oscillator_observations()

    def _refresh_oscillator_observations(self):
        # Preserve cached observation references across integration and resets.
        self.oscillator_obs[:, :4].copy_(torch.cos(self.oscillators))
        self.oscillator_obs[:, 4:].copy_(torch.sin(self.oscillators))

    def _reset_oscillators(self, reset_mask):
        patterns = {
            "standing": [1.5 * math.pi] * 4,
            "trot": [0.0, math.pi, math.pi, 0.0],
            "pace": [0.0, math.pi, 0.0, math.pi],
            "bound": [0.0, 0.0, math.pi, math.pi],
            "pronk": [0.0] * 4,
        }
        if self.cfg.osc.init_to == "random":
            candidate = 2 * math.pi * torch.rand_like(self.oscillators)
        elif self.cfg.osc.init_to in patterns:
            candidate = torch.tensor(
                patterns[self.cfg.osc.init_to], device=self.device
            ).expand_as(self.oscillators)
        else:
            raise ValueError(
                f"Unknown oscillator initialization: {self.cfg.osc.init_to}"
            )
        if self.cfg.osc.init_w_offset:
            candidate = candidate + 2 * math.pi * torch.rand(
                self.num_envs, 1, device=self.device
            )
        masked_update(self.oscillators, candidate.remainder(2 * math.pi), reset_mask)
        self.oscillators_vel.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self._refresh_oscillator_observations()

    def _reset_idx(self, reset_mask):
        # The parent performs the single atomic physics-state reset.
        previous_target = self.dof_pos_target.clone()
        super()._reset_idx(reset_mask)
        self._reset_oscillators(reset_mask)
        # Home posture has zero residual. Continued timeout trajectories retain
        # their current target so a logical reset does not command a new pose.
        self.dof_pos_target.masked_fill_(self._physical_reset_mask.unsqueeze(1), 0.0)
        continued = reset_mask & ~self._physical_reset_mask
        masked_update(self.dof_pos_target, previous_target, continued)
        self.dof_pos_history.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.grf.copy_(self._compute_grf())
        self.compute_osc_slope(reset_mask)

    def _clock_forward_command(self):
        """Forward speed [m/s] the clock schedule of Equation (5) follows.

        The commanded forward speed; variants may substitute a target that
        approaches the command gradually.
        """
        return self.commands[:, :1]

    def _slow_command_mask(self):
        """Environments on the standing branch of Equation (5)."""
        return self._clock_forward_command().abs() <= self.cfg.osc.stop_threshold

    def compute_osc_slope(self, update_mask=None):
        """Equation (5), with optional explicitly configured parameter noise."""
        speed = self._clock_forward_command().abs()
        slow = self._slow_command_mask()
        omega = torch.where(
            slow,
            torch.full_like(speed, self.cfg.osc.omega_stop),
            (self.cfg.osc.omega_offset + speed).clamp(max=self.cfg.osc.omega_max),
        )
        coupling = torch.where(
            slow,
            torch.full_like(speed, self.cfg.osc.coupling_stop),
            torch.full_like(speed, self.cfg.osc.coupling_moving),
        )
        offset = slow.to(speed.dtype) * self.cfg.osc.stance_offset
        omega = (omega + self.cfg.osc.omega_noise_std * torch.randn_like(omega)).clamp(
            min=0.1
        )
        if self.cfg.osc.coupling:
            coupling = (
                coupling + self.cfg.osc.coupling_noise_std * torch.randn_like(coupling)
            ).clamp(min=0.0)
        else:
            # C=0 is an intervention, including the low-speed branch and jitter.
            coupling = torch.zeros_like(coupling)
        if update_mask is None:
            self.osc_omega.copy_(omega)
            self.osc_coupling.copy_(coupling)
            self.osc_offset.copy_(offset)
        else:
            masked_update(self.osc_omega, omega, update_mask)
            masked_update(self.osc_coupling, coupling, update_mask)
            masked_update(self.osc_offset, offset, update_mask)

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self.compute_osc_slope()

    def _post_physics_step(self):
        super()._post_physics_step()
        self._step_oscillators(self.dt / self.cfg.control.decimation)

    def _step_oscillators(self, dt):
        """Integrate Equation (4) using freshly refreshed world-frame forces."""
        self.grf.copy_(self._compute_grf())
        velocity = self._compute_oscillator_drift()
        velocity = velocity + self.cfg.osc.process_noise_std * torch.randn_like(
            velocity
        )
        self.oscillators_vel.copy_(2 * math.pi * velocity)
        self.oscillators.add_(dt * self.oscillators_vel).remainder_(2 * math.pi)
        self._refresh_oscillator_observations()

    def _compute_oscillator_drift(self):
        """Deterministic phase drift in cycles/s, before process noise."""
        local_feedback = self.osc_coupling * (
            torch.cos(self.oscillators) + self.osc_offset
        )
        return self.osc_omega - self.grf * local_feedback

    def _compute_grf(self, grf_norm=True):
        forces = self.contact_forces.index_select(1, self.feet_indices)
        magnitude = torch.linalg.vector_norm(forces, dim=-1)
        if not grf_norm:
            return magnitude
        # link_mass is live and includes current per-environment randomization.
        weight = self._backend.link_mass.sum(dim=1, keepdim=True) * 9.81
        return (magnitude / weight).clamp(min=0.0, max=1.0)

    def _resample_commands(self, command_mask):
        super()._resample_commands(command_mask)
        bounds = getattr(self.cfg.commands, "forward_uniform_range", None)
        if bounds is None:
            # Keep the original draws and arithmetic for saved configurations.
            centers = torch.tensor(
                self.cfg.commands.forward_centers, device=self.device
            )
            indices = torch.randint(len(centers), (self.num_envs,), device=self.device)
            forward = centers[
                indices
            ] + self.cfg.commands.forward_noise_std * torch.randn(
                self.num_envs, device=self.device
            )
        else:
            forward = torch.empty(self.num_envs, device=self.device).uniform_(*bounds)
            probability = getattr(
                self.cfg.commands, "forward_boundary_probability", 0.0
            )
            if probability > 0.0:
                draw = torch.rand(self.num_envs, device=self.device)
                forward = torch.where(draw < probability / 2, bounds[0], forward)
                forward = torch.where(
                    (draw >= probability / 2) & (draw < probability), bounds[1], forward
                )
        masked_update(self.commands[:, 0], forward, command_mask)
        if self.cfg.commands.include_standing:
            for selected in (slice(1, None), slice(None, 2), slice(None)):
                probability = 0.1 if selected == slice(None) else 0.2
                drop = command_mask & (
                    torch.rand(self.num_envs, device=self.device) < probability
                )
                self.commands[:, selected].masked_fill_(drop.unsqueeze(1), 0.0)

    def _reward_swing_grf(self):
        phase = torch.sin(self.oscillators).clamp(min=0.0)
        return -(phase * self._compute_grf()).sum(dim=1)

    def _reward_stance_grf(self):
        phase = (-torch.sin(self.oscillators)).clamp(min=0.0)
        return (phase * self._compute_grf()).sum(dim=1)

    def _reward_cursorial(self):
        hip = self.dof_pos.index_select(1, self.hip_dof_indices)
        scales = self.scales["dof_pos"].index_select(0, self.hip_dof_indices)
        return -(hip / scales).square().mean(dim=1)

    def _reward_torques(self):
        """Author implementation uses total squared joint torque."""
        return -self.torques.square().sum(dim=1)

    def _reward_action_rate(self):
        n = self.num_actuators
        change = self.dof_pos_history[:, :n] - self.dof_pos_history[:, n : 2 * n]
        interval = self.dt * self.cfg.control.decimation
        return -change.square().sum(dim=1) / interval**2

    def _reward_action_rate2(self):
        n = self.num_actuators
        change = (
            self.dof_pos_history[:, :n]
            - 2 * self.dof_pos_history[:, n : 2 * n]
            + self.dof_pos_history[:, 2 * n :]
        )
        interval = self.dt * self.cfg.control.decimation
        return -change.square().sum(dim=1) / interval**2

    def _reward_tracking_ang_vel(self):
        # Preserve the public implementation's quartic yaw-error exponent.
        error = ((self.commands[:, 2] - self.base_ang_vel[:, 2]) / 5.0).square()
        return self._sqrdexp(error)


class Go2OscGRF(Go2Osc):
    """Add foot forces in a level heading frame to the oscillator policy."""

    def _init_buffers(self):
        super()._init_buffers()
        self.foot_grf_obs = torch.zeros(self.num_envs, 12, device=self.device)
        self._refresh_grf_observations()

    def _refresh_grf_observations(self, reset_mask=None):
        """FL/FR/RL/RR, each [normal, forward, left], divided by body weight.

        Supported terrain is flat. Remove yaw only, so body roll/pitch cannot
        turn vertical support into apparent tangential traction. Preserve signed
        tangential forces and their magnitude without component-wise clipping.
        """
        forces = self.contact_forces.index_select(1, self.feet_indices)
        x, y, z, w = self.base_quat.unbind(dim=-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        cosine = yaw.cos().unsqueeze(1)
        sine = yaw.sin().unsqueeze(1)
        components = torch.stack(
            (
                forces[..., 2].clamp(min=0.0),
                cosine * forces[..., 0] + sine * forces[..., 1],
                -sine * forces[..., 0] + cosine * forces[..., 1],
            ),
            dim=-1,
        )
        weight = self._backend.link_mass.sum(dim=1, keepdim=True) * 9.81
        observation = (components / weight.unsqueeze(-1)).flatten(start_dim=1)
        if reset_mask is None:
            self.foot_grf_obs.copy_(observation)
        else:
            masked_update(self.foot_grf_obs, observation, reset_mask)

    def _post_physics_step(self):
        super()._post_physics_step()
        self._refresh_grf_observations()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self._refresh_grf_observations(reset_mask)


class Go2OscDirectional(Go2Osc):
    """Observe four force inclinations and use them in moving-phase feedback.

    Inclination is measured from the ground plane: vertical GRF is pi/2.
    Angle feedback can add an offset, slow advancing stance drift, or delay
    release with a bounded angle clock. Force inclination does not detect slip.
    """

    def __init__(self, cfg, device, headless, backend):
        self._validate_directional_cfg(cfg)
        super().__init__(cfg, device, headless, backend)

    @staticmethod
    def _validate_directional_cfg(cfg):
        gain = cfg.osc.directional_gain
        if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
            raise ValueError("directional_gain must be finite and in [0, 1]")
        minimum = cfg.osc.directional_min_load
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("directional_min_load must be finite and positive")
        progress = cfg.osc.directional_min_progress
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError("directional_min_progress must be finite and in [0, 1]")
        # Saved pre-gate configs must retain their original oscillator dynamics.
        if not isinstance(getattr(cfg.osc, "directional_stance_gate", False), bool):
            raise ValueError("directional_stance_gate must be a bool")
        mode = getattr(cfg.osc, "directional_feedback_mode", "offset")
        if not isinstance(mode, str) or mode not in (
            "offset",
            "fractional_stance",
            "angle_release",
        ):
            raise ValueError(
                "directional_feedback_mode must be 'offset', 'fractional_stance', "
                "or 'angle_release'"
            )
        frequency = cfg.osc.moving_frequency_scale
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("moving_frequency_scale must be finite and positive")
        speed = cfg.reward_settings.mechanical_power_reference_speed
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("mechanical_power_reference_speed must be positive")
        tolerance = cfg.reward_settings.velocity_tracking_tolerance
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0.0):
            raise ValueError("velocity_tracking_tolerance must be positive or None")
        tracking_reward = getattr(
            cfg.reward_settings, "velocity_tracking_reward", "gaussian"
        )
        if not isinstance(tracking_reward, str) or tracking_reward not in (
            "gaussian",
            "linear",
        ):
            raise ValueError("velocity_tracking_reward must be 'gaussian' or 'linear'")
        reference = getattr(
            cfg.reward_settings, "velocity_tracking_reference_speed", 1.0
        )
        if (
            not isinstance(reference, (int, float))
            or isinstance(reference, bool)
            or not math.isfinite(reference)
            or reference <= 0.0
        ):
            raise ValueError(
                "velocity_tracking_reference_speed must be finite and positive"
            )
        vertical_reference = getattr(
            cfg.reward_settings, "stance_foot_vertical_speed_reference", 0.1
        )
        if (
            not isinstance(vertical_reference, (int, float))
            or isinstance(vertical_reference, bool)
            or not math.isfinite(vertical_reference)
            or vertical_reference <= 0.0
        ):
            raise ValueError(
                "stance_foot_vertical_speed_reference must be finite and positive"
            )
        touchdown_reference = getattr(
            cfg.reward_settings, "touchdown_vertical_speed_reference", 0.1
        )
        if (
            not isinstance(touchdown_reference, (int, float))
            or isinstance(touchdown_reference, bool)
            or not math.isfinite(touchdown_reference)
            or touchdown_reference <= 0.0
        ):
            raise ValueError(
                "touchdown_vertical_speed_reference must be finite and positive"
            )
        soft_support_reference = getattr(
            cfg.reward_settings, "soft_support_load_reference", 0.05
        )
        if (
            not isinstance(soft_support_reference, (int, float))
            or isinstance(soft_support_reference, bool)
            or not math.isfinite(soft_support_reference)
            or soft_support_reference <= 0.0
        ):
            raise ValueError("soft_support_load_reference must be finite and positive")
        sustained_duration = getattr(
            cfg.reward_settings, "sustained_support_duration", 0.1
        )
        if (
            not isinstance(sustained_duration, (int, float))
            or isinstance(sustained_duration, bool)
            or not math.isfinite(sustained_duration)
            or sustained_duration <= 0.0
        ):
            raise ValueError("sustained_support_duration must be finite and positive")
        for name, default in (
            ("sliding_speed_reference", 0.2),
            ("sliding_vertical_speed_reference", 0.05),
            ("sliding_lateral_speed_reference", 0.1),
            ("sliding_contact_ramp_duration", 0.02),
            ("sliding_clearance_reference", 0.001),
            ("sliding_tracking_width", 0.3),
        ):
            value = getattr(cfg.reward_settings, name, default)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        speed_reward = getattr(cfg.reward_settings, "sliding_speed_reward", "saturated")
        if not isinstance(speed_reward, str) or speed_reward not in (
            "saturated",
            "tracking",
            "tracking_excess",
            "bidirectional",
        ):
            raise ValueError("Unknown sliding_speed_reward")
        deadzone = getattr(cfg.reward_settings, "sliding_speed_deadzone", 0.0)
        if (
            not isinstance(deadzone, (int, float))
            or isinstance(deadzone, bool)
            or not math.isfinite(deadzone)
            or not 0.0
            <= deadzone
            < getattr(cfg.reward_settings, "sliding_speed_reference", 0.2)
        ):
            raise ValueError(
                "sliding_speed_deadzone must be finite and in [0, speed reference)"
            )
        initial_fraction = getattr(
            cfg.reward_settings, "sliding_contact_initial_fraction", 0.0
        )
        if (
            not isinstance(initial_fraction, (int, float))
            or isinstance(initial_fraction, bool)
            or not math.isfinite(initial_fraction)
            or not 0.0 <= initial_fraction <= 1.0
        ):
            raise ValueError(
                "sliding_contact_initial_fraction must be finite and in [0, 1]"
            )
        age_mode = getattr(cfg.reward_settings, "sliding_contact_age_mode", "force")
        if not isinstance(age_mode, str) or age_mode not in ("force", "engagement"):
            raise ValueError("sliding_contact_age_mode must be force or engagement")
        for name, default, allow_zero in (
            ("sliding_engagement_min_load", 0.05, False),
            ("sliding_engagement_min_speed", 0.05, False),
            ("sliding_engagement_max_gap", 0.01, True),
            ("sliding_engagement_max_clearance", 0.001, True),
        ):
            value = getattr(cfg.reward_settings, name, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if allow_zero else value <= 0)
            ):
                raise ValueError(
                    f"{name} must be finite and "
                    f"{'nonnegative' if allow_zero else 'positive'}"
                )
        body_match_width = getattr(
            cfg.reward_settings, "sliding_body_match_width", None
        )
        if body_match_width is not None and (
            not isinstance(body_match_width, (int, float))
            or isinstance(body_match_width, bool)
            or not math.isfinite(body_match_width)
            or body_match_width <= 0.0
        ):
            raise ValueError("sliding_body_match_width must be positive or None")
        clearance_reference = getattr(
            cfg.reward_settings, "foot_clearance_height_reference", 0.01
        )
        if (
            not isinstance(clearance_reference, (int, float))
            or isinstance(clearance_reference, bool)
            or not math.isfinite(clearance_reference)
            or clearance_reference <= 0.0
        ):
            raise ValueError(
                "foot_clearance_height_reference must be finite and positive"
            )
        yaw_tolerance = cfg.reward_settings.yaw_tracking_tolerance
        if yaw_tolerance is not None and (
            not math.isfinite(yaw_tolerance) or yaw_tolerance <= 0.0
        ):
            raise ValueError("yaw_tracking_tolerance must be positive or None")
        Go2OscDirectional._validate_stance_force_cfg(cfg)

    @staticmethod
    def _validate_stance_force_cfg(cfg):
        defaults = {
            "stance_force_sliding_gain": 0.0,
            "stance_force_sliding_ratio_reference": 0.2,
            "stance_force_sliding_min_load": 0.05,
            "stance_force_sliding_epsilon": 0.0001,
        }
        for name, default in defaults.items():
            value = getattr(cfg.reward_settings, name, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if name == "stance_force_sliding_gain" else value <= 0)
            ):
                qualifier = (
                    "nonnegative" if name == "stance_force_sliding_gain" else "positive"
                )
                raise ValueError(f"{name} must be finite and {qualifier}")

    def _reward_stance_grf(self):
        """Original stance reward plus bounded force-utilization shaping.

        This cue measures tangential traction relative to normal load; it does
        not measure material slip. No foot velocity, geometry or age is read.
        The runner's stance_grf weight scales the entire returned quantity.
        """
        baseline = super()._reward_stance_grf()
        settings = self.cfg.reward_settings
        gain = getattr(settings, "stance_force_sliding_gain", 0.0)
        if gain == 0.0:
            return baseline
        reference = getattr(settings, "stance_force_sliding_ratio_reference", 0.2)
        minimum = getattr(settings, "stance_force_sliding_min_load", 0.05)
        epsilon = getattr(settings, "stance_force_sliding_epsilon", 1.0e-4)
        forces = self.contact_forces.index_select(1, self.feet_indices)
        weight = self._backend.link_mass.sum(dim=1, keepdim=True) * 9.81
        normalized = forces / weight.unsqueeze(-1)
        normal = normalized[..., 2].clamp(min=0.0)
        loaded = torch.isfinite(normalized).all(dim=-1) & (normal > minimum)
        load = torch.where(loaded, normal, torch.zeros_like(normal))
        rho = normalized[..., :2].norm(dim=-1) / (normal + epsilon)
        utilization = torch.where(
            loaded, (rho / reference).clamp(0, 1), torch.zeros_like(normal)
        )
        stance = (-self.oscillators.sin()).clamp(min=0.0)
        cue = (load * stance * utilization).sum(dim=1)
        # All meaningfully loaded feet enter the denominator, including swing.
        cue = cue / load.sum(dim=1).clamp(min=1.0)

        command = self.commands[:, :2]
        speed = command.norm(dim=1)
        denominator = speed.clamp(min=1.0e-6)
        local_direction = command / denominator.unsqueeze(1)
        x, y, z, w = self.root_states[:, 3:7].unbind(dim=1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        cosine, sine = yaw.cos(), yaw.sin()
        direction = torch.stack(
            (
                cosine * local_direction[:, 0] - sine * local_direction[:, 1],
                sine * local_direction[:, 0] + cosine * local_direction[:, 1],
            ),
            dim=1,
        )
        velocity = self.root_states[:, 7:9]
        progress = ((velocity * direction).sum(dim=1) / denominator).clamp(0, 1)
        tolerance = getattr(settings, "velocity_tracking_tolerance", None)
        tolerance = 0.4 if tolerance is None else tolerance
        yaw_tolerance = getattr(settings, "yaw_tracking_tolerance", None)
        yaw_tolerance = 0.5 if yaw_tolerance is None else yaw_tolerance
        planar_error = (velocity - direction * speed.unsqueeze(1)) / tolerance
        yaw_error = (self.root_states[:, 12] - self.commands[:, 2]) / yaw_tolerance
        tracking = torch.exp(-planar_error.square().sum(dim=1) - yaw_error.square())
        bonus = progress * tracking * cue
        bonus = torch.where(
            (speed > 1.0e-6) & torch.isfinite(bonus), bonus, torch.zeros_like(bonus)
        )
        return baseline + gain * bonus

    def compute_osc_slope(self, update_mask=None):
        super().compute_osc_slope(update_mask)
        scale = self.cfg.osc.moving_frequency_scale
        if scale == 1.0:
            return
        moving = ~self._slow_command_mask()
        if update_mask is not None:
            moving = moving & update_mask.unsqueeze(1)
        torch.where(moving, self.osc_omega * scale, self.osc_omega, out=self.osc_omega)

    def _init_buffers(self):
        super()._init_buffers()
        self.foot_grf_angle = torch.zeros(self.num_envs, 4, device=self.device)
        self.normal_load_squared = torch.zeros(self.num_envs, device=self.device)
        self._normal_load_squared_sum = torch.zeros_like(self.normal_load_squared)
        self.normal_load_change_squared = torch.zeros_like(self.normal_load_squared)
        self._normal_load_change_squared_sum = torch.zeros_like(
            self.normal_load_squared
        )
        self.stance_foot_vertical_velocity = torch.zeros_like(self.normal_load_squared)
        self._stance_foot_vertical_velocity_sum = torch.zeros_like(
            self.normal_load_squared
        )
        self.foot_vertical_velocity = torch.zeros_like(self.normal_load_squared)
        self._foot_vertical_velocity_sum = torch.zeros_like(self.normal_load_squared)
        self.loaded_foot_fraction = torch.zeros_like(self.normal_load_squared)
        self._loaded_foot_fraction_sum = torch.zeros_like(self.normal_load_squared)
        self.soft_foot_support = torch.zeros_like(self.normal_load_squared)
        self._soft_foot_support_sum = torch.zeros_like(self.normal_load_squared)
        self.sustained_foot_support = torch.zeros_like(self.normal_load_squared)
        self._sustained_foot_support_sum = torch.zeros_like(self.normal_load_squared)
        self._sustained_support_age = torch.zeros_like(self.foot_grf_angle)
        self._sliding_engagement_age = torch.zeros_like(self.foot_grf_angle)
        self._sliding_engagement_off_steps = torch.zeros(
            self.num_envs, 4, dtype=torch.long, device=self.device
        )
        self._sliding_engagement_direction = torch.zeros_like(self.foot_grf_angle)
        self._pre_sliding_engagement_clearance = torch.zeros_like(self.foot_grf_angle)
        self.forward_sliding = torch.zeros_like(self.normal_load_squared)
        self._forward_sliding_sum = torch.zeros_like(self.normal_load_squared)
        self.loaded_contact_switches = torch.zeros_like(self.normal_load_squared)
        self._loaded_contact_switches_sum = torch.zeros_like(self.normal_load_squared)
        self.touchdown_vertical_velocity = torch.zeros_like(self.normal_load_squared)
        self._touchdown_vertical_velocity_sum = torch.zeros_like(
            self.normal_load_squared
        )
        self._pre_touchdown_center_vertical_velocity = torch.zeros_like(
            self.foot_grf_angle
        )
        self._previous_normal_load = self._refresh_force_angle_observations().clone()
        self.foot_clearance = torch.zeros_like(self.normal_load_squared)
        self._foot_clearance_sum = torch.zeros_like(self.normal_load_squared)
        self._init_foot_clearance_geometry()

    def _init_foot_clearance_geometry(self):
        """Read actual canonical-foot collision spheres once from the URDF."""
        root = ET.parse(self.cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR)).getroot()
        links = {link.attrib["name"]: link for link in root.findall("link")}
        offsets, radii = [], []
        for name in self.robot_layout.body_groups["feet"]:
            collisions = links[name].findall("collision")
            if len(collisions) != 1 or collisions[0].find("geometry/sphere") is None:
                raise ValueError(
                    f"Foot clearance requires one collision sphere for {name!r}"
                )
            collision = collisions[0]
            origin = collision.find("origin")
            xyz = "0 0 0" if origin is None else origin.attrib.get("xyz", "0 0 0")
            offset = [float(value) for value in xyz.split()]
            radius = float(collision.find("geometry/sphere").attrib["radius"])
            if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
                raise ValueError(f"Invalid collision-sphere offset for {name!r}")
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"Invalid collision-sphere radius for {name!r}")
            offsets.append(offset)
            radii.append(radius)
        self._foot_collision_offsets = (
            torch.tensor(offsets, dtype=torch.float, device=self.device)
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1)
        )
        self._foot_collision_radii = torch.tensor(
            radii, dtype=torch.float, device=self.device
        )

    def _foot_sphere_clearance(self):
        """Sphere-bottom height above the fixed flat plane at world z=0."""
        positions = self._rigid_body_pos.index_select(1, self.feet_indices)
        quaternions = self._rigid_body_quat.index_select(1, self.feet_indices)
        centers = positions + quat_apply(quaternions, self._foot_collision_offsets)
        return centers[..., 2] - self._foot_collision_radii

    def _compute_foot_clearance_cost(self):
        """All-phase linear airborne height cost without a contact-force gate."""
        reference = getattr(
            self.cfg.reward_settings, "foot_clearance_height_reference", 0.01
        )
        return self._foot_sphere_clearance().clamp(min=0.0).mean(dim=1) / reference

    def _reward_foot_clearance(self):
        """Negative substep mean; the runner applies the timestep once."""
        return -self.foot_clearance

    def _compute_directional_force_signals(self):
        """Return normal load in body weights and inclination in radians."""
        forces = self.contact_forces.index_select(1, self.feet_indices)
        weight = self._backend.link_mass.sum(dim=1, keepdim=True) * 9.81
        forces = forces / weight.unsqueeze(-1)
        normal = forces[..., 2].clamp(min=0.0)
        tangent = forces[..., :2].norm(dim=-1)
        angle = torch.atan2(normal, tangent)
        angle = torch.where(normal > self.cfg.osc.directional_min_load, angle, 0.0)
        return normal, angle

    def _refresh_force_angle_observations(self, reset_mask=None):
        """FL/FR/RL/RR inclinations divided by pi/2; unloaded feet are zero."""
        normal, angle = self._compute_directional_force_signals()
        observation = angle / (math.pi / 2)
        if reset_mask is None:
            self.foot_grf_angle.copy_(observation)
        else:
            masked_update(self.foot_grf_angle, observation, reset_mask)
        return normal

    def _compute_directional_phase_offset(self):
        # Read live forces: the cached actor observations refresh after phase
        # integration and must not introduce a one-substep feedback delay.
        _, angle = self._compute_directional_force_signals()
        moving = self._clock_forward_command().abs() > self.cfg.osc.stop_threshold
        offset = self.cfg.osc.directional_gain * angle.sin() * moving
        if getattr(self.cfg.osc, "directional_stance_gate", False):
            offset = offset * (-self.oscillators.sin()).clamp(min=0.0)
        return offset

    def _compute_oscillator_drift(self):
        drift = super()._compute_oscillator_drift()
        if self.cfg.osc.directional_gain == 0.0 or not self.cfg.osc.coupling:
            return drift
        if (
            getattr(self.cfg.osc, "directional_feedback_mode", "offset")
            == "angle_release"
        ):
            _, angle = self._compute_directional_force_signals()
            moving = self._clock_forward_command().abs() > self.cfg.osc.stop_threshold
            release = (1 + self.oscillators.cos()) / 2
            activation = angle.sin() * release
            # Cycles/s; common process noise and 2*pi conversion remain outside.
            # Full gain is bounded/angle-only; intermediate gains blend legacy.
            clock = self.osc_omega * (
                1 - (1 - self.cfg.osc.directional_min_progress) * activation
            )
            gain = self.cfg.osc.directional_gain
            blended = (1 - gain) * drift + gain * clock
            return torch.where(moving, blended, drift)
        if (
            getattr(self.cfg.osc, "directional_feedback_mode", "offset")
            == "fractional_stance"
        ):
            _, angle = self._compute_directional_force_signals()
            moving = self._clock_forward_command().abs() > self.cfg.osc.stop_threshold
            stance = (-self.oscillators.sin()).clamp(min=0.0)
            activation = self.cfg.osc.directional_gain * angle.sin() * moving * stance
            # This mode always gates in stance; the legacy offset-mode gate is
            # independent. Nonpositive baseline drift remains unchanged.
            return drift - (
                (1 - self.cfg.osc.directional_min_progress)
                * drift.clamp(min=0.0)
                * activation
            )
        extra_feedback = (
            self.osc_coupling * self.grf * self._compute_directional_phase_offset()
        )
        # A configurable positive-progress floor allows testing long loaded
        # contacts; zero permits phase hold. Existing reverse drift is unchanged.
        feedback_limit = (1 - self.cfg.osc.directional_min_progress) * drift.clamp(
            min=0.0
        )
        return drift - torch.minimum(extra_feedback, feedback_limit)

    def _reward_tracking_lin_vel(self):
        if (
            getattr(self.cfg.reward_settings, "velocity_tracking_reward", "gaussian")
            == "linear"
        ):
            reference = getattr(
                self.cfg.reward_settings, "velocity_tracking_reference_speed", 1.0
            )
            error = self.commands[:, :2] - self.base_lin_vel[:, :2]
            return 1.0 - error.norm(dim=1) / reference
        tolerance = self.cfg.reward_settings.velocity_tracking_tolerance
        if tolerance is None:
            return super()._reward_tracking_lin_vel()
        error = (self.commands[:, :2] - self.base_lin_vel[:, :2]) / tolerance
        return torch.exp(-error.square().sum(dim=1))

    def _reward_joint_velocity_squared(self):
        """Negative mean squared actual joint speed, in radian²/second²."""
        return -self.dof_vel.square().mean(dim=1)

    def _reward_mechanical_power(self):
        """Absolute joint-power proxy, normalized by weight and a fixed speed."""
        velocity = self.dof_vel.index_select(1, self.actuated_dof_indices)
        power = (self.torques * velocity).abs().sum(dim=1)
        weight = self._backend.link_mass.sum(dim=1) * 9.81
        reference = self.cfg.reward_settings.mechanical_power_reference_speed
        return -power / (weight * reference)

    def _reward_normalized_torques(self):
        """Holding-effort proxy; remains nonzero when joint speed is zero."""
        return -(self.torques / self.actuated_torque_limits).square().mean(dim=1)

    def _reward_normal_load_squared(self):
        """Negative mean squared foot normal load over every physics substep.

        Loads are normalized by live body weight without clipping. The runner
        applies the control timestep once, as for other ordinary rewards.
        """
        return -self.normal_load_squared

    def _reward_normal_load_change_squared(self):
        """Negative mean squared normal-load increments at the physics rate.

        This prices increments rather than a time derivative: there is no dt
        division. The runner integrates the resulting control-frame mean once.
        """
        return -self.normal_load_change_squared

    def _compute_stance_foot_vertical_velocity_cost(self):
        """Mean stance-weighted squared world-z foot-LINK-origin speed.

        Backend rigid-body velocities are world-frame body-origin velocities.
        This is neither contact-material slip nor sphere-geometry-center speed.
        Force-off intervals are included when the oscillator indicates stance.
        """
        vertical = self._rigid_body_lin_vel.index_select(1, self.feet_indices)[..., 2]
        reference = getattr(
            self.cfg.reward_settings, "stance_foot_vertical_speed_reference", 0.1
        )
        stance = (-self.oscillators.sin()).clamp(min=0.0)
        return (stance * (vertical / reference).square()).mean(dim=1)

    def _reward_stance_foot_vertical_velocity(self):
        """Negative control-frame mean; the runner applies the timestep once."""
        return -self.stance_foot_vertical_velocity

    def _compute_foot_vertical_velocity_cost(self):
        """Mean squared world-z foot-link-origin speed across all phases."""
        vertical = self._rigid_body_lin_vel.index_select(1, self.feet_indices)[..., 2]
        reference = getattr(
            self.cfg.reward_settings, "stance_foot_vertical_speed_reference", 0.1
        )
        return (vertical / reference).square().mean(dim=1)

    def _reward_foot_vertical_velocity(self):
        """Negative physics-substep mean; the runner applies the timestep once."""
        return -self.foot_vertical_velocity

    def _compute_loaded_foot_fraction(self, normal):
        """Fraction of canonical feet exceeding the existing meaningful-load gate."""
        loaded = normal > self.cfg.osc.directional_min_load
        return loaded.to(normal.dtype).mean(dim=1)

    def _reward_loaded_foot_fraction(self):
        """Positive substep-mean support occupancy; contains no slip-velocity term."""
        return self.loaded_foot_fraction

    def _compute_soft_foot_support(self, normal):
        """Smooth support score from positive, unclipped loads in body weights."""
        positive = normal.clamp(min=0.0)
        reference = getattr(
            self.cfg.reward_settings, "soft_support_load_reference", 0.05
        )
        return (positive / (positive + reference)).mean(dim=1)

    def _reward_soft_foot_support(self):
        """Positive physics-substep mean; the runner applies the timestep once."""
        return self.soft_foot_support

    def _update_sustained_foot_support(self, normal):
        """Advance capped loaded-contact ages once and return their soft score."""
        duration = getattr(self.cfg.reward_settings, "sustained_support_duration", 0.1)
        reference = getattr(
            self.cfg.reward_settings, "soft_support_load_reference", 0.05
        )
        loaded = normal > self.cfg.osc.directional_min_load
        self._sustained_support_age.add_(self.dt / self.cfg.control.decimation)
        self._sustained_support_age.clamp_(max=duration)
        self._sustained_support_age.masked_fill_(~loaded, 0.0)
        positive = normal.clamp(min=0.0)
        score = positive / (positive + reference)
        return (score * (self._sustained_support_age / duration)).mean(dim=1)

    def _reward_sustained_foot_support(self):
        """Positive substep mean of age-weighted support; no foot-motion term."""
        return self.sustained_foot_support

    def _foot_sliding_velocities(self):
        """World sphere-center and lowest material-point velocities."""
        quaternion = self._rigid_body_quat.index_select(1, self.feet_indices)
        linear = self._rigid_body_lin_vel.index_select(1, self.feet_indices)
        angular = self._rigid_body_ang_vel.index_select(1, self.feet_indices)
        offset = quat_apply(quaternion, self._foot_collision_offsets)
        center = linear + torch.cross(angular, offset, dim=-1)
        lever = offset.clone()
        lever[..., 2] -= self._foot_collision_radii
        material = linear + torch.cross(angular, lever, dim=-1)
        return center, material

    def _sliding_engagement_samples(
        self,
        normal,
        material_parallel,
        center_parallel,
        clearance,
        *,
        advance=False,
        finite_motion=None,
    ):
        """Causal body-force/point-motion engagement; not native ground evidence.

        Age counts only loaded, same-direction moving samples near the plane.
        Valid short force gaps preserve history but add no age or reward.
        """
        settings = self.cfg.reward_settings
        minimum_load = getattr(settings, "sliding_engagement_min_load", 0.05)
        minimum_speed = getattr(settings, "sliding_engagement_min_speed", 0.05)
        maximum_clearance = getattr(settings, "sliding_engagement_max_clearance", 0.001)
        maximum_gap = getattr(settings, "sliding_engagement_max_gap", 0.01)
        interval = self.dt / self.cfg.control.decimation
        maximum_steps = math.floor(maximum_gap / interval + 1.0e-9)
        loaded = normal > minimum_load
        moving = (
            (
                torch.minimum(material_parallel.abs(), center_parallel.abs())
                > minimum_speed
            )
            & (material_parallel * center_parallel > 0)
            & torch.isfinite(material_parallel)
            & torch.isfinite(center_parallel)
        )
        near = (
            (clearance <= maximum_clearance)
            & (self._pre_sliding_engagement_clearance <= maximum_clearance)
            & torch.isfinite(clearance)
            & torch.isfinite(self._pre_sliding_engagement_clearance)
        )
        eligible = moving & near & torch.isfinite(normal)
        if finite_motion is not None:
            eligible &= finite_motion
        valid = loaded & eligible
        if not advance:
            return valid
        direction = material_parallel.sign()
        age = self._sliding_engagement_age
        gap = self._sliding_engagement_off_steps
        previous_direction = self._sliding_engagement_direction
        history_valid = (
            torch.isfinite(age)
            & (age > 0)
            & torch.isfinite(previous_direction)
            & torch.isfinite(gap)
            & (gap >= 0)
            & (gap <= maximum_steps)
        )
        same_direction = previous_direction == direction
        continuing = history_valid & same_direction
        off_candidate = ~loaded & eligible & continuing
        next_gap = gap + 1
        hold = off_candidate & (next_gap <= maximum_steps)
        duration = getattr(settings, "sustained_support_duration", 0.1)
        advanced = torch.where(
            continuing, age + interval, torch.full_like(age, interval)
        ).clamp(max=duration)
        next_age = torch.where(
            valid, advanced, torch.where(hold, age, torch.zeros_like(age))
        )
        age.copy_(next_age)
        gap.copy_(torch.where(hold, next_gap, torch.zeros_like(gap)))
        previous_direction.copy_(
            torch.where(valid | hold, direction, torch.zeros_like(direction))
        )
        return valid

    def _compute_forward_sliding(self, normal, *, advance_engagement=False):
        """Bounded, task-gated forward sliding with live substep state."""
        settings = self.cfg.reward_settings
        speed_reference = getattr(settings, "sliding_speed_reference", 0.2)
        vertical_reference = getattr(settings, "sliding_vertical_speed_reference", 0.05)
        lateral_reference = getattr(settings, "sliding_lateral_speed_reference", 0.1)
        ramp_duration = getattr(settings, "sliding_contact_ramp_duration", 0.02)
        clearance_reference = getattr(settings, "sliding_clearance_reference", 0.001)
        tolerance = getattr(settings, "velocity_tracking_tolerance", None)
        tolerance = 0.4 if tolerance is None else tolerance
        yaw_tolerance = getattr(settings, "yaw_tracking_tolerance", None)
        yaw_tolerance = 0.5 if yaw_tolerance is None else yaw_tolerance

        command = self.commands[:, :2]
        speed = command.norm(dim=1)
        denominator = speed.clamp(min=1.0e-6)
        local_direction = command / denominator.unsqueeze(1)
        quaternion = self.root_states[:, 3:7]
        x, y, z, w = quaternion.unbind(dim=1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        cosine, sine = yaw.cos(), yaw.sin()
        direction = torch.stack(
            (
                cosine * local_direction[:, 0] - sine * local_direction[:, 1],
                sine * local_direction[:, 0] + cosine * local_direction[:, 1],
            ),
            dim=1,
        )
        command_world = direction * speed.unsqueeze(1)
        world_velocity = self.root_states[:, 7:9]
        progress = ((world_velocity * direction).sum(dim=1) / denominator).clamp(0, 1)
        tracking_error = (world_velocity - command_world) / tolerance
        yaw_error = (self.root_states[:, 12] - self.commands[:, 2]) / yaw_tolerance
        task_gate = torch.exp(-tracking_error.square().sum(dim=1) - yaw_error.square())

        center, material = self._foot_sliding_velocities()
        center_parallel = (center[..., :2] * direction.unsqueeze(1)).sum(dim=-1)
        material_parallel = (material[..., :2] * direction.unsqueeze(1)).sum(dim=-1)
        speed_reward = getattr(settings, "sliding_speed_reward", "saturated")
        if speed_reward in ("tracking", "tracking_excess"):
            # Both the material point and sphere center must match the target.
            # Pure rolling cannot improve this score over stationary contact.
            width = getattr(settings, "sliding_tracking_width", 0.3)
            error = torch.maximum(
                (material_parallel - speed_reference).square(),
                (center_parallel - speed_reference).square(),
            )
            parallel_score = torch.exp(-error / width**2)
            if speed_reward == "tracking_excess":
                baseline = math.exp(-((speed_reference / width) ** 2))
                parallel_score = (
                    (parallel_score - baseline) / max(1 - baseline, 1.0e-12)
                ).clamp(0, 1)
        elif speed_reward == "bidirectional":
            forward = torch.minimum(center_parallel, material_parallel)
            backward = torch.minimum(-center_parallel, -material_parallel)
            shared_speed = torch.maximum(forward, backward).clamp(min=0.0)
            deadzone = getattr(settings, "sliding_speed_deadzone", 0.0)
            parallel_score = (
                (shared_speed - deadzone) / (speed_reference - deadzone)
            ).clamp(0, 1)
        else:
            parallel = torch.minimum(center_parallel, material_parallel)
            parallel_score = (parallel / speed_reference).clamp(0, 1)
            body_match_width = getattr(settings, "sliding_body_match_width", None)
            if body_match_width is not None:
                body_parallel = (world_velocity * direction).sum(dim=1, keepdim=True)
                match_error = (material_parallel - body_parallel).square()
                match_error += (center_parallel - body_parallel).square()
                parallel_score = parallel_score * torch.exp(
                    -match_error / (2 * body_match_width**2)
                )
        lateral = (
            direction[:, None, 0] * material[..., 1]
            - direction[:, None, 1] * material[..., 0]
        )
        engagement = (
            getattr(settings, "sliding_contact_age_mode", "force") == "engagement"
        )
        if engagement:
            raw_clearance = self._foot_sphere_clearance()
            clearance = raw_clearance.clamp(min=0.0)
        else:
            clearance = self._foot_sphere_clearance().clamp(min=0.0)
        motion_gate = torch.exp(
            -(material[..., 2] / vertical_reference).square()
            - (lateral / lateral_reference).square()
            - (clearance / clearance_reference).square()
        )
        if engagement:
            valid_engagement = self._sliding_engagement_samples(
                normal,
                material_parallel,
                center_parallel,
                raw_clearance,
                advance=advance_engagement,
                finite_motion=(
                    torch.isfinite(material).all(dim=-1)
                    & torch.isfinite(center).all(dim=-1)
                ),
            )
            age = (self._sliding_engagement_age / ramp_duration).clamp(0, 1)
            minimum_load = getattr(settings, "sliding_engagement_min_load", 0.05)
        else:
            age = (self._sustained_support_age / ramp_duration).clamp(0, 1)
            minimum_load = self.cfg.osc.directional_min_load
        initial_fraction = getattr(settings, "sliding_contact_initial_fraction", 0.0)
        if initial_fraction != 0.0:
            age = initial_fraction + (1 - initial_fraction) * age
        if engagement:
            load = torch.where(
                torch.isfinite(normal) & (normal > minimum_load),
                normal.clamp(min=0.0),
                torch.zeros_like(normal),
            )
            score = torch.where(
                valid_engagement,
                load * age * parallel_score * motion_gate,
                torch.zeros_like(normal),
            ).sum(dim=1)
        else:
            load = torch.where(
                normal > minimum_load,
                normal.clamp(min=0.0),
                torch.zeros_like(normal),
            )
            score = (load * age * parallel_score * motion_gate).sum(dim=1)
        score = score / load.sum(dim=1).clamp(min=1.0)
        reward = progress * task_gate * score
        if engagement:
            return torch.where(
                (speed > 1.0e-6) & torch.isfinite(reward),
                reward,
                torch.zeros_like(reward),
            )
        return torch.where(speed > 1.0e-6, reward, torch.zeros_like(reward))

    def _reward_forward_sliding(self):
        """Positive cached physics-substep mean; the runner applies dt once."""
        return self.forward_sliding

    def _compute_loaded_contact_switches(self, normal):
        """Count changed foot-load states since the preceding physics sample."""
        threshold = self.cfg.osc.directional_min_load
        loaded = normal > threshold
        previously_loaded = self._previous_normal_load > threshold
        return (loaded ^ previously_loaded).to(normal.dtype).sum(dim=1)

    def _reward_loaded_contact_switches(self):
        """Negative substep-mean event count; no dt or foot-count division."""
        return -self.loaded_contact_switches

    def _compute_touchdown_vertical_velocity_cost(self, normal):
        """Sum squared incoming sphere-center approach speed at load onsets."""
        threshold = self.cfg.osc.directional_min_load
        onset = (normal > threshold) & (self._previous_normal_load <= threshold)
        reference = getattr(
            self.cfg.reward_settings, "touchdown_vertical_speed_reference", 0.1
        )
        approach = (-self._pre_touchdown_center_vertical_velocity).clamp(min=0.0)
        return (onset * (approach / reference).square()).sum(dim=1)

    def _reward_touchdown_vertical_velocity(self):
        """Negative cached physics mean; no foot-count or dt division."""
        return -self.touchdown_vertical_velocity

    def _pre_compute_torques(self):
        super()._pre_compute_torques()
        center, _ = self._foot_sliding_velocities()
        self._pre_touchdown_center_vertical_velocity.copy_(center[..., 2])
        if (
            getattr(self.cfg.reward_settings, "sliding_contact_age_mode", "force")
            == "engagement"
        ):
            self._pre_sliding_engagement_clearance.copy_(self._foot_sphere_clearance())

    def _reward_tracking_ang_vel(self):
        tolerance = self.cfg.reward_settings.yaw_tracking_tolerance
        if tolerance is None:
            return super()._reward_tracking_ang_vel()
        error = (self.commands[:, 2] - self.base_ang_vel[:, 2]) / tolerance
        return torch.exp(-error.square())

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self._normal_load_squared_sum.zero_()
        self._normal_load_change_squared_sum.zero_()
        self._stance_foot_vertical_velocity_sum.zero_()
        self._foot_vertical_velocity_sum.zero_()
        self._loaded_foot_fraction_sum.zero_()
        self._soft_foot_support_sum.zero_()
        self._sustained_foot_support_sum.zero_()
        self._forward_sliding_sum.zero_()
        self._foot_clearance_sum.zero_()
        self._loaded_contact_switches_sum.zero_()
        self._touchdown_vertical_velocity_sum.zero_()

    def _post_physics_step(self):
        super()._post_physics_step()
        normal = self._refresh_force_angle_observations()
        self._normal_load_squared_sum.add_(normal.square().sum(dim=1))
        change = normal - self._previous_normal_load
        self._normal_load_change_squared_sum.add_(change.square().sum(dim=1))
        self._loaded_contact_switches_sum.add_(
            self._compute_loaded_contact_switches(normal)
        )
        self._touchdown_vertical_velocity_sum.add_(
            self._compute_touchdown_vertical_velocity_cost(normal)
        )
        self._previous_normal_load.copy_(normal)
        self._stance_foot_vertical_velocity_sum.add_(
            self._compute_stance_foot_vertical_velocity_cost()
        )
        self._foot_vertical_velocity_sum.add_(
            self._compute_foot_vertical_velocity_cost()
        )
        self._loaded_foot_fraction_sum.add_(self._compute_loaded_foot_fraction(normal))
        self._soft_foot_support_sum.add_(self._compute_soft_foot_support(normal))
        self._sustained_foot_support_sum.add_(
            self._update_sustained_foot_support(normal)
        )
        if (
            getattr(self.cfg.reward_settings, "sliding_contact_age_mode", "force")
            == "engagement"
        ):
            self._forward_sliding_sum.add_(
                self._compute_forward_sliding(normal, advance_engagement=True)
            )
        else:
            self._forward_sliding_sum.add_(self._compute_forward_sliding(normal))
        self._foot_clearance_sum.add_(self._compute_foot_clearance_cost())

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._refresh_force_angle_observations()
        self.normal_load_squared.copy_(
            self._normal_load_squared_sum / self.cfg.control.decimation
        )
        self.normal_load_change_squared.copy_(
            self._normal_load_change_squared_sum / self.cfg.control.decimation
        )
        self.stance_foot_vertical_velocity.copy_(
            self._stance_foot_vertical_velocity_sum / self.cfg.control.decimation
        )
        self.foot_vertical_velocity.copy_(
            self._foot_vertical_velocity_sum / self.cfg.control.decimation
        )
        self.loaded_foot_fraction.copy_(
            self._loaded_foot_fraction_sum / self.cfg.control.decimation
        )
        self.soft_foot_support.copy_(
            self._soft_foot_support_sum / self.cfg.control.decimation
        )
        self.sustained_foot_support.copy_(
            self._sustained_foot_support_sum / self.cfg.control.decimation
        )
        self.forward_sliding.copy_(
            self._forward_sliding_sum / self.cfg.control.decimation
        )
        self.foot_clearance.copy_(
            self._foot_clearance_sum / self.cfg.control.decimation
        )
        self.loaded_contact_switches.copy_(
            self._loaded_contact_switches_sum / self.cfg.control.decimation
        )

        self.touchdown_vertical_velocity.copy_(
            self._touchdown_vertical_velocity_sum / self.cfg.control.decimation
        )

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        normal = self._refresh_force_angle_observations(reset_mask)
        self.normal_load_squared.masked_fill_(reset_mask, 0.0)
        self._normal_load_squared_sum.masked_fill_(reset_mask, 0.0)
        self.normal_load_change_squared.masked_fill_(reset_mask, 0.0)
        self._normal_load_change_squared_sum.masked_fill_(reset_mask, 0.0)
        self.stance_foot_vertical_velocity.masked_fill_(reset_mask, 0.0)
        self._stance_foot_vertical_velocity_sum.masked_fill_(reset_mask, 0.0)
        self.foot_vertical_velocity.masked_fill_(reset_mask, 0.0)
        self._foot_vertical_velocity_sum.masked_fill_(reset_mask, 0.0)
        self.loaded_foot_fraction.masked_fill_(reset_mask, 0.0)
        self._loaded_foot_fraction_sum.masked_fill_(reset_mask, 0.0)
        self.soft_foot_support.masked_fill_(reset_mask, 0.0)
        self._soft_foot_support_sum.masked_fill_(reset_mask, 0.0)
        self.sustained_foot_support.masked_fill_(reset_mask, 0.0)
        self._sustained_foot_support_sum.masked_fill_(reset_mask, 0.0)
        self._sustained_support_age.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self._sliding_engagement_age.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self._sliding_engagement_off_steps.masked_fill_(reset_mask.unsqueeze(1), 0)
        self._sliding_engagement_direction.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        masked_update(
            self._pre_sliding_engagement_clearance,
            self._foot_sphere_clearance(),
            reset_mask,
        )
        self.forward_sliding.masked_fill_(reset_mask, 0.0)
        self._forward_sliding_sum.masked_fill_(reset_mask, 0.0)
        self.foot_clearance.masked_fill_(reset_mask, 0.0)
        self._foot_clearance_sum.masked_fill_(reset_mask, 0.0)
        self.loaded_contact_switches.masked_fill_(reset_mask, 0.0)
        self._loaded_contact_switches_sum.masked_fill_(reset_mask, 0.0)
        self.touchdown_vertical_velocity.masked_fill_(reset_mask, 0.0)
        self._touchdown_vertical_velocity_sum.masked_fill_(reset_mask, 0.0)
        self._pre_touchdown_center_vertical_velocity.masked_fill_(
            reset_mask.unsqueeze(1), 0.0
        )
        # Begin from the reset state's current solved load, not an artificial
        # zero. Preserve unselected environments' cross-frame load history.
        masked_update(self._previous_normal_load, normal, reset_mask)


class Go2OscPushGlide(Go2OscSpeedTargetMixin, Go2OscDirectional):
    """Read each phase as push versus glide instead of contact versus flight.

    While the forward command exceeds the stop threshold, oscillator feedback
    and the push reward use normal load gated by foot motion relative to the
    base, and the glide reward penalizes tangential force instead of load. A
    loaded foot may therefore stay on low-friction ground through its glide
    half, where the drag it pays is small, but not on high-friction ground.
    At or below the threshold every signal is plain normal load, which keeps
    the standing fixed point of Equation (4) in mid-stance. With the steering
    thresholds set, a sideways or turning command also selects the moving
    clocks, so the robot can step in place.

    Optional configuration adds what hardware adds: errors on the foot-force
    and friction signals (``cfg.sensing``), a delay before joint targets reach
    the motors (``cfg.control.actuation_delay_range``) and ground friction
    that changes during an episode and differs between feet
    (``cfg.friction_changes``). Each is off unless configured.

    ``cfg.commands.sliding_stop_probability`` adds stop commands for robots
    that are moving, so training sees more slides to a stop.
    """

    def __init__(self, cfg, device, headless, backend):
        self._validate_push_glide_cfg(cfg)
        super().__init__(cfg, device, headless, backend)

    @staticmethod
    def _validate_push_glide_cfg(cfg):
        for name in ("push_gate_speed", "traction_ratio_limit"):
            value = getattr(cfg.osc, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if cfg.osc.directional_gain != 0.0:
            raise ValueError(
                "directional_gain must be 0: the push signal replaces angle feedback"
            )
        # Configurations saved before these fields existed keep the plain sine.
        for name in ("push_load_cap", "push_window_saturation"):
            value = getattr(cfg.reward_settings, name, 1.0)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 < value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in (0, 1]")
        cap = getattr(cfg.reward_settings, "tilt_rate_cap", 4.0)
        if (
            not isinstance(cap, (int, float))
            or isinstance(cap, bool)
            or not math.isfinite(cap)
            or cap <= 0.0
        ):
            raise ValueError("tilt_rate_cap must be finite and positive")
        # Configurations saved before this field existed leave targets unbounded.
        limit = getattr(cfg.control, "action_limit", None)
        if limit is not None and (
            not isinstance(limit, (int, float))
            or isinstance(limit, bool)
            or not math.isfinite(limit)
            or limit <= 0.0
        ):
            raise ValueError("action_limit must be None or finite and positive")
        margin = getattr(cfg.control, "joint_target_margin", None)
        if margin is not None and (
            not isinstance(margin, (int, float))
            or isinstance(margin, bool)
            or not math.isfinite(margin)
            or margin < 0.0
        ):
            raise ValueError("joint_target_margin must be None or finite and >= 0")
        for name in ("cost_of_transport_min_speed", "cost_of_transport_cap"):
            value = getattr(cfg.reward_settings, name, 1.0)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        Go2OscPushGlide._validate_variation_cfg(cfg)
        # Both steering thresholds, or neither for the forward-only rule.
        steering = [
            getattr(cfg.osc, name, None)
            for name in ("sideways_stop_threshold", "turning_stop_threshold")
        ]
        if any(value is not None for value in steering) and any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
            for value in steering
        ):
            raise ValueError(
                "sideways_stop_threshold and turning_stop_threshold must both be "
                "None or both finite and nonnegative"
            )

    def _init_buffers(self):
        super()._init_buffers()
        self.foot_traction_obs = torch.zeros(self.num_envs, 8, device=self.device)
        self.contact_friction_obs = torch.zeros(self.num_envs, 1, device=self.device)
        self._init_variation_buffers()
        self._refresh_traction_observations()
        self._refresh_friction_observation()
        self._refresh_friction_estimate(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )

    def _pre_decimation_step(self):
        self._limit_joint_targets()
        self._record_joint_targets()
        super()._pre_decimation_step()

    def _limit_joint_targets(self):
        limit_joint_targets(self)

    def _reward_target_saturation(self):
        return target_saturation_reward(self)

    def _reward_standing_saturation(self):
        return standing_saturation_reward(self)

    def _reward_joint_limit_margin(self):
        return joint_limit_margin_reward(self)

    def _joint_target_margin(self):
        return getattr(self.cfg.control, "joint_target_margin", None)

    def _joint_target_bounds(self):
        return joint_target_bounds(self)

    def _keep_targets_inside_joint_limits(self):
        keep_joint_targets(self)

    def _slow_command_mask(self):
        """Standing clocks only when no command asks the robot to move.

        Equation (5) switches on the forward command alone, which leaves a
        turn or sidestep in place on clocks that hold loaded feet still.
        Configurations without the two steering thresholds keep that rule.
        """
        slow = super()._slow_command_mask()
        sideways = getattr(self.cfg.osc, "sideways_stop_threshold", None)
        turning = getattr(self.cfg.osc, "turning_stop_threshold", None)
        if sideways is None or turning is None:
            return slow
        steering = self.commands[:, 1:3].abs()
        return slow & (steering[:, :1] <= sideways) & (steering[:, 1:] <= turning)

    def _moving(self):
        """Environments on the moving branch of the oscillator schedule."""
        return ~self._slow_command_mask()

    def _translating(self):
        """Commands with a travel direction for the push gate to act along."""
        forward = self._clock_forward_command().abs() > self.cfg.osc.stop_threshold
        sideways = getattr(self.cfg.osc, "sideways_stop_threshold", None)
        if sideways is None:
            return forward
        return forward | (self.commands[:, 1:2].abs() > sideways)

    def _foot_loads(self):
        """Normal and tangential foot force in body weights, each at most 1."""
        forces = self.contact_forces.index_select(1, self.feet_indices)
        weight = self._backend.link_mass.sum(dim=1, keepdim=True) * 9.81
        normal = (forces[..., 2] / weight).clamp(min=0.0, max=1.0)
        tangent = (forces[..., :2].norm(dim=-1) / weight).clamp(max=1.0)
        return normal, tangent

    def _foot_velocity_relative_to_base(self):
        """Foot-centre velocity seen from the moving, rotating base.

        World-frame components of what leg kinematics and joint encoders
        measure: rigid motion of the whole robot contributes nothing.
        """
        velocity, _ = self._foot_sliding_velocities()
        quaternion = self._rigid_body_quat.index_select(1, self.feet_indices)
        position = self._rigid_body_pos.index_select(1, self.feet_indices)
        center = position + quat_apply(quaternion, self._foot_collision_offsets)
        lever = center - self.root_states[:, None, :3]
        base_linear = self.root_states[:, None, 7:10]
        base_angular = self.root_states[:, None, 10:13].expand_as(lever)
        return velocity - base_linear - torch.cross(base_angular, lever, dim=-1)

    def _compute_push_gate(self):
        """Near 1 for a foot stroking against commanded travel, near 0 with it."""
        command = torch.zeros(self.num_envs, 3, device=self.device)
        command[:, :1] = self._clock_forward_command()
        command[:, 1] = self.commands[:, 1]
        heading = command / command.norm(dim=1, keepdim=True).clamp(min=1.0e-6)
        direction = quat_apply(self.base_quat, heading)
        relative = self._foot_velocity_relative_to_base()
        along = (relative * direction.unsqueeze(1)).sum(dim=-1)
        return torch.sigmoid(-along / self.cfg.osc.push_gate_speed)

    def _gated(self, load):
        """Apply the foot-motion gate when translating; else pass the load through.

        A turn in place has no travel direction, so its load is not gated.
        """
        gated = load * self._compute_push_gate()
        return torch.where(self._translating(), gated, load)

    def _compute_push_signal(self):
        """Gated normal load when moving; plain normal load otherwise.

        With foot-load errors configured, the clocks receive the sensed load.
        """
        if self._foot_load_is_sensed():
            return self._gated(self._sensed_foot_load)
        normal, _ = self._foot_loads()
        return self._gated(normal)

    def _step_oscillators(self, dt):
        self._refresh_sensed_foot_load(dt)
        super()._step_oscillators(dt)

    def _compute_oscillator_drift(self):
        """Equation (4) with the push signal in place of force magnitude."""
        local_feedback = self.osc_coupling * (
            torch.cos(self.oscillators) + self.osc_offset
        )
        return self.osc_omega - self._compute_push_signal() * local_feedback

    def _reward_push_load(self):
        """Push half: reward load on stroking feet, up to a per-foot cap.

        Credit stops at push_load_cap body weights, and the phase weight
        saturates once -sin(phase) reaches push_window_saturation, so a short
        hard burst earns no more than steady support through the push half.
        Both default to 1, which is the plain sine-weighted load. Oscillator
        feedback keeps the uncapped signal.
        """
        settings = self.cfg.reward_settings
        cap = getattr(settings, "push_load_cap", 1.0)
        saturation = getattr(settings, "push_window_saturation", 1.0)
        phase = (-torch.sin(self.oscillators) / saturation).clamp(min=0.0, max=1.0)
        normal, _ = self._foot_loads()
        return (phase * self._gated(normal.clamp(max=cap))).sum(dim=1)

    def _reward_glide_drag(self):
        """Glide half: penalize tangential force when moving, else normal load.

        Drag is in body weights and is not divided by friction, so a foot
        left on the ground costs in proportion to the friction it meets.
        """
        phase = torch.sin(self.oscillators).clamp(min=0.0)
        normal, tangent = self._foot_loads()
        cost = torch.where(self._moving(), tangent, normal)
        return -(phase * cost).sum(dim=1)

    def _reward_cost_of_transport(self):
        """Negative joint power per unit weight and unit speed.

        With ``cost_of_transport_speed = "command"`` the speed is the commanded
        one: the mechanical cost of transport when the robot moves as told, so
        every speed is charged per metre and not per second. With
        ``"travelled"`` it is the speed the body actually makes along the
        commanded direction, counted up to the commanded speed, so walking
        slower than told no longer lowers the charge. Speeds below the floor
        are charged as if at the floor, which keeps standing and slow walking
        from being priced out. The cap bounds what a fallen, thrashing robot
        can cost.
        """
        settings = self.cfg.reward_settings
        velocity = self.dof_vel.index_select(1, self.actuated_dof_indices)
        power = (self.torques * velocity).abs().sum(dim=1)
        weight = self._backend.link_mass.sum(dim=1) * 9.81
        floor = getattr(settings, "cost_of_transport_min_speed", 0.5)
        command = self.commands[:, :2]
        commanded = command.norm(dim=1)
        if getattr(settings, "cost_of_transport_speed", "command") == "travelled":
            direction = command / commanded.clamp(min=1.0e-6).unsqueeze(1)
            along = (self.base_lin_vel[:, :2] * direction).sum(dim=1)
            speed = torch.minimum(along, commanded).clamp(min=floor)
        else:
            speed = commanded.clamp(min=floor)
        cap = getattr(settings, "cost_of_transport_cap", 5.0)
        return -(power / (weight * speed)).clamp(max=cap)

    def _reward_tilt_rate(self):
        """Negative squared roll and pitch rate of the base [rad^2/s^2], capped.

        Quadratic, so it still has a slope at the rocking a trot produces,
        where the squared-exponential ang_vel_xy term is nearly flat. The cap
        bounds what untrained thrashing can cost.
        """
        rate = self.base_ang_vel[:, :2].square().sum(dim=1)
        return -rate.clamp(max=self.cfg.reward_settings.tilt_rate_cap)

    def _refresh_traction_observations(self, reset_mask=None):
        """FL/FR/RL/RR, each [forward, left] tangential over normal force.

        Positive forward is thrust on the robot along its heading; negative
        is drag. The level heading frame removes yaw only, as in Go2OscGRF.
        Feet at or below the directional load threshold read zero.
        """
        forces = self.contact_forces.index_select(1, self.feet_indices)
        x, y, z, w = self.base_quat.unbind(dim=-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        cosine = yaw.cos().unsqueeze(1)
        sine = yaw.sin().unsqueeze(1)
        tangent = torch.stack(
            (
                cosine * forces[..., 0] + sine * forces[..., 1],
                -sine * forces[..., 0] + cosine * forces[..., 1],
            ),
            dim=-1,
        )
        normal = forces[..., 2].clamp(min=0.0)
        weight = self._backend.link_mass.sum(dim=1, keepdim=True) * 9.81
        loaded = normal / weight > self.cfg.osc.directional_min_load
        limit = self.cfg.osc.traction_ratio_limit
        ratio = (tangent / normal.clamp(min=1.0e-9).unsqueeze(-1)).clamp(-limit, limit)
        observation = torch.where(
            loaded.unsqueeze(-1), ratio, torch.zeros_like(ratio)
        ).flatten(start_dim=1)
        observation = self._sense_traction(observation, reset_mask)
        if reset_mask is None:
            self.foot_traction_obs.copy_(observation)
        else:
            masked_update(self.foot_traction_obs, observation, reset_mask)

    def _refresh_friction_observation(self, reset_mask=None):
        """Cache the applied per-environment sliding friction for the critic."""
        friction = self._backend.contact_friction.unsqueeze(1)
        if reset_mask is None:
            self.contact_friction_obs.copy_(friction)
        else:
            masked_update(self.contact_friction_obs, friction, reset_mask)
        # Without per-foot draws every foot meets the environment's value.
        per_foot = getattr(self, "foot_friction_obs", None)
        if per_foot is not None and not self._ground_friction_varies():
            if reset_mask is None:
                per_foot.copy_(friction.expand_as(per_foot))
            else:
                masked_update(per_foot, friction.expand_as(per_foot), reset_mask)

    def _post_physics_step(self):
        super()._post_physics_step()
        self._refresh_traction_observations()
        self._refresh_friction_observation()

    def _post_compute_torques(self):
        super()._post_compute_torques()
        self._physics_step_in_control_step += 1

    def _post_decimation_step(self):
        self._count_down_sliding_stops()
        super()._post_decimation_step()
        self._change_ground_friction()
        self._refresh_friction_observation()
        self._refresh_friction_estimate()
        self._advance_speed_reference()
        self._advance_clock_speed()

    def _resample_commands(self, command_mask):
        ended = getattr(self, "_sliding_stop_ended", None)
        if ended is not None:
            # A stop that has run its time is followed by a new command.
            command_mask = command_mask | ended
            self._sliding_stop_ended = None
        super()._resample_commands(command_mask)
        self._command_sliding_stops(command_mask)
        self._draw_braking(command_mask)

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self._resample_variation(reset_mask)
        self._start_fast_slides(reset_mask)
        self._refresh_sensed_foot_load(reset_mask=reset_mask)
        self._refresh_traction_observations(reset_mask)
        self._refresh_friction_observation(reset_mask)
        self._refresh_friction_estimate(reset_mask)
        self._advance_speed_reference(reset_mask)
        self._advance_clock_speed(reset_mask)

    # ── Sensing errors, actuation delay and ground variation ───────────────

    def _sensing(self, name, default=None):
        return getattr(getattr(self.cfg, "sensing", None), name, default)

    @staticmethod
    def _validate_variation_cfg(cfg):
        """Reject malformed optional sensing, delay and ground settings."""

        def number(value):
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )

        def pair(value):
            return (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(number(item) for item in value)
            )

        mode = getattr(cfg.reward_settings, "cost_of_transport_speed", "command")
        if mode not in ("command", "travelled"):
            raise ValueError("cost_of_transport_speed must be 'command' or 'travelled'")
        delay = getattr(cfg.control, "actuation_delay_range", None)
        if delay is not None and not (pair(delay) and 0.0 <= delay[0] <= delay[1]):
            raise ValueError(
                "actuation_delay_range must be None or [low, high] seconds with "
                "0 <= low <= high"
            )
        sensing = getattr(cfg, "sensing", None)
        for name in (
            "friction_estimate_gain_range",
            "foot_load_gain_range",
            "foot_traction_gain_range",
        ):
            value = getattr(sensing, name, None)
            if value is not None and not (
                pair(value) and 0.0 < value[0] <= 1.0 <= value[1]
            ):
                raise ValueError(
                    f"{name} must be None or [low, high] with 0 < low <= 1 <= high"
                )
        for name in (
            "friction_estimate_noise_std",
            "friction_estimate_time_constant",
            "foot_load_noise_std",
            "foot_force_time_constant",
        ):
            value = getattr(sensing, name, 0.0)
            if not number(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        changes = getattr(cfg, "friction_changes", None)
        rate = getattr(changes, "rate", 0.0)
        if not number(rate) or rate < 0.0:
            raise ValueError("friction_changes rate must be finite and nonnegative")
        split = getattr(changes, "split_probability", 0.0)
        if not number(split) or not 0.0 <= split <= 1.0:
            raise ValueError("friction_changes split_probability must lie in [0, 1]")
        settings = cfg.reward_settings
        if getattr(settings, "speed_reference", False):
            for name in (
                "speed_reference_gain",
                "speed_reference_brake",
                "speed_reference_max_gain",
                "speed_reference_max_brake",
                "velocity_tracking_tolerance",
            ):
                value = getattr(settings, name, None)
                if not number(value) or value <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
        braking = getattr(cfg.commands, "braking_range", None)
        if braking is not None and not (
            pair(braking) and 0.0 < braking[0] <= braking[1]
        ):
            raise ValueError(
                "braking_range must be None or [low, high] fractions of "
                "friction times gravity with 0 < low <= high"
            )
        if getattr(cfg.osc, "clock_follows_speed_target", False):
            if not getattr(settings, "speed_reference", False):
                raise ValueError("clock_follows_speed_target needs speed_reference")
            floor = getattr(cfg.osc, "clock_speed_min_friction", None)
            if not number(floor) or floor <= 0.0:
                raise ValueError("clock_speed_min_friction must be finite and positive")
        if getattr(settings, "speed_reference_carve", False):
            if not getattr(settings, "speed_reference", False):
                raise ValueError("speed_reference_carve needs speed_reference")
            floor = getattr(settings, "speed_reference_carve_min_brake", None)
            if not number(floor) or not 0.0 <= floor <= 1.0:
                raise ValueError("speed_reference_carve_min_brake must lie in [0, 1]")
        for name in (
            "standing_turn_weight",
            "standing_lift_weight",
            "standing_lift_load",
        ):
            value = getattr(settings, name, 0.0)
            if not number(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        probability = getattr(cfg.commands, "sliding_stop_probability", 0.0)
        if not number(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("sliding_stop_probability must lie in [0, 1]")
        if probability > 0.0:
            speed = getattr(cfg.commands, "sliding_stop_min_speed", None)
            if not number(speed) or speed < 0.0:
                raise ValueError(
                    "sliding_stop_min_speed must be finite and nonnegative"
                )
            duration = getattr(cfg.commands, "sliding_stop_duration", None)
            if not number(duration) or duration <= 0.0:
                raise ValueError("sliding_stop_duration must be finite and positive")
        slides = getattr(cfg, "fast_slides", None)
        slide_probability = getattr(slides, "probability", 0.0)
        if not number(slide_probability) or not 0.0 <= slide_probability <= 1.0:
            raise ValueError("fast_slides probability must lie in [0, 1]")
        if slide_probability > 0.0:
            friction = getattr(slides, "friction_range", None)
            if not (pair(friction) and 0.0 < friction[0] <= friction[1]):
                raise ValueError(
                    "fast_slides friction_range must be [low, high] with "
                    "0 < low <= high"
                )
            speed = getattr(slides, "speed_range", None)
            if not (pair(speed) and 0.0 <= speed[0] <= speed[1]):
                raise ValueError(
                    "fast_slides speed_range must be [low, high] m/s with "
                    "0 <= low <= high"
                )
        tilt = getattr(getattr(cfg, "falls", None), "max_tilt", None)
        if tilt is not None and not (number(tilt) and 0.0 < tilt < 180.0):
            raise ValueError("falls max_tilt must be None or degrees in (0, 180)")
        overshoot = getattr(settings, "yaw_overshoot_tolerance", 0.0)
        if not number(overshoot) or overshoot < 0.0:
            raise ValueError("yaw_overshoot_tolerance must be finite and nonnegative")

    def _init_variation_buffers(self):
        n, device = self.num_envs, self.device
        # Dedicated stream: turning an option on leaves other draws unchanged.
        self._variation_rng = None
        seed = getattr(self.cfg, "seed", -1)
        if isinstance(seed, int) and seed >= 0:
            self._variation_rng = torch.Generator(device=device)
            self._variation_rng.manual_seed(seed + 7919)
        self.friction_estimate_obs = torch.zeros(n, 1, device=device)
        self.foot_friction_obs = self._backend.contact_friction.unsqueeze(1).repeat(
            1, 4
        )
        self._pre_fast_slide_friction = self.foot_friction_obs.clone()
        self._fast_slide_friction_override = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        self._friction_estimate_gain = torch.ones(n, 1, device=device)
        self._friction_estimate_filter = self.foot_friction_obs.mean(
            dim=1, keepdim=True
        )
        self._foot_load_gain = torch.ones(n, 4, device=device)
        self._foot_load_filter = torch.zeros(n, 4, device=device)
        self._sensed_foot_load = torch.zeros(n, 4, device=device)
        self._foot_traction_gain = torch.ones(n, 4, device=device)
        self._traction_filter = torch.zeros(n, 8, device=device)
        self._physics_step_in_control_step = 0
        self._env_rows = torch.arange(n, device=device)
        self._actuation_delay_steps = torch.zeros(n, dtype=torch.long, device=device)
        self._target_history = None
        delay = getattr(self.cfg.control, "actuation_delay_range", None)
        if delay is not None:
            physics_dt = self.dt / self.cfg.control.decimation
            longest = round(delay[1] / physics_dt)
            depth = -(-longest // self.cfg.control.decimation) + 1
            self._target_history = torch.zeros(
                n, depth, self.num_actuators, device=device
            )
        self._ground_friction_range = contact_friction_range(self.cfg)
        self._sliding_stop_steps = torch.zeros(n, dtype=torch.long, device=device)
        self._sliding_stop_ended = None
        self._speed_reference = torch.zeros(n, 2, device=device)
        brake = getattr(self.cfg.reward_settings, "speed_reference_brake", 0.9)
        self._braking_coefficient = torch.full((n,), float(brake), device=device)
        self._yaw_reference = torch.zeros(n, device=device)
        self.clock_speed_obs = torch.zeros(n, 1, device=device)

    def _uniform(self, shape, low, high):
        generator = getattr(self, "_variation_rng", None)
        sample = torch.rand(shape, generator=generator, device=self.device)
        return low + (high - low) * sample

    def _normal(self, shape):
        generator = getattr(self, "_variation_rng", None)
        return torch.randn(shape, generator=generator, device=self.device)

    def _resample_variation(self, reset_mask):
        """Draw the per-episode sensor gains, delay and ground of reset robots."""
        # Gains and delay follow logical episodes; only ground waits for a
        # fresh physical reset.
        for name, gain in (
            ("friction_estimate_gain_range", self._friction_estimate_gain),
            ("foot_load_gain_range", self._foot_load_gain),
            ("foot_traction_gain_range", self._foot_traction_gain),
        ):
            bounds = self._sensing(name)
            if bounds is not None:
                masked_update(gain, self._uniform(gain.shape, *bounds), reset_mask)
        if self._target_history is not None:
            low, high = self.cfg.control.actuation_delay_range
            physics_dt = self.dt / self.cfg.control.decimation
            steps = (self._uniform((self.num_envs,), low, high) / physics_dt).round()
            masked_update(self._actuation_delay_steps, steps.long(), reset_mask)
            held = self.dof_pos_target.unsqueeze(1).expand_as(self._target_history)
            masked_update(self._target_history, held, reset_mask)
        fresh = reset_mask & self._physical_reset_mask
        if self._ground_friction_varies():
            # A continued timeout keeps walking on the ground it was on.
            self._draw_ground_friction(fresh)
        elif getattr(self.cfg, "fast_slides", None) is not None:
            restore = fresh & self._fast_slide_friction_override
            ids = restore.nonzero(as_tuple=False).flatten()
            if ids.numel():
                self._backend.set_contact_friction_by_body(
                    ids,
                    self.feet_indices,
                    self._pre_fast_slide_friction.index_select(0, ids),
                )
                masked_update(
                    self.foot_friction_obs, self._pre_fast_slide_friction, restore
                )
        self._fast_slide_friction_override.masked_fill_(fresh, False)

    # Sliding stops: a robot that is moving when it gets a new command is
    # sometimes told to stop, with a turn rate, for a short fixed time.

    def _sliding_stops_enabled(self):
        commands = self.cfg.commands
        return (
            getattr(commands, "sliding_stop_probability", 0.0) > 0.0
            # Commands held for a whole episode (play, evaluation) stay as set.
            and commands.resampling_time < self.cfg.env.episode_length_s
        )

    def _command_sliding_stops(self, command_mask):
        """Replace the new command of some moving robots by a stop.

        The stop is zero forward and sideways speed with a turn rate drawn
        over the whole range, so one in five is a plain stop. Each robot whose
        command is redrawn starts a new count or clears its old one.
        """
        if not self._sliding_stops_enabled():
            return
        commands = self.cfg.commands
        speed = self.root_states[:, 7:9].norm(dim=1)
        draw = self._uniform((self.num_envs,), 0.0, 1.0)
        chosen = (
            command_mask
            & (speed > commands.sliding_stop_min_speed)
            & (draw < commands.sliding_stop_probability)
        )
        limit = self.command_ranges["yaw_vel"]
        turn = self._uniform((self.num_envs,), -limit, limit)
        self.commands[:, :2].masked_fill_(chosen.unsqueeze(1), 0.0)
        masked_update(self.commands[:, 2], turn, chosen)
        steps = round(commands.sliding_stop_duration / self.dt)
        self._sliding_stop_steps.masked_fill_(command_mask, 0)
        self._sliding_stop_steps.masked_fill_(chosen, steps)

    def _count_down_sliding_stops(self):
        """Mark the stops whose time ends on this control step."""
        if not self._sliding_stops_enabled():
            return
        running = self._sliding_stop_steps > 0
        self._sliding_stop_steps -= running.long()
        self._sliding_stop_ended = running & (self._sliding_stop_steps == 0)

    # Fast slides: some fresh episodes start sliding fast on slippery ground
    # with a stop already commanded, a case that is rare otherwise.

    def _fast_slides_enabled(self):
        slides = getattr(self.cfg, "fast_slides", None)
        return (
            getattr(slides, "probability", 0.0) > 0.0
            and self._sliding_stops_enabled()
            and getattr(self, "_ground_friction_range", None) is not None
        )

    def _start_fast_slides(self, reset_mask):
        """Start some freshly placed robots fast on low friction, told to stop.

        The robot moves along its heading at a speed from speed_range on
        ground of one friction from friction_range under every foot, and gets
        a sliding stop with a turn rate drawn over the whole range.
        """
        if not self._fast_slides_enabled():
            return
        slides = self.cfg.fast_slides
        draw = self._uniform((self.num_envs,), 0.0, 1.0)
        chosen = reset_mask & self._physical_reset_mask & (draw < slides.probability)
        friction = self._uniform((self.num_envs, 1), *slides.friction_range)
        speed = self._uniform((self.num_envs,), *slides.speed_range)
        limit = self.command_ranges["yaw_vel"]
        turn = self._uniform((self.num_envs,), -limit, limit)
        ids = chosen.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        # Startup randomization can postdate buffer initialization. Read the
        # current ground before saving the temporary slide's replacement.
        self._refresh_friction_observation(chosen)
        masked_update(self._pre_fast_slide_friction, self.foot_friction_obs, chosen)
        self._fast_slide_friction_override |= chosen
        friction = friction.expand_as(self.foot_friction_obs)
        self._backend.set_contact_friction_by_body(
            ids, self.feet_indices, friction.index_select(0, ids)
        )
        masked_update(self.foot_friction_obs, friction, chosen)
        x, y, z, w = self.root_states[:, 3:7].unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        velocity = torch.stack((yaw.cos() * speed, yaw.sin() * speed), dim=1)
        self.root_states[ids, 7:9] = velocity[ids]
        self._backend.reset_state(chosen)
        self._refresh_base_observations(chosen)
        self.commands[:, :2].masked_fill_(chosen.unsqueeze(1), 0.0)
        masked_update(self.commands[:, 2], turn, chosen)
        steps = round(self.cfg.commands.sliding_stop_duration / self.dt)
        self._sliding_stop_steps.masked_fill_(chosen, steps)

    # Falls: a robot tilted past falls.max_tilt degrees has fallen, even if
    # its base has not touched the ground.

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        tilt = getattr(getattr(self.cfg, "falls", None), "max_tilt", None)
        if tilt is not None:
            upright = -self.projected_gravity[:, 2]
            self.terminated |= upright < math.cos(math.radians(tilt))

    def _reward_yaw_overshoot(self):
        """Penalise turning faster than commanded, beyond a tolerance [rad/s]."""
        tolerance = getattr(self.cfg.reward_settings, "yaw_overshoot_tolerance", 0.0)
        excess = self.base_ang_vel[:, 2].abs() - self._yaw_target().abs() - tolerance
        return -excess.clamp(min=0.0).square()

    # Actuation delay: the motors follow the target the policy issued a little
    # earlier. Delays are whole physics steps and constant within an episode.

    def _record_joint_targets(self):
        self._physics_step_in_control_step = 0
        if getattr(self, "_target_history", None) is None:
            return
        self._target_history = self._target_history.roll(1, dims=1)
        self._target_history[:, 0] = self.dof_pos_target

    def _joint_position_target(self):
        history = getattr(self, "_target_history", None)
        if history is None:
            return self.dof_pos_target
        decimation = self.cfg.control.decimation
        late = self._actuation_delay_steps - self._physics_step_in_control_step
        back = ((late + decimation - 1) // decimation).clamp(min=0)
        return history[self._env_rows, back]

    # Foot-force sensing: the clocks and the traction input see each foot's
    # force through a gain that is fixed for the episode, a first-order lag
    # and, for the load, noise. Rewards keep the true forces.

    def _foot_load_is_sensed(self):
        return (
            self._sensing("foot_load_gain_range") is not None
            or self._sensing("foot_load_noise_std", 0.0) > 0.0
            or self._sensing("foot_force_time_constant", 0.0) > 0.0
        )

    def _refresh_sensed_foot_load(self, dt=None, reset_mask=None):
        if not self._foot_load_is_sensed():
            return
        normal, _ = self._foot_loads()
        measured = normal * self._foot_load_gain
        if reset_mask is not None:
            masked_update(self._foot_load_filter, measured, reset_mask)
            masked_update(self._sensed_foot_load, measured.clamp(max=1.0), reset_mask)
            return
        lag = self._sensing("foot_force_time_constant", 0.0)
        self._foot_load_filter.add_(
            (dt / (lag + dt)) * (measured - self._foot_load_filter)
        )
        noise = self._sensing("foot_load_noise_std", 0.0) * self._normal(normal.shape)
        self._sensed_foot_load.copy_((self._foot_load_filter + noise).clamp(0.0, 1.0))

    def _sense_traction(self, ratio, reset_mask=None):
        """Lag and scale the traction ratios the actor observes."""
        lag = self._sensing("foot_force_time_constant", 0.0)
        if self._sensing("foot_traction_gain_range") is None and lag <= 0.0:
            return ratio
        if reset_mask is None:
            dt = self.dt / self.cfg.control.decimation
            self._traction_filter.add_(
                (dt / (lag + dt)) * (ratio - self._traction_filter)
            )
        else:
            masked_update(self._traction_filter, ratio, reset_mask)
        return self._traction_filter * self._foot_traction_gain.repeat_interleave(
            2, dim=1
        )

    # Ground friction: with ``cfg.friction_changes`` the ground under a robot
    # is redrawn at every fresh episode and at random moments within one, and
    # with some probability each foot gets its own value.

    def _ground_friction_varies(self):
        changes = getattr(self.cfg, "friction_changes", None)
        if changes is None or getattr(self, "_ground_friction_range", None) is None:
            return False
        return (
            getattr(changes, "rate", 0.0) > 0.0
            or getattr(changes, "split_probability", 0.0) > 0.0
        )

    def _draw_ground_friction(self, mask):
        """Redraw the friction under the feet of the selected robots."""
        ids = mask.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        low, high = self._ground_friction_range
        shape = self.foot_friction_obs.shape
        shared = self._uniform((self.num_envs, 1), low, high).expand(shape)
        separate = self._uniform(shape, low, high)
        probability = getattr(self.cfg.friction_changes, "split_probability", 0.0)
        split = self._uniform((self.num_envs, 1), 0.0, 1.0) < probability
        friction = torch.where(split, separate, shared)
        self._backend.set_contact_friction_by_body(
            ids, self.feet_indices, friction.index_select(0, ids)
        )
        masked_update(self.foot_friction_obs, friction, mask)

    def _change_ground_friction(self):
        """Poisson changes of ground friction, checked once per control step."""
        if not self._ground_friction_varies():
            return
        rate = getattr(self.cfg.friction_changes, "rate", 0.0)
        if rate <= 0.0:
            return
        self._draw_ground_friction(
            self._uniform((self.num_envs,), 0.0, 1.0) < rate * self.dt
        )

    def _refresh_friction_estimate(self, reset_mask=None):
        """What a friction estimator would tell the actor.

        The mean friction under the four feet, through a first-order lag, a
        gain fixed for the episode and noise. With none of these configured
        it equals contact_friction_obs.
        """
        truth = self.foot_friction_obs.mean(dim=1, keepdim=True)
        if reset_mask is None:
            lag = self._sensing("friction_estimate_time_constant", 0.0)
            self._friction_estimate_filter.add_(
                (self.dt / (lag + self.dt)) * (truth - self._friction_estimate_filter)
            )
        else:
            masked_update(self._friction_estimate_filter, truth, reset_mask)
        estimate = self._friction_estimate_gain * self._friction_estimate_filter
        noise = self._sensing("friction_estimate_noise_std", 0.0)
        if noise > 0.0:
            estimate = estimate + noise * self._normal(estimate.shape)
        estimate = estimate.clamp(min=0.0)
        if reset_mask is None:
            self.friction_estimate_obs.copy_(estimate)
        else:
            masked_update(self.friction_estimate_obs, estimate, reset_mask)
