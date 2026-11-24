"""Friction-limited clock targets and standing rewards for Go2 locomotion."""

import torch

from gym.utils.sampling import masked_update


class Go2OscSpeedTargetMixin:
    """Keep the speed target, clock schedule and standing objective together."""

    def _yaw_target(self):
        """Turn rate the rewards follow: the command, or its carved share."""
        reference = getattr(self, "_yaw_reference", None)
        if reference is None or not getattr(
            self.cfg.reward_settings, "speed_reference_carve", False
        ):
            return self.commands[:, 2]
        return reference

    def _reward_tracking_ang_vel(self):
        if not getattr(self.cfg.reward_settings, "speed_reference_carve", False):
            return super()._reward_tracking_ang_vel()
        tolerance = self.cfg.reward_settings.yaw_tracking_tolerance
        error = (self._yaw_target() - self.base_ang_vel[:, 2]) / tolerance
        return torch.exp(-error.square())

    def _reward_tracking_lin_vel(self):
        if not self._speed_reference_enabled():
            return super()._reward_tracking_lin_vel()
        tolerance = self.cfg.reward_settings.velocity_tracking_tolerance
        error = (self._speed_reference - self.root_states[:, 7:9]) / tolerance
        return torch.exp(-error.square().sum(dim=1))

    # Braking: each command draw sets how hard slowing down is asked for, as
    # a fraction of friction times gravity. Low fractions are long slides.

    def _draw_braking(self, command_mask):
        braking = getattr(self.cfg.commands, "braking_range", None)
        if braking is None:
            return
        draw = self._uniform((self.num_envs,), *braking)
        masked_update(self._braking_coefficient, draw, command_mask)

    # Clock speed target v_t: the forward speed the clocks follow. It moves
    # toward the forward command no faster than the friction estimate allows,
    # so the stepping rate builds with the speed the robot can reach. The
    # actor observes it.

    def _clock_speed_enabled(self):
        return bool(
            getattr(self.cfg.osc, "clock_follows_speed_target", False)
        ) and hasattr(self, "clock_speed_obs")

    def _clock_forward_command(self):
        if self._clock_speed_enabled():
            return self.clock_speed_obs
        return super()._clock_forward_command()

    def _clock_friction(self):
        """Friction [N] the clock speed target assumes: the estimate."""
        return self.friction_estimate_obs[:, 0]

    def _clock_target_command(self):
        """Forward command [N] the speed target moves toward."""
        return self.commands[:, 0]

    def _clock_start_speed(self):
        """Forward speed [N, 1] a fresh episode starts v_t at."""
        return self.base_lin_vel[:, :1]

    def _advance_clock_speed(self, reset_mask=None):
        """Move v_t toward the forward command at friction-limited rates.

        Speeding up: speed_reference_gain of the estimated friction times
        gravity; slowing or reversing: the braking coefficient of it, less
        what a carved turn takes (see _carve).
        Both use the friction estimate, as a robot would. A fresh episode
        starts v_t at the body's forward speed.
        """
        if not self._clock_speed_enabled():
            return
        if reset_mask is not None:
            masked_update(
                self.clock_speed_obs, self._clock_start_speed(), reset_mask.unsqueeze(1)
            )
            return
        settings = self.cfg.reward_settings
        target = self._clock_target_command()
        speed = self.clock_speed_obs[:, 0]
        friction = self._clock_friction().clamp(
            min=self.cfg.osc.clock_speed_min_friction
        )
        limit = friction * 9.81
        speeding = (target.abs() > speed.abs()) & (target * speed >= 0.0)
        budget = (self._braking_coefficient * limit).clamp(
            max=settings.speed_reference_max_brake
        )
        _, _, brake = self._carve(speed.abs(), budget)
        rate = torch.where(
            speeding,
            (settings.speed_reference_gain * limit).clamp(
                max=settings.speed_reference_max_gain
            ),
            brake,
        )
        step = rate * self.dt
        speed.add_(torch.maximum(torch.minimum(target - speed, step), -step))

    def _reward_standing(self):
        """Penalise motion while the clocks stand.

        Planar body speed squared [m^2/s^2], plus standing_turn_weight times
        the yaw rate squared, plus standing_lift_weight per foot carrying less
        than standing_lift_load body weights. Zero on the moving branch.
        """
        settings = self.cfg.reward_settings
        normal, _ = self._foot_loads()
        lifted = (normal < settings.standing_lift_load).float().sum(dim=1)
        cost = (
            self.root_states[:, 7:9].square().sum(dim=1)
            + settings.standing_turn_weight * self.base_ang_vel[:, 2].square()
            + settings.standing_lift_weight * lifted
        )
        return -cost * self._slow_command_mask()[:, 0].float()

    # Speed reference: the horizontal velocity that speed tracking is measured
    # against. It moves toward the commanded velocity no faster than the ground
    # lets a robot speed up or brake, so a command that friction cannot deliver
    # at once still pays for doing what is possible.

    def _speed_reference_enabled(self):
        return bool(getattr(self.cfg.reward_settings, "speed_reference", False))

    def _commanded_world_velocity(self):
        x, y, z, w = self.root_states[:, 3:7].unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        cos, sin = yaw.cos(), yaw.sin()
        forward, sideways = self.commands[:, 0], self.commands[:, 1]
        return torch.stack(
            (cos * forward - sin * sideways, sin * forward + cos * sideways), dim=1
        )

    def _advance_speed_reference(self, reset_mask=None):
        """Move the reference one control step, or restart it at a reset.

        World frame, so that a body turning while it slides to a stop is not
        charged for the turn. A reset starts the reference at the speed the
        body has. The rate is a fraction of true friction times gravity, one
        fraction for speeding up and one for braking, each with a cap.
        """
        if not self._speed_reference_enabled():
            return
        if reset_mask is not None:
            masked_update(
                self._speed_reference,
                self.root_states[:, 7:9],
                reset_mask.unsqueeze(1),
            )
            return
        settings = self.cfg.reward_settings
        target = self._commanded_world_velocity()
        reference = self._speed_reference
        limit = self.foot_friction_obs.mean(dim=1) * 9.81
        budget = (self._braking_coefficient * limit).clamp(
            max=settings.speed_reference_max_brake
        )
        speed = reference.norm(dim=1)
        carving, turn, brake = self._carve(speed, budget)
        self._yaw_reference.copy_(torch.where(carving, turn, self.commands[:, 2]))
        # A carve turns the direction of travel with the commanded turn.
        angle = turn * self.dt * carving
        cos, sin = angle.cos(), angle.sin()
        reference.copy_(
            torch.stack(
                (
                    cos * reference[:, 0] - sin * reference[:, 1],
                    sin * reference[:, 0] + cos * reference[:, 1],
                ),
                dim=1,
            )
        )
        change = target - reference
        slowing = (target * reference).sum(dim=1) < reference.square().sum(dim=1)
        rate = torch.where(
            slowing,
            brake,
            (settings.speed_reference_gain * limit).clamp(
                max=settings.speed_reference_max_gain
            ),
        )
        distance = change.norm(dim=1).clamp(min=1.0e-9)
        fraction = (rate * self.dt / distance).clamp(max=1.0)
        reference += change * fraction.unsqueeze(1)

    def _carve(self, speed, budget):
        """Split a braking budget [m/s^2] between turning and slowing.

        With speed_reference_carve on, a robot told zero planar speed and a
        turn rate carves: its direction of travel turns at that rate, which
        takes speed times turn rate of sideways acceleration, and it brakes
        with what is left, so the two together use the whole budget. The
        turn is capped so that at least speed_reference_carve_min_brake of
        the budget still slows the robot. Returns the carving mask, the turn
        rate to follow [rad/s] and the braking rate [m/s^2].
        """
        settings = self.cfg.reward_settings
        turn = self.commands[:, 2]
        if not getattr(settings, "speed_reference_carve", False):
            return torch.zeros_like(speed, dtype=torch.bool), turn, budget
        carving = (self.commands[:, :2].abs().sum(dim=1) == 0.0) & (turn != 0.0)
        floor = settings.speed_reference_carve_min_brake
        sideways_limit = budget * (1.0 - floor**2) ** 0.5
        cap = sideways_limit / speed.clamp(min=1.0e-6)
        turn = torch.maximum(torch.minimum(turn, cap), -cap)
        sideways = (speed * turn.abs()).clamp(max=sideways_limit)
        carved = (budget.square() - sideways.square()).clamp(min=0.0).sqrt()
        brake = torch.where(carving, carved, budget)
        return carving, torch.where(carving, turn, self.commands[:, 2]), brake
