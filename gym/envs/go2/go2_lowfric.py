import torch

from gym.envs.go2.go2 import Go2
from gym.utils.locomotion_metrics import (
    FOOT_LOAD_THRESHOLD_N,
    GRAVITY_M_S2,
    contact_point_velocity,
)
from gym.utils.sampling import masked_update
from gym.utils.torch_quat import quat_rotate_inverse


def slide_speed_profile(time_s, speed, friction, runup_s, decel_scale=1.0):
    """Commanded forward speed of a run-up-and-slide episode at ``time_s``.

    Holds ``speed`` until ``runup_s``, then decays at decel_scale * mu * g,
    the deceleration of a body sliding on all feet, down to zero. Works
    elementwise on tensors (or broadcastable scalars).
    """
    decel = decel_scale * friction * GRAVITY_M_S2
    sliding = (time_s - runup_s).clamp_min(0.0)
    return (speed - decel * sliding).clamp_min(0.0)


def slide_phase_profile(time_s, speed, friction, runup_s, hold_s, decel_scale=1.0):
    """Slide-phase flag of a run-up-and-slide episode at ``time_s``.

    True from the end of the run-up until ``hold_s`` after the commanded
    speed reaches zero, matching the observation Go2LowFric trains with.
    """
    decel = decel_scale * friction * GRAVITY_M_S2
    end_s = runup_s + speed / decel + hold_s
    return (time_s >= runup_s) & (time_s < end_s)


class Go2LowFric(Go2):
    """Go2 locomotion across bimodal (ice / ground) contact friction.
    the true per-environment friction is a
    privileged observation for both actor and critic (a teacher policy).
    """

    def _init_buffers(self):
        super()._init_buffers()
        self.contact_friction = self._backend.contact_friction.unsqueeze(1)
        # IMU accelerometer: specific force (acceleration minus gravity) in
        # the body frame, as a real IMU reports it; (0, 0, g) standing still.
        # Finite difference of the base velocity over one physics step.
        # Read through get_state in units of g (scaling below); a sensor
        # for state estimators, not a policy input.
        self.scales.setdefault("base_lin_acc", GRAVITY_M_S2)
        self._physics_dt = self.dt / self.cfg.control.decimation
        self._prev_world_lin_vel = self.root_states[:, 7:10].clone()
        self.base_lin_acc = self._at_rest_specific_force()
        num_feet = len(self.feet_indices)
        # Seconds each foot has been continuously loaded (0 while in the air)
        # and whether it touched down during the last control step.
        self.foot_contact_time = torch.zeros(
            self.num_envs, num_feet, device=self.device
        )
        self.foot_touchdown = torch.zeros(
            self.num_envs, num_feet, dtype=torch.bool, device=self.device
        )
        # Run-up-and-slide episodes: which envs run one, and their run-up speed.
        self.slide_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.slide_speed = torch.zeros(self.num_envs, device=self.device)
        # Observation: 1 from the start of a slide until its post-stop hold
        # ends, else 0. On hardware this is part of the command ("slide now").
        self.slide_phase = torch.zeros(self.num_envs, 1, device=self.device)
        # True when a script drives slide_phase itself (evaluate_slide.py,
        # play.py --slide): the env then leaves it alone, so everything that
        # reads it inside step() (the residual hold, an in-env estimator)
        # sees the scripted phase instead of 0.
        self.external_slide_phase = False
        # Mid-episode friction switch: when (s after reset) and to what mu.
        # inf = no switch pending. time_since_friction_change restarts at
        # every reset and every switch (estimator settling is measured from it).
        self.friction_switch_time = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self.friction_switch_target = torch.zeros(self.num_envs, device=self.device)
        self.time_since_friction_change = torch.zeros(self.num_envs, device=self.device)
        # Writing friction every step costs a per-env loop on the CPU
        # backend, so switches are only checked when something can switch.
        self._friction_switch_enabled = self._friction_switch_settings() is not None

    def _friction_switch_settings(self):
        """The friction_switch config if switching is on, else None.

        Run snapshots that predate the option lack the section; they
        trained without switches.
        """
        settings = getattr(self.cfg, "friction_switch", None)
        if settings is None or settings.probability <= 0.0:
            return None
        return settings

    def _at_rest_specific_force(self):
        up = torch.zeros(self.num_envs, 3, device=self.device)
        up[:, 2] = GRAVITY_M_S2
        # root_states, not base_quat: right after a reset only the former
        # holds the new orientation.
        return quat_rotate_inverse(self.root_states[:, 3:7], up)

    def _post_physics_step(self):
        super()._post_physics_step()  # refreshes base_quat
        world_vel = self.root_states[:, 7:10]
        accel = (world_vel - self._prev_world_lin_vel) / self._physics_dt
        accel[:, 2] += GRAVITY_M_S2  # specific force = a - g, g = (0, 0, -g)
        self.base_lin_acc = quat_rotate_inverse(self.base_quat, accel)
        self._prev_world_lin_vel = world_vel.clone()

    def _slide_settings(self):
        """The commands config if it defines slide episodes, else None.

        Config snapshots saved with runs that predate slide episodes lack
        these fields; replaying them with --original_cfg must reproduce how
        they were trained, which is with no slide episodes.
        """
        commands = self.cfg.commands
        return commands if hasattr(commands, "slide_probability") else None

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self._apply_slide_action_scale()

    def _apply_slide_action_scale(self):
        """Residual hold: shrink joint targets toward the standing pose.

        dof_pos_target is an offset from the standing pose, so scaling it by
        slide_action_scale during the slide phase leaves the standing pose
        plus a small learned correction: enough to balance and lean, not to
        splay or step. A rigid standing pose brakes at ~mu*g but tips over
        at higher friction (scripted test on run Nov15_07-35-27_), so the
        policy keeps a little authority. Applied before the PD loop, so
        training, evaluation and playback all see it.
        """
        settings = self._slide_settings()
        # Snapshots of runs before the residual hold lack the setting; they
        # were trained with full authority in the slide.
        scale = getattr(settings, "slide_action_scale", None) if settings else None
        if scale is None:
            return
        factor = 1.0 - (1.0 - scale) * self.slide_phase
        self.dof_pos_target *= factor

    def _post_decimation_step(self):
        super()._post_decimation_step()  # may resample commands
        loaded = self.contact_forces[:, self.feet_indices, 2] > FOOT_LOAD_THRESHOLD_N
        self.foot_touchdown = loaded & (self.foot_contact_time == 0)
        self.foot_contact_time = (self.foot_contact_time + self.dt) * loaded
        self._apply_friction_switch()
        self._apply_slide_commands()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)  # resamples friction, then commands
        # A reset teleports the base: restart the accelerometer's difference
        # (out of place: these may be inference tensors after a PPO rollout).
        mask = reset_mask.unsqueeze(1)
        self._prev_world_lin_vel = torch.where(
            mask, self.root_states[:, 7:10], self._prev_world_lin_vel
        )
        self.base_lin_acc = torch.where(
            mask, self._at_rest_specific_force(), self.base_lin_acc
        )
        masked_update(
            self.foot_contact_time, torch.zeros_like(self.foot_contact_time), reset_mask
        )
        masked_update(
            self.foot_touchdown, torch.zeros_like(self.foot_touchdown), reset_mask
        )
        self._sample_slide_episodes(reset_mask)
        self._sample_friction_switches(reset_mask)
        self.time_since_friction_change.masked_fill_(reset_mask, 0.0)
        self._apply_slide_commands()

    def _sample_friction_switches(self, reset_mask):
        """Schedule a mid-episode switch to a different friction bin.

        Each resetting env that is not a slide episode switches with
        friction_switch.probability, at a time uniform in time_range, to a
        mu uniform in another bin, chosen in proportion to the bin weights
        (with two bins: always the other one).
        """
        settings = self._friction_switch_settings()
        if settings is None:
            return
        startup = self.cfg.domain_randomization.startup
        bins = startup.contact_friction_bins
        if bins is None or len(bins) < 2:
            raise ValueError("friction_switch needs at least two friction bins")
        bounds = torch.tensor(bins, dtype=torch.float, device=self.device)
        weights = torch.tensor(
            startup.contact_friction_bin_weights or [1.0] * len(bins),
            dtype=torch.float,
            device=self.device,
        )
        friction = self._backend.contact_friction
        # Bin of the current mu (bins are ordered; edges belong to the lower).
        current = (friction.unsqueeze(1) > bounds[:, 1]).sum(dim=1)
        current = current.clamp(max=len(bins) - 1)
        others = weights.expand(self.num_envs, -1).clone()
        others.scatter_(1, current.unsqueeze(1), 0.0)
        target_bin = torch.multinomial(others, 1).squeeze(1)
        rand = lambda: torch.rand(self.num_envs, device=self.device)  # noqa: E731
        chosen = ~self.slide_active & (rand() < settings.probability)
        low, high = settings.time_range
        when = low + (high - low) * rand()
        when = torch.where(chosen, when, torch.full_like(when, float("inf")))
        lo, hi = bounds[target_bin, 0], bounds[target_bin, 1]
        target = lo + (hi - lo) * rand()
        masked_update(self.friction_switch_time, when, reset_mask)
        masked_update(self.friction_switch_target, target, reset_mask)

    def schedule_friction_switch(self, env_ids, time_s, friction):
        """Force a switch (evaluation): env_ids change to ``friction`` at time_s."""
        self.friction_switch_time[env_ids] = torch.as_tensor(
            time_s, dtype=torch.float, device=self.device
        )
        self.friction_switch_target[env_ids] = torch.as_tensor(
            friction, dtype=torch.float, device=self.device
        )
        self._friction_switch_enabled = True

    def _apply_friction_switch(self):
        """Apply due switches; keeps time_since_friction_change current.

        One full masked write, like DomainRandomizer.randomize_episode, so no
        GPU->host sync is needed to find the switching envs.
        """
        self.time_since_friction_change += self.dt
        if not self._friction_switch_enabled:
            return
        time_s = self.episode_length_buf * self.dt
        due = time_s >= self.friction_switch_time
        current = self._backend.contact_friction
        self._backend.set_contact_friction(
            torch.arange(self.num_envs, device=self.device),
            torch.where(due, self.friction_switch_target, current),
        )
        self.friction_switch_time.masked_fill_(due, float("inf"))
        self.time_since_friction_change.masked_fill_(due, 0.0)

    def _resample_commands(self, command_mask):
        """Sample commands, then turn some of them into pure turn-in-place.

        The base sampler only zeroes forward/sideways speed when it is small,
        so pure turns are ~1% of commands, and both residual-hold seeds
        (runs Nov15_19-11-13_, Nov15_19-31-15_) failed turn-in-place.
        A fraction turn_in_place_probability of resamples become: no
        forward/sideways speed, yaw rate of at least turn_in_place_min_yaw.
        With turn_in_place_min_friction set, only envs on at least that
        friction get them (ground-only): turn-in-place on every surface made
        the ice slide fragile (runs 13, 13b), so ice keeps run 12's mix.
        """
        super()._resample_commands(command_mask)
        # Snapshots of runs before this option lack it; they trained without.
        probability = getattr(self.cfg.commands, "turn_in_place_probability", 0.0)
        if probability == 0.0:
            return
        turn = command_mask & (
            torch.rand(self.num_envs, device=self.device) < probability
        )
        # Snapshots of runs 13/13b lack the friction limit: turns everywhere.
        min_friction = getattr(self.cfg.commands, "turn_in_place_min_friction", None)
        if min_friction is not None:
            # True mu from the backend, not the (possibly misreported) input.
            turn &= self._backend.contact_friction >= min_friction
        yaw = self.commands[:, 2]
        minimum = self.cfg.commands.turn_in_place_min_yaw
        # Keep the sampled direction (and size, if already large enough).
        direction = torch.where(yaw < 0, -1.0, 1.0)
        turn_yaw = direction * yaw.abs().clamp_min(minimum)
        self.commands[:, :2].masked_fill_(turn.unsqueeze(1), 0.0)
        torch.where(turn, turn_yaw, yaw, out=self.commands[:, 2])

    def _sample_slide_episodes(self, reset_mask):
        """Choose which resetting envs run a run-up-and-slide episode.

        Only envs whose (just resampled) friction is at most
        slide_max_friction are eligible; each is chosen with
        slide_probability and gets a run-up speed from slide_speed_range.
        """
        settings = self._slide_settings()
        if settings is None:
            return
        eligible = self._backend.contact_friction <= settings.slide_max_friction
        chosen = eligible & (
            torch.rand(self.num_envs, device=self.device) < settings.slide_probability
        )
        low, high = settings.slide_speed_range
        speed = low + (high - low) * torch.rand(self.num_envs, device=self.device)
        masked_update(self.slide_active, chosen, reset_mask)
        masked_update(self.slide_speed, speed, reset_mask)

    def _apply_slide_commands(self):
        """Drive slide envs' commands and the slide-phase observation.

        A slide episode holds the run-up speed, decays it at the sliding
        deceleration, holds zero for slide_stop_hold_time, and then ends:
        the env gets a normal random command for the rest of the episode.
        Ending the slide keeps "zero forward command" from meaning "stand
        still" so often that turning in place is forgotten (run
        Nov15_06-10-02_ held zero until the episode ended and lost it).
        """
        settings = self._slide_settings()
        if settings is None:
            return
        time_s = self.episode_length_buf * self.dt
        friction = self._backend.contact_friction
        runup = settings.slide_runup_time
        # Snapshots of run Nov15_06-10-02_ predate the hold setting; that run
        # held zero until the episode ended, so replay it that way.
        hold = getattr(settings, "slide_stop_hold_time", None)
        if hold is not None:
            decel = settings.slide_decel_scale * friction * GRAVITY_M_S2
            end_time = runup + self.slide_speed / decel + hold
            finished = self.slide_active & (time_s >= end_time)
            self.slide_active &= ~finished
            self._resample_commands(finished)
        forward = slide_speed_profile(
            time_s, self.slide_speed, friction, runup, settings.slide_decel_scale
        )
        profile = torch.zeros_like(self.commands)
        profile[:, 0] = forward
        torch.where(
            self.slide_active.unsqueeze(1), profile, self.commands, out=self.commands
        )
        if not self.external_slide_phase:
            self.slide_phase[:, 0] = (self.slide_active & (time_s >= runup)).float()

    def _reward_slide_still(self):
        """Penalize joint motion during the slide phase (else 0).

        Mean squared joint velocity, normalized by the dof_vel scaling, times
        the slide-phase flag: rewards holding a posture and coasting instead
        of stepping through the slide.
        """
        normalized = self.dof_vel / self.scales["dof_vel"]
        return -self.slide_phase[:, 0] * normalized.square().mean(dim=1)

    def _reward_slide_pose(self):
        """Penalize leaving the standing pose during the slide phase (else 0).

        Mean squared joint deviation from the default standing pose (rad^2),
        times the slide-phase flag. On flat ground with isotropic friction,
        standing still on sliding feet already gives the maximum braking,
        mu*g; splaying legs only makes feet slide partly sideways, wasting
        friction. Run Nov15_06-56-01_ slid on three splayed legs with the
        rear-left leg held up (thigh ~0.73 rad off the pose).
        """
        deviation = self.dof_pos - self.default_dof_pos
        return -self.slide_phase[:, 0] * deviation.square().mean(dim=1)

    def _ice_gate(self):
        """~1 on low friction, ~0 on high: sigmoid((threshold - mu) / width).

        Uses the true mu from the backend, not the observation, which
        evaluation may overwrite to misreport friction (rewards may use
        privileged information; they only exist in training).
        """
        settings = self.cfg.reward_settings
        friction = self._backend.contact_friction
        return torch.sigmoid(
            (settings.glide_friction_threshold - friction)
            / settings.glide_friction_width
        )

    def _reward_glide(self):
        """Reward weight-bearing feet that stay down and slide with the body.

        Per foot, the product of three factors in [0, 1]:

        - slide: how far the contact point moves in the body's direction of
          travel, relative to body speed (1 = gliding at body speed; planted,
          rolling, backward or sideways sliding feet earn 0). Body speed
          below glide_min_command scales it down, so shuffling feet under a
          stationary body cannot earn it.
        - load share: vertical force over glide_full_load_fraction of body
          weight, so a foot lightly scuffing the ground earns little.
        - persistence: time continuously loaded over glide_min_contact_time,
          so taps and brief scuffs earn little; a sustained glide earns all.

        Averaged over the feet, gated to low friction, and zero unless the
        command asks to move. The first version used only slide x (load >
        5 N); run Nov15_04-33-19_ exploited it by dragging lightly loaded
        feet forward in rapid taps, which braked the robot.
        """
        settings = self.cfg.reward_settings
        feet = self.feet_indices
        foot_velocity = contact_point_velocity(
            self._rigid_body_lin_vel[:, feet],
            self._rigid_body_ang_vel[:, feet],
            settings.foot_radius,
        )[..., :2]
        body_velocity = self.root_states[:, 7:9]
        body_speed = body_velocity.norm(dim=1, keepdim=True)
        direction = body_velocity / body_speed.clamp_min(1e-6)
        along = (foot_velocity * direction.unsqueeze(1)).sum(dim=-1)
        slide = (along / body_speed.clamp_min(settings.glide_min_command)).clamp(
            0.0, 1.0
        )
        # link_mass is live and includes per-env mass randomization.
        weight = self._backend.link_mass.sum(dim=1, keepdim=True) * GRAVITY_M_S2
        load = self.contact_forces[:, feet, 2]
        load_share = (load / (settings.glide_full_load_fraction * weight)).clamp(
            0.0, 1.0
        )
        persistence = (self.foot_contact_time / settings.glide_min_contact_time).clamp(
            0.0, 1.0
        )
        glide = (slide * load_share * persistence).mean(dim=1)
        moving = self.commands[:, :2].norm(dim=1) > settings.glide_min_command
        return self._ice_gate() * glide * moving

    def _reward_ice_touchdowns(self):
        """Penalize each new foot contact on low friction (gated like glide).

        Returns minus the number of feet that touched down this control
        step, so rapid tapping costs in proportion to its rate.
        """
        return -self._ice_gate() * self.foot_touchdown.sum(dim=1)
