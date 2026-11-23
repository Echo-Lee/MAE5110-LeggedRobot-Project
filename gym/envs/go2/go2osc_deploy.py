"""Robust push-glide Go2 whose actor and clocks use only on-board signals.

``go2osc_push_glide_deploy`` is ``go2osc_push_glide_robust`` (friction-
limited speed target, carved sliding stops, hardware variation) trained
together with the state estimator of ``go2osc_push_glide_est``:

    actor       IMU, joints, governed command, previous targets, oscillator
                phases, the estimator's mu-hat and v-hat, and the speed
                target v_t
    command     the forward command the actor and v_t get leads v-hat by
    governor    at most command_governor.margin: at rest mu-hat cannot
                tell ice from grip, so a fast command is handed over as
                the body speeds up
    speed       v_t follows the governed command at rates set by mu-hat. An
    target      episode starts it at the body's forward speed; a robot
                starts at rest, so its controller starts v_t at zero
    clocks      driven by foot load from joint torques (F = -(J^T)^-1 tau)
                and gated by foot velocity from joint speeds (J qdot),
                both sampled once per control step, as the robot's 100 Hz
                loop samples them
    estimator   trained on the true friction and velocity while PPO trains
                the policy; saved in every checkpoint
    joint       targets are clamped inside the joint limits (hard stops on
    targets     the robot); control.joint_target_ramp closes that clamp
                during training so that a policy trained without it adapts
    hand-over   early in training the actor and the speed target read
                stand-ins for the estimates (the simulated friction
                estimate and the true body velocity with noise) while the
                estimator learns beside them; state_estimator.handover
                then blends them over to mu-hat and v-hat. On the robot
                there are only the estimates.

The critic and the rewards keep the simulator's true values. Everything the
robot has to compute is owned by the standalone custom configuration in
``go2_deploy/custom_implementation/go2osc_push_glide.py``.
"""

import torch

from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from go2_deploy.custom_implementation.go2osc_push_glide import (
    GRAVITY_M_S2,
    govern_command,
    leg_geometry,
    push_signal,
)
from gym.utils.sampling import masked_update

from learning.modules.history_estimator import FRICTION_CLIP

JOINT_TORQUE = "joint_torque"
CONTACT_FORCE = "contact_force"


class Go2OscPushGlideDeploy(Go2OscPushGlideEst):
    """Go2OscPushGlideEst with deployable clocks and speed target."""

    def _init_buffers(self):
        source = self._sensing("foot_load_source", CONTACT_FORCE)
        if source not in (JOINT_TORQUE, CONTACT_FORCE):
            raise ValueError(
                f"sensing.foot_load_source must be {JOINT_TORQUE!r} or "
                f"{CONTACT_FORCE!r}, got {source!r}"
            )
        self._load_from_torque = source == JOINT_TORQUE
        # Share of the estimates in what the actor and the speed target
        # read: 1 on the robot and in any environment not told a training
        # iteration; set_iteration moves it along state_estimator.handover.
        self.estimate_share = 1.0
        # How far the joint-target clamp has closed: 1 on the robot and in
        # any environment not told a training iteration (the clamp is at
        # control.joint_target_margin); control.joint_target_ramp moves it.
        self.joint_target_progress = 1.0
        self._check_handover()
        self._check_joint_target_ramp()
        super()._init_buffers()
        self._govern_commands()
        self._push_signal = torch.zeros(self.num_envs, 4, device=self.device)
        self._refresh_onboard_push_signal(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )

    # ── Hand-over from the simulator's stand-ins to the estimates ──────────

    def _handover(self):
        return getattr(self.cfg.state_estimator, "handover", None)

    def _check_handover(self):
        handover = self._handover()
        if handover is None:
            return
        if not 0 <= handover.start <= handover.end:
            raise ValueError(
                "state_estimator.handover needs 0 <= start <= end, got "
                f"{handover.start} and {handover.end}"
            )

    def set_iteration(self, iteration):
        """Runner hook: the training iteration moves the task's schedules.

        The joint-target clamp closes over control.joint_target_ramp, then
        the estimates' share rises over state_estimator.handover: each 0
        until its start, 1 from its end, linear between. A loaded
        checkpoint is therefore evaluated as it was trained.
        """
        self.joint_target_progress = _progress(self._joint_target_ramp(), iteration)
        self.estimate_share = _progress(self._handover(), iteration)
        stage = (
            int(4.0 * self.joint_target_progress),
            int(4.0 * self.estimate_share),
        )
        if stage != getattr(self, "_stage_reported", stage):
            margin = self._joint_target_margin()
            clamp = "none" if margin is None else f"{margin:+.2f} rad inside"
            print(
                f"[go2osc schedule] iteration {iteration}: joint targets kept "
                f"{clamp} the limits; the estimates' share is "
                f"{self.estimate_share:.2f}",
                flush=True,
            )
        self._stage_reported = stage
        if hasattr(self, "estimator"):
            self._publish_estimate()

    # ── Joint-target clamp that closes during training ─────────────────────

    def _joint_target_ramp(self):
        return getattr(self.cfg.control, "joint_target_ramp", None)

    def _check_joint_target_ramp(self):
        ramp = self._joint_target_ramp()
        if ramp is None:
            return
        if not 0 <= ramp.start <= ramp.end or ramp.slack < 0.0:
            raise ValueError(
                "control.joint_target_ramp needs 0 <= start <= end and "
                f"slack >= 0, got {ramp.start}, {ramp.end} and {ramp.slack}"
            )

    def _joint_target_margin(self):
        """The clamp's margin now: from `slack` beyond the limits to the
        configured margin inside them as joint_target_progress goes 0 to 1."""
        margin = super()._joint_target_margin()
        ramp = self._joint_target_ramp()
        if margin is None or ramp is None:
            return margin
        return margin - (1.0 - self.joint_target_progress) * (ramp.slack + margin)

    def _publish_estimate(self):
        """mu-hat and v-hat, or their stand-ins while the estimator learns.

        The stand-ins are what go2osc_push_glide_robust fed its actor: the
        simulated friction estimate (lag, gain, noise) and the true body
        velocity with uniform noise.
        """
        super()._publish_estimate()
        share = self.estimate_share
        if share >= 1.0:
            return
        noise = self._handover().velocity_noise
        velocity = self.base_lin_vel + noise * (
            2.0 * torch.rand_like(self.base_lin_vel) - 1.0
        )
        self.contact_friction_hat = torch.lerp(
            self.friction_estimate_obs.clamp(*FRICTION_CLIP),
            self.contact_friction_hat,
            share,
        )
        self.base_lin_vel_hat = torch.lerp(velocity, self.base_lin_vel_hat, share)

    # ── Speed target from the estimates ────────────────────────────────────

    def _estimates_ready(self):
        return hasattr(self, "contact_friction_hat")

    def _clock_friction(self):
        if not self._estimates_ready():
            return super()._clock_friction()
        return self.contact_friction_hat[:, 0]

    # v_t starts an episode at the body's true forward speed (the parent's
    # _clock_start_speed), not at v-hat: at a reset the estimator's history
    # is one repeated frame, which carries no motion. The robot's controller
    # starts at rest with v_t = 0 (go2osc_onboard.Go2OscOnboard.reset).

    # ── Command governor ───────────────────────────────────────────────────

    def _command_margin(self):
        governor = getattr(self.cfg, "command_governor", None)
        return None if governor is None else governor.margin

    def _govern_commands(self):
        """governed_commands [N, 3]: what the actor and the speed target get."""
        self.governed_commands = govern_command(
            self.commands, self.base_lin_vel_hat[:, 0], self._command_margin()
        )

    def _clock_target_command(self):
        if not hasattr(self, "governed_commands"):
            return super()._clock_target_command()
        return self.governed_commands[:, 0]

    def _advance_clock_speed(self, reset_mask=None):
        # Governed from the command now and the v-hat of the step before,
        # the order go2osc_onboard.Go2OscOnboard.step keeps.
        if self._estimates_ready():
            self._govern_commands()
        super()._advance_clock_speed(reset_mask)

    def _estimator_targets(self):
        """True labels: the mean friction under the feet, and body velocity.

        The feet can stand on different ground (friction_changes), which
        the environment-wide friction of the parent's labels does not show.
        """
        friction = self.foot_friction_obs.mean(dim=1, keepdim=True)
        return torch.cat((friction, self.base_lin_vel), dim=1)

    # ── Clock feedback from joint torques and speeds ───────────────────────

    def _refresh_sensed_foot_load(self, dt=None, reset_mask=None):
        if not self._load_from_torque:
            super()._refresh_sensed_foot_load(dt, reset_mask)

    def _compute_push_signal(self):
        if not self._load_from_torque or not hasattr(self, "_push_signal"):
            return super()._compute_push_signal()
        return self._push_signal

    def _sensor_noise(self, name, like):
        """Noise on a joint sensor [physical units], as the estimator's inputs."""
        if not getattr(self, "sensor_noise_enabled", False):
            return torch.zeros_like(like)
        std = getattr(self.cfg.state_estimator.sensor_noise, "std", {})
        return float(std.get(name, 0.0)) * self._normal(like.shape)

    def _refresh_onboard_push_signal(self, reset_mask=None):
        """Sample the clocks' push signal for the coming control step.

        Normal load per foot from the joint torques, through the load gain,
        lag and noise of ``sensing``; gated by the foot's velocity seen from
        the base, J qdot, along the commanded travel.
        """
        if not self._load_from_torque:
            return
        n = self.num_envs
        torques = self.torques + self._sensor_noise("torques", self.torques)
        dof_vel = self.dof_vel + self._sensor_noise("dof_vel", self.dof_vel)
        force, _, _ = self._leg_features(torques, dof_vel)
        measured = force.view(n, 4, 3)[..., 2].clamp(0.0, 1.0) * self._foot_load_gain
        if reset_mask is None:
            lag = self._sensing("foot_force_time_constant", 0.0)
            self._foot_load_filter.add_(
                (self.dt / (lag + self.dt)) * (measured - self._foot_load_filter)
            )
        else:
            masked_update(self._foot_load_filter, measured, reset_mask)
        noise = self._sensing("foot_load_noise_std", 0.0)
        load = self._foot_load_filter
        if noise > 0.0 and reset_mask is None:
            load = load + noise * self._normal(load.shape)
        load = load.clamp(0.0, 1.0)
        stroke = (self._leg_jacobian @ dof_vel.view(n, 4, 3).unsqueeze(-1)).squeeze(-1)
        signal = push_signal(
            load,
            stroke,
            self._clock_forward_command(),
            self.commands,
            self._translating(),
            self.cfg.osc.push_gate_speed,
        )
        if reset_mask is None:
            self._push_signal = signal
        else:
            self._push_signal = torch.where(
                reset_mask.unsqueeze(1), signal, self._push_signal
            )
        self._sensed_foot_load.copy_(load)

    # ── Non-finite values ──────────────────────────────────────────────────
    #
    # The estimator and the actor's input normaliser are shared by every
    # robot, so one non-finite signal can contaminate every prediction.
    # Derived values are made finite and counted; a simulator state that
    # is itself non-finite stops the run
    # with the robot and signal named.

    def _leg_features(self, torques, dof_vel):
        features = super()._leg_features(torques, dof_vel)
        if not hasattr(self, "_replaced_leg_values"):
            self._replaced_leg_values = torch.zeros(
                (), dtype=torch.long, device=self.device
            )
        finite = []
        for value in features:
            bad = ~torch.isfinite(value)
            self._replaced_leg_values += bad.sum()
            finite.append(torch.where(bad, torch.zeros_like(value), value))
        return tuple(finite)

    def _simulator_signals(self):
        return {
            "root_states": self.root_states,
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            "torques": self.torques,
        }

    def _stop_on_broken_simulation(self):
        """Raise, naming robot and signal, if the simulator state is not finite."""
        signals = self._simulator_signals()
        finite = torch.stack(
            [torch.isfinite(value).flatten(1).all(dim=1) for value in signals.values()]
        )
        if bool(finite.all()):
            return
        broken = (~finite).any(dim=0).nonzero().flatten()
        robot = int(broken[0])
        names = [name for name, ok in zip(signals, finite[:, robot]) if not ok]
        raise FloatingPointError(
            f"simulator state of robot {robot} is not finite ({', '.join(names)}); "
            f"{len(broken)} of {self.num_envs} robots affected. "
            f"Episode time {float(self.episode_length_buf[robot] * self.dt):.2f} s, "
            f"command {self.commands[robot].tolist()}, "
            f"foot friction {self.foot_friction_obs[robot].tolist()}."
        )

    def guard_counts(self):
        """Non-finite values handled so far (one GPU read): name -> count."""
        return {
            "leg_values_replaced": int(getattr(self, "_replaced_leg_values", 0)),
            "estimator_inputs_replaced": int(self.estimator.replaced_values),
            "estimator_samples_dropped": int(self.estimator.dropped_samples),
            "estimator_steps_skipped": int(self.estimator.skipped_steps),
        }

    def _report_estimator_loss(self):
        updates = self.estimator.updates
        if updates < self._estimator_updates_reported + 50:
            return
        self._estimator_updates_reported = updates
        loss = self.estimator.last_loss
        loss = "none yet" if loss is None else f"{loss:.4f}"
        print(f"[state estimator] update {updates}: loss {loss}", flush=True)
        counts = self.guard_counts()
        if counts != getattr(self, "_guard_counts_reported", None) and any(
            counts.values()
        ):
            self._guard_counts_reported = counts
            print(f"[go2osc guard] non-finite values handled: {counts}", flush=True)

    # ── Step and reset ─────────────────────────────────────────────────────

    def _post_decimation_step(self):
        self._stop_on_broken_simulation()
        super()._post_decimation_step()  # ends with the estimator's step
        self._refresh_onboard_push_signal()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        if not hasattr(self, "estimator"):  # reset() also runs during __init__
            return
        # The parent governed the command before the estimator saw the new
        # episode: govern it again from the fresh v-hat.
        self._advance_clock_speed(reset_mask)
        if hasattr(self, "_push_signal"):
            self._refresh_onboard_push_signal(reset_mask)


def _progress(schedule, iteration):
    """0 until schedule.start, 1 from schedule.end, linear between; 1 without one."""
    if schedule is None or iteration >= schedule.end:
        return 1.0
    if iteration <= schedule.start:
        return 0.0
    return (iteration - schedule.start) / (schedule.end - schedule.start)


def onboard_settings(env):
    """Everything go2osc_onboard.Go2OscOnboard needs, as plain numbers.

    Read from a built environment so that it carries the scales, posture
    and estimator layout the policy was trained with. JSON-serializable.
    """
    cfg = env.cfg
    osc, rewards = cfg.osc, cfg.reward_settings
    actuated = env.actuated_dof_indices
    features = list(env.estimator_features)

    def scale(name):
        value = torch.as_tensor(env.scales[name], dtype=torch.float).flatten()
        return value.tolist()

    bounds = env._joint_target_bounds()
    action_scale = torch.as_tensor(
        env.scales["dof_pos_target"], dtype=torch.float, device=env.device
    ).expand(len(actuated))
    limit = getattr(cfg.control, "action_limit", None)
    low = torch.full((len(actuated),), -float("inf"), device=env.device)
    high = -low
    if limit is not None:
        low, high = -limit * action_scale, limit * action_scale
    if bounds is not None:
        low = torch.maximum(low, bounds[0][0])
        high = torch.minimum(high, bounds[1][0])
    limits = env.dof_pos_limits.index_select(0, actuated)
    braking = getattr(cfg.commands, "braking_range", None)
    brake = rewards.speed_reference_brake if braking is None else braking[0]
    if braking is not None and braking[0] != braking[1]:
        raise ValueError("a deployed speed target needs one braking coefficient")
    return {
        "dt": float(env.dt),
        "substeps": int(cfg.control.decimation),
        "weight": float(env._backend.link_mass.sum(dim=1).mean() * GRAVITY_M_S2),
        "default_dof_pos": env.default_dof_pos[0, actuated].tolist(),
        "leg_geometry": leg_geometry(),
        "friction_clip": list(FRICTION_CLIP),
        "load_time_constant": float(env._sensing("foot_force_time_constant", 0.0)),
        "osc": {
            "stop_threshold": float(osc.stop_threshold),
            "sideways_stop_threshold": float(osc.sideways_stop_threshold),
            "turning_stop_threshold": float(osc.turning_stop_threshold),
            "omega_stop": float(osc.omega_stop),
            "omega_offset": float(osc.omega_offset),
            "omega_max": float(osc.omega_max),
            "coupling_stop": float(osc.coupling_stop),
            "coupling_moving": float(osc.coupling_moving),
            "stance_offset": float(osc.stance_offset),
            "push_gate_speed": float(osc.push_gate_speed),
            "directional_min_load": float(osc.directional_min_load),
        },
        "speed_target": {
            "min_friction": float(osc.clock_speed_min_friction),
            "gain": float(rewards.speed_reference_gain),
            "max_gain": float(rewards.speed_reference_max_gain),
            "brake": float(brake),
            "max_brake": float(rewards.speed_reference_max_brake),
            "carve": bool(getattr(rewards, "speed_reference_carve", False)),
            "carve_min_brake": float(
                getattr(rewards, "speed_reference_carve_min_brake", 0.0)
            ),
            # Lead [m/s] of the governed forward command over v-hat; None: off.
            "command_margin": env._command_margin(),
        },
        "estimator": {
            "features": features,
            "feature_dims": env._feature_dims(features),
            "frames": int(env.estimator.model.config["frames"]),
            "stride": int(env.estimator.model.config["stride"]),
            "scales": {name: scale(name) for name in features if name in env.scales},
        },
        "joint_limits": limits.tolist(),
        # Joint-target offsets [rad] the exported policy is clamped to.
        "target_offset_bounds": [low.tolist(), high.tolist()],
        "action_scale": action_scale.tolist(),
        "stiffness": _single_gain(cfg.control.stiffness, "stiffness"),
        "damping": _single_gain(cfg.control.damping, "damping"),
    }


def _single_gain(gains, name):
    """The one PD gain all joints share (the deploy runner sends one)."""
    values = set(float(value) for value in gains.values())
    if len(values) != 1:
        raise ValueError(f"the deploy runner needs one {name} for all joints")
    return values.pop()
