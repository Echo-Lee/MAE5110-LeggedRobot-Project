"""Robust push-glide Go2 whose actor and clocks use only on-board signals.

``go2osc_push_glide_deploy`` is ``go2osc_push_glide_robust`` (friction-
limited speed target, carved sliding stops, hardware variation) trained
together with the state estimator of ``go2osc_push_glide_est``:

    actor       IMU, joints, command, previous targets, oscillator phases,
                the estimator's mu-hat and v-hat, and the speed target v_t
    speed       v_t follows the command at rates set by mu-hat, and starts
    target      an episode at the estimated forward speed
    clocks      driven by foot load from joint torques (F = -(J^T)^-1 tau)
                and gated by foot velocity from joint speeds (J qdot),
                both sampled once per control step, as the robot's 100 Hz
                loop samples them
    estimator   trained on the true friction and velocity while PPO trains
                the policy; saved in every checkpoint

The critic and the rewards keep the simulator's true values. Everything the
robot has to compute is in ``go2osc_onboard.py``, which shares this file's
arithmetic.
"""

import torch

from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from gym.envs.go2.go2osc_onboard import (
    GRAVITY_M_S2,
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
        super()._init_buffers()
        self._push_signal = torch.zeros(self.num_envs, 4, device=self.device)
        self._refresh_onboard_push_signal(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )

    # ── Speed target from the estimates ────────────────────────────────────

    def _estimates_ready(self):
        return hasattr(self, "contact_friction_hat")

    def _clock_friction(self):
        if not self._estimates_ready():
            return super()._clock_friction()
        return self.contact_friction_hat[:, 0]

    def _clock_start_speed(self):
        if not self._estimates_ready():
            return super()._clock_start_speed()
        return self.base_lin_vel_hat[:, :1]

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

    # ── Step and reset ─────────────────────────────────────────────────────

    def _post_decimation_step(self):
        super()._post_decimation_step()  # ends with the estimator's step
        self._refresh_onboard_push_signal()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        if not hasattr(self, "estimator"):  # reset() also runs during __init__
            return
        # The parent restarted v_t before the estimator saw the new episode.
        self._advance_clock_speed(reset_mask)
        if hasattr(self, "_push_signal"):
            self._refresh_onboard_push_signal(reset_mask)


class Go2OscPushGlideDeployOracle(Go2OscPushGlideDeploy):
    """Simulation-only ablation: true friction and velocity drive the clock."""

    def _clock_friction(self):
        return self.foot_friction_obs.mean(dim=1)

    def _clock_start_speed(self):
        return self.base_lin_vel[:, :1]


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
