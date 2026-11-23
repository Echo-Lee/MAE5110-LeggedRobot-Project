"""Push-glide Go2 with a learned friction and body-velocity estimator."""

from pathlib import Path
import warnings

import torch

from gym import GYM_ROOT_DIR
from gym.envs.base.domain_randomization import (
    contact_friction_bins,
    contact_friction_range,
)
from gym.envs.go2.go2osc import Go2OscPushGlide
from gym.envs.go2.go2osc_est_config import GRAVITY_M_S2
from gym.utils.sampling import masked_update
from gym.utils.torch_quat import quat_apply, quat_rotate_inverse

from learning.modules.history_estimator import (
    FRICTION_CLIP,
    ConcurrentEstimator,
    HistoryStateEstimator,
    expand_estimator_inputs,
    targets_from_env,
)

LEGS = ("FL", "FR", "RL", "RR")  # order of the actuators and feet_indices
# Joint axes in each joint's child-body frame (resources/robots/go2/urdf/go2.urdf):
# hip abduction about x, thigh and calf about y. Body origins are the joints.
LEG_JOINT_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
# Damping of the Jacobian inverse [m]: keeps forces finite near a straight leg.
JACOBIAN_DAMPING = 0.01
# Forces are in body weights; clip swing-phase and near-singular spikes.
FOOT_FORCE_CLIP = 3.0
TRACTION_RATIO_CLIP = 2.0
# Estimator inputs computed from the joint torques and velocities.
JACOBIAN_FEATURES = ("foot_force_est", "foot_traction_est", "foot_vel_rel_base")


class Go2OscPushGlideEst(Go2OscPushGlide):
    """Go2OscPushGlide plus an IMU accelerometer, friction switches and an
    in-environment state estimator.

    Every control step a history estimator reads the last 0.5 s of
    ``state_estimator.features`` (signals a real Go2 has) and publishes
    ``contact_friction_hat`` (mu-hat, clipped) and ``base_lin_vel_hat``
    (body-frame v-hat). Physics, rewards and the critic keep the true
    values. While the runner trains (``set_learning(True)``) the estimator
    also trains on the true mu and velocity, and its weights are saved in
    every policy checkpoint (``checkpoint_state``).
    """

    def _init_buffers(self):
        super()._init_buffers()
        # IMU accelerometer: specific force (acceleration minus gravity) in
        # the body frame, as a real IMU reports it; (0, 0, g) standing still.
        # Finite difference of the base velocity over one physics step.
        self._physics_dt = self.dt / self.cfg.control.decimation
        self._prev_world_lin_vel = self.root_states[:, 7:10].clone()
        self.base_lin_acc = self._at_rest_specific_force()
        # Mid-episode friction switch: when (s after reset) and to what mu.
        # inf = no switch pending. time_since_friction_change restarts at
        # every reset and every switch (estimator settling is measured from it).
        self.friction_switch_time = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self.friction_switch_target = torch.zeros(self.num_envs, device=self.device)
        self.time_since_friction_change = torch.zeros(self.num_envs, device=self.device)
        # Writing friction every step costs a per-env loop on the CPU
        # backend, so switches are only applied when something can switch.
        self._friction_switch_enabled = self._friction_switch_settings() is not None
        self._init_leg_kinematics()
        self._init_estimator()

    # ------------------------------------------------------------------
    # Leg kinematics: Jacobian-transpose foot forces (deployable GRF)
    # ------------------------------------------------------------------
    def _init_leg_kinematics(self):
        """Body indices per leg and zeroed feature buffers.

        Features (all in the base frame, from joint encoders, motor torques
        and the IMU, i.e. what the real robot can compute):
            foot_force_est     [N, 12] ground force on each foot from
                               F = -(J^T)^-1 tau, in body weights
            foot_traction_est  [N, 8]  tangential / normal force per foot
                               (x, y), zero for unloaded feet
            foot_vel_rel_base  [N, 12] foot velocity relative to the base
                               origin, omega x r + J qdot [m/s]; for a stance
                               foot that does not slip it equals -v_base
        Leg inertia is ignored, so swing-phase forces are not ground forces.
        """
        names = self._backend.body_names
        self._leg_bodies = torch.tensor(
            [
                [names.index(f"{leg}_{part}") for part in ("hip", "thigh", "calf")]
                for leg in LEGS
            ],
            device=self.device,
        )  # [4 legs, 3 joints]
        self._leg_axes = torch.tensor(LEG_JOINT_AXES, device=self.device)
        self.foot_force_est = torch.zeros(self.num_envs, 12, device=self.device)
        self.foot_traction_est = torch.zeros(self.num_envs, 8, device=self.device)
        self.foot_vel_rel_base = torch.zeros(self.num_envs, 12, device=self.device)
        self._refresh_leg_kinematics()

    def _leg_jacobians(self):
        """(J [N, 4, 3, 3] base frame, foot position r [N, 4, 3] from base).

        Geometric Jacobian of the foot point: column j = axis_j x (p - o_j).
        """
        n = self.num_envs
        bodies = self._leg_bodies.flatten()
        origins = self._rigid_body_pos.index_select(1, bodies).view(n, 4, 3, 3)
        quats = self._rigid_body_quat.index_select(1, bodies).view(n, 4, 3, 4)
        axes = quat_apply(quats.reshape(-1, 4), self._leg_axes.repeat(n * 4, 1)).view(
            n, 4, 3, 3
        )
        foot_quat = self._rigid_body_quat.index_select(1, self.feet_indices)
        foot = self._rigid_body_pos.index_select(1, self.feet_indices) + quat_apply(
            foot_quat.reshape(-1, 4), self._foot_collision_offsets.reshape(-1, 3)
        ).view(n, 4, 3)
        columns = torch.cross(axes, foot.unsqueeze(2) - origins, dim=-1)
        base_quat = self.base_quat.repeat_interleave(12, dim=0)
        columns = quat_rotate_inverse(base_quat, columns.reshape(-1, 3)).view(
            n, 4, 3, 3
        )
        lever = quat_rotate_inverse(
            self.base_quat.repeat_interleave(4, dim=0),
            (foot - self.root_states[:, None, :3]).reshape(-1, 3),
        ).view(n, 4, 3)
        return columns.transpose(-1, -2), lever  # J[..., i, j]: row i, joint j

    def _refresh_leg_kinematics(self):
        self._leg_jacobian, self._leg_lever = self._leg_jacobians()
        force, ratio, velocity = self._leg_features(self.torques, self.dof_vel)
        self.foot_force_est = force
        self.foot_traction_est = ratio
        self.foot_vel_rel_base = velocity

    def _leg_features(self, torques, dof_vel):
        """(foot force [N, 12], traction ratio [N, 8], foot velocity [N, 12]).

        From joint torques and velocities [N, 12] with the Jacobians of the
        last _refresh_leg_kinematics; sensor noise passes in through them.
        """
        jacobian, lever = self._leg_jacobian, self._leg_lever
        n = self.num_envs
        tau = torques.view(n, 4, 3).unsqueeze(-1)
        # tau_motor + J^T F_ground = 0 (static leg): F = -J (J^T J + l^2 I)^-1 tau
        damping = JACOBIAN_DAMPING**2 * torch.eye(3, device=self.device)
        gram = jacobian.transpose(-1, -2) @ jacobian + damping
        force = -(jacobian @ torch.linalg.solve(gram, tau)).squeeze(-1)
        weight = (self._backend.link_mass.sum(dim=1) * GRAVITY_M_S2).view(n, 1, 1)
        force = (force / weight).clamp(-FOOT_FORCE_CLIP, FOOT_FORCE_CLIP)
        normal = force[..., 2]
        loaded = normal > self.cfg.osc.directional_min_load
        ratio = (force[..., :2] / normal.clamp(min=1e-6).unsqueeze(-1)).clamp(
            -TRACTION_RATIO_CLIP, TRACTION_RATIO_CLIP
        )
        ratio = torch.where(loaded.unsqueeze(-1), ratio, torch.zeros_like(ratio))
        qdot = dof_vel.view(n, 4, 3).unsqueeze(-1)
        omega = self.base_ang_vel.unsqueeze(1).expand(n, 4, 3)
        velocity = torch.cross(omega, lever, dim=-1) + (jacobian @ qdot).squeeze(-1)
        return force.reshape(n, 12), ratio.reshape(n, 8), velocity.reshape(n, 12)

    # ------------------------------------------------------------------
    # Accelerometer
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Friction switches
    # ------------------------------------------------------------------
    def _friction_switch_settings(self):
        """The friction_switch config if switching is on, else None.

        Switching needs the friction axis: with friction randomization off
        (e.g. apply_domain_randomization_override "off") the backend has one
        fixed friction and no per-env values to switch, so the task runs on
        fixed friction instead.
        """
        settings = getattr(self.cfg, "friction_switch", None)
        if settings is None or settings.probability <= 0.0:
            return None
        if contact_friction_range(self.cfg) is None:
            return None
        return settings

    def _friction_switch_bins(self):
        """(bounds [B, 2], weights [B]); one bin = the whole friction range."""
        binned = contact_friction_bins(self.cfg)
        if binned is None:
            envelope = contact_friction_range(self.cfg)
            if envelope is None:
                raise ValueError("friction_switch needs contact_friction_range")
            bins, weights = [envelope], [1.0]
        else:
            bins, weights = binned
        bounds = torch.tensor(bins, dtype=torch.float, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        return bounds, weights

    def _sample_friction_switches(self, reset_mask):
        """Schedule one mid-episode switch for some resetting envs.

        Each resetting env switches with friction_switch.probability, at a
        time uniform in time_range, to a mu uniform in another bin chosen in
        proportion to the bin weights (with a single bin: anywhere in it).
        """
        settings = self._friction_switch_settings()
        if settings is None:
            return
        bounds, weights = self._friction_switch_bins()
        others = weights.expand(self.num_envs, -1).clone()
        if len(bounds) > 1:
            # Bin of the current mu (bins are ordered; edges belong to the lower).
            friction = self._backend.contact_friction
            current = (friction.unsqueeze(1) > bounds[:, 1]).sum(dim=1)
            current = current.clamp(max=len(bounds) - 1)
            others.scatter_(1, current.unsqueeze(1), 0.0)
        target_bin = torch.multinomial(others, 1).squeeze(1)

        def rand():
            return torch.rand(self.num_envs, device=self.device)

        chosen = rand() < settings.probability
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
        self._refresh_friction_observation()  # critic sees the new mu now

    # ------------------------------------------------------------------
    # Estimator
    # ------------------------------------------------------------------
    def _init_estimator(self):
        settings = self.cfg.state_estimator
        model, untrained = self._load_estimator_model(settings)
        model, new_inputs = self._widen(model)
        if untrained:
            # Every input still has mean 0, std 1: take them from the first
            # batch of training data (a pretraining script may set them first).
            new_inputs = torch.ones(
                model.input_mean.numel(), dtype=torch.bool, device=self.device
            )
        self.estimator_features = model.config["features"]
        self._init_sensor_noise()
        self.estimator = ConcurrentEstimator(
            model,
            self.num_envs,
            self.device,
            learning_rate=settings.learning_rate,
            samples_per_step=settings.samples_per_step,
            update_interval=settings.update_interval,
            gradient_steps=settings.gradient_steps,
            normalization_updates=getattr(settings, "normalization_updates", 1),
        )
        self.estimator.unnormalized_inputs = new_inputs
        # Per-feature sensor-noise std for the estimator's inputs (set by
        # evaluations that perturb sensors; None = clean).
        self.estimator_noise_std = None
        self.contact_friction_hat = torch.zeros(self.num_envs, 1, device=self.device)
        self.base_lin_vel_hat = torch.zeros(self.num_envs, 3, device=self.device)
        self._estimator_updates_reported = 0

    def _feature_dims(self, features):
        return [self.get_states([name]).shape[1] for name in features]

    # ------------------------------------------------------------------
    # Sensor noise on the estimator's inputs
    # ------------------------------------------------------------------
    def _init_sensor_noise(self):
        """Per-input Gaussian noise std and the accelerometer-bias slot.

        ``state_estimator.sensor_noise`` gives a std per sensor in physical
        units (m/s^2, rad, rad/s, N m); they are converted to the scaled
        units get_states returns. The accelerometer also gets a constant
        bias per episode and axis, uniform in +-accelerometer_bias, as a real
        IMU's offset drifts between power-ups. Only the estimator's inputs
        are perturbed; the policy's observations keep the runner's noise.
        ``sensor_noise_enabled`` turns it off (e.g. for clean evaluation).
        """
        settings = getattr(self.cfg.state_estimator, "sensor_noise", None)
        self.sensor_noise_enabled = bool(settings and settings.enabled)
        std = getattr(settings, "std", {}) if settings else {}
        dims = self._feature_dims(self.estimator_features)
        pieces, offset, self._acc_slice = [], 0, None
        self._feature_slices = {}
        for name, dim in zip(self.estimator_features, dims):
            self._feature_slices[name] = slice(offset, offset + dim)
            value = torch.full((dim,), float(std.get(name, 0.0)), device=self.device)
            if name in self.scales:
                value = value / torch.as_tensor(
                    self.scales[name], dtype=torch.float, device=self.device
                )
            pieces.append(value)
            if name == "base_lin_acc":
                self._acc_slice = slice(offset, offset + dim)
            offset += dim
        self._sensor_noise_std = torch.cat(pieces)
        bias = getattr(settings, "accelerometer_bias", 0.0) if settings else 0.0
        self._acc_bias_limit = float(bias) / GRAVITY_M_S2  # in g, like the feature
        self.accelerometer_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._resample_accelerometer_bias(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )

    def _resample_accelerometer_bias(self, mask):
        drawn = (2 * torch.rand(self.num_envs, 3, device=self.device) - 1) * (
            self._acc_bias_limit
        )
        self.accelerometer_bias = torch.where(
            mask.unsqueeze(1), drawn, self.accelerometer_bias
        )

    def _estimator_inputs(self):
        """The estimator's input frame: clean features plus sensor noise.

        The Jacobian features are recomputed from the noisy torques and joint
        velocities, with the same noise draw as those inputs, so they carry
        the sensors' noise instead of a clean copy of it. Encoder angle noise
        (0.01 rad) would shift the Jacobian by millimetres and is left out.
        """
        features = self.get_states(self.estimator_features)
        if not self.sensor_noise_enabled:
            return features
        noise = self._sensor_noise_std * torch.randn_like(features)
        features = features + noise
        if self._acc_slice is not None:
            features[:, self._acc_slice] += self.accelerometer_bias
        if any(name in self._feature_slices for name in JACOBIAN_FEATURES):
            torques = self.torques + self._physical_noise("torques", noise)
            dof_vel = self.dof_vel + self._physical_noise("dof_vel", noise)
            noisy = dict(zip(JACOBIAN_FEATURES, self._leg_features(torques, dof_vel)))
            for name in JACOBIAN_FEATURES:
                if name in self._feature_slices:
                    value = noisy[name]
                    if name in self.scales:
                        value = value / torch.as_tensor(
                            self.scales[name], dtype=torch.float, device=self.device
                        )
                    features[:, self._feature_slices[name]] = value
        return features

    def _physical_noise(self, name, noise):
        """Noise on ``name`` in physical units: the draw already applied to that
        estimator input if it is one, else a fresh draw with the config std."""
        if name in self._feature_slices:
            value = noise[:, self._feature_slices[name]]
            if name in self.scales:
                value = value * torch.as_tensor(
                    self.scales[name], dtype=torch.float, device=self.device
                )
            return value
        std = getattr(self.cfg.state_estimator.sensor_noise, "std", {})
        return float(std.get(name, 0.0)) * torch.randn_like(getattr(self, name))

    def _load_estimator_model(self, settings):
        """(initial estimator, untrained?): the configured file, else untrained."""
        path = Path(settings.init_path) if settings.init_path else None
        if path is not None and not path.is_absolute():
            path = Path(GYM_ROOT_DIR) / path
        if path is not None:
            if not path.exists():
                raise FileNotFoundError(f"state estimator file {path} not found")
            model, _ = HistoryStateEstimator.load(path, self.device)
            return model, False
        warnings.warn(
            "starting from an untrained state estimator: valid for pretraining "
            "or when a go2osc_push_glide_est checkpoint supplies its weights"
        )
        features = list(settings.features)
        model = HistoryStateEstimator(
            features=features,
            feature_dims=self._feature_dims(features),
            frames=settings.frames,
            stride=settings.stride,
            hidden_dims=settings.hidden_dims,
        )
        return model, True

    def _widen(self, model):
        """(model, mask of new inputs or None) taking the configured features."""
        wanted = list(self.cfg.state_estimator.features)
        if list(model.config["features"]) == wanted:
            return model.to(self.device), None
        model, new_inputs = expand_estimator_inputs(
            model, wanted, self._feature_dims(wanted)
        )
        return model.to(self.device), new_inputs.to(self.device)

    def _estimator_targets(self):
        """True (mu, vx, vy, vz) labels [N, 4] the estimator learns."""
        return targets_from_env(self)

    def _publish_estimate(self):
        self.contact_friction_hat = self.estimator.mu_hat.clamp(
            *FRICTION_CLIP
        ).unsqueeze(1)
        self.base_lin_vel_hat = self.estimator.v_hat

    def _report_estimator_loss(self):
        updates = self.estimator.updates
        if updates >= self._estimator_updates_reported + 50:
            self._estimator_updates_reported = updates
            print(
                f"[state estimator] update {updates}: "
                f"loss {self.estimator.last_loss:.4f}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Step and reset
    # ------------------------------------------------------------------
    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._apply_friction_switch()
        self._refresh_leg_kinematics()
        self.estimator.step(
            self._estimator_inputs(),
            self._estimator_targets(),
            self.estimator_noise_std,
        )
        self._publish_estimate()
        self._report_estimator_loss()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)  # resamples friction, then commands
        if not hasattr(self, "estimator"):  # reset() also runs during __init__
            return
        # A reset teleports the base: restart the accelerometer's difference
        # (out of place: these may be inference tensors after a PPO rollout).
        mask = reset_mask.unsqueeze(1)
        self._prev_world_lin_vel = torch.where(
            mask, self.root_states[:, 7:10], self._prev_world_lin_vel
        )
        self.base_lin_acc = torch.where(
            mask, self._at_rest_specific_force(), self.base_lin_acc
        )
        self._sample_friction_switches(reset_mask)
        self.time_since_friction_change.masked_fill_(reset_mask, 0.0)
        self._refresh_leg_kinematics()
        self._resample_accelerometer_bias(reset_mask)
        self.estimator.reset(reset_mask, self._estimator_inputs())
        self._publish_estimate()

    # ------------------------------------------------------------------
    # Hooks called by OnPolicyRunner
    # ------------------------------------------------------------------
    def set_learning(self, enabled):
        self.estimator.learning = bool(enabled) and self.cfg.state_estimator.learn

    def checkpoint_state(self):
        return {
            "state_estimator": self.estimator.state_dict(),
            "estimator_config": self.estimator.model.config,
        }

    def load_checkpoint_state(self, state):
        if "state_estimator" not in state:
            return
        saved = state["state_estimator"]
        config = state["estimator_config"]
        model = HistoryStateEstimator(
            features=config["features"],
            feature_dims=config["feature_dims"],
            frames=config["frames"],
            stride=config["stride"],
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
        )
        model.load_state_dict(saved["model"])
        model, new_inputs = self._widen(model)
        self.estimator.set_model(model, new_inputs)
        if new_inputs is None:  # same architecture: keep Adam's moments
            self.estimator.optimizer.load_state_dict(saved["optimizer"])
        self.estimator_features = model.config["features"]
        self._init_sensor_noise()
        everyone = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.estimator.reset(everyone, self._estimator_inputs())
        self._publish_estimate()
