"""History-based state estimator: friction mu and base velocity from proprioception.

A policy that reads the true contact friction is not deployable (no robot
measures mu). This estimator predicts mu-hat and the body-frame base linear
velocity v-hat from a short history of what a real Go2 does measure: IMU
angular velocity and gravity direction, joint positions/velocities, the
previous joint targets, and the operator's commands. It is trained by
supervised regression in simulation (scripts/train_state_estimator.py).

    ProprioHistory      ring buffer of per-step feature vectors, per env
    HistoryStateEstimator   MLP with frozen input normalization and output
                        scaling; save/load as one checkpoint file
    EstimatorInTheLoop  replaces the policy's friction input with mu-hat
                        while an environment runs (evaluation, playback)
    ConcurrentEstimator history + model owned by an environment (task
                        go2osc_push_glide_est); optionally keeps training on
                        the true labels while PPO trains the policy
"""

import torch
from torch import nn

from learning.modules.state_estimator import StateEstimatorNN

# Inputs a real Go2 can provide (names of environment attributes, read
# through get_states so they carry the same scaling the policy sees).
# Estimators trained before the accelerometer (stage 2a/2b files, runs
# 15-17 checkpoints) use LEGACY_FEATURES.
LEGACY_FEATURES = (
    "base_ang_vel",
    "projected_gravity",
    "commands",
    "dof_pos_obs",
    "dof_vel",
    "dof_pos_target",
    "slide_phase",
)
# + IMU accelerometer (specific force, body frame, in g). While all feet
# slide, deceleration = mu * g, so mu is observable during a slide.
DEFAULT_FEATURES = LEGACY_FEATURES + ("base_lin_acc",)
# Estimated quantities and the scale each is divided by inside the loss, so
# mu (0.05-1.2) and velocity (~+-3 m/s) carry comparable weight.
TARGETS = ("contact_friction", "base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z")
TARGET_SCALES = (0.5, 1.0, 1.0, 0.5)
# mu-hat handed to the policy is clipped to the range it was trained on.
FRICTION_CLIP = (0.02, 1.5)


class ProprioHistory:
    """Last ``frames * stride`` per-step feature vectors for every env.

    ``window()`` returns every ``stride``-th frame (oldest first), so the
    estimator sees ``frames * stride`` control steps with ``frames`` inputs.
    After a reset an env's whole buffer is filled with its current
    features, so no data from the previous episode leaks in.
    """

    def __init__(self, num_envs, num_features, frames, stride, device):
        self.frames = frames
        self.stride = stride
        self.length = frames * stride
        self.buffer = torch.zeros(num_envs, self.length, num_features, device=device)

    # Out-of-place updates: the buffer may be written inside and outside
    # torch.inference_mode (PPO rollout vs. evaluation), and a masked
    # where() needs no GPU->host sync.
    def reset(self, mask, features):
        self.buffer = torch.where(
            mask.view(-1, 1, 1), features.unsqueeze(1), self.buffer
        )

    def append(self, features):
        self.buffer = torch.cat((self.buffer[:, 1:], features.unsqueeze(1)), dim=1)

    def window(self):
        # Newest frame last; take every stride-th step counting back from it.
        picked = self.buffer[:, self.stride - 1 :: self.stride]
        return picked.reshape(picked.shape[0], -1)


class HistoryStateEstimator(nn.Module):
    """MLP from a feature history to (mu-hat, v-hat)."""

    def __init__(
        self,
        features=DEFAULT_FEATURES,
        feature_dims=None,
        frames=25,
        stride=2,
        hidden_dims=(512, 256, 128),
        activation="elu",
    ):
        super().__init__()
        if feature_dims is None:
            raise ValueError("feature_dims (size of each feature) is required")
        self.config = {
            "features": list(features),
            "feature_dims": list(feature_dims),
            "frames": frames,
            "stride": stride,
            "hidden_dims": list(hidden_dims),
            "activation": activation,
            "targets": list(TARGETS),
            "target_scales": list(TARGET_SCALES),
        }
        self.num_features = sum(feature_dims)
        num_inputs = self.num_features * frames
        self.net = StateEstimatorNN(
            num_inputs,
            len(TARGETS),
            hidden_dims=list(hidden_dims),
            activation=activation,
        )
        self.register_buffer("input_mean", torch.zeros(num_inputs))
        self.register_buffer("input_std", torch.ones(num_inputs))
        self.register_buffer("target_scale", torch.tensor(TARGET_SCALES))

    def set_input_normalization(self, inputs):
        """Freeze per-input mean/std from a batch of history windows."""
        self.input_mean.copy_(inputs.mean(dim=0))
        self.input_std.copy_(inputs.std(dim=0).clamp_min(1e-3))

    def forward(self, window):
        """Normalized prediction (targets divided by their scales)."""
        return self.net.evaluate((window - self.input_mean) / self.input_std)

    def predict(self, window):
        """(mu-hat [N], v-hat [N, 3]) in physical units."""
        out = self.forward(window) * self.target_scale
        return out[:, 0], out[:, 1:4]

    def loss(self, window, targets):
        """MSE between normalized prediction and normalized targets [N, 4]."""
        return nn.functional.mse_loss(self.forward(window), targets / self.target_scale)

    def save(self, path, extra=None):
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = cls(
            features=config["features"],
            feature_dims=config["feature_dims"],
            frames=config["frames"],
            stride=config["stride"],
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval(), checkpoint["extra"]


def expand_estimator_inputs(model, features, feature_dims):
    """Copy of ``model`` that also takes the extra ``features``.

    ``features`` must contain all of the model's features (any order). The
    new inputs get zero first-layer weights, so the copy predicts exactly
    what ``model`` does until it is trained. Returns the copy and a bool
    mask over its inputs marking the new columns (their normalization is
    still unset: mean 0, std 1).
    """
    old = model.config
    missing = set(old["features"]) - set(features)
    if missing:
        raise ValueError(f"cannot drop estimator features {sorted(missing)}")
    new = HistoryStateEstimator(
        features=features,
        feature_dims=feature_dims,
        frames=old["frames"],
        stride=old["stride"],
        hidden_dims=old["hidden_dims"],
        activation=old["activation"],
    )
    offset, start = 0, {}
    for name, dim in zip(features, feature_dims):
        start[name] = offset
        offset += dim
    per_frame = torch.cat(
        [
            torch.arange(start[name], start[name] + dim)
            for name, dim in zip(old["features"], old["feature_dims"])
        ]
    )
    # Window layout is frame-major: input = frame * num_features + feature.
    frames = torch.arange(old["frames"]).unsqueeze(1) * new.num_features
    columns = (frames + per_frame).flatten()
    is_new = torch.ones(new.input_mean.numel(), dtype=torch.bool)
    is_new[columns] = False
    source = model.state_dict()
    target = new.state_dict()
    first = "net.NN.0.weight"
    with torch.no_grad():
        for key, value in source.items():
            if key == first:
                target[key].zero_()
                target[key][:, columns] = value.cpu()
            elif key in ("input_mean", "input_std"):
                target[key][columns] = value.cpu()
            else:
                target[key].copy_(value.cpu())
    new.load_state_dict(target)
    return new, is_new


def targets_from_env(env):
    """True (mu, vx, vy, vz) labels [N, 4] from the simulator."""
    return torch.cat(
        (env._backend.contact_friction.unsqueeze(1), env.base_lin_vel), dim=1
    )


def features_from_env(env, features):
    return env.get_states(list(features))


class EstimatorInTheLoop:
    """Feed mu-hat to a policy instead of the true friction.

    Call ``reset()`` after the environment resets and ``step()`` once per
    control step before the policy acts. ``step`` appends the current
    features, predicts, and overwrites ``env.contact_friction`` (the
    policy's friction input) with clipped mu-hat. Physics and rewards keep
    the true friction. The latest predictions stay in ``mu_hat`` / ``v_hat``.
    With ``feed_policy=False`` it only observes: the policy keeps the true
    mu (a diagnostic for how much mu-hat depends on its own feedback).
    """

    def __init__(self, env, path, device, feed_policy=True):
        if hasattr(env, "contact_friction_hat"):
            raise ValueError(
                "this task runs its own estimator (contact_friction_hat); "
                "do not add --state_estimator"
            )
        self.model, self.extra = HistoryStateEstimator.load(path, device)
        config = self.model.config
        self.features = config["features"]
        self.env = env
        self.feed_policy = feed_policy
        self.history = ProprioHistory(
            env.num_envs,
            self.model.num_features,
            config["frames"],
            config["stride"],
            device,
        )
        self.mu_hat = None
        self.v_hat = None
        self.reset(torch.ones(env.num_envs, dtype=torch.bool, device=device))

    @torch.no_grad()
    def reset(self, mask):
        self.history.reset(mask, features_from_env(self.env, self.features))
        self._predict()

    @torch.no_grad()
    def step(self, noise_std=None, generator=None):
        """Append current features (plus optional sensor noise) and predict.

        ``noise_std`` is a per-feature standard deviation in the scaled
        units ``get_states`` returns, as BaseRunner.get_noise produces.
        """
        features = features_from_env(self.env, self.features)
        if noise_std is not None:
            features = features + noise_std * torch.randn(
                features.shape, device=features.device, generator=generator
            )
        self.history.append(features)
        self._predict()

    def _predict(self):
        self.mu_hat, self.v_hat = self.model.predict(self.history.window())
        if not self.feed_policy:
            return
        clipped = self.mu_hat.clamp(*FRICTION_CLIP)
        # A plain tensor, not the backend's live view: the policy reads it.
        self.env.contact_friction = clipped.unsqueeze(1)


# Bound on an estimator input, in the scaled units get_states returns
# (real signals stay below about 100). A simulator fault can produce
# readings many orders above it.
FEATURE_LIMIT = 1.0e4


class ConcurrentEstimator:
    """An estimator run, and optionally trained, inside an environment.

    The environment calls ``reset(mask, features)`` for resetting envs and
    ``step(features, targets)`` once per control step; ``mu_hat`` / ``v_hat``
    then hold the current predictions. While ``learning`` is on, each step
    keeps ``samples_per_step`` random (window, true target) pairs and every
    ``update_interval`` steps the model takes ``gradient_steps`` Adam steps
    on them. Training the estimator alongside the policy keeps it matched
    to the states the changing policy visits (labels stay the true mu, v).

    Steps may run under ``torch.inference_mode`` (the PPO rollout); updates
    leave it locally, so the stored samples are cloned into normal tensors.
    """

    def __init__(
        self,
        model,
        num_envs,
        device,
        learning_rate=3e-4,
        samples_per_step=256,
        update_interval=24,
        gradient_steps=4,
        normalization_updates=1,
    ):
        self.num_envs = num_envs
        self.device = device
        self.learning_rate = learning_rate
        # Updates over which unnormalized inputs keep averaging their
        # mean/std before those freeze. 1 takes them from the first batch;
        # more suits an estimator trained from scratch beside a policy
        # whose first states (falling over) differ from the later ones.
        self.normalization_updates = max(1, int(normalization_updates))
        self.set_model(model)
        self.samples_per_step = min(samples_per_step, num_envs)
        self.update_interval = update_interval
        self.gradient_steps = gradient_steps
        self.learning = False
        self.updates = 0
        self.last_loss = None
        self._windows, self._targets = [], []
        # Guard counters (device tensors: counting never waits for the GPU).
        # One non-finite sample would turn every weight into NaN, and with
        # them the estimates of every robot.
        self.replaced_values = torch.zeros((), dtype=torch.long, device=device)
        self.dropped_samples = torch.zeros((), dtype=torch.long, device=device)
        self.skipped_steps = 0
        self.mu_hat = torch.zeros(num_envs, device=device)
        self.v_hat = torch.zeros(num_envs, 3, device=device)

    def set_model(self, model, unnormalized_inputs=None):
        """Use ``model`` from now on: fresh history and optimizer.

        ``unnormalized_inputs`` (bool mask over inputs, from
        expand_estimator_inputs) get their mean/std from the first batch of
        training samples; with zero weights this does not change outputs.
        Call ``reset`` for all envs afterwards to fill the new history.
        """
        self.model = model.to(self.device).eval()
        config = model.config
        self.history = ProprioHistory(
            self.num_envs,
            model.num_features,
            config["frames"],
            config["stride"],
            self.device,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.unnormalized_inputs = (
            None if unnormalized_inputs is None else unnormalized_inputs.to(self.device)
        )
        self._normalization_batches = 0
        self._windows, self._targets = [], []

    def reset(self, mask, features):
        self.history.reset(mask, self._finite(features))
        self._predict()

    def step(self, features, targets=None, noise_std=None):
        """Append one frame, predict, and (while learning) collect and train.

        ``noise_std`` adds sensor noise to the frame (evaluation only).
        """
        if noise_std is not None:
            features = features + noise_std * torch.randn_like(features)
        self.history.append(self._finite(features))
        window = self._predict()
        if not self.learning or targets is None:
            return
        picked = torch.randint(
            self.num_envs, (self.samples_per_step,), device=self.device
        )
        self._windows.append(window[picked])
        self._targets.append(targets[picked])
        if len(self._windows) >= self.update_interval:
            self._update()

    def _finite(self, features):
        """Features with non-finite entries zeroed and absurd ones bounded."""
        bad = ~torch.isfinite(features)
        bounded = torch.where(bad, torch.zeros_like(features), features).clamp(
            -FEATURE_LIMIT, FEATURE_LIMIT
        )
        self.replaced_values += (bounded != features).sum()
        return bounded

    def _predict(self):
        window = self.history.window()
        with torch.no_grad():
            self.mu_hat, self.v_hat = self.model.predict(window)
        return window

    def _update(self):
        with torch.inference_mode(False), torch.enable_grad():
            windows = torch.cat(self._windows).clone()
            targets = torch.cat(self._targets).clone()
            self._windows, self._targets = [], []
            # A sample without a finite label (a robot whose simulation
            # broke) teaches nothing and would poison the whole update.
            usable = torch.isfinite(windows).all(dim=1) & (
                torch.isfinite(targets).all(dim=1)
            )
            self.dropped_samples += (~usable).sum()
            windows, targets = windows[usable], targets[usable]
            if windows.shape[0] == 0:
                self.skipped_steps += self.gradient_steps
                return
            if self.unnormalized_inputs is not None:
                # Running average of the batch statistics, then frozen.
                new = self.unnormalized_inputs
                rate = 1.0 / (self._normalization_batches + 1)
                mean = windows[:, new].mean(dim=0)
                correction = 0 if windows.shape[0] == 1 else 1
                std = windows[:, new].std(dim=0, correction=correction).clamp_min(1e-3)
                model = self.model
                model.input_mean[new] = torch.lerp(model.input_mean[new], mean, rate)
                blended = torch.lerp(model.input_std[new], std, rate)
                model.input_std[new] = blended.clamp_min(1e-3)
                self._normalization_batches += 1
                if self._normalization_batches >= self.normalization_updates:
                    self.unnormalized_inputs = None
            self.model.train()
            order = torch.randperm(windows.shape[0], device=self.device)
            for chunk in order.chunk(self.gradient_steps):
                loss = self.model.loss(windows[chunk], targets[chunk])
                self.optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float("inf")
                )
                if not torch.isfinite(norm):
                    # As in PPO2: one non-finite gradient step would turn
                    # every weight into NaN.
                    self.optimizer.zero_grad()
                    self.skipped_steps += 1
                    continue
                self.optimizer.step()
                self.last_loss = loss.item()
            self.model.eval()
        self.updates += 1

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._windows, self._targets = [], []
