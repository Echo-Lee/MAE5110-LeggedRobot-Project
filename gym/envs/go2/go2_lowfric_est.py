from pathlib import Path
import warnings

import torch

from gym import GYM_ROOT_DIR
from gym.envs.go2.go2_lowfric import Go2LowFric
from learning.modules.history_estimator import (
    FRICTION_CLIP,
    LEGACY_FEATURES,
    ConcurrentEstimator,
    HistoryStateEstimator,
    expand_estimator_inputs,
    targets_from_env,
)


class Go2LowFricEst(Go2LowFric):
    """Deployable student of Go2LowFric: the actor reads mu-hat, not mu.

    A history estimator runs inside the environment every control step and
    publishes ``contact_friction_hat`` (clipped mu-hat, the actor's friction
    input) and ``base_lin_vel_hat``. Physics, rewards, command sampling and
    the critic keep the true friction. During PPO training (the runner calls
    ``set_learning(True)``) the estimator also keeps training on the true
    labels, so it follows the states the fine-tuned policy visits; its
    weights are saved inside every policy checkpoint (``checkpoint_state``)
    so a checkpoint always replays with its own estimator.

    With ``state_estimator.latch_in_slide`` the actor's mu-hat is frozen at
    its value when the slide phase starts and held until the phase ends:
    the surface does not change during a slide, mu-hat is accurate while
    running up, and in the held slide pose the history carries little
    friction information (run 16: mu 0.1 read as ~0.6 mid-slide).

    The estimator's inputs are ``state_estimator.features``. An estimator
    (initial file or checkpoint) trained on fewer features is widened with
    zero weights for the new ones (expand_estimator_inputs): it predicts as
    before and learns to use them while training (run 18: accelerometer).
    """

    def _init_buffers(self):
        super()._init_buffers()
        settings = self.cfg.state_estimator
        model = self._load_estimator_model(settings)
        model, new_inputs = self._widen(model)
        self.estimator_features = model.config["features"]
        self.estimator = ConcurrentEstimator(
            model,
            self.num_envs,
            self.device,
            learning_rate=settings.learning_rate,
            samples_per_step=settings.samples_per_step,
            update_interval=settings.update_interval,
            gradient_steps=settings.gradient_steps,
        )
        # Per-feature sensor-noise std for the estimator's inputs (set by
        # evaluations that perturb sensors; None = clean).
        self.estimator_noise_std = None
        self.estimator.unnormalized_inputs = new_inputs
        self.contact_friction_hat = torch.zeros(self.num_envs, 1, device=self.device)
        self.base_lin_vel_hat = torch.zeros(self.num_envs, 3, device=self.device)
        # Runs 15-16 predate the latch: their config snapshots lack it.
        self.latch_in_slide = getattr(settings, "latch_in_slide", False)
        self.latched_mu_hat = torch.zeros(self.num_envs, device=self.device)
        self._was_sliding = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._estimator_updates_reported = 0

    def _load_estimator_model(self, settings):
        """The initial estimator: trained file if present, else untrained.

        The untrained model is only a shell for a checkpoint that carries
        estimator weights (runner.load replaces them); starting PPO from it
        would feed the policy garbage, hence the warning.
        """
        path = Path(settings.init_path) if settings.init_path else None
        if path is not None and not path.is_absolute():
            path = Path(GYM_ROOT_DIR) / path
        if path is not None and path.exists():
            model, _ = HistoryStateEstimator.load(path, self.device)
            return model
        warnings.warn(
            f"state estimator file {path} not found: starting from an untrained "
            "estimator; only valid when loading a go2_lowfric_est checkpoint"
        )
        features = list(settings.features)
        dims = [self.get_states([name]).shape[1] for name in features]
        return HistoryStateEstimator(
            features=features,
            feature_dims=dims,
            frames=settings.frames,
            stride=settings.stride,
            hidden_dims=settings.hidden_dims,
        )

    def _widen(self, model):
        """(model, mask of new inputs or None) taking the configured features."""
        wanted = list(self.cfg.state_estimator.features)
        if list(model.config["features"]) == wanted:
            return model, None
        dims = [self.get_states([name]).shape[1] for name in wanted]
        model, new_inputs = expand_estimator_inputs(model, wanted, dims)
        return model.to(self.device), new_inputs.to(self.device)

    def _legacy_estimator_config(self):
        """Architecture of checkpoints saved without one (runs 15-17)."""
        settings = self.cfg.state_estimator
        features = list(LEGACY_FEATURES)
        return {
            "features": features,
            "feature_dims": [self.get_states([f]).shape[1] for f in features],
            "frames": settings.frames,
            "stride": settings.stride,
            "hidden_dims": list(settings.hidden_dims),
            "activation": "elu",
        }

    def _post_decimation_step(self):
        super()._post_decimation_step()  # friction switch, slide commands
        self.estimator.step(
            self.get_states(self.estimator_features),
            targets_from_env(self),
            self.estimator_noise_std,
        )
        self._publish_estimate()
        self._report_estimator_loss()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        if hasattr(self, "estimator"):  # reset() also runs during __init__
            self.estimator.reset(reset_mask, self.get_states(self.estimator_features))
            self._publish_estimate()

    def _publish_estimate(self):
        mu_hat = self.estimator.mu_hat
        if self.latch_in_slide:
            sliding = self.slide_phase[:, 0] > 0.5
            starting = sliding & ~self._was_sliding
            self.latched_mu_hat = torch.where(starting, mu_hat, self.latched_mu_hat)
            mu_hat = torch.where(sliding, self.latched_mu_hat, mu_hat)
            self._was_sliding = sliding
        self.contact_friction_hat = mu_hat.clamp(*FRICTION_CLIP).unsqueeze(1)
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

    # Hooks called by OnPolicyRunner.
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
        config = state.get("estimator_config") or self._legacy_estimator_config()
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
        everyone = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.estimator.reset(everyone, self.get_states(self.estimator_features))
        self._publish_estimate()
