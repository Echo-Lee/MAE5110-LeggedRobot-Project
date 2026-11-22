"""Robust push-glide Go2 (sliding stops, turn-rate tracking) with the
learned friction and body-velocity estimator."""

from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from gym.utils.sampling import masked_update


class Go2OscPushGlideRobustEst(Go2OscPushGlideEst):
    """Go2OscPushGlideEst on the go2osc_push_glide_robust ground.

    Differences from Go2OscPushGlideEst, all from the robust task's per-foot
    friction:

    * ``friction_estimate_obs``, the friction the speed clock brakes and
      carves with, is mu-hat instead of the robust task's synthetic estimate
      (true mean friction through a lag, gain and noise).
    * The estimator's friction label is ``contact_friction``, which the
      backends report as the mean over the feet after per-foot writes; it
      is the quantity the synthetic estimate stood for.
    * Every ground redraw (fresh episode or Poisson change) restarts
      ``time_since_friction_change``.
    * Scheduled switches (evaluation) set all four feet, so the per-foot
      friction the critic and rewards read stays in step with the physics.
    """

    # ------------------------------------------------------------------
    # mu-hat drives the speed clock
    # ------------------------------------------------------------------
    def _refresh_friction_estimate(self, reset_mask=None):
        hat = getattr(self, "contact_friction_hat", None)
        if hat is None:  # during __init__, before the estimator exists
            super()._refresh_friction_estimate(reset_mask)
            return
        if reset_mask is None:
            self.friction_estimate_obs.copy_(hat)
        else:
            masked_update(self.friction_estimate_obs, hat, reset_mask)

    def _post_decimation_step(self):
        # The robust step advances the clock with the previous mu-hat (one
        # control step old, as on the robot); then the estimator steps.
        super()._post_decimation_step()
        self._refresh_friction_estimate()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)  # estimator reset and published last
        if hasattr(self, "estimator"):
            self._refresh_friction_estimate(reset_mask)

    # ------------------------------------------------------------------
    # Per-foot ground friction
    # ------------------------------------------------------------------
    def _draw_ground_friction(self, mask):
        super()._draw_ground_friction(mask)
        since = getattr(self, "time_since_friction_change", None)
        if since is not None:
            since.masked_fill_(mask, 0.0)

    def _apply_friction_switch(self):
        """Apply due scheduled switches to all four feet."""
        self.time_since_friction_change += self.dt
        if not self._friction_switch_enabled:
            return
        time_s = self.episode_length_buf * self.dt
        due = time_s >= self.friction_switch_time
        ids = due.nonzero(as_tuple=False).flatten()
        if ids.numel():
            friction = self.friction_switch_target.unsqueeze(1).expand_as(
                self.foot_friction_obs
            )
            self._backend.set_contact_friction_by_body(
                ids, self.feet_indices, friction.index_select(0, ids)
            )
            masked_update(self.foot_friction_obs, friction, due)
            # A fast slide's temporary ground is replaced, not restored later.
            self._fast_slide_friction_override.masked_fill_(due, False)
        self.friction_switch_time.masked_fill_(due, float("inf"))
        self.time_since_friction_change.masked_fill_(due, 0.0)
        self._refresh_friction_observation()
