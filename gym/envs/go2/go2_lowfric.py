import torch

from gym.envs.go2.go2 import Go2
from gym.utils.locomotion_metrics import FOOT_LOAD_THRESHOLD_N, contact_point_velocity


class Go2LowFric(Go2):
    """Go2 locomotion across bimodal (ice / ground) contact friction.
    the true per-environment friction is a
    privileged observation for both actor and critic (a teacher policy).
    """

    def _init_buffers(self):
        super()._init_buffers()
        self.contact_friction = self._backend.contact_friction.unsqueeze(1)

    def _reward_glide(self):
        """Reward loaded feet sliding along with the body, on low friction only.

        Per foot: if it carries load, how far its contact point moves in the
        body's direction of travel, relative to the body's speed, clipped to
        [0, 1] (1 = gliding at body speed; planted, rolling, backward or
        sideways sliding feet earn 0). Averaged over the four feet, then:

        - gated by friction: sigmoid((threshold - mu) / width), so it is
          ~1 on ice and ~0 on ground (rewards may use the true mu);
        - zero unless the command asks to move, so sliding is never
          rewarded while told to stand still;
        - body speed below glide_min_command scales the reward down, so
          feet shuffling under a stationary body cannot earn it.
        """
        settings = self.cfg.reward_settings
        # True mu from the backend, not the observation, which evaluation
        # may overwrite to misreport friction.
        friction = self._backend.contact_friction
        gate = torch.sigmoid(
            (settings.glide_friction_threshold - friction)
            / settings.glide_friction_width
        )
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
        ratio = along / body_speed.clamp_min(settings.glide_min_command)
        loaded = self.contact_forces[:, feet, 2] > FOOT_LOAD_THRESHOLD_N
        glide = (ratio.clamp(0.0, 1.0) * loaded).mean(dim=1)
        moving = self.commands[:, :2].norm(dim=1) > settings.glide_min_command
        return gate * glide * moving
