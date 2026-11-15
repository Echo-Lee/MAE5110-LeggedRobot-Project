import torch

from gym.envs.go2.go2 import Go2
from gym.utils.locomotion_metrics import (
    FOOT_LOAD_THRESHOLD_N,
    GRAVITY_M_S2,
    contact_point_velocity,
)
from gym.utils.sampling import masked_update


class Go2LowFric(Go2):
    """Go2 locomotion across bimodal (ice / ground) contact friction.
    the true per-environment friction is a
    privileged observation for both actor and critic (a teacher policy).
    """

    def _init_buffers(self):
        super()._init_buffers()
        self.contact_friction = self._backend.contact_friction.unsqueeze(1)
        num_feet = len(self.feet_indices)
        # Seconds each foot has been continuously loaded (0 while in the air)
        # and whether it touched down during the last control step.
        self.foot_contact_time = torch.zeros(
            self.num_envs, num_feet, device=self.device
        )
        self.foot_touchdown = torch.zeros(
            self.num_envs, num_feet, dtype=torch.bool, device=self.device
        )

    def _post_decimation_step(self):
        super()._post_decimation_step()
        loaded = self.contact_forces[:, self.feet_indices, 2] > FOOT_LOAD_THRESHOLD_N
        self.foot_touchdown = loaded & (self.foot_contact_time == 0)
        self.foot_contact_time = (self.foot_contact_time + self.dt) * loaded

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        masked_update(
            self.foot_contact_time, torch.zeros_like(self.foot_contact_time), reset_mask
        )
        masked_update(
            self.foot_touchdown, torch.zeros_like(self.foot_touchdown), reset_mask
        )

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
