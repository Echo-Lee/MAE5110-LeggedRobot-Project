"""Joint-target offsets constrained by actuator position limits."""

import torch


def joint_target_bounds(env):
    """(lowest, highest) joint-target offset [rad] the hard stops allow.

    The joint limits less control.joint_target_margin [rad], relative to
    the default posture; None without the setting. MuJoCo's limits are
    soft and the real stops are not, so a target is never placed on or
    past a stop.
    """
    margin = getattr(env.cfg.control, "joint_target_margin", None)
    if margin is None:
        return None
    limits = env.dof_pos_limits.index_select(0, env.actuated_dof_indices)
    default = env.default_dof_pos.index_select(1, env.actuated_dof_indices)
    return limits[:, 0] + margin - default, limits[:, 1] - margin - default


def keep_joint_targets(env):
    """Clamp targets through the task's joint-bound calculation."""
    bounds = env._joint_target_bounds()
    if bounds is None:
        return
    low, high = bounds
    env.dof_pos_target.copy_(
        torch.maximum(torch.minimum(env.dof_pos_target, high), low)
    )
