"""Joint-target offsets constrained by actuator position limits."""

import torch


def joint_target_bounds(env):
    """(lowest, highest) joint-target offset [rad] the hard stops allow.

    The joint limits less the task's joint-target margin [rad]
    (control.joint_target_margin), relative to the default posture; None
    without the setting. A positive margin keeps targets inside the
    limits; the simulation curriculum may temporarily use a negative one.
    """
    margin = env._joint_target_margin()
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


def target_excess(env, requested):
    """Requested minus applied joint-target offsets [N, 12], in action units."""
    scale = torch.as_tensor(
        env.scales["dof_pos_target"], dtype=requested.dtype, device=requested.device
    )
    return (requested - env.dof_pos_target) / scale


def limit_joint_targets(env):
    """Bound the commanded joint offsets, in units of the policy action.

    The policy sees its previous target, so an unbounded target can feed on
    itself once the robot leaves the states it was trained on.
    """
    requested = env.dof_pos_target.clone()
    limit = getattr(env.cfg.control, "action_limit", None)
    if limit is not None:
        bound = limit * env.scales["dof_pos_target"]
        env.dof_pos_target.clamp_(-bound, bound)
    env._keep_targets_inside_joint_limits()
    env.dof_pos_target_excess = target_excess(env, requested)


def target_saturation_reward(env):
    """Minus how far the requested joint targets lie beyond the applied ones.

    A clamped target hides the policy's output from the motors: past a
    bound every output acts the same, so nothing pulls the output back
    and its spread grows until the joint is never commanded inside its
    range again. This term restores that pull. Per joint, with the
    excess e in action units: e^2 / 2 below one unit, |e| - 1/2 above.
    """
    excess = getattr(env, "dof_pos_target_excess", None)
    if excess is None:
        return torch.zeros(env.num_envs, device=env.device)
    size = excess.abs()
    cost = torch.where(size < 1.0, 0.5 * size.square(), size - 0.5)
    return -cost.sum(dim=1)


def standing_saturation_reward(env):
    """The target-saturation cost while the clocks stand; zero otherwise."""
    standing = env._slow_command_mask()[:, 0].float()
    return target_saturation_reward(env) * standing


def joint_limit_margin_reward(env):
    """Minus how far the joint angles are into the margin before their limits.

    Per joint: 0 while the angle keeps reward_settings.joint_limit_margin
    [rad] from both limits, 1 at a limit, more beyond it. The real joint
    limits are hard stops, so the measured angles must stay inside them.
    The targets may lie beyond a limit: with position control a target is
    how the policy asks for torque, and a walking Go2 extends its knees
    with targets past the stop while the knee itself stays short of it.
    """
    margin = env.cfg.reward_settings.joint_limit_margin
    limits = env.dof_pos_limits.index_select(0, env.actuated_dof_indices)
    angle = env.dof_pos.index_select(1, env.actuated_dof_indices)
    depth = torch.maximum(
        angle - (limits[:, 1] - margin), (limits[:, 0] + margin) - angle
    )
    return -(depth.clamp(min=0.0) / margin).sum(dim=1)
