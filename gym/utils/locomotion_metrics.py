"""
Policy-independent locomotion metrics computed from recorded traces.
"""

import torch

GRAVITY_M_S2 = 9.81
# A foot carrying less than this vertical force counts as swing. Matches the
# contact-duty threshold of the Go2Trot performance gate.
FOOT_LOAD_THRESHOLD_N = 5.0
# Distance-normalized metrics (slip ratio, CoT, touchdowns per meter) are
# reported only when replicas travel at least this far on average.
MIN_DISTANCE_M = 0.25
TORQUE_SATURATION_FRACTION = 0.99


def contact_point_velocity(
    lin_vel: torch.Tensor, ang_vel: torch.Tensor, radius: float
) -> torch.Tensor:
    """World velocity of the lowest point of a spherical foot.
    """
    offset = torch.zeros_like(ang_vel)
    offset[..., 2] = -radius
    return lin_vel + torch.cross(ang_vel, offset, dim=-1)


def _first_failure_time(alive: torch.Tensor, dt: float) -> list[float]:
    times = []
    for replica in range(alive.shape[1]):
        dead = (~alive[:, replica]).nonzero()
        if dead.numel():
            times.append(float(dead[0, 0] + 1) * dt)
    return times


def summarize_locomotion(
    trace: dict[str, torch.Tensor],
    *,
    dt: float,
    warmup_steps: int,
    command: torch.Tensor,
    mass_kg: float,
    torque_limits: torch.Tensor,
    load_threshold: float = FOOT_LOAD_THRESHOLD_N,
    min_distance: float = MIN_DISTANCE_M,
) -> dict:
    """Summarize one fixed-command, fixed-friction case.

    Required trace keys and shapes (T steps, R replicas, J joints, F feet):

    - ``alive`` [T, R] bool, absorbing
    - ``body_velocity`` [T, R, 3]: body-frame vx, vy and yaw rate
    - ``world_planar_speed`` [T, R]: horizontal base speed in m/s
    - ``torques`` / ``joint_velocity`` [T, R, J]
    - ``foot_load`` [T, R, F]: vertical contact force in N
    - ``foot_contact_velocity`` [T, R, F, 3]: see ``contact_point_velocity``
    - ``base_height`` [T, R]
    """
    alive = trace["alive"].bool()
    replicas = alive.shape[1]
    failure_times = _first_failure_time(alive, dt)
    result = {
        "fall_rate": len(failure_times) / replicas,
        "mean_time_to_fall_s": (
            sum(failure_times) / len(failure_times) if failure_times else None
        ),
    }
    valid = alive.clone()
    valid[:warmup_steps] = False
    result["samples"] = int(valid.sum())
    undefined = (
        "linear_rmse_m_s",
        "yaw_rmse_rad_s",
        "mean_stance_slip_speed_m_s",
        "slip_ratio",
        "distance_per_replica_m",
        "mean_mechanical_power_w",
        "cost_of_transport",
        "touchdowns_per_m",
        "contact_duty",
        "torque_saturation_fraction",
        "mean_base_height_m",
    )
    if result["samples"] == 0:
        return result | dict.fromkeys(undefined)

    def mean(value, mask=valid):
        return value[mask].float().mean().item()

    # Planar vector error: per-axis averaging would dilute a forward error
    # with zero lateral error and let a stationary policy look half right.
    planar_error = (trace["body_velocity"][..., :2] - command[:2]).norm(dim=-1)
    result["linear_rmse_m_s"] = mean(planar_error.square()) ** 0.5
    yaw_error = trace["body_velocity"][..., 2] - command[2]
    result["yaw_rmse_rad_s"] = mean(yaw_error.square()) ** 0.5

    loaded = trace["foot_load"] > load_threshold
    stance = loaded & valid[..., None]
    slip_speed = trace["foot_contact_velocity"][..., :2].norm(dim=-1)
    result["mean_stance_slip_speed_m_s"] = (
        slip_speed[stance].mean().item() if stance.any() else None
    )

    distance = (trace["world_planar_speed"] * valid).sum().item() * dt
    result["distance_per_replica_m"] = distance / replicas
    travelled = distance / replicas >= min_distance
    moving = bool(command[:2].norm() > 0)

    # Per-foot slip distance over body distance: 0 for planted stance feet,
    # about 1 when stance feet glide as far as the body travels.
    num_feet = loaded.shape[-1]
    slip_distance = (slip_speed * stance).sum().item() * dt / num_feet
    result["slip_ratio"] = slip_distance / distance if travelled else None

    power = (trace["torques"] * trace["joint_velocity"]).abs().sum(dim=-1)
    result["mean_mechanical_power_w"] = mean(power)
    energy = (power * valid).sum().item() * dt
    result["cost_of_transport"] = (
        energy / (mass_kg * GRAVITY_M_S2 * distance) if moving and travelled else None
    )

    pair_valid = (valid[1:] & valid[:-1])[..., None]
    touchdowns = (loaded[1:] & ~loaded[:-1] & pair_valid).sum().item()
    result["touchdowns_per_m"] = touchdowns / distance if travelled else None
    result["contact_duty"] = loaded[valid].float().mean(dim=0).tolist()

    saturated = trace["torques"].abs() >= TORQUE_SATURATION_FRACTION * torque_limits
    result["torque_saturation_fraction"] = saturated[valid].float().mean().item()
    result["mean_base_height_m"] = mean(trace["base_height"])
    return result
