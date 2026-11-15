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
    """World velocity of the lowest point of a spherical foot."""
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


def replica_speed_and_cot(
    trace: dict[str, torch.Tensor],
    *,
    dt: float,
    warmup_steps: int,
    mass_kg: float,
) -> dict[str, torch.Tensor]:
    """Per-replica steady forward speed and cost of transport.

    Required trace keys (T steps, R replicas): ``alive`` [T, R] bool,
    absorbing; ``forward_speed`` [T, R] body-frame vx in m/s; ``position``
    [T, R, 2] world xy; ``power_abs`` and ``power_positive`` [T, R] in W,
    i.e. sum |tau*qdot| and sum max(tau*qdot, 0). CoT uses net displacement
    over the measured window, so sideways wobble does not count as distance.
    Returns [R] tensors; ``survived`` marks replicas that never fell.
    """
    window = slice(warmup_steps, None)
    position = trace["position"][window]
    distance = (position[-1] - position[0]).norm(dim=-1)
    weight = mass_kg * GRAVITY_M_S2

    def cot(key):
        energy = trace[key][window].sum(dim=0) * dt
        return energy / (weight * distance.clamp_min(1e-9))

    return {
        "survived": trace["alive"].bool()[-1],
        "speed": trace["forward_speed"][window].mean(dim=0),
        "cot": cot("power_abs"),
        "cot_positive_work": cot("power_positive"),
    }


def cot_at_speed(
    points: list[dict], target_speed: float, *, max_gap: float = 0.5
) -> dict:
    """Cost of transport at an exact actual speed, by linear interpolation.

    ``points`` holds one entry per commanded speed at a single friction, each
    with the mean actual ``speed`` reached and the mean ``cot`` (plus its
    replica ``cot_std``) over surviving replicas. Points are ordered by
    actual speed, so commanded-speed tracking error does not matter: the CoT
    is read off at the speed actually achieved. The target must be bracketed
    by two points no more than ``max_gap`` m/s apart; otherwise the result
    is ``None`` (``unreachable`` when the target exceeds every reached speed).
    """
    points = sorted(points, key=lambda p: p["speed"])
    top_speed = points[-1]["speed"] if points else None
    result = {"target_speed_m_s": target_speed, "top_speed_m_s": top_speed}
    for low, high in zip(points, points[1:]):
        if low["speed"] <= target_speed <= high["speed"]:
            if high["speed"] - low["speed"] > max_gap:
                break
            span = high["speed"] - low["speed"]
            w = 0.0 if span == 0 else (target_speed - low["speed"]) / span
            mix = lambda key: (1 - w) * low[key] + w * high[key]  # noqa: E731
            return result | {
                "status": "ok",
                "cot": mix("cot"),
                "cot_std": mix("cot_std"),
                "cot_positive_work": mix("cot_positive_work"),
                "cot_positive_work_std": mix("cot_positive_work_std"),
            }
    status = (
        "unreachable"
        if top_speed is None or target_speed > top_speed
        else "not_bracketed"
    )
    return result | {
        "status": status,
        "cot": None,
        "cot_std": None,
        "cot_positive_work": None,
        "cot_positive_work_std": None,
    }
