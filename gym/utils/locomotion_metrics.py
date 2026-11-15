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


# Speed (m/s) below which a slide counts as stopped.
SLIDE_STOP_SPEED = 0.1
# Control steps before the slide starts that are averaged for its entry speed.
SLIDE_ENTRY_WINDOW = 20
# A foot must be unloaded at least this long before landing again to count
# as a step; shorter gaps are contact chatter (a loaded foot bouncing).
STEP_MIN_AIR_TIME_S = 0.05


def count_steps(loaded: torch.Tensor, min_air_steps: int) -> int:
    """Landings of one foot after at least ``min_air_steps`` unloaded steps.

    ``loaded`` is a 1-D bool sequence. Contact chatter (brief unloading of
    a foot that stays down) is not counted.
    """
    steps = 0
    air = 0
    for is_loaded in loaded.tolist():
        if is_loaded:
            if air >= min_air_steps and air > 0:
                steps += 1
            air = 0
        else:
            air += 1
    return steps


def summarize_slide(
    trace: dict[str, torch.Tensor],
    *,
    dt: float,
    slide_start_step,
    friction: float,
    stop_speed: float = SLIDE_STOP_SPEED,
    load_threshold: float = FOOT_LOAD_THRESHOLD_N,
    min_air_time: float = STEP_MIN_AIR_TIME_S,
) -> dict:
    """Summarize run-up-and-slide trials on one friction coefficient.

    Required trace keys (T steps, R replicas, F feet): ``alive`` [T, R]
    bool, absorbing; ``position`` [T, R, 2] world xy; ``speed`` [T, R]
    horizontal base speed; ``forward_speed`` [T, R] body-frame vx;
    ``command`` [T, R] commanded forward speed; ``foot_load`` [T, R, F]
    vertical contact force; ``power_abs`` [T, R] sum |tau*qdot| in W.
    Optional ``pose_deviation`` [T, R]: RMS joint deviation from the
    standing pose (rad), reported as ``pose_rmse_rad``.

    ``slide_start_step`` is one step index for all replicas, or a sequence
    with one per replica (slides triggered at different times).

    Each surviving replica's slide runs from its start step until its
    speed first drops below ``stop_speed`` (or the trace ends). On friction
    mu nothing can stop faster than mu*g, which pure sliding on all feet
    achieves; so a coasting slide has distance_ratio and decel_ratio near 1,
    all four feet loaded, few touchdowns, and little joint energy, while
    stepping to slow down shows touchdowns, energy, and a longer distance.

    ``touchdowns_per_s`` counts only landings after at least ``min_air_time``
    in the air (real steps); ``contact_breaks_per_s`` counts every reloading,
    including chatter of feet that stay down.
    """
    alive = trace["alive"].bool()
    survived = alive[-1]
    result = {
        "friction": friction,
        "replicas": alive.shape[1],
        "fall_rate": 1.0 - survived.float().mean().item(),
    }
    keys = (
        "stopped_rate",
        "entry_speed_m_s",
        "slide_distance_m",
        "predicted_distance_m",
        "distance_ratio",
        "slide_duration_s",
        "decel_ratio",
        "all_feet_loaded_fraction",
        "min_foot_loaded_fraction",
        "touchdowns_per_s",
        "contact_breaks_per_s",
        "slide_energy_j",
        "slide_energy_per_m_j",
        "slide_tracking_rmse_m_s",
        "pose_rmse_rad",
    )
    if not survived.any():
        return result | dict.fromkeys(keys)
    starts = (
        [int(slide_start_step)] * alive.shape[1]
        if isinstance(slide_start_step, int)
        else [int(step) for step in slide_start_step]
    )
    loaded = trace["foot_load"] > load_threshold
    per_replica = {key: [] for key in keys}
    for r in torch.nonzero(survived).flatten().tolist():
        start = starts[r]
        speed = trace["speed"][:, r]
        entry = speed[max(0, start - SLIDE_ENTRY_WINDOW) : start].mean().item()
        below = torch.nonzero(speed[start:] < stop_speed).flatten()
        stopped = below.numel() > 0
        stop = start + (below[0].item() if stopped else speed.shape[0] - 1 - start)
        stop = max(stop, start + 1)
        window = slice(start, stop)
        duration = (stop - start) * dt
        distance = (trace["position"][stop, r] - trace["position"][start, r]).norm()
        predicted = entry**2 / (2 * friction * GRAVITY_M_S2)
        feet = loaded[window, r]
        min_air_steps = max(1, round(min_air_time / dt))
        touchdowns = sum(
            count_steps(feet[:, f], min_air_steps) for f in range(feet.shape[1])
        )
        contact_breaks = (feet[1:] & ~feet[:-1]).sum().item()
        energy = trace["power_abs"][window, r].sum().item() * dt
        error = trace["forward_speed"][window, r] - trace["command"][window, r]
        values = {
            "stopped_rate": float(stopped),
            "entry_speed_m_s": entry,
            "slide_distance_m": distance.item(),
            "predicted_distance_m": predicted,
            "distance_ratio": distance.item() / predicted if predicted else None,
            "slide_duration_s": duration,
            "decel_ratio": (entry / duration) / (friction * GRAVITY_M_S2),
            "all_feet_loaded_fraction": feet.all(dim=1).float().mean().item(),
            # The least-used foot: ~0 when a leg is held up through the slide.
            "min_foot_loaded_fraction": feet.float().mean(dim=0).min().item(),
            "touchdowns_per_s": touchdowns / duration,
            "contact_breaks_per_s": contact_breaks / duration,
            "slide_energy_j": energy,
            "slide_energy_per_m_j": energy / max(distance.item(), 1e-6),
            "slide_tracking_rmse_m_s": error.square().mean().sqrt().item(),
            "pose_rmse_rad": (
                trace["pose_deviation"][window, r].mean().item()
                if "pose_deviation" in trace
                else None
            ),
        }
        for key, value in values.items():
            per_replica[key].append(value)
    for key, values in per_replica.items():
        defined = [v for v in values if v is not None]
        result[key] = sum(defined) / len(defined) if defined else None
    return result
