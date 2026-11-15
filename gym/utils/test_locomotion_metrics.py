import pytest
import torch

from gym.utils.locomotion_metrics import (
    GRAVITY_M_S2,
    contact_point_velocity,
    summarize_locomotion,
)

DT = 0.01
MASS = 15.0


def _trace(steps=200, replicas=2, speed=1.0, feet=4, joints=12):
    """Straight walking at ``speed`` with all feet loaded and planted."""
    body = torch.zeros(steps, replicas, 3)
    body[..., 0] = speed
    return {
        "alive": torch.ones(steps, replicas, dtype=torch.bool),
        "body_velocity": body,
        "world_planar_speed": torch.full((steps, replicas), speed),
        "torques": torch.zeros(steps, replicas, joints),
        "joint_velocity": torch.zeros(steps, replicas, joints),
        "foot_load": torch.full((steps, replicas, feet), 30.0),
        "foot_contact_velocity": torch.zeros(steps, replicas, feet, 3),
        "base_height": torch.full((steps, replicas), 0.3),
    }


def _summarize(trace, command=(1.0, 0.0, 0.0), warmup=0):
    return summarize_locomotion(
        trace,
        dt=DT,
        warmup_steps=warmup,
        command=torch.tensor(command),
        mass_kg=MASS,
        torque_limits=torch.full((12,), 20.0),
    )


def test_rolling_sphere_contact_point_is_stationary():
    radius, spin = 0.022, 5.0
    lin = torch.tensor([radius * spin, 0.0, 0.0])
    ang = torch.tensor([0.0, spin, 0.0])
    torch.testing.assert_close(contact_point_velocity(lin, ang, radius), torch.zeros(3))


def test_perfect_planted_walking_has_zero_error_and_slip():
    result = _summarize(_trace())
    assert result["fall_rate"] == 0.0
    assert result["mean_time_to_fall_s"] is None
    assert result["linear_rmse_m_s"] == pytest.approx(0.0)
    assert result["yaw_rmse_rad_s"] == pytest.approx(0.0)
    assert result["slip_ratio"] == pytest.approx(0.0)
    assert result["mean_stance_slip_speed_m_s"] == pytest.approx(0.0)
    assert result["contact_duty"] == [1.0] * 4


def test_cost_of_transport_matches_hand_computation():
    trace = _trace(steps=100, replicas=1, speed=2.0)
    # 3 joints each delivering |tau * qdot| = 10 W -> 30 W total.
    trace["torques"][..., :3] = 5.0
    trace["joint_velocity"][..., :3] = -2.0
    result = _summarize(trace, command=(2.0, 0.0, 0.0))
    distance = 2.0 * 100 * DT
    expected = 30.0 * 100 * DT / (MASS * GRAVITY_M_S2 * distance)
    assert result["mean_mechanical_power_w"] == pytest.approx(30.0)
    assert result["cost_of_transport"] == pytest.approx(expected)


def test_gliding_feet_give_unit_slip_ratio():
    trace = _trace(speed=1.5)
    trace["foot_contact_velocity"][..., 0] = 1.5
    result = _summarize(trace, command=(1.5, 0.0, 0.0))
    assert result["slip_ratio"] == pytest.approx(1.0)
    assert result["mean_stance_slip_speed_m_s"] == pytest.approx(1.5)


def test_swing_foot_motion_is_not_slip():
    trace = _trace()
    trace["foot_load"][..., 0] = 0.0
    trace["foot_contact_velocity"][..., 0, 0] = 3.0
    result = _summarize(trace)
    assert result["slip_ratio"] == pytest.approx(0.0)
    assert result["contact_duty"] == [0.0, 1.0, 1.0, 1.0]


def test_failure_is_absorbing_and_excluded_from_tracking():
    trace = _trace(steps=100, replicas=2)
    trace["alive"][40:, 1] = False
    trace["body_velocity"][40:, 1, 0] = -50.0  # post-fall garbage
    result = _summarize(trace)
    assert result["fall_rate"] == 0.5
    # Sample k is recorded after step k + 1, so the fall happened at 41 dt.
    assert result["mean_time_to_fall_s"] == pytest.approx(41 * DT)
    assert result["linear_rmse_m_s"] == pytest.approx(0.0)


def test_touchdowns_per_meter_counts_load_onsets():
    trace = _trace(steps=100, replicas=1, speed=1.0)
    # Foot 0 lifts and lands every 20 steps: touchdowns at 10, 30, 50, 70, 90.
    for start in range(0, 100, 20):
        trace["foot_load"][start : start + 10, 0, 0] = 0.0
    result = _summarize(trace)
    assert result["touchdowns_per_m"] == pytest.approx(5 / 1.0)


def test_stationary_command_reports_no_distance_metrics():
    trace = _trace(speed=0.0)
    result = _summarize(trace, command=(0.0, 0.0, 0.0))
    assert result["cost_of_transport"] is None
    assert result["slip_ratio"] is None
    assert result["touchdowns_per_m"] is None
    assert result["linear_rmse_m_s"] == pytest.approx(0.0)


def test_all_failed_during_warmup_has_undefined_metrics():
    trace = _trace(steps=50, replicas=2)
    trace["alive"][5:] = False
    result = _summarize(trace, warmup=10)
    assert result["fall_rate"] == 1.0
    assert result["samples"] == 0
    assert result["linear_rmse_m_s"] is None
    assert result["cost_of_transport"] is None


def _straight_trace(steps=100, replicas=4, speed=1.0, power=150.0):
    t = torch.arange(steps, dtype=torch.float)[:, None].repeat(1, replicas)
    position = torch.zeros(steps, replicas, 2)
    position[..., 0] = speed * DT * (t + 1)
    return {
        "alive": torch.ones(steps, replicas, dtype=torch.bool),
        "forward_speed": torch.full((steps, replicas), speed),
        "position": position,
        "power_abs": torch.full((steps, replicas), power),
        "power_positive": torch.full((steps, replicas), power / 2),
    }


def test_replica_cot_matches_hand_computation():
    from gym.utils.locomotion_metrics import replica_speed_and_cot

    trace = _straight_trace(speed=1.0, power=150.0)
    trace["alive"][60:, 3] = False
    result = replica_speed_and_cot(trace, dt=DT, warmup_steps=0, mass_kg=MASS)
    # Window spans 99 position intervals but 100 power samples.
    expected = 150.0 * 100 * DT / (MASS * GRAVITY_M_S2 * 0.99)
    torch.testing.assert_close(result["cot"], torch.full((4,), expected))
    torch.testing.assert_close(result["cot_positive_work"], result["cot"] / 2)
    torch.testing.assert_close(result["speed"], torch.ones(4))
    assert result["survived"].tolist() == [True, True, True, False]


def test_sideways_wobble_is_not_distance():
    from gym.utils.locomotion_metrics import replica_speed_and_cot

    trace = _straight_trace(steps=101, replicas=1)
    trace["position"][1::2, 0, 1] = 0.05  # zig-zag 5 cm every step
    result = replica_speed_and_cot(trace, dt=DT, warmup_steps=0, mass_kg=MASS)
    straight = replica_speed_and_cot(
        _straight_trace(steps=101, replicas=1), dt=DT, warmup_steps=0, mass_kg=MASS
    )
    torch.testing.assert_close(result["cot"], straight["cot"])


def _point(speed, cot):
    return {
        "speed": speed,
        "cot": cot,
        "cot_std": 0.1,
        "cot_positive_work": cot / 2,
        "cot_positive_work_std": 0.05,
    }


def test_cot_is_interpolated_at_the_actual_speed_not_the_command():
    from gym.utils.locomotion_metrics import cot_at_speed

    # Out of order on purpose: points are sorted by the speed actually reached.
    points = [_point(1.4, 3.0), _point(0.6, 1.0), _point(1.0, 2.0)]
    result = cot_at_speed(points, 1.2)
    assert result["status"] == "ok"
    assert result["cot"] == pytest.approx(2.5)
    assert result["cot_positive_work"] == pytest.approx(1.25)
    assert result["top_speed_m_s"] == pytest.approx(1.4)
    assert cot_at_speed(points, 1.0)["cot"] == pytest.approx(2.0)


def test_speed_above_top_speed_is_unreachable():
    from gym.utils.locomotion_metrics import cot_at_speed

    result = cot_at_speed([_point(0.4, 1.0), _point(0.8, 1.5)], 1.5)
    assert result["status"] == "unreachable"
    assert result["cot"] is None
    assert result["top_speed_m_s"] == pytest.approx(0.8)


def test_wide_gap_is_not_interpolated():
    from gym.utils.locomotion_metrics import cot_at_speed

    result = cot_at_speed([_point(0.2, 1.0), _point(1.6, 1.5)], 1.0, max_gap=0.5)
    assert result["status"] == "not_bracketed"
    assert result["cot"] is None


def _slide_trace(friction=0.1, v0=1.2, replicas=2, start=50, steps=300, feet=4):
    """Pure coasting: hold v0 until ``start``, then decelerate at mu*g."""
    decel = friction * GRAVITY_M_S2
    t = torch.arange(steps, dtype=torch.float) * DT
    slide_t = (t - start * DT).clamp_min(0)
    speed = (v0 - decel * slide_t).clamp_min(0.0)
    x = torch.cumsum(speed * DT, dim=0)
    rep = lambda v: v[:, None].repeat(1, replicas)  # noqa: E731
    position = torch.zeros(steps, replicas, 2)
    position[..., 0] = rep(x)
    return {
        "alive": torch.ones(steps, replicas, dtype=torch.bool),
        "position": position,
        "speed": rep(speed),
        "forward_speed": rep(speed),
        "command": rep(speed),
        "foot_load": torch.full((steps, replicas, feet), 40.0),
        "power_abs": torch.zeros(steps, replicas),
    }


def _slide(trace, friction=0.1, start=50):
    from gym.utils.locomotion_metrics import summarize_slide

    return summarize_slide(trace, dt=DT, slide_start_step=start, friction=friction)


def test_pure_coasting_slide_matches_friction_physics():
    result = _slide(_slide_trace())
    assert result["fall_rate"] == 0.0 and result["stopped_rate"] == 1.0
    assert result["entry_speed_m_s"] == pytest.approx(1.2)
    assert result["predicted_distance_m"] == pytest.approx(
        1.2**2 / (2 * 0.1 * GRAVITY_M_S2)
    )
    assert result["distance_ratio"] == pytest.approx(1.0, abs=0.05)
    assert result["decel_ratio"] == pytest.approx(1.0, abs=0.1)
    assert result["all_feet_loaded_fraction"] == 1.0
    assert result["touchdowns_per_s"] == 0.0
    assert result["slide_energy_j"] == 0.0
    assert result["slide_tracking_rmse_m_s"] == pytest.approx(0.0)


def test_stepping_during_the_slide_shows_touchdowns_and_energy():
    trace = _slide_trace()
    # One foot lifts and lands every 10 steps; joints spend 100 W.
    for s in range(0, 300, 10):
        trace["foot_load"][s : s + 5, :, 0] = 0.0
    trace["power_abs"][:] = 100.0
    result = _slide(trace)
    assert result["all_feet_loaded_fraction"] < 0.6
    assert result["touchdowns_per_s"] == pytest.approx(10.0, rel=0.2)
    assert result["slide_energy_j"] == pytest.approx(100.0 * result["slide_duration_s"])


def test_slower_than_friction_braking_has_ratios_above_and_below_one():
    # Decelerating at half of mu*g: twice the distance, half the decel.
    result = _slide(_slide_trace(friction=0.05), friction=0.1)
    assert result["distance_ratio"] == pytest.approx(2.0, rel=0.06)
    assert result["decel_ratio"] == pytest.approx(0.5, rel=0.1)


def test_fallen_replicas_are_excluded_and_unstopped_slides_reported():
    trace = _slide_trace(steps=120)  # ends before the robot stops
    trace["alive"][80:, 1] = False
    result = _slide(trace)
    assert result["fall_rate"] == 0.5
    assert result["stopped_rate"] == 0.0
    assert result["entry_speed_m_s"] == pytest.approx(1.2)


def test_all_fallen_reports_no_slide_metrics():
    trace = _slide_trace()
    trace["alive"][10:] = False
    result = _slide(trace)
    assert result["fall_rate"] == 1.0
    assert result["distance_ratio"] is None
