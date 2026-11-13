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
