"""Independent signals expose gait, tracking, jitter, and survival failures."""

import math

import pytest
import torch

from scripts.evaluate_go2trot import (
    high_frequency_rms,
    mechanical_energy_metrics,
    summarize_case,
)


def ideal_trace():
    phase = torch.arange(800).float()[:, None, None] * (2 * math.pi * 2 * 0.01)
    phase += 0.1
    legs = phase + torch.tensor([0.0, math.pi, math.pi, 0.0])
    return {
        "alive": torch.ones(800, 1, dtype=torch.bool),
        "velocity": torch.zeros(800, 1, 3),
        "residual": torch.zeros(800, 1, 12),
        "joint_velocity": torch.zeros(800, 1, 12),
        "joint_velocity_before": torch.zeros(800, 1, 12),
        "torque": torch.zeros(800, 1, 12),
        "position": torch.zeros(800, 1, 3),
        "position_before": torch.zeros(800, 1, 3),
        "clipped": torch.zeros(800, 1, 12, dtype=torch.bool),
        "clip_error": torch.zeros(800, 1, 12),
        "load": (legs.sin() > 0).float() * 50,
        "body_weight": torch.full((800, 1), 100.0),
        "phase": phase,
    }


def summarize(trace):
    return summarize_case(trace, 0.01, 200, 2.0, torch.zeros(3))


def test_mechanical_energy_metrics_use_gross_work_and_projected_progress():
    trace = {
        "torque": torch.tensor([[[2.0]], [[-2.0]]]),
        "joint_velocity_before": torch.tensor([[[3.0]], [[3.0]]]),
        "joint_velocity": torch.tensor([[[3.0]], [[3.0]]]),
        "position_before": torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        "position": torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]),
        "body_weight": torch.full((2, 1), 98.1),
    }
    result = mechanical_energy_metrics(
        trace, 0.1, torch.ones(2, 1, dtype=torch.bool), torch.tensor([1.0, 0.0, 0.0])
    )
    assert result["gross_mechanical_energy_j"] == pytest.approx(1.2)
    assert result["projected_progress_m"] == pytest.approx(2.0)
    assert result["gross_mechanical_energy_j_per_m"] == pytest.approx(0.6)
    assert result["gross_mechanical_cost_of_transport"] == pytest.approx(0.006116208)
    assert result["gross_mechanical_energy_j_per_path_m"] == pytest.approx(0.6)
    assert result["gross_mechanical_path_cost_of_transport"] == pytest.approx(
        0.006116208
    )


def test_mechanical_energy_metrics_do_not_cancel_opposing_joint_power():
    trace = {
        "torque": torch.tensor([[[2.0, -2.0]]]),
        "joint_velocity_before": torch.tensor([[[3.0, 3.0]]]),
        "joint_velocity": torch.tensor([[[3.0, 3.0]]]),
        "position_before": torch.zeros(1, 1, 3),
        "position": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "body_weight": torch.full((1, 1), 98.1),
    }
    result = mechanical_energy_metrics(
        trace, 0.1, torch.ones(1, 1, dtype=torch.bool), torch.tensor([1.0, 0.0, 0.0])
    )
    assert result["gross_mechanical_energy_j"] == pytest.approx(1.2)


def test_mechanical_energy_metrics_omit_distance_metrics_without_planar_motion():
    trace = {
        "torque": torch.ones(1, 1, 1),
        "joint_velocity_before": torch.ones(1, 1, 1),
        "joint_velocity": torch.ones(1, 1, 1),
        "position_before": torch.zeros(1, 1, 3),
        "position": torch.zeros(1, 1, 3),
        "body_weight": torch.full((1, 1), 98.1),
    }
    result = mechanical_energy_metrics(
        trace, 0.1, torch.ones(1, 1, dtype=torch.bool), torch.zeros(3)
    )
    assert result["gross_mechanical_energy_j"] == pytest.approx(0.1)
    assert "gross_mechanical_energy_j_per_m" not in result


def test_mechanical_energy_metrics_omit_distance_metrics_without_progress():
    trace = {
        "torque": torch.ones(1, 1, 1),
        "joint_velocity_before": torch.ones(1, 1, 1),
        "joint_velocity": torch.ones(1, 1, 1),
        "position_before": torch.zeros(1, 1, 3),
        "position": torch.tensor([[[0.0, 1.0, 0.0]]]),
        "body_weight": torch.full((1, 1), 98.1),
    }
    result = mechanical_energy_metrics(
        trace, 0.1, torch.ones(1, 1, dtype=torch.bool), torch.tensor([1.0, 0.0, 0.0])
    )
    assert result["projected_progress_m"] == pytest.approx(0.0)
    assert "gross_mechanical_energy_j_per_m" not in result
    assert result["planar_path_length_m"] == pytest.approx(1.0)
    assert result["gross_mechanical_energy_j_per_path_m"] == pytest.approx(0.1)


def test_balanced_scheduled_trot_passes_with_complete_strides():
    metrics = summarize(ideal_trace())
    assert metrics["passed"]
    assert metrics["complete_strides"] >= 10
    assert metrics["swing_load_fraction"] == 0
    assert metrics["load_share"] == pytest.approx([0.25] * 4, abs=0.01)


@pytest.mark.parametrize(
    "failure", ["standing", "wrong_phase", "one_leg", "jitter", "tracking"]
)
def test_bad_policies_cannot_pass_on_reward_or_average_load_alone(failure):
    trace = ideal_trace()
    if failure == "standing":
        trace["load"].fill_(25)
        expected = "swing_load_fraction"
    elif failure == "wrong_phase":
        trace["load"] = 50 - trace["load"]
        expected = "swing_load_fraction"
    elif failure == "one_leg":
        trace["load"][:, :, :3] = 0
        expected = "stride_load_max_deviation_p90"
    elif failure == "jitter":
        trace["residual"][::2] = 0.2
        trace["residual"][1::2] = -0.2
        expected = "residual_jitter_rms_rad"
    else:
        trace["velocity"][:, :, 0] = 1.0
        expected = "linear_rmse_m_s"
    metrics = summarize(trace)
    assert not metrics["passed"]
    assert expected in metrics["failures"]


def test_failed_replica_is_not_hidden_by_good_survivors():
    trace = {
        key: value.repeat_interleave(2, dim=1) for key, value in ideal_trace().items()
    }
    trace["alive"][100:, 1] = False
    result = summarize(trace)
    assert result["survival"] == 0.5
    assert "survival" in result["failures"]


def test_stationary_policy_fails_a_small_forward_command():
    result = summarize_case(
        ideal_trace(), 0.01, 200, 2.0, torch.tensor([0.5, 0.0, 0.0])
    )
    assert result["linear_rmse_m_s"] == pytest.approx(0.5)
    assert "linear_rmse_m_s" in result["failures"]


def test_balanced_feet_must_carry_the_body_weight():
    trace = ideal_trace()
    trace["load"] *= 0.25
    result = summarize(trace)
    assert result["load_share"] == pytest.approx([0.25] * 4, abs=0.01)
    assert "foot_support_weight_ratio" in result["failures"]


def test_clipping_gate_measures_discarded_motion_in_radians():
    trace = ideal_trace()
    trace["clipped"][:, :, 2] = True
    trace["clip_error"][:, :, 2] = 0.001
    assert summarize(trace)["passed"]
    trace["clip_error"][:, :, 2] = 0.1
    assert "max_discarded_target_rms_rad" in summarize(trace)["failures"]


@pytest.mark.parametrize("death_step", [100, 201])
def test_no_survivors_fails_without_nan_report(death_step):
    trace = ideal_trace()
    trace["alive"][death_step:] = False
    result = summarize(trace)
    assert not result["passed"]
    assert result["samples"] == max(0, death_step - 200)


def test_stride_measurement_detects_load_swapping_between_strides():
    trace = ideal_trace()
    # Whole-run shares can be equal while alternate strides neglect half the feet.
    cycles = (trace["phase"][:, 0, 0] / (2 * math.pi)).floor().long()
    trace["load"][cycles % 2 == 0, :, :2] = 0
    trace["load"][cycles % 2 == 1, :, 2:] = 0
    result = summarize(trace)
    assert result["load_share"] == pytest.approx([0.25] * 4, abs=0.01)
    assert "stride_load_max_deviation_p90" in result["failures"]


def test_jitter_separates_smooth_gait_motion_from_fast_oscillation():
    time = torch.arange(600).float() * 0.01
    slow = torch.sin(2 * math.pi * 3 * time)[:, None, None]
    fast = 0.1 * torch.sin(2 * math.pi * 25 * time)[:, None, None]
    valid = torch.ones(600, 1, dtype=torch.bool)
    assert high_frequency_rms(slow, valid, 0.01) < 1e-4
    assert high_frequency_rms(slow + fast, valid, 0.01) == pytest.approx(
        0.1 / math.sqrt(2), rel=0.01
    )
