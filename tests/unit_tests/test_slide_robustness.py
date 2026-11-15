"""Slide robustness suite: per-friction summary and acceptance grading."""

from pathlib import Path
import sys

import pytest

# The evaluation scripts import each other as top-level modules, as when run
# with `uv run scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from evaluate_slide import SlidePerturbations, trial_seconds  # noqa: E402
from evaluate_slide_robustness import (  # noqa: E402
    SCENARIOS,
    by_friction,
    grade,
)


def _case(friction, speed, fall=0.0, decel=0.9, distance=1.2, all4=0.6):
    return {
        "friction": friction,
        "runup_command_m_s": speed,
        "fall_rate": fall,
        "decel_ratio": decel,
        "distance_ratio": distance,
        "all_feet_loaded_fraction": all4,
    }


def test_by_friction_takes_worst_falls_and_mean_braking():
    summary = by_friction(
        [_case(0.1, 0.8, fall=0.0, decel=0.8), _case(0.1, 1.2, fall=0.5, decel=1.0)]
    )
    assert summary[0.1]["fall_rate"] == 0.5
    assert summary[0.1]["decel_ratio"] == pytest.approx(0.9)


def test_grade_passes_a_good_slide_and_ignores_untrained_friction():
    summary = by_friction(
        [_case(0.05, 0.8), _case(0.2, 1.2), _case(0.3, 1.2, fall=1.0, decel=0.1)]
    )
    result = grade(summary)
    assert result["passed"] and result["failures"] == []


@pytest.mark.parametrize(
    ("case", "failure"),
    [
        (dict(fall=0.125), "falls"),
        (dict(decel=0.6), "decel"),
        (dict(distance=1.8), "distance"),
    ],
)
def test_grade_reports_each_criterion(case, failure):
    summary = by_friction([_case(0.1, 0.8, **case)])
    assert failure in grade(summary)["failures"]


def test_grade_compares_against_nominal_braking():
    summary = by_friction([_case(0.1, 0.8, decel=0.80)])
    assert "worse than nominal" in grade(summary, nominal_decel=0.95)["failures"]
    assert grade(summary, nominal_decel=0.85)["passed"]


def test_trial_covers_the_latest_run_up_and_slowest_stop():
    nominal = trial_seconds([0.05, 0.2], [0.8, 1.2], 4.0)
    late = trial_seconds(
        [0.05, 0.2], [0.8, 1.2], 4.0, SlidePerturbations(runup_range=(2.5, 6.0))
    )
    assert late == pytest.approx(nominal + 2.0)


def test_every_scenario_is_well_formed():
    assert "nominal" in SCENARIOS and "combined" in SCENARIOS
    for spec in SCENARIOS.values():
        assert set(spec) <= {"perturb", "speeds", "randomize_robot", "graded_on"}
        if "perturb" in spec:
            assert isinstance(spec["perturb"], SlidePerturbations)


def test_push_scenarios_are_graded_on_falls_only():
    from evaluate_slide_robustness import grade as grade_fn

    pushed = by_friction([_case(0.1, 0.8, decel=0.5, distance=2.0)])
    assert grade_fn(pushed, graded_on=("falls",))["passed"]
    fallen = by_friction([_case(0.1, 0.8, fall=0.25, decel=0.5, distance=2.0)])
    assert grade_fn(fallen, graded_on=("falls",))["failures"] == ["falls"]
    assert SCENARIOS["push"]["graded_on"] == ("falls",)
    assert SCENARIOS["combined"]["graded_on"] == ("falls",)
