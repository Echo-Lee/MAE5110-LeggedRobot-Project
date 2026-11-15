"""Robustness suite for the commanded run-up-and-slide of a Go2 policy.

Runs the slide evaluation (scripts/evaluate_slide.py) under off-nominal
conditions and checks each against the stage-1 acceptance criteria:

    nominal          ideal conditions (reference)
    randomized       training randomization of link masses and PD gains on
    random_trigger   slide starts after a random 2.5-6 s run-up
    entry_speeds     run-up speeds 0.5 / 1.0 / 1.5 m/s
    messy_entry      sideways 0.3 m/s and turning 0.5 rad/s until the slide
    push             0.5 m/s sideways kick 0.3 s into the slide
    sensor_noise     IMU / joint encoder noise on the policy's inputs
    friction_error   policy told mu off by up to +-30%
    combined         randomized + random_trigger + sensor_noise + push

Criteria on the trained slide range (mu <= 0.2), per scenario:
    no falls; mean decel >= 0.75 mu*g; mean distance ratio <= 1.5;
    mean decel within 10% of the nominal scenario.
Scenarios with a push are graded on falls only: the kick adds sideways
speed after the entry speed is measured, so the predicted distance and
deceleration ratios are biased against them by construction.
mu = 0.3 (never trained for slides) is reported but not graded.

    uv run --frozen scripts/evaluate_slide_robustness.py --task go2_lowfric \\
        --load_run RUN --original_cfg --device cuda:0 \\
        --output logs/eval/robustness_RUN.json
"""

import argparse
import json
from pathlib import Path
import statistics

from evaluate_slide import (
    PROTOCOL,
    SlidePerturbations,
    checkpoint_metadata,
    evaluate,
    make_slide_env,
    trial_seconds,
)

FRICTIONS = (0.05, 0.1, 0.15, 0.2, 0.3)
TRAINED_MAX_FRICTION = 0.2
SPEEDS = (0.8, 1.2)
# Physical-unit noise per observation: IMU gyro, gravity direction, joint
# encoders and joint velocity estimates of a real robot (rough magnitudes).
SENSOR_NOISE = {
    "base_ang_vel": 0.1,
    "projected_gravity": 0.03,
    "dof_pos_obs": 0.01,
    "dof_vel": 0.3,
}
CRITERIA = {
    "max_fall_rate": 0.0,
    "min_decel_ratio": 0.75,
    "max_distance_ratio": 1.5,
    "max_decel_drop_vs_nominal": 0.10,
}

SCENARIOS = {
    "nominal": dict(),
    "randomized": dict(randomize_robot=True),
    "random_trigger": dict(perturb=SlidePerturbations(runup_range=(2.5, 6.0))),
    "entry_speeds": dict(speeds=(0.5, 1.0, 1.5)),
    "messy_entry": dict(perturb=SlidePerturbations(entry_lateral=0.3, entry_yaw=0.5)),
    "push": dict(
        perturb=SlidePerturbations(push_speed=0.5, push_delay=0.3),
        graded_on=("falls",),
    ),
    "sensor_noise": dict(perturb=SlidePerturbations(noise=SENSOR_NOISE)),
    "friction_error": dict(perturb=SlidePerturbations(friction_error=0.3)),
    "combined": dict(
        randomize_robot=True,
        perturb=SlidePerturbations(
            runup_range=(2.5, 6.0),
            noise=SENSOR_NOISE,
            push_speed=0.5,
            push_delay=0.3,
        ),
        graded_on=("falls",),
    ),
}
ALL_CHECKS = ("falls", "decel", "distance", "worse than nominal")


def by_friction(results):
    """Per friction: worst fall rate and mean decel / distance over speeds."""
    summary = {}
    for friction in sorted({r["friction"] for r in results}):
        cases = [r for r in results if r["friction"] == friction]

        def mean(key):
            values = [r[key] for r in cases if r[key] is not None]
            return statistics.mean(values) if values else None

        summary[friction] = {
            "fall_rate": max(r["fall_rate"] for r in cases),
            "decel_ratio": mean("decel_ratio"),
            "distance_ratio": mean("distance_ratio"),
            "all_feet_loaded_fraction": mean("all_feet_loaded_fraction"),
        }
    return summary


def grade(summary, nominal_decel=None, graded_on=ALL_CHECKS):
    """Check the trained range (mu <= 0.2) against CRITERIA.

    Only the checks named in ``graded_on`` can fail the scenario; the
    others are still reported in the decel and distance values.
    """
    trained = {f: s for f, s in summary.items() if f <= TRAINED_MAX_FRICTION}
    failures = []
    if max(s["fall_rate"] for s in trained.values()) > CRITERIA["max_fall_rate"]:
        failures.append("falls")
    decels = [s["decel_ratio"] for s in trained.values() if s["decel_ratio"]]
    distances = [s["distance_ratio"] for s in trained.values() if s["distance_ratio"]]
    decel = statistics.mean(decels) if decels else None
    distance = statistics.mean(distances) if distances else None
    if decel is None or decel < CRITERIA["min_decel_ratio"]:
        failures.append("decel")
    if distance is None or distance > CRITERIA["max_distance_ratio"]:
        failures.append("distance")
    if (
        nominal_decel is not None
        and decel is not None
        and decel < nominal_decel * (1 - CRITERIA["max_decel_drop_vs_nominal"])
    ):
        failures.append("worse than nominal")
    failures = [failure for failure in failures if failure in graded_on]
    return {
        "decel_ratio": decel,
        "distance_ratio": distance,
        "graded_on": list(graded_on),
        "passed": not failures,
        "failures": failures,
    }


def table(report):
    frictions = FRICTIONS
    header = (
        "| scenario | "
        + " | ".join(
            f"μ {f:g}{'*' if f > TRAINED_MAX_FRICTION else ''}" for f in frictions
        )
        + " | decel (μ≤0.2) | dist (μ≤0.2) | verdict |"
    )
    lines = [
        "cells: fall rate / decel ÷ μg / distance ratio; * = untrained for slides",
        "",
        header,
        "|---" * (len(frictions) + 4) + "|",
    ]
    for name, scenario in report["scenarios"].items():
        cells = []
        for f in frictions:
            s = scenario["by_friction"].get(str(f)) or scenario["by_friction"].get(f)
            if s is None:
                cells.append("–")
                continue
            decel = "–" if s["decel_ratio"] is None else f"{s['decel_ratio']:.2f}"
            dist = "–" if s["distance_ratio"] is None else f"{s['distance_ratio']:.2f}"
            cells.append(f"{s['fall_rate']:.0%} / {decel} / {dist}")
        g = scenario["grade"]
        verdict = "PASS" if g["passed"] else "FAIL: " + ", ".join(g["failures"])
        if list(g.get("graded_on", ALL_CHECKS)) != list(ALL_CHECKS):
            verdict += " (falls only)"
        decel = "–" if g["decel_ratio"] is None else f"{g['decel_ratio']:.2f}"
        dist = "–" if g["distance_ratio"] is None else f"{g['distance_ratio']:.2f}"
        lines.append(
            f"| {name} | " + " | ".join(cells) + f" | {decel} | {dist} | {verdict} |"
        )
    return "\n".join(lines)


def regrade(report):
    """Re-apply the current grading to a saved report's per-friction data."""
    nominal = report["scenarios"].get("nominal")
    nominal_decel = nominal["grade"]["decel_ratio"] if nominal else None
    for name, scenario in report["scenarios"].items():
        summary = {float(f): s for f, s in scenario["by_friction"].items()}
        scenario["grade"] = grade(
            summary,
            nominal_decel if name != "nominal" else None,
            SCENARIOS[name].get("graded_on", ALL_CHECKS),
        )
    report["criteria"] = CRITERIA
    report["passed"] = all(s["grade"]["passed"] for s in report["scenarios"].values())
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="go2_lowfric")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--load_run", default="-1", help="run directory; -1 = latest")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--label", default=None)
    parser.add_argument("--backend", choices=("mujoco", "vsim"), default="mujoco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--runup", type=float, default=4.0)
    parser.add_argument("--hold", type=float, default=0.5)
    parser.add_argument(
        "--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="re-grade the existing report at --output without simulating",
    )
    args = parser.parse_args()
    if args.regrade:
        report = regrade(json.loads(args.output.read_text()))
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(table(report))
        print(f"\noverall: {'PASS' if report['passed'] else 'FAIL'}")
        return

    report = {
        "protocol": PROTOCOL + "-robustness-v1",
        "criteria": CRITERIA,
        "trained_max_friction": TRAINED_MAX_FRICTION,
        "sensor_noise": SENSOR_NOISE,
        "arguments": vars(args) | {"output": str(args.output)},
        "scenarios": {},
    }
    nominal_decel = None
    for name in args.scenarios:
        spec = SCENARIOS[name]
        perturb = spec.get("perturb", SlidePerturbations())
        speeds = spec.get("speeds", SPEEDS)
        cases = [(f, s) for f in FRICTIONS for s in speeds]
        seconds = trial_seconds(FRICTIONS, speeds, args.runup, perturb)
        env, runner, _, _, path = make_slide_env(
            task=args.task,
            load_run=args.load_run,
            original_cfg=args.original_cfg,
            experiment_name=args.experiment_name,
            checkpoint=args.checkpoint,
            device=args.device,
            backend=args.backend,
            seed=args.seed,
            frictions=FRICTIONS,
            num_envs=len(cases) * args.replicas,
            seconds=seconds,
            randomize_robot=spec.get("randomize_robot", False),
        )
        try:
            results = evaluate(
                env,
                runner,
                cases,
                replicas=args.replicas,
                runup_s=args.runup,
                hold_s=args.hold,
                seconds=seconds,
                perturb=perturb,
                seed=args.seed,
            )
        finally:
            env._backend.close()
        summary = by_friction(results)
        graded = grade(
            summary,
            nominal_decel if name != "nominal" else None,
            spec.get("graded_on", ALL_CHECKS),
        )
        if name == "nominal":
            nominal_decel = graded["decel_ratio"]
        report["scenarios"][name] = {
            "by_friction": {str(f): s for f, s in summary.items()},
            "grade": graded,
            "cases": results,
        }
        report.setdefault("label", args.label or f"{args.task}/{path.parent.name}")
        report.update(checkpoint_metadata(path))
        verdict = "PASS" if graded["passed"] else "FAIL " + ",".join(graded["failures"])
        print(f"{name}: {verdict}", flush=True)
    report["passed"] = all(s["grade"]["passed"] for s in report["scenarios"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print()
    print(table(report))
    print(f"\nreport: {args.output}  overall: {'PASS' if report['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
