"""
Fixed-friction, fixed-command evaluation of a Go2-family policy.

    uv run --frozen scripts/evaluate_go2_friction.py --task go2_lowfric \\
        --device cuda:0 --output logs/eval/go2_lowfric.json
    uv run --frozen scripts/evaluate_go2_friction.py --task go2 \\
        --device cuda:0 --output logs/eval/go2.json
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import class_to_dict, get_load_path, set_seed
from gym.utils.locomotion_metrics import contact_point_velocity, summarize_locomotion
from gym.utils.task_registry import task_registry

PROTOCOL = "go2-friction-v1"
FRICTIONS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
COMMANDS = {
    "stopped": (0.0, 0.0, 0.0),
    "forward_slow": (0.5, 0.0, 0.0),
    "forward": (1.0, 0.0, 0.0),
    "forward_fast": (2.0, 0.0, 0.0),
    "left": (0.0, 0.5, 0.0),
    "turn_left": (0.0, 0.0, 1.0),
}
# Go2 foot collision sphere radius (resources/robots/go2/urdf/go2.urdf).
FOOT_RADIUS_M = 0.022
# Friction regimes that receive acceptance bounds. Coefficients in between are
# reported but outside both training bins of go2_lowfric, so ungated.
ICE_MAX_FRICTION = 0.2
GROUND_MIN_FRICTION = 0.6


def case_bounds(friction: float, command: torch.Tensor) -> dict | None:
    """Declared acceptance bounds, fixed before any checkpoint is evaluated."""
    speed = command[:2].norm().item()
    if friction >= GROUND_MIN_FRICTION:
        return {
            "fall_rate": (0.0, 0.05),
            "linear_rmse_m_s": (0.0, 0.20 + 0.10 * speed),
            "yaw_rmse_rad_s": (0.0, 0.35),
        }
    if friction <= ICE_MAX_FRICTION:
        return {
            "fall_rate": (0.0, 0.05),
            "linear_rmse_m_s": (0.0, 0.30 + 0.15 * speed),
            "yaw_rmse_rad_s": (0.0, 0.50),
        }
    return None


def check_case(result: dict, bounds: dict | None) -> dict:
    if bounds is None:
        return {"bounds": None, "passed": None, "failures": []}
    failures = [
        name
        for name, (low, high) in bounds.items()
        if result[name] is None or not low <= result[name] <= high
    ]
    return {"bounds": bounds, "passed": not failures, "failures": failures}


@torch.no_grad()
def evaluate(env, runner, cases, *, replicas, seconds, warmup):
    num_envs = len(cases) * replicas
    frictions = torch.tensor(
        [friction for friction, _ in cases], device=env.device
    ).repeat_interleave(replicas)
    commands = torch.tensor(
        [COMMANDS[name] for _, name in cases], device=env.device
    ).repeat_interleave(replicas, dim=0)
    env._backend.set_contact_friction(
        torch.arange(num_envs, device=env.device), frictions
    )
    # Start every replica from the same reset distribution on its surface.
    env.reset()

    feet = env.feet_indices
    joints = env.actuated_dof_indices
    alive = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    records = {
        key: []
        for key in (
            "alive",
            "body_velocity",
            "world_planar_speed",
            "torques",
            "joint_velocity",
            "foot_load",
            "foot_contact_velocity",
            "base_height",
        )
    }
    for _ in range(round(seconds / env.dt)):
        # Pin commands even when the task's periodic sampler fires.
        env.commands[:] = commands
        actions = runner.get_inference_actions()
        runner.set_actions(runner.actor_cfg["actions"], actions)
        env.step()
        # Terminated environments are not reset: failure is absorbing.
        alive &= ~env.terminated
        values = {
            "alive": alive,
            "body_velocity": torch.cat(
                (env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), dim=1
            ),
            "world_planar_speed": env.root_states[:, 7:9].norm(dim=1),
            # With decimation > 1 this samples the last physics substep.
            "torques": env.torques,
            "joint_velocity": env.dof_vel[:, joints],
            "foot_load": env.contact_forces[:, feet, 2],
            "foot_contact_velocity": contact_point_velocity(
                env._rigid_body_lin_vel[:, feet],
                env._rigid_body_ang_vel[:, feet],
                FOOT_RADIUS_M,
            ),
            "base_height": env.root_states[:, 2],
        }
        for key, value in values.items():
            records[key].append(value.clone())
    trace = {key: torch.stack(value).cpu() for key, value in records.items()}
    if not all(torch.isfinite(value.float()).all() for value in trace.values()):
        raise RuntimeError("Nonfinite evaluation trajectory")

    mass_kg = env._backend.link_mass[0].sum().item()
    torque_limits = env.actuated_torque_limits.cpu()
    results = {}
    for i, (friction, name) in enumerate(cases):
        selection = slice(i * replicas, (i + 1) * replicas)
        command = torch.tensor(COMMANDS[name])
        result = summarize_locomotion(
            {key: value[:, selection] for key, value in trace.items()},
            dt=env.dt,
            warmup_steps=round(warmup / env.dt),
            command=command,
            mass_kg=mass_kg,
            torque_limits=torque_limits,
        )
        result |= check_case(result, case_bounds(friction, command))
        result |= {
            "friction": friction,
            "command_name": name,
            "command": {
                "vx_m_s": COMMANDS[name][0],
                "vy_m_s": COMMANDS[name][1],
                "yaw_rad_s": COMMANDS[name][2],
            },
        }
        results[f"mu={friction:g}/{name}"] = result
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="go2_lowfric")
    parser.add_argument(
        "--experiment_name", default=None, help="defaults to the task's experiment"
    )
    parser.add_argument("--load_run", default="-1", help="run directory; -1 = latest")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--backend", choices=("mujoco", "vsim"), default="mujoco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--frictions", type=float, nargs="+", default=FRICTIONS)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--misreport_friction", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.replicas < 1 or args.warmup < 0 or args.seconds - args.warmup < 3:
        parser.error("use positive replicas and at least 3 measured seconds")
    if min(args.frictions) <= 0:
        parser.error("friction coefficients must be positive")
    load_run = -1 if args.load_run == "-1" else args.load_run
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            args.task,
            original_cfg=args.original_cfg,
            experiment_name=args.experiment_name,
            load_run=args.load_run if args.original_cfg else None,
        )
    )
    experiment = args.experiment_name or train_cfg.runner.experiment_name
    cases = [(friction, name) for friction in args.frictions for name in COMMANDS]
    env_cfg.env.num_envs = len(cases) * args.replicas
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.push_robots.toggle = False
    # Every replica starts at rest; the policy must build speed on its surface.
    env_cfg.init_state.root_vel_range = [[0.0, 0.0]] * 6
    apply_domain_randomization_override(env_cfg, "off")
    # Re-enable only the friction axis as an envelope, so every backend
    # allocates per-environment friction; values are then set explicitly.
    env_cfg.domain_randomization.startup.contact_friction_range = [
        min(args.frictions),
        max(args.frictions),
    ]
    env_cfg.seed = args.seed
    train_cfg.runner.device = args.device
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(args.seed)
    path = Path(
        get_load_path(experiment, load_run=load_run, checkpoint=args.checkpoint)
    )
    env = task_registry.make_env(args.task, env_cfg, args.device, True, args.backend)
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        runner.load(path, load_optimizer=False)
        runner.switch_to_eval()
        if args.misreport_friction is not None:
            if "contact_friction" not in runner.actor_cfg["obs"]:
                parser.error("--misreport_friction needs a contact_friction policy")
            # Replace the live backend view with a constant: the actor sees
            # this value while the physics keeps each case's true friction.
            env.contact_friction = torch.full(
                (env.num_envs, 1), args.misreport_friction, device=env.device
            )
        results = evaluate(
            env,
            runner,
            cases,
            replicas=args.replicas,
            seconds=args.seconds,
            warmup=args.warmup,
        )
    finally:
        env._backend.close()
    gated = [case["passed"] for case in results.values() if case["passed"] is not None]
    report = {
        "protocol": PROTOCOL,
        "task": args.task,
        "experiment": experiment,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree": subprocess.check_output(["git", "status", "--short"], text=True),
        "arguments": vars(args) | {"output": str(args.output)},
        "env_config": class_to_dict(env_cfg),
        "runner_config": class_to_dict(train_cfg),
        "foot_order": list(env.robot_layout.body_groups["feet"]),
        "cases": results,
        "passed": all(gated),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report": str(args.output),
                "passed_cases": sum(gated),
                "gated_cases": len(gated),
                "total_cases": len(results),
            }
        )
    )
    if args.check and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
