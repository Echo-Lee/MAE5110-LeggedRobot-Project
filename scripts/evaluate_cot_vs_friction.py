"""Matched-speed cost of transport versus contact friction for Go2 policies.

Policies do not hold commanded speed exactly (they overshoot or undershoot,
differently on each surface), so a fixed command does not give a fixed speed.
Instead, on each friction coefficient every replica walks straight ahead from
rest at one of a sweep of commanded speeds; each command yields an (actual
speed, CoT) point, and CoT is interpolated at exact actual target speeds.
A target faster than any speed reached on that surface is "unreachable", and
the top speed reached on each surface is reported as well.

    uv run --frozen scripts/evaluate_cot_vs_friction.py --task go2_lowfric \\
        --load_run RUN --original_cfg --device cuda:0 \\
        --output logs/eval/cot_lowfric.json
    uv run --frozen scripts/plot_cot_vs_friction.py logs/eval/cot_*.json \\
        --output logs/eval/cot_vs_friction.png

Two energy definitions are reported: sum |tau*qdot| (braking work counts as
cost, matching evaluate_go2_friction.py) and positive work only.
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
from gym.utils.locomotion_metrics import cot_at_speed, replica_speed_and_cot
from gym.utils.task_registry import task_registry

PROTOCOL = "go2-cot-vs-friction-v2"
FRICTIONS = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
# Commanded forward speeds swept on every surface (m/s).
COMMAND_SWEEP = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
# Actual speeds at which CoT is compared across friction (m/s).
TARGET_SPEEDS = (0.5, 1.0, 1.5)


@torch.no_grad()
def evaluate(env, runner, cases, *, replicas, seconds, warmup):
    num_envs = len(cases) * replicas
    frictions = torch.tensor(
        [friction for friction, _ in cases], device=env.device
    ).repeat_interleave(replicas)
    commands = torch.zeros(num_envs, 3, device=env.device)
    commands[:, 0] = torch.tensor(
        [speed for _, speed in cases], device=env.device
    ).repeat_interleave(replicas)
    env._backend.set_contact_friction(
        torch.arange(num_envs, device=env.device), frictions
    )
    env.reset()

    joints = env.actuated_dof_indices
    alive = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    keys = ("alive", "forward_speed", "position", "power_abs", "power_positive")
    records = {key: [] for key in keys}
    for _ in range(round(seconds / env.dt)):
        env.commands[:] = commands
        actions = runner.get_inference_actions()
        runner.set_actions(runner.actor_cfg["actions"], actions)
        env.step()
        alive &= ~env.terminated
        # With decimation > 1 this samples the last physics substep.
        joint_power = env.torques * env.dof_vel[:, joints]
        values = {
            "alive": alive,
            "forward_speed": env.base_lin_vel[:, 0],
            "position": env.root_states[:, :2],
            "power_abs": joint_power.abs().sum(dim=1),
            "power_positive": joint_power.clamp_min(0).sum(dim=1),
        }
        for key, value in values.items():
            records[key].append(value.clone())
    trace = {key: torch.stack(value).cpu() for key, value in records.items()}
    if not all(torch.isfinite(value.float()).all() for value in trace.values()):
        raise RuntimeError("Nonfinite evaluation trajectory")

    mass_kg = env._backend.link_mass[0].sum().item()
    sweep = {}
    for i, (friction, command) in enumerate(cases):
        selection = slice(i * replicas, (i + 1) * replicas)
        per_replica = replica_speed_and_cot(
            {key: value[:, selection] for key, value in trace.items()},
            dt=env.dt,
            warmup_steps=round(warmup / env.dt),
            mass_kg=mass_kg,
        )
        alive = per_replica["survived"]
        point = {"command_m_s": command, "survived": int(alive.sum())}
        if alive.any():
            for key in ("speed", "cot", "cot_positive_work"):
                values = per_replica[key][alive]
                point[key] = values.mean().item()
                point[key + "_std"] = values.std().item() if values.numel() > 1 else 0.0
        sweep.setdefault(friction, []).append(point)
    return sweep


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="go2_lowfric")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--load_run", default="-1", help="run directory; -1 = latest")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--label", default=None, help="legend name for plots")
    parser.add_argument("--backend", choices=("mujoco", "vsim"), default="mujoco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--frictions", type=float, nargs="+", default=FRICTIONS)
    parser.add_argument("--commands", type=float, nargs="+", default=COMMAND_SWEEP)
    parser.add_argument("--target_speeds", type=float, nargs="+", default=TARGET_SPEEDS)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicas < 2 or args.warmup < 0 or args.seconds - args.warmup < 3:
        parser.error("use at least 2 replicas and 3 measured seconds")
    if min(args.frictions) <= 0 or min(args.commands) <= 0:
        parser.error("frictions and commanded speeds must be positive")
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
    cases = [(f, c) for f in args.frictions for c in args.commands]
    env_cfg.env.num_envs = len(cases) * args.replicas
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.push_robots.toggle = False
    if hasattr(env_cfg.commands, "slide_probability"):
        # Fixed commands are pinned below; no run-up-and-slide episodes.
        env_cfg.commands.slide_probability = 0.0
    if hasattr(env_cfg, "friction_switch"):
        # No random mid-episode friction switches (go2_lowfric_est trains
        # with them); evaluate_friction_switch.py schedules its own.
        env_cfg.friction_switch.probability = 0.0
    env_cfg.init_state.root_vel_range = [[0.0, 0.0]] * 6
    apply_domain_randomization_override(env_cfg, "off")
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
        sweep = evaluate(
            env,
            runner,
            cases,
            replicas=args.replicas,
            seconds=args.seconds,
            warmup=args.warmup,
        )
    finally:
        env._backend.close()
    report = {
        "protocol": PROTOCOL,
        "label": args.label or f"{args.task}/{path.parent.name}",
        "task": args.task,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "arguments": vars(args) | {"output": str(args.output)},
        "env_config": class_to_dict(env_cfg),
        "runner_config": class_to_dict(train_cfg),
        "cases": [
            {"friction": friction}
            | cot_at_speed([p for p in points if "speed" in p], target)
            for friction, points in sweep.items()
            for target in args.target_speeds
        ],
        "sweep": [
            {"friction": friction, "points": points}
            for friction, points in sweep.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    measured = sum(case["status"] == "ok" for case in report["cases"])
    print(
        json.dumps(
            {
                "report": str(args.output),
                "measured_cases": measured,
                "total_cases": len(report["cases"]),
            }
        )
    )


if __name__ == "__main__":
    main()
