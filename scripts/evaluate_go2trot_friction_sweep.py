"""Measure Go2-family gross mechanical CoT across fixed friction and speed.

The sweep uses the policy checkpoint without retraining. Each friction value
creates a fresh batch of otherwise nominal environments, so the configured
coefficient is fixed for every replica and is never resampled at reset.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry
from evaluate_go2trot import mechanical_energy_metrics


FRICTIONS = tuple(round(value / 10, 1) for value in range(1, 11))
SPEEDS_M_S = (0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


@torch.no_grad()
def evaluate_batch(env, runner, speeds, replicas, seconds, warmup, gait_frequency):
    """Evaluate one fixed-friction environment batch at each forward speed."""
    if env.cfg.control.decimation != 1:
        raise ValueError(
            "mechanical-energy evaluation requires one physics step per control "
            "step; add substep accounting before using a different decimation"
        )
    commands = torch.zeros(len(speeds) * replicas, 3, device=env.device)
    commands[:, 0] = torch.tensor(speeds, device=env.device).repeat_interleave(replicas)
    env.commands[:] = commands
    if hasattr(env, "phase_frequency"):
        env.phase_frequency[:, 0] = gait_frequency
    alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    records = {
        key: []
        for key in (
            "alive",
            "velocity",
            "joint_velocity",
            "joint_velocity_before",
            "torque",
            "position",
            "position_before",
            "body_weight",
        )
    }
    body_weight = env._backend.link_mass.sum(dim=1) * 9.81
    for _ in range(round(seconds / env.dt)):
        env.commands[:] = commands
        actions = runner.get_inference_actions()
        runner.set_actions(runner.actor_cfg["actions"], actions)
        joint_velocity_before = env.dof_vel.clone()
        position_before = env.root_states[:, :3].clone()
        env.step()
        alive &= ~env.terminated
        values = {
            "alive": alive,
            "velocity": env.base_lin_vel,
            "joint_velocity": env.dof_vel,
            "joint_velocity_before": joint_velocity_before,
            "torque": env.torques,
            "position": env.root_states[:, :3],
            "position_before": position_before,
            "body_weight": body_weight,
        }
        for key, value in values.items():
            records[key].append(value.clone())
    trace = {key: torch.stack(value).cpu() for key, value in records.items()}
    if not all(torch.isfinite(value).all() for value in trace.values()):
        raise RuntimeError("Nonfinite evaluation trajectory")

    results = []
    for speed_index, speed in enumerate(speeds):
        selection = slice(speed_index * replicas, (speed_index + 1) * replicas)
        case_trace = {key: value[:, selection] for key, value in trace.items()}
        valid = case_trace["alive"].bool()
        valid[: round(warmup / env.dt)] = False
        metrics = mechanical_energy_metrics(
            case_trace,
            env.dt,
            valid,
            torch.tensor([speed, 0.0, 0.0]),
        )
        metrics.update(
            speed_m_s=speed,
            survival=case_trace["alive"][-1].float().mean().item(),
            samples=int(valid.sum()),
            mean_forward_velocity_m_s=(
                case_trace["velocity"][..., 0][valid].mean().item()
                if valid.any()
                else None
            ),
        )
        results.append(metrics)
    return results


def write_plot(rows, output_path, task_name):
    """Write path-energy-intensity-vs-friction plot in J/m."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for speed in SPEEDS_M_S:
        speed_rows = [row for row in rows if row["speed_m_s"] == speed]
        valid_rows = [
            row
            for row in speed_rows
            if row.get("gross_mechanical_energy_j_per_path_m") is not None
        ]
        if valid_rows:
            axis.plot(
                [row["friction"] for row in valid_rows],
                [row["gross_mechanical_energy_j_per_path_m"] for row in valid_rows],
                marker="o",
                label=f"{speed:g} m/s",
            )
    axis.set(
        xlabel="Sliding friction coefficient",
        ylabel="Gross mechanical energy per distance (J/m)",
        title=f"{task_name} energy consumption across friction and commanded speed",
    )
    axis.grid(True, alpha=0.3)
    axis.legend(title="Forward command")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("go2", "go2trot"), default="go2trot")
    parser.add_argument("--experiment_name", default="go2trot")
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--backend", choices=("mujoco", "vsim"), default="mujoco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--gait_frequency", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicas < 1 or args.warmup < 0 or args.seconds - args.warmup < 3:
        parser.error("use positive replicas and at least 3 measured seconds")
    if args.gait_frequency <= 0:
        parser.error("--gait_frequency must be positive")

    import gym.envs  # noqa: F401

    path = Path(
        get_load_path(
            name=args.experiment_name,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
        )
    )
    all_rows = []
    for friction in FRICTIONS:
        env_cfg, train_cfg = copy.deepcopy(
            task_registry.get_cfgs(
                args.task,
                original_cfg=args.original_cfg,
                experiment_name=args.experiment_name,
                load_run=args.load_run,
            )
        )
        env_cfg.env.num_envs = len(SPEEDS_M_S) * args.replicas
        env_cfg.env.episode_length_s = args.seconds + 1
        env_cfg.commands.resampling_time = args.seconds + 1
        env_cfg.push_robots.toggle = False
        apply_domain_randomization_override(env_cfg, "friction-only")
        env_cfg.domain_randomization.startup.contact_friction_range = [
            friction,
            friction,
        ]
        env_cfg.seed = args.seed
        train_cfg.runner.device = args.device
        train_cfg.runner.resume = False
        train_cfg.logging.enable_local_saving = False
        train_cfg.log_dir = None
        task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
        set_seed(args.seed)
        env = task_registry.make_env(
            args.task, env_cfg, args.device, True, args.backend
        )
        try:
            runner = task_registry.make_alg_runner(env, train_cfg)
            runner.load(path, load_optimizer=False)
            runner.switch_to_eval()
            rows = evaluate_batch(
                env,
                runner,
                SPEEDS_M_S,
                args.replicas,
                args.seconds,
                args.warmup,
                args.gait_frequency,
            )
        finally:
            env._backend.close()
        for row in rows:
            row["friction"] = friction
            row["gait_frequency_hz"] = args.gait_frequency
        all_rows.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    plot_path = args.output.with_suffix(".png")
    report = {
        "protocol": "go2-friction-sweep-v2",
        "task": args.task,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "arguments": vars(args) | {"output": str(args.output)},
        "frictions": FRICTIONS,
        "speeds_m_s": SPEEDS_M_S,
        "rows": all_rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_plot(all_rows, plot_path, args.task)
    print(
        json.dumps(
            {"report": str(args.output), "csv": str(csv_path), "plot": str(plot_path)}
        )
    )


if __name__ == "__main__":
    main()
