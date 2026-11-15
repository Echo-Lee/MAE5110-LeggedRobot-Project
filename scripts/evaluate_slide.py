"""Run-up-and-slide evaluation of a Go2 policy on low-friction surfaces.

Every replica starts at rest, tracks a forward run-up speed v0 for the run-up
time, then follows a command that decays at mu*g (the deceleration of a body
sliding on all four feet) down to a stop. Per friction x run-up speed it
reports whether the robot stays up and how it slows down: slide distance and
deceleration against the friction prediction, fraction of the slide with all
four feet loaded, touchdowns per second, and joint energy spent sliding (see
summarize_slide in gym/utils/locomotion_metrics.py).

    uv run --frozen scripts/evaluate_slide.py --task go2_lowfric \\
        --load_run RUN --original_cfg --device cuda:0 \\
        --output logs/eval/slide.json

Works for any Go2-family checkpoint, including runs trained without slide
episodes, so a policy's slide can be compared before and after training it.
Off-nominal conditions (random trigger time, messy entry, pushes, sensor
noise, a wrong friction estimate, randomized robot properties) are set with
SlidePerturbations; scripts/evaluate_slide_robustness.py runs them as a suite.
"""

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.envs.go2.go2_lowfric import slide_phase_profile, slide_speed_profile
from gym.utils.helpers import class_to_dict, get_load_path, set_seed
from gym.utils.locomotion_metrics import (
    GRAVITY_M_S2,
    SLIDE_STOP_SPEED,
    summarize_slide,
)
from gym.utils.task_registry import task_registry

PROTOCOL = "go2-slide-v1"
FRICTIONS = (0.05, 0.075, 0.1, 0.15, 0.2)
RUNUP_SPEEDS = (0.8, 1.2)
# Seconds simulated after the slowest predicted stop, to see it settle.
SETTLE_S = 2.0


@dataclass
class SlidePerturbations:
    """Off-nominal conditions for a slide trial (defaults = nominal).

    runup_range: (low, high) s; each replica's run-up length is drawn
        uniformly, so slides start at different times and gait phases.
    entry_lateral, entry_yaw: sideways speed (m/s) and turn rate (rad/s)
        commanded during the run-up, so the slide starts while moving
        sideways or turning.
    push_speed, push_delay: a sideways velocity kick (m/s, perpendicular to
        the direction of travel, random side) push_delay seconds after the
        slide starts.
    noise: physical-unit standard deviation per observation name, added to
        the actor's observations every step (sensor noise).
    friction_error: the policy is told mu * U(1 - e, 1 + e) instead of the
        true mu (a friction estimate that is off by up to e).
    """

    runup_range: tuple[float, float] | None = None
    entry_lateral: float = 0.0
    entry_yaw: float = 0.0
    push_speed: float = 0.0
    push_delay: float = 0.3
    noise: dict | None = None
    friction_error: float = 0.0


def make_slide_env(
    *,
    task,
    load_run,
    original_cfg,
    experiment_name,
    checkpoint,
    device,
    backend,
    seed,
    frictions,
    num_envs,
    seconds,
    randomize_robot=False,
):
    """Build the env and load the policy for slide trials.

    Friction is pinned per environment by the caller. With randomize_robot,
    the task's training randomization of link masses and PD gains is kept
    (friction randomization still off); otherwise all randomization is off.
    """
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            task,
            original_cfg=original_cfg,
            experiment_name=experiment_name,
            load_run=load_run if original_cfg else None,
        )
    )
    experiment = experiment_name or train_cfg.runner.experiment_name
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = seconds + 1
    env_cfg.commands.resampling_time = seconds + 1
    env_cfg.push_robots.toggle = False
    if hasattr(env_cfg.commands, "slide_probability"):
        # The slide profile is pinned by the caller; no random episodes.
        env_cfg.commands.slide_probability = 0.0
    if hasattr(env_cfg.commands, "turn_in_place_probability"):
        env_cfg.commands.turn_in_place_probability = 0.0
    env_cfg.init_state.root_vel_range = [[0.0, 0.0]] * 6
    if not randomize_robot:
        apply_domain_randomization_override(env_cfg, "off")
    startup = env_cfg.domain_randomization.startup
    startup.contact_friction_range = [min(frictions), max(frictions)]
    startup.contact_friction_bins = None
    startup.contact_friction_bin_weights = None
    env_cfg.domain_randomization.episode.resample_contact_friction = False
    env_cfg.seed = seed
    train_cfg.runner.device = device
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(seed)
    load_index = -1 if str(load_run) == "-1" else load_run
    path = Path(get_load_path(experiment, load_run=load_index, checkpoint=checkpoint))
    env = task_registry.make_env(task, env_cfg, device, True, backend)
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.load(path, load_optimizer=False)
    runner.switch_to_eval()
    return env, runner, env_cfg, train_cfg, path


def _push_direction(env, generator):
    """Unit vector perpendicular to travel, on a random side per env."""
    velocity = env.root_states[:, 7:9]
    heading = velocity / velocity.norm(dim=1, keepdim=True).clamp_min(1e-6)
    side = torch.where(
        torch.rand(env.num_envs, device=env.device, generator=generator) < 0.5,
        -1.0,
        1.0,
    )
    return torch.stack((-heading[:, 1], heading[:, 0]), dim=1) * side.unsqueeze(1)


@torch.no_grad()
def evaluate(
    env,
    runner,
    cases,
    *,
    replicas,
    runup_s,
    hold_s,
    seconds,
    perturb=None,
    seed=0,
):
    perturb = perturb or SlidePerturbations()
    generator = torch.Generator(device=env.device).manual_seed(seed)
    num_envs = len(cases) * replicas
    frictions = torch.tensor(
        [friction for friction, _ in cases], device=env.device
    ).repeat_interleave(replicas)
    speeds = torch.tensor(
        [speed for _, speed in cases], device=env.device
    ).repeat_interleave(replicas)
    if perturb.runup_range is None:
        runups = torch.full((num_envs,), float(runup_s), device=env.device)
    else:
        low, high = perturb.runup_range
        runups = low + (high - low) * torch.rand(
            num_envs, device=env.device, generator=generator
        )
    # Snap run-ups to control steps so the trace's slide start is exact.
    runups = torch.round(runups / env.dt) * env.dt
    env._backend.set_contact_friction(
        torch.arange(num_envs, device=env.device), frictions
    )
    env.reset()

    observes_phase = "slide_phase" in runner.actor_cfg["obs"]
    observes_friction = "contact_friction" in runner.actor_cfg["obs"]
    if perturb.friction_error and observes_friction:
        error = perturb.friction_error
        told = frictions * (
            1.0
            + error
            * (2.0 * torch.rand(num_envs, device=env.device, generator=generator) - 1)
        )
        # Replace the live view: the policy reads this, physics keeps mu.
        env.contact_friction = told.unsqueeze(1)
    if perturb.noise:
        noise_scale = runner.get_noise(runner.actor_cfg["obs"], perturb.noise)

    feet = env.feet_indices
    joints = env.actuated_dof_indices
    alive = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    pushed = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    push_time = runups + perturb.push_delay
    keys = (
        "alive",
        "position",
        "speed",
        "forward_speed",
        "command",
        "foot_load",
        "power_abs",
        "pose_deviation",
    )
    records = {key: [] for key in keys}
    for step in range(round(seconds / env.dt)):
        time_s = torch.tensor(step * env.dt, device=env.device)
        forward = slide_speed_profile(time_s, speeds, frictions, runups)
        running_up = time_s < runups
        env.commands[:] = 0.0
        env.commands[:, 0] = forward
        env.commands[:, 1] = perturb.entry_lateral * running_up
        env.commands[:, 2] = perturb.entry_yaw * running_up
        if observes_phase:
            env.slide_phase[:, 0] = slide_phase_profile(
                time_s, speeds, frictions, runups, hold_s
            ).float()
        if perturb.push_speed:
            due = (~pushed) & (time_s >= push_time)
            if due.any():
                kick = perturb.push_speed * _push_direction(env, generator)
                env.root_states[:, 7:9] += kick * due.unsqueeze(1)
                env._backend.set_all_root_states()
                pushed |= due
        observation = runner.get_obs(runner.actor_cfg["obs"])
        if perturb.noise:
            noise = noise_scale * torch.randn(
                observation.shape, device=env.device, generator=generator
            )
            observation = observation + noise
        actions = runner.alg.actor.act_inference(observation)
        runner.set_actions(runner.actor_cfg["actions"], actions)
        env.step()
        alive &= ~env.terminated
        values = {
            "alive": alive,
            "position": env.root_states[:, :2],
            "speed": env.root_states[:, 7:9].norm(dim=1),
            "forward_speed": env.base_lin_vel[:, 0],
            "command": forward,
            "foot_load": env.contact_forces[:, feet, 2],
            "power_abs": (env.torques * env.dof_vel[:, joints]).abs().sum(dim=1),
            "pose_deviation": (env.dof_pos - env.default_dof_pos)[:, joints]
            .square()
            .mean(dim=1)
            .sqrt(),
        }
        for key, value in values.items():
            records[key].append(value.clone())
    trace = {key: torch.stack(value).cpu() for key, value in records.items()}
    if not all(torch.isfinite(value.float()).all() for value in trace.values()):
        raise RuntimeError("Nonfinite evaluation trajectory")

    # Commands are set before step k, which the trace records at k.
    starts = torch.round(runups / env.dt).long().cpu()
    results = []
    for i, (friction, speed) in enumerate(cases):
        selection = slice(i * replicas, (i + 1) * replicas)
        result = summarize_slide(
            {key: value[:, selection] for key, value in trace.items()},
            dt=env.dt,
            slide_start_step=starts[selection].tolist(),
            friction=friction,
        )
        results.append({"runup_command_m_s": speed} | result)
    return results


def trial_seconds(frictions, speeds, runup_s, perturb=None):
    """Simulated seconds that cover the latest run-up and slowest stop."""
    perturb = perturb or SlidePerturbations()
    latest = max(perturb.runup_range) if perturb.runup_range else runup_s
    longest_slide = max(speeds) / (min(frictions) * GRAVITY_M_S2)
    return latest + longest_slide + SETTLE_S


def markdown_table(results) -> str:
    def fmt(value, spec):
        return "–" if value is None else format(value, spec)

    lines = [
        "| μ | v0 cmd | fell | stopped | entry v | distance (pred.) | dist ratio "
        "| decel ratio | all 4 feet | least-used foot | steps/s (chatter/s) "
        "| energy (J/m) | pose dev (rad) |",
        "|---" * 13 + "|",
    ]
    for r in results:
        distance = (
            f"{fmt(r['slide_distance_m'], '.2f')} "
            f"({fmt(r['predicted_distance_m'], '.2f')})"
        )
        lines.append(
            f"| {r['friction']:g} | {r['runup_command_m_s']:g} "
            f"| {r['fall_rate']:.0%} | {fmt(r['stopped_rate'], '.0%')} "
            f"| {fmt(r['entry_speed_m_s'], '.2f')} | {distance} "
            f"| {fmt(r['distance_ratio'], '.2f')} | {fmt(r['decel_ratio'], '.2f')} "
            f"| {fmt(r['all_feet_loaded_fraction'], '.0%')} "
            f"| {fmt(r['min_foot_loaded_fraction'], '.0%')} "
            f"| {fmt(r['touchdowns_per_s'], '.1f')} "
            f"({fmt(r['contact_breaks_per_s'], '.0f')}) "
            f"| {fmt(r['slide_energy_per_m_j'], '.0f')} "
            f"| {fmt(r['pose_rmse_rad'], '.2f')} |"
        )
    return "\n".join(lines)


def checkpoint_metadata(path):
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }


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
    parser.add_argument("--frictions", type=float, nargs="+", default=FRICTIONS)
    parser.add_argument("--speeds", type=float, nargs="+", default=RUNUP_SPEEDS)
    parser.add_argument("--runup", type=float, default=4.0)
    parser.add_argument(
        "--hold",
        type=float,
        default=0.5,
        help="seconds the slide-phase flag stays on after the stop (as trained)",
    )
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.frictions) <= 0 or min(args.speeds) <= 0 or args.runup <= 0:
        parser.error("frictions, speeds and run-up time must be positive")
    seconds = trial_seconds(args.frictions, args.speeds, args.runup)
    cases = [(f, s) for f in args.frictions for s in args.speeds]
    env, runner, env_cfg, train_cfg, path = make_slide_env(
        task=args.task,
        load_run=args.load_run,
        original_cfg=args.original_cfg,
        experiment_name=args.experiment_name,
        checkpoint=args.checkpoint,
        device=args.device,
        backend=args.backend,
        seed=args.seed,
        frictions=args.frictions,
        num_envs=len(cases) * args.replicas,
        seconds=seconds,
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
            seed=args.seed,
        )
    finally:
        env._backend.close()
    report = {
        "protocol": PROTOCOL,
        "label": args.label or f"{args.task}/{path.parent.name}",
        "task": args.task,
        **checkpoint_metadata(path),
        "stop_speed_m_s": SLIDE_STOP_SPEED,
        "arguments": vars(args) | {"output": str(args.output)},
        "env_config": class_to_dict(env_cfg),
        "runner_config": class_to_dict(train_cfg),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(markdown_table(results))
    print(f"\nreport: {args.output}")


if __name__ == "__main__":
    main()
