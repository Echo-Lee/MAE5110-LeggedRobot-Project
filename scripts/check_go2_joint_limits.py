"""Check that a Go2 policy keeps its joints inside the robot's hard stops.

MuJoCo joint limits are soft: a joint can sink past its stop. The real Go2
has hard mechanical stops, so a policy has to stay inside them by itself.
This rolls a checkpoint out on several frictions and commands (walking,
turning, backward, standing and sliding stops) and compares every joint's
measured angle and commanded target with the URDF limits.

Both are checked. A target beyond a limit drives an unloaded joint into its
stop (tau = kp (target - angle)), so a policy for the robot keeps its
targets inside as well.

    uv run scripts/check_go2_joint_limits.py --task go2osc_push_glide_deploy \
        --checkpoint logs/go2osc_push_glide_deploy/<run>/model_4000.pt

Exit code 0 when no joint angle comes within ``--angle_margin`` [rad] of a
hard limit and no commanded target passes the limits less ``--margin``
[rad]; 1 otherwise. ``--angles_only`` leaves the targets out of the result.
``--json`` writes the table.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

import gym.envs  # noqa: F401
from gym.envs.base.domain_randomization import (
    apply_domain_randomization_override,
    set_domain_randomization_range,
)
from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry

# [forward m/s, sideways m/s, turn rad/s]
COMMANDS = (
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.5, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.5),
    (1.0, 0.0, -0.5),
    (0.0, 0.3, 0.0),
    (0.0, -0.3, 0.0),
    (0.0, 0.0, 0.5),
    (1.0, 0.3, 0.5),
)
# Commands that end in a sliding stop: after stop_time the command becomes
# zero speed with this turn rate.
STOP_TURNS = (0.0, 0.5, -0.5)
JOINTS = ("hip", "thigh", "calf")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frictions", default="0.05,0.1,0.3,1.0")
    parser.add_argument("--duration", type=float, default=8.0, help="s per rollout")
    parser.add_argument("--stop_time", type=float, default=5.0)
    parser.add_argument("--stop_speed", type=float, default=2.0)
    parser.add_argument(
        "--angle_margin", type=float, default=0.0, help="rad kept from a limit"
    )
    parser.add_argument(
        "--angles_only",
        action="store_true",
        help="report the commanded targets without failing on them",
    )
    parser.add_argument("--margin", type=float, default=0.0, help="rad for targets")
    parser.add_argument(
        "--variation",
        action="store_true",
        help="keep sensor noise, actuation delay, mass and gain variation",
    )
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def make_env(task, frictions, commands, duration, variation):
    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            task, original_cfg=False, experiment_name=None, load_run=None
        )
    )
    env_cfg.env.num_envs = len(frictions) * len(commands)
    env_cfg.env.episode_length_s = duration + 2.0
    env_cfg.commands.resampling_time = duration + 2.0
    env_cfg.push_robots.toggle = False
    if hasattr(env_cfg.commands, "sliding_stop_probability"):
        env_cfg.commands.sliding_stop_probability = 0.0
    if getattr(env_cfg, "fast_slides", None) is not None:
        env_cfg.fast_slides.probability = 0.0
    if getattr(env_cfg, "friction_changes", None) is not None:
        env_cfg.friction_changes.rate = 0.0
        env_cfg.friction_changes.split_probability = 0.0
    if getattr(env_cfg, "friction_switch", None) is not None:
        env_cfg.friction_switch.probability = 0.0
    if not variation:
        apply_domain_randomization_override(env_cfg, "off")
    set_domain_randomization_range(env_cfg, "contact_friction_range", [0.05, 1.0])
    root = env_cfg.init_state.root_vel_range
    env_cfg.init_state.root_vel_range = [[0.0, 0.0] for _ in root]
    env_cfg.seed = 0
    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(0)
    env = task_registry.make_env(task, env_cfg, "cpu", True, "mujoco")
    return env, train_cfg


def set_friction(env, friction):
    """One friction per robot, under every foot, for the whole rollout."""
    rows = torch.arange(env.num_envs)
    env._backend.set_contact_friction(rows, friction)
    if hasattr(env, "foot_friction_obs"):
        table = friction.unsqueeze(1).repeat(1, 4)
        env._backend.set_contact_friction_by_body(rows, env.feet_indices, table)
        env.foot_friction_obs.copy_(table)
    everyone = torch.ones(env.num_envs, dtype=torch.bool)
    for refresh in ("_refresh_friction_observation", "_refresh_friction_estimate"):
        if hasattr(env, refresh):
            getattr(env, refresh)(everyone)


def main():
    arguments = parse_arguments()
    frictions = [float(value) for value in arguments.frictions.split(",")]
    stops = [(arguments.stop_speed, 0.0, 0.0, turn) for turn in STOP_TURNS]
    plans = [command + (None,) for command in COMMANDS] + stops
    env, train_cfg = make_env(
        arguments.task, frictions, plans, arguments.duration, arguments.variation
    )
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        runner.load(arguments.checkpoint, load_optimizer=False)
        runner.switch_to_eval()
        count = len(plans)
        friction = torch.tensor(frictions).repeat_interleave(count)
        set_friction(env, friction)
        before = torch.tensor([plan[:3] for plan in plans]).repeat(len(frictions), 1)
        after = before.clone()
        for index, plan in enumerate(plans):
            if plan[3] is not None:
                after[index::count] = torch.tensor([0.0, 0.0, plan[3]])
        actuated = env.actuated_dof_indices
        limits = env.dof_pos_limits.index_select(0, actuated)
        default = env.default_dof_pos.index_select(1, actuated)
        steps = round(arguments.duration / env.dt)
        stop_step = round(arguments.stop_time / env.dt)
        position_low = torch.full((env.num_envs, len(actuated)), float("inf"))
        position_high = -position_low
        target_low, target_high = position_low.clone(), position_high.clone()
        past = torch.zeros(env.num_envs, len(actuated))
        pressed = torch.zeros(env.num_envs, len(actuated))
        alive = torch.ones(env.num_envs, dtype=torch.bool)
        with torch.inference_mode():
            for step in range(steps):
                env.commands[:] = before if step < stop_step else after
                actions = runner.get_inference_actions()
                runner.set_actions(runner.actor_cfg["actions"], actions)
                env.step()
                alive &= ~(env.terminated | env.timed_out)
                position = env.dof_pos.index_select(1, actuated)
                target = default + env._joint_position_target()
                keep = alive.unsqueeze(1)
                position_low = torch.where(
                    keep, torch.minimum(position_low, position), position_low
                )
                position_high = torch.where(
                    keep, torch.maximum(position_high, position), position_high
                )
                target_low = torch.where(
                    keep, torch.minimum(target_low, target), target_low
                )
                target_high = torch.where(
                    keep, torch.maximum(target_high, target), target_high
                )
                outside = (position < limits[:, 0]) | (position > limits[:, 1])
                past += (outside & keep).float()
                beyond = (target < limits[:, 0]) | (target > limits[:, 1])
                pressed += (beyond & keep).float()
    finally:
        env._backend.close()

    names = [env.robot_layout.dof_names[index] for index in actuated.tolist()]
    rows, failed = [], False
    print(f"{arguments.task}  {arguments.checkpoint}")
    print(f"upright {int(alive.sum())}/{env.num_envs} rollouts")
    print(
        f"{'joint':<16}{'limits [rad]':<20}{'angle range':<20}"
        f"{'target range':<20}{'angle past stop':<17}{'target past stop':<16}"
    )
    for index, name in enumerate(names):
        low, high = limits[index].tolist()
        angle = (
            position_low[:, index].min().item(),
            position_high[:, index].max().item(),
        )
        target = (target_low[:, index].min().item(), target_high[:, index].max().item())
        share = (past[:, index].sum() / (steps * env.num_envs)).item()
        push = (pressed[:, index].sum() / (steps * env.num_envs)).item()
        angle_bad = (
            angle[0] < low + arguments.angle_margin
            or angle[1] > high - arguments.angle_margin
        )
        target_bad = not arguments.angles_only and (
            target[0] < low + arguments.margin or target[1] > high - arguments.margin
        )
        failed |= angle_bad or target_bad
        flag = " ANGLE" if angle_bad else ""
        flag += " TARGET" if target_bad else ""
        print(
            f"{name:<16}{f'{low:+.3f} {high:+.3f}':<20}"
            f"{f'{angle[0]:+.3f} {angle[1]:+.3f}':<20}"
            f"{f'{target[0]:+.3f} {target[1]:+.3f}':<20}"
            f"{share:<17.2%}{push:<10.2%}{flag}"
        )
        rows.append(
            {
                "joint": name,
                "limits": [low, high],
                "angle": list(angle),
                "closest_to_a_limit": min(angle[0] - low, high - angle[1]),
                "target": list(target),
                "time_angle_past_stop": share,
                "time_target_past_stop": push,
            }
        )
    print("RESULT:", "outside the limits" if failed else "inside the limits")
    if arguments.json:
        arguments.json.write_text(
            json.dumps(
                {
                    "task": arguments.task,
                    "checkpoint": str(arguments.checkpoint),
                    "upright": int(alive.sum()),
                    "rollouts": env.num_envs,
                    "inside": not failed,
                    "angle_margin": arguments.angle_margin,
                    "targets_checked": not arguments.angles_only,
                    "joints": rows,
                },
                indent=2,
            )
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
