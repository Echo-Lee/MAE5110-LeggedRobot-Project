"""
Mid-episode friction switch: does the gait (and mu-hat) adapt to a new surface?

Every replica starts at rest on mu_before, follows a fixed command, and at
--switch_time its friction jumps to mu_after (stepping onto a patch). The
report gives, per (transition, command):

    fall rate, and falls after the switch
    velocity tracking before the switch, in the first second after it
    (transient), and after that (adapted)
    with --state_estimator: mu-hat after the switch and its settling time
    (first time after which mu-hat stays within --settle_tolerance of the
    new mu)

Each case is graded with evaluate_go2_friction's bounds for the new surface,
applied to the adapted window. Unchanged-surface controls (0.1 -> 0.1,
0.8 -> 0.8) separate the cost of the switch from the surface itself.

    uv run --frozen scripts/evaluate_friction_switch.py --load_run RUN \\
        --original_cfg --device cuda:0 --output logs/eval/switch.json \\
        [--state_estimator logs/go2_lowfric/RUN/state_estimator/estimator.pt]
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from evaluate_go2_friction import COMMANDS, case_bounds, check_case
from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import class_to_dict, get_load_path, set_seed
from gym.utils.task_registry import task_registry

PROTOCOL = "go2-friction-switch-v1"
TRANSITIONS = (
    (0.1, 0.8),
    (0.05, 1.0),
    (0.8, 0.1),
    (1.0, 0.05),
    (0.1, 0.1),  # controls: no change
    (0.8, 0.8),
)
SWITCH_COMMANDS = ("forward_slow", "forward", "left", "turn_left")
TRANSIENT_S = 1.0
MU_HAT_PROBES_S = (0.1, 0.25, 0.5, 1.0, 2.0)


def _rmse(error, alive):
    """RMSE over alive samples of a [T, R] error trace, or None if none."""
    count = alive.sum()
    if count == 0:
        return None
    return (error.square() * alive).sum().div(count).sqrt().item()


def settle_time(mu_hat, mu_true, alive, tolerance, dt):
    """Seconds after the switch until mu-hat stays within tolerance.

    ``mu_hat`` [T, R] starts at the switch. A replica that is never inside
    the band for the rest of the trial (or falls) has no settling time.
    """
    inside = ((mu_hat - mu_true).abs() <= tolerance * mu_true) | ~alive
    # inside_rest[t] = inside at every step from t to the end.
    inside_rest = inside.flip(0).cummin(0).values.flip(0)
    times = []
    for r in range(mu_hat.shape[1]):
        if not alive[-1, r] or not inside_rest[-1, r]:
            times.append(None)
            continue
        first = torch.nonzero(inside_rest[:, r])[0, 0].item()
        times.append(first * dt)
    return times


@torch.no_grad()
def evaluate(env, runner, cases, args):
    replicas = args.replicas
    num_envs = len(cases) * replicas
    before = torch.tensor(
        [b for (b, _), _ in cases], device=env.device
    ).repeat_interleave(replicas)
    after = torch.tensor(
        [a for (_, a), _ in cases], device=env.device
    ).repeat_interleave(replicas)
    commands = torch.tensor(
        [COMMANDS[name] for _, name in cases], device=env.device
    ).repeat_interleave(replicas, dim=0)
    ids = torch.arange(num_envs, device=env.device)
    env._backend.set_contact_friction(ids, before)
    env.reset()
    env.schedule_friction_switch(ids, args.switch_time, after)
    estimator = None
    if args.state_estimator is not None:
        from learning.modules.history_estimator import EstimatorInTheLoop

        estimator = EstimatorInTheLoop(
            env,
            args.state_estimator,
            env.device,
            feed_policy=not args.estimator_observe_only,
        )

    alive = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    rec = {"alive": [], "lin_err": [], "yaw_err": [], "mu": [], "mu_hat": []}
    steps = round(args.seconds / env.dt)
    for _ in range(steps):
        env.commands[:] = commands
        if estimator is not None:
            estimator.step()
        actions = runner.get_inference_actions()
        runner.set_actions(runner.actor_cfg["actions"], actions)
        env.step()
        alive &= ~env.terminated
        rec["alive"].append(alive.clone())
        rec["lin_err"].append((env.base_lin_vel[:, :2] - commands[:, :2]).norm(dim=1))
        rec["yaw_err"].append(env.base_ang_vel[:, 2] - commands[:, 2])
        rec["mu"].append(env._backend.contact_friction.clone())
        if estimator is not None:
            rec["mu_hat"].append(estimator.mu_hat.clone())
        elif hasattr(env, "contact_friction_hat"):  # the task's own estimator
            rec["mu_hat"].append(env.contact_friction_hat[:, 0].clone())
    trace = {k: torch.stack(v).cpu() for k, v in rec.items() if v}
    # Step k ends at time (k + 1) * dt.
    t = (torch.arange(steps) + 1) * env.dt
    switch = args.switch_time
    windows = {
        "before": (t >= switch - 2.0) & (t < switch),
        "transient": (t >= switch) & (t < switch + TRANSIENT_S),
        "adapted": t >= switch + TRANSIENT_S,
    }
    first_after = int(torch.nonzero(t >= switch)[0, 0])
    alive_at_switch = trace["alive"][first_after - 1]

    results = {}
    for i, ((mu_before, mu_after), name) in enumerate(cases):
        sel = slice(i * replicas, (i + 1) * replicas)
        a = trace["alive"][:, sel].float()
        result = {
            "mu_before": mu_before,
            "mu_after": mu_after,
            "command_name": name,
            "fall_rate": 1.0 - a[-1].mean().item(),
            "fall_after_switch_rate": (
                alive_at_switch[sel].float() - trace["alive"][-1, sel].float()
            )
            .mean()
            .item(),
        }
        for window, mask in windows.items():
            m = a * mask.unsqueeze(1)
            result[f"linear_rmse_{window}_m_s"] = _rmse(trace["lin_err"][:, sel], m)
            result[f"yaw_rmse_{window}_rad_s"] = _rmse(trace["yaw_err"][:, sel], m)
        if "mu_hat" in trace:
            hat = trace["mu_hat"][first_after:, sel]
            mu = trace["mu"][first_after:, sel]
            live = trace["alive"][first_after:, sel]
            result["mu_hat_after_switch"] = {
                f"{p:g}s": hat[min(round(p / env.dt), hat.shape[0] - 1)].median().item()
                for p in MU_HAT_PROBES_S
            }
            times = settle_time(hat, mu, live, args.settle_tolerance, env.dt)
            settled = [x for x in times if x is not None]
            result["mu_hat_settle_s"] = {
                "median": (
                    sorted(settled)[len(settled) // 2]
                    if len(settled) == len(times)
                    else None
                ),
                "max": max(settled) if len(settled) == len(times) else None,
                "unsettled_rate": 1.0 - len(settled) / len(times),
            }
        graded = {
            "fall_rate": result["fall_rate"],
            "linear_rmse_m_s": result["linear_rmse_adapted_m_s"],
            "yaw_rmse_rad_s": result["yaw_rmse_adapted_rad_s"],
        }
        command = torch.tensor(COMMANDS[name])
        result |= check_case(graded, case_bounds(mu_after, command))
        results[f"{mu_before:g}->{mu_after:g}/{name}"] = result
    return results


def print_table(results):
    print(f"{'case':28s} falls  after  lin_before lin_trans lin_adapt  settle")
    for key, r in results.items():

        def fmt(x, spec=".2f"):
            return "  -  " if x is None else format(x, spec)

        settle = r.get("mu_hat_settle_s", {}).get("median")
        status = {True: "ok", False: "FAIL", None: ""}[r["passed"]]
        print(
            f"{key:28s} {r['fall_rate']:.2f}  {r['fall_after_switch_rate']:.2f}"
            f"   {fmt(r['linear_rmse_before_m_s'])}      "
            f"{fmt(r['linear_rmse_transient_m_s'])}      "
            f"{fmt(r['linear_rmse_adapted_m_s'])}     {fmt(settle)}  {status}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="go2_lowfric")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--load_run", default="-1")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--switch_time", type=float, default=4.0)
    parser.add_argument("--seconds", type=float, default=9.0)
    parser.add_argument("--settle_tolerance", type=float, default=0.3)
    parser.add_argument("--state_estimator", type=Path, default=None)
    parser.add_argument(
        "--estimator_observe_only",
        action="store_true",
        help="policy keeps the true mu; mu-hat is only recorded",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.seconds < args.switch_time + TRANSIENT_S + 2.0 or args.switch_time < 2.0:
        parser.error("need >= 2 s before the switch and >= 3 s after it")
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
    cases = [(pair, name) for pair in TRANSITIONS for name in SWITCH_COMMANDS]
    env_cfg.env.num_envs = len(cases) * args.replicas
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.push_robots.toggle = False
    if hasattr(env_cfg.commands, "slide_probability"):
        env_cfg.commands.slide_probability = 0.0
    if hasattr(env_cfg, "friction_switch"):
        env_cfg.friction_switch.probability = 0.0  # switches are scheduled
    env_cfg.init_state.root_vel_range = [[0.0, 0.0]] * 6
    apply_domain_randomization_override(env_cfg, "off")
    frictions = [mu for pair in TRANSITIONS for mu in pair]
    env_cfg.domain_randomization.startup.contact_friction_range = [
        min(frictions),
        max(frictions),
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
    env = task_registry.make_env(args.task, env_cfg, args.device, True, "mujoco")
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        runner.load(path, load_optimizer=False)
        runner.switch_to_eval()
        results = evaluate(env, runner, cases, args)
    finally:
        env._backend.close()
    gated = [r["passed"] for r in results.values() if r["passed"] is not None]
    report = {
        "protocol": PROTOCOL,
        "task": args.task,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "env_config": class_to_dict(env_cfg),
        "cases": results,
        "passed_cases": sum(gated),
        "gated_cases": len(gated),
        "passed": all(gated),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(text + "\n")
    print_table(results)
    print(
        json.dumps(
            {
                "passed_cases": sum(gated),
                "gated_cases": len(gated),
                "report": str(args.output),
            }
        )
    )
    if args.check and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
