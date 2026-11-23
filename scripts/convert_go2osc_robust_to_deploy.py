"""Turn a go2osc_push_glide_robust policy into a go2osc_push_glide_deploy start.

The deploy task's actor reads the estimates where the robust actor read
their simulated stand-ins, and has no foot_traction_obs:

    friction_estimate_obs  ->  contact_friction_hat   (same scale)
    base_lin_vel           ->  base_lin_vel_hat       (same scale)
    commands               ->  governed_commands      (same scale)
    foot_traction_obs      ->  dropped (as feeding its running mean)

The critic's inputs are the same in both tasks and are copied. The result
is a deploy-task checkpoint that walks while the deploy task still feeds
the stand-ins (before state_estimator.handover.start), so the estimator can
learn on a gait that exists and the hand-over then moves the policy onto
mu-hat and v-hat:

    uv run scripts/convert_go2osc_robust_to_deploy.py \
        --checkpoint logs/go2osc_push_glide_robust_carve_4000/<run>/model_4000.pt
    uv run scripts/train.py --task go2osc_push_glide_deploy --resume \
        --load_run from_robust --max_iterations 2000 --device cuda:0 --headless

Writes logs/<experiment_name>/<run_name>/model_<iteration>.pt and
convert.json. ``--iteration`` is the training iteration the checkpoint is
labelled with: the hand-over schedule counts from it, so 1000 leaves 500
iterations for the estimator before a hand-over that starts at 1500.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gym.envs  # noqa: E402, F401
from gym.utils.helpers import set_seed  # noqa: E402
from gym.utils.task_registry import task_registry  # noqa: E402

SOURCE_TASK = "go2osc_push_glide_robust"
TASK = "go2osc_push_glide_deploy"
RENAMED = {
    "friction_estimate_obs": "contact_friction_hat",
    "base_lin_vel": "base_lin_vel_hat",
    "commands": "governed_commands",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run_name", default="from_robust")
    parser.add_argument(
        "--experiment_name",
        help="logs/<experiment_name>/ to write into; pass the same name to "
        "scripts/train.py (default: the task's own)",
    )
    parser.add_argument("--iteration", type=int, default=1000)
    return parser.parse_args()


def make(output):
    cfg, train_cfg = task_registry.get_cfgs(TASK)
    cfg.env.num_envs = 2
    cfg.seed = train_cfg.seed = 0
    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = str(output)
    task_registry.convert_frequencies_to_params(cfg, train_cfg)
    set_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = task_registry.make_env(TASK, cfg, device="cpu", headless=True)
    return env, task_registry.make_alg_runner(env, train_cfg)


def input_columns(env, source_names, target_names):
    """Source column of every target column, and the dropped source inputs."""
    scales = env.scales
    for old, new in RENAMED.items():
        same = torch.equal(
            torch.as_tensor(scales[old], dtype=torch.float).flatten(),
            torch.as_tensor(scales[new], dtype=torch.float).flatten(),
        )
        if not same:
            raise ValueError(f"{old} and {new} are scaled differently")
    start, where = 0, {}
    for name in source_names:
        size = env.get_states([name]).shape[1]
        where[RENAMED.get(name, name)] = (start, size)
        start += size
    columns = []
    for name in target_names:
        if name not in where:
            raise ValueError(f"the source policy has no input for {name}")
        begin, size = where[name]
        if env.get_states([name]).shape[1] != size:
            raise ValueError(f"{name} has a different size in the two tasks")
        columns.extend(range(begin, begin + size))
    kept = {RENAMED.get(name, name) for name in source_names} & set(target_names)
    dropped = [name for name in source_names if RENAMED.get(name, name) not in kept]
    return torch.tensor(columns), start, dropped


def narrowed(value, columns, width):
    """Keep the target's columns of any tensor laid out over the source inputs."""
    if torch.is_tensor(value) and value.dim() >= 1 and value.shape[-1] == width:
        return value.index_select(value.dim() - 1, columns)
    return value


def convert(source, columns, width):
    actor = {
        k: narrowed(v, columns, width) for k, v in source["actor_state_dict"].items()
    }
    optimizer = dict(source["optimizer_state_dict"])
    optimizer["state"] = {
        index: {k: narrowed(v, columns, width) for k, v in state.items()}
        for index, state in optimizer["state"].items()
    }
    return actor, optimizer


def main():
    args = arguments()
    _, train_cfg = task_registry.get_cfgs(TASK)
    experiment = args.experiment_name or train_cfg.runner.experiment_name
    output = ROOT / "logs" / experiment / args.run_name
    if output.exists() and any(output.glob("model_*.pt")):
        raise FileExistsError(f"{output} already holds a checkpoint")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _, source_cfg = task_registry.get_cfgs(SOURCE_TASK)
    source_names = list(source_cfg.actor.obs)

    env, runner = make(output)
    try:
        target_names = list(runner.actor_cfg["obs"])
        if list(source_cfg.critic.obs) != list(runner.critic_cfg["obs"]):
            raise ValueError("the two tasks' critics read different inputs")
        columns, width, dropped = input_columns(env, source_names, target_names)
        found = source["actor_state_dict"]["obs_rms.running_mean"].shape[0]
        if found != width:
            raise ValueError(
                f"{args.checkpoint} has {found} actor inputs; {SOURCE_TASK} "
                f"now has {width}"
            )
        actor, optimizer = convert(source, columns, width)
        # Loading checks every shape against the deploy task's networks.
        runner.alg.actor.load_state_dict(actor)
        runner.alg.critic.load_state_dict(source["critic_state_dict"])
        runner.alg.optimizer.load_state_dict(optimizer)
        runner.alg.critic_optimizer.load_state_dict(
            source["critic_optimizer_state_dict"]
        )
    finally:
        env._backend.close()

    output.mkdir(parents=True, exist_ok=True)
    path = output / f"model_{args.iteration}.pt"
    torch.save(
        {
            "actor_state_dict": runner.alg.actor.state_dict(),
            "critic_state_dict": runner.alg.critic.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "critic_optimizer_state_dict": runner.alg.critic_optimizer.state_dict(),
            "iter": args.iteration,
        },
        path,
    )
    info = {
        "task": TASK,
        "source_task": SOURCE_TASK,
        "source_checkpoint": str(args.checkpoint),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_iteration": int(source["iter"]),
        "iteration": args.iteration,
        "renamed_inputs": RENAMED,
        "dropped_inputs": dropped,
        "actor_inputs": [width, len(columns)],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output / "convert.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
