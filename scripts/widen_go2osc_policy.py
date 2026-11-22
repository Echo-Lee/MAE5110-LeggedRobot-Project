"""Turn a go2osc_push_glide policy into a go2osc_push_glide_est starting point.

The actor gains the estimate inputs (mu-hat, v-hat) with zero weights and
loses the privileged foot_traction_obs (equivalent to feeding its running
mean, about 0); the estimates' input normalization is measured from a short
rollout. The pretrained estimator rides in the same checkpoint, so
fine-tuning is a plain resume:

    uv run scripts/widen_go2osc_policy.py \
        --policy-dir logs/go2osc_posture1800/go2osc_push_glide \
        --estimator \
        logs/go2osc_push_glide_est/pretrain_posture1800_noise_jacobian_s1/estimator.pt
    uv run scripts/train.py --task go2osc_push_glide_est --resume \
        --load_run widened_posture1800 --max_iterations 500 --device cuda:0 --headless

Writes logs/<experiment>/<run_name>/model_<iter>.pt and widen.json. The
stage 5 comparisons start the same way (same policy, estimator and run name):

    uv run scripts/widen_go2osc_policy.py --task go2osc_push_glide_blind
    uv run scripts/widen_go2osc_policy.py --task go2osc_push_glide_oracle
    uv run scripts/train.py --task go2osc_push_glide_blind --resume \
        --load_run widened_posture1800 --max_iterations 500 --device cuda:0 --headless

Blind only drops foot_traction_obs; the oracle gets the true mu and v
(contact_friction_obs, base_lin_vel) with zero weights instead of the
estimates.
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
from gym.envs.go2.go2osc_config import Go2OscPushGlideRunnerCfg  # noqa: E402
from gym.utils.helpers import class_to_dict, set_seed  # noqa: E402
from gym.utils.task_registry import task_registry  # noqa: E402
from learning.modules import Actor  # noqa: E402
from learning.modules.input_widening import widen_actor_state  # noqa: E402

TASKS = (
    "go2osc_push_glide_est",
    "go2osc_push_glide_blind",
    "go2osc_push_glide_oracle",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=ROOT / "logs/go2osc_posture1800/go2osc_push_glide",
    )
    parser.add_argument(
        "--estimator",
        type=Path,
        default=ROOT
        / "logs/go2osc_push_glide_est"
        / "pretrain_posture1800_noise_jacobian_s1/estimator.pt",
    )
    parser.add_argument("--run_name", default="widened_posture1800")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--stats_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_policy(policy_dir):
    metadata = json.loads((policy_dir / "policy.json").read_text())
    path = policy_dir / metadata["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Checkpoint SHA256 does not match policy.json")
    if metadata["task"] != "go2osc_push_glide":
        raise ValueError(f"expected a go2osc_push_glide policy, got {metadata['task']}")
    return torch.load(path, map_location="cpu", weights_only=True), metadata


def make(args, output):
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs
    cfg.seed = train_cfg.seed = args.seed
    cfg.state_estimator.init_path = str(args.estimator.resolve())
    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = str(output)
    task_registry.convert_frequencies_to_params(cfg, train_cfg)
    set_seed(args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = task_registry.make_env(args.task, cfg, device="cpu", headless=True)
    runner = task_registry.make_alg_runner(env, train_cfg)
    return env, runner


def main():
    args = arguments()
    old, metadata = load_policy(args.policy_dir)
    _, train_cfg = task_registry.get_cfgs(args.task)
    output = ROOT / "logs" / train_cfg.runner.experiment_name / args.run_name
    if output.exists() and any(output.glob("model_*.pt")):
        raise FileExistsError(f"{output} already holds a checkpoint")
    env, runner = make(args, output)

    old_names = list(Go2OscPushGlideRunnerCfg.actor.obs)
    new_names = list(runner.actor_cfg["obs"])
    dims = {n: env.get_states([n]).shape[1] for n in set(old_names) | set(new_names)}
    old_inputs = [(name, dims[name]) for name in old_names]
    new_inputs = [(name, dims[name]) for name in new_names]
    added = [name for name in new_names if name not in old_names]
    dropped = [name for name in old_names if name not in new_names]

    # Zero-weight estimates and dropped columns: the rollout below runs the
    # original policy fed the dropped inputs' running means.
    widened = widen_actor_state(
        old["actor_state_dict"], old_inputs, new_inputs, drop=dropped
    )
    runner.alg.actor.load_state_dict(widened)
    runner.alg.critic.load_state_dict(old["critic_state_dict"])
    runner.alg.critic_optimizer.load_state_dict(old["critic_optimizer_state_dict"])
    runner.switch_to_eval()  # no normalizer updates, estimator frozen

    reference = Actor(
        sum(dim for _, dim in old_inputs),
        runner.alg.actor.num_actions,
        **class_to_dict(Go2OscPushGlideRunnerCfg.actor),
    ).eval()
    reference.load_state_dict(old["actor_state_dict"])

    rms_mean = old["actor_state_dict"]["obs_rms.running_mean"].float()
    columns, start = {}, 0
    for name, dim in old_inputs:
        columns[name] = slice(start, start + dim)
        start += dim
    samples, max_diff, max_drop_effect = {name: [] for name in added}, 0.0, 0.0
    with torch.no_grad():
        for step in range(args.stats_steps):
            obs = runner.get_obs(runner.actor_cfg["obs"])
            actions = runner.alg.actor.act_inference(obs)
            original_obs = runner.get_obs(old_names)
            mean_fed = original_obs.clone()
            for name in dropped:  # dropping == feeding the running mean
                mean_fed[:, columns[name]] = rms_mean[columns[name]]
            expected = reference.act_inference(mean_fed)
            max_diff = max(max_diff, (actions - expected).abs().max().item())
            effect = actions - reference.act_inference(original_obs)
            max_drop_effect = max(max_drop_effect, effect.abs().mean().item())
            runner.set_actions(
                runner.actor_cfg["actions"],
                actions,
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            if step >= 50:  # skip the spawn
                for name in added:
                    samples[name].append(env.get_states([name]))
    stats = {}
    for name in added:
        data = torch.cat(samples[name])
        stats[name] = (data.mean(dim=0), data.var(dim=0).clamp_min(1e-6))
    widened = widen_actor_state(
        old["actor_state_dict"], old_inputs, new_inputs, new_stats=stats, drop=dropped
    )
    runner.alg.actor.load_state_dict(widened)
    runner.it = int(old["iter"])
    runner.save()  # fresh actor optimizer, old critic optimizer, estimator

    info = {
        "task": args.task,
        "source_policy": str(args.policy_dir),
        "source_sha256": metadata["sha256"],
        "estimator": str(args.estimator),
        "iteration": runner.it,
        "old_inputs": old_inputs,
        "new_inputs": new_inputs,
        "added_input_stats": {
            name: {"mean": m.tolist(), "var": v.tolist()}
            for name, (m, v) in stats.items()
        },
        "dropped_inputs": dropped,
        "max_action_difference_vs_original_fed_dropped_means": max_diff,
        "max_mean_action_change_from_dropping": max_drop_effect,
    }
    (output / "widen.json").write_text(json.dumps(info, indent=2))
    env._backend.close()
    print(
        json.dumps({k: info[k] for k in ("iteration", "added_input_stats")}, indent=1)
    )
    print(
        f"max |action - original fed dropped means| over {args.stats_steps} steps: "
        f"{max_diff:.2e}; dropped {dropped}, largest mean action change "
        f"{max_drop_effect:.3f}"
    )
    print(f"saved {output / f'model_{runner.it}.pt'}")


if __name__ == "__main__":
    main()
