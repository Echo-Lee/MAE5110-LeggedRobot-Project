"""Pretrain the go2osc state estimator on a frozen push-glide policy.

The policy only walks (deterministic actions, no PPO update); the estimator
inside go2osc_push_glide_est trains on the true friction and body velocity
of the states that policy visits.

    uv run scripts/pretrain_go2osc_estimator.py --device cuda:0 \
        --policy-dir logs/go2osc_posture1800/go2osc_push_glide
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gym.envs  # noqa: E402, F401
from gym.utils.helpers import set_seed  # noqa: E402
from gym.envs.go2.go2osc_config import Go2OscPushGlideRunnerCfg  # noqa: E402
from gym.utils.task_registry import task_registry  # noqa: E402
from learning.modules.input_widening import widen_actor_state  # noqa: E402

TASK = "go2osc_push_glide_est"
# Histories right after a reset hold copies of the reset frame; skip them.
SETTLE_S = 0.5


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=ROOT / "logs/go2osc_posture1800/go2osc_push_glide",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--eval_steps", type=int, default=800)
    parser.add_argument("--samples_per_step", type=int, default=1024)
    parser.add_argument("--update_interval", type=int, default=4)
    parser.add_argument("--gradient_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_sensor_noise",
        action="store_true",
        help="Train and evaluate on clean sensors (stages 2-4 did)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Exact estimator inputs, replacing the config default",
    )
    parser.add_argument(
        "--extra_features",
        nargs="*",
        default=[],
        help="Env attributes appended to the estimator inputs (e.g. torques)",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_policy(args):
    metadata = json.loads((args.policy_dir / "policy.json").read_text())
    checkpoint = args.policy_dir / metadata["checkpoint"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("Checkpoint SHA256 does not match policy.json")
    if metadata["task"] != "go2osc_push_glide":
        raise ValueError(f"expected a go2osc_push_glide policy, got {metadata['task']}")
    return checkpoint, metadata


def make(args, checkpoint):
    cfg, train_cfg = task_registry.get_cfgs(TASK)
    cfg.env.num_envs = args.num_envs
    cfg.seed = train_cfg.seed = args.seed
    settings = cfg.state_estimator
    settings.init_path = None
    base = args.features if args.features else settings.features
    settings.features = list(base) + list(args.extra_features)
    settings.sensor_noise.enabled = not args.no_sensor_noise
    settings.samples_per_step = args.samples_per_step
    settings.update_interval = args.update_interval
    settings.gradient_steps = args.gradient_steps
    settings.learning_rate = args.learning_rate
    train_cfg.runner.device = args.device
    # The frozen walker is the original policy, exactly: its own inputs
    # (foot_traction_obs included) plus zero-weight estimate inputs.
    train_cfg.actor.obs = list(Go2OscPushGlideRunnerCfg.actor.obs) + [
        name
        for name in train_cfg.actor.obs
        if name not in Go2OscPushGlideRunnerCfg.actor.obs
    ]
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(cfg, train_cfg)
    set_seed(args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # untrained-estimator notice
        env = task_registry.make_env(
            TASK, cfg, device=args.device, headless=True, backend="mujoco"
        )
    runner = task_registry.make_alg_runner(env, train_cfg)
    load_widened(env, runner, checkpoint)
    runner.switch_to_eval()  # also turns estimator learning off
    return env, runner


def load_widened(env, runner, checkpoint):
    """Load a go2osc_push_glide policy into the runner's actor.

    Estimate inputs get zero weights and inputs the runner's actor lacks are
    dropped (widen_actor_state). Pretraining keeps every original input, so
    there the policy walks exactly as the original; only the estimator learns.
    """
    saved = torch.load(checkpoint, map_location=runner.device, weights_only=True)
    names = list(runner.actor_cfg["obs"])
    old_names = list(Go2OscPushGlideRunnerCfg.actor.obs)
    dims = {n: env.get_states([n]).shape[1] for n in set(names) | set(old_names)}
    actor = saved["actor_state_dict"]
    if names != old_names:
        actor = widen_actor_state(
            actor,
            [(n, dims[n]) for n in old_names],
            [(n, dims[n]) for n in names],
            drop=[n for n in old_names if n not in names],
        )
    runner.alg.actor.load_state_dict(actor)
    runner.alg.critic.load_state_dict(saved["critic_state_dict"])
    runner.it = int(saved["iter"])


def act(env, runner, record=None):
    """One policy step, then reset finished envs as PPO's rollout does.

    env.step() never resets: falls and timeouts are reset by the runner,
    which also resamples friction and restarts the estimator's history and
    accelerometer bias. ``record`` runs between the step and that reset, so
    terminal states are measured. Returns the (timed_out, terminated) masks.
    """
    runner.set_actions(
        runner.actor_cfg["actions"],
        runner.get_inference_actions(),
        runner.actor_cfg["disable_actions"],
    )
    env.step()
    timed_out, terminated = env.timed_out.clone(), env.terminated.clone()
    if record is not None:
        record()
    runner.reset_envs()
    return timed_out, terminated


def set_normalization(env, runner, steps=200, start=50):
    """Freeze the estimator's input mean/std from walking data, not the spawn."""
    windows = []
    for step in range(steps):
        act(env, runner)
        if step >= start:
            picked = torch.randint(env.num_envs, (256,), device=env.device)
            windows.append(env.estimator.history.window()[picked])
    env.estimator.model.set_input_normalization(torch.cat(windows))
    env.estimator.unnormalized_inputs = None


def train(env, runner, steps):
    env.set_learning(True)
    losses, start = [], time.time()
    for step in range(1, steps + 1):
        act(env, runner)
        if env.estimator.last_loss is not None:
            losses.append(env.estimator.last_loss)
        if step % 250 == 0:
            recent = np.mean(losses[-50:]) if losses else float("nan")
            print(
                f"[pretrain] step {step}/{steps}  updates {env.estimator.updates}  "
                f"loss {recent:.4f}  {time.time() - start:.0f} s",
                flush=True,
            )
    env.set_learning(False)
    return losses


def evaluate(env, runner, steps):
    """Fresh episodes, learning off: per-step truth and estimates."""
    env.reset()
    keys = ("mu", "mu_hat", "v", "v_hat", "t_reset", "t_switch")
    rows = {k: [] for k in keys + ("timed_out", "terminated")}

    def record():  # after the step, before finished envs are reset
        rows["mu"].append(env._backend.contact_friction.clone())
        rows["mu_hat"].append(env.estimator.mu_hat.clone())
        rows["v"].append(env.base_lin_vel.clone())
        rows["v_hat"].append(env.estimator.v_hat.clone())
        rows["t_reset"].append(env.episode_length_buf * env.dt)
        rows["t_switch"].append(env.time_since_friction_change.clone())

    for _ in range(steps):
        timed_out, terminated = act(env, runner, record)
        rows["timed_out"].append(timed_out)
        rows["terminated"].append(terminated)
    return {k: torch.stack(v).cpu().numpy() for k, v in rows.items()}


def metrics(trace):
    keep = trace["t_reset"] >= SETTLE_S
    mu, mu_hat = trace["mu"][keep], trace["mu_hat"][keep]
    err = np.abs(mu_hat - mu)
    after_switch = trace["t_switch"][keep]
    v_err = trace["v_hat"][keep] - trace["v"][keep]
    out = {
        "samples": int(keep.sum()),
        "mu_mae": float(err.mean()),
        "mu_mae_predict_mean_baseline": float(np.abs(mu - mu.mean()).mean()),
        "mu_mae_by_true_mu": {},
        "mu_mae_by_time_since_switch_s": {},
        "v_rmse_xyz_m_s": np.sqrt((v_err**2).mean(axis=0)).tolist(),
        "v_rmse_predict_mean_baseline_xyz_m_s": trace["v"][keep].std(axis=0).tolist(),
        # Absent from traces saved before evaluation reset finished envs.
        "episodes_timed_out": (
            int(trace["timed_out"].sum()) if "timed_out" in trace else None
        ),
        "episodes_terminated": (
            int(trace["terminated"].sum()) if "terminated" in trace else None
        ),
    }
    for low, high in ((0.1, 0.3), (0.3, 0.6), (0.6, 1.0)):
        band = (mu >= low) & (mu <= high)
        if band.any():
            out["mu_mae_by_true_mu"][f"{low}-{high}"] = float(err[band].mean())
    for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99.0)):
        band = (after_switch >= low) & (after_switch < high)
        if band.any():
            out["mu_mae_by_time_since_switch_s"][f"{low}-{high}"] = float(
                err[band].mean()
            )
    return out


def git_state():
    """Commit and whether tracked or new files differ from it."""

    def git(*command):
        return subprocess.run(
            ["git", *command], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--short"))}


def snapshot_code(output):
    """Uncommitted code used by this run: tracked diff + new Python files."""

    def git(*command):
        return subprocess.run(
            ["git", *command], cwd=ROOT, capture_output=True, text=True
        ).stdout

    (output / "code.patch").write_text(git("diff", "HEAD"))
    for name in git("ls-files", "--others", "--exclude-standard").split():
        if name.endswith(".py"):
            target = output / "untracked" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((ROOT / name).read_text())


def main():
    args = arguments()
    checkpoint, metadata = load_policy(args)
    stamp = datetime.now().strftime("%b%d_%H-%M-%S")
    output = args.output or ROOT / f"logs/go2osc_push_glide_est/pretrain_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    snapshot_code(output)
    env, runner = make(args, checkpoint)
    with torch.no_grad():
        set_normalization(env, runner)
        losses = train(env, runner, args.train_steps)
        trace = evaluate(env, runner, args.eval_steps)
    result = metrics(trace)
    result.update(
        code=git_state(),
        policy=str(args.policy_dir),
        policy_sha256=metadata["sha256"],
        features=env.estimator_features,
        settings=vars(args) | {"policy_dir": str(args.policy_dir), "output": None},
        updates=env.estimator.updates,
        final_train_loss=float(np.mean(losses[-50:])) if losses else None,
    )
    env.estimator.model.save(output / "estimator.pt", extra=result)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    np.savez_compressed(output / "eval_trace.npz", **trace)
    np.save(output / "train_loss.npy", np.asarray(losses))
    print(json.dumps({k: result[k] for k in list(result)[:7]}, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
