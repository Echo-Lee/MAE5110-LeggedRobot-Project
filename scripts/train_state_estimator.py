"""Train the history state estimator (mu-hat, v-hat) for a frozen Go2 policy.

The frozen policy drives the training distribution (ice/ground friction
redrawn each episode, random commands, run-up-and-slide episodes). Each
control step adds (feature history, true mu, true body velocity) samples;
the estimator is trained online on short rollouts, so nothing large is
stored. After ``--dagger_start`` iterations, a fraction of the envs give the
policy mu-hat instead of the true mu (labels stay true), so the estimator
also sees the states the policy reaches when it acts on its own estimate.

A separate validation rollout (new seed, policy on true mu) reports mu-hat
error by friction bin and by time since reset (how fast the estimate
settles), and v-hat error per axis. With ``--friction_switch_probability``
episodes also switch friction mid-episode (ice <-> ground), so the estimator
learns to re-estimate on a new surface; validation then also reports the
error by time since the switch.

    uv run --frozen scripts/train_state_estimator.py --load_run RUN \\
        --original_cfg --device cuda:0
"""

import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

import gym.envs  # noqa: F401
from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry
from learning.modules.history_estimator import (
    DEFAULT_FEATURES,
    FRICTION_CLIP,
    HistoryStateEstimator,
    ProprioHistory,
    features_from_env,
    targets_from_env,
)

ICE_MAX = 0.2
SETTLE_BUCKETS = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 1e9))


def make_env(args, seed, num_envs):
    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            args.task,
            original_cfg=args.original_cfg,
            experiment_name=args.experiment_name,
            load_run=args.load_run if args.original_cfg else None,
        )
    )
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = seed
    if args.friction_switch_probability > 0:
        # Run snapshots older than the option lack the section.
        switch = getattr(env_cfg, "friction_switch", None) or SimpleNamespace(
            time_range=[2.0, 8.0]
        )
        switch.probability = args.friction_switch_probability
        env_cfg.friction_switch = switch
    train_cfg.runner.device = args.device
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(seed)
    experiment = args.experiment_name or train_cfg.runner.experiment_name
    load_run = -1 if args.load_run == "-1" else args.load_run
    path = Path(
        get_load_path(experiment, load_run=load_run, checkpoint=args.checkpoint)
    )
    env = task_registry.make_env(args.task, env_cfg, args.device, True, "mujoco")
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.load(path, load_optimizer=False)
    runner.switch_to_eval()
    return env, runner, path


class Rollout:
    """Steps a frozen policy and keeps each env's feature history current."""

    def __init__(self, env, runner, features, frames, stride):
        self.env, self.runner, self.features = env, runner, features
        self.true_friction_view = env.contact_friction  # live backend view
        first = features_from_env(env, features)
        self.history = ProprioHistory(
            env.num_envs, first.shape[1], frames, stride, env.device
        )
        self.history.reset(
            torch.ones(env.num_envs, dtype=torch.bool, device=env.device), first
        )

    @torch.no_grad()
    def step(self, estimator=None, dagger_mask=None):
        """One control step.

        Returns (window, targets, time since reset, time since the friction
        last changed: the reset or a mid-episode switch).
        """
        env = self.env
        window = self.history.window()
        targets = targets_from_env(env)
        age = env.episode_length_buf.float() * env.dt
        since_change = getattr(env, "time_since_friction_change", age).clone()
        if estimator is not None and dagger_mask is not None and dagger_mask.any():
            mu_hat, _ = estimator.predict(window)
            told = torch.where(
                dagger_mask, mu_hat.clamp(*FRICTION_CLIP), self.true_friction_view[:, 0]
            )
            env.contact_friction = told.unsqueeze(1)
        else:
            env.contact_friction = self.true_friction_view
        actions = self.runner.get_inference_actions()
        self.runner.set_actions(self.runner.actor_cfg["actions"], actions)
        env.step()
        self.runner.reset_envs()
        current = features_from_env(env, self.features)
        self.history.append(current)
        self.history.reset(env.to_be_reset, current)
        return window, targets, age, since_change


def evaluate(estimator, rollout, steps):
    mu_true, mu_hat, v_true, v_hat, ages, changes = [], [], [], [], [], []
    for _ in range(steps):
        window, targets, age, since_change = rollout.step()
        with torch.no_grad():
            m, v = estimator.predict(window)
        mu_true.append(targets[:, 0])
        mu_hat.append(m)
        v_true.append(targets[:, 1:])
        v_hat.append(v)
        ages.append(age)
        changes.append(since_change)
    mu_true, mu_hat = torch.cat(mu_true), torch.cat(mu_hat)
    v_true, v_hat = torch.cat(v_true), torch.cat(v_hat)
    ages = torch.cat(ages)
    changes = torch.cat(changes)
    # Samples whose friction changed after the reset (a mid-episode switch).
    switched = changes < ages - 1e-6
    err = (mu_hat - mu_true).abs()
    rel = err / mu_true
    report = {"samples": int(mu_true.numel())}

    def mu_stats(sel):
        if not sel.any():
            return None
        return {
            "mae": err[sel].mean().item(),
            "median_rel_err": rel[sel].median().item(),
            "p90_rel_err": rel[sel].quantile(0.9).item(),
            "n": int(sel.sum()),
        }

    ice, ground = mu_true <= ICE_MAX, mu_true > ICE_MAX
    # Settled: at least 1 s on the current surface (since reset or switch).
    settled = changes >= 1.0
    report["mu"] = {
        "ice_settled": mu_stats(ice & settled),
        "ground_settled": mu_stats(ground & settled),
        "ice_by_time_since_reset": {
            f"{lo:g}-{hi:g}s": mu_stats(ice & (ages >= lo) & (ages < hi))
            for lo, hi in SETTLE_BUCKETS
        },
        "ground_by_time_since_reset": {
            f"{lo:g}-{hi:g}s": mu_stats(ground & (ages >= lo) & (ages < hi))
            for lo, hi in SETTLE_BUCKETS
        },
    }
    if switched.any():
        report["mu"] |= {
            f"after_switch_to_{name}_by_time": {
                f"{lo:g}-{hi:g}s": mu_stats(
                    switched & bin_ & (changes >= lo) & (changes < hi)
                )
                for lo, hi in SETTLE_BUCKETS
            }
            for name, bin_ in (("ice", ice), ("ground", ground))
        }
    v_err = (v_hat - v_true)[settled]
    report["v"] = {
        axis: {
            "rmse_m_s": v_err[:, i].square().mean().sqrt().item(),
            "true_std_m_s": v_true[settled][:, i].std().item(),
        }
        for i, axis in enumerate(("x", "y", "z"))
    }
    return report


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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--steps_per_iteration", type=int, default=24)
    parser.add_argument("--gradient_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dagger_start", type=int, default=300)
    parser.add_argument("--dagger_fraction", type=float, default=0.5)
    parser.add_argument("--warmup_iterations", type=int, default=10)
    parser.add_argument("--validation_steps", type=int, default=1500)
    parser.add_argument(
        "--friction_switch_probability",
        type=float,
        default=0.0,
        help="fraction of non-slide episodes that switch friction bin mid-episode",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    env, runner, policy_path = make_env(args, args.seed, args.num_envs)
    features = [f for f in DEFAULT_FEATURES if hasattr(env, f)]
    rollout = Rollout(env, runner, features, args.frames, args.stride)
    feature_dims = [env.get_states([f]).shape[1] for f in features]
    estimator = HistoryStateEstimator(
        features=features,
        feature_dims=feature_dims,
        frames=args.frames,
        stride=args.stride,
    ).to(args.device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.learning_rate)
    dagger_mask = (
        torch.arange(env.num_envs, device=env.device)
        < args.dagger_fraction * env.num_envs
    )

    # Freeze input normalization from warm-up rollouts.
    warm = []
    for _ in range(args.warmup_iterations):
        for _ in range(args.steps_per_iteration):
            warm.append(rollout.step()[0])
    estimator.set_input_normalization(torch.cat(warm))
    del warm

    start = time.time()
    for it in range(args.iterations):
        use_dagger = it >= args.dagger_start
        windows, targets = [], []
        for _ in range(args.steps_per_iteration):
            w, t, *_ = rollout.step(
                estimator if use_dagger else None, dagger_mask if use_dagger else None
            )
            windows.append(w)
            targets.append(t)
        windows, targets = torch.cat(windows), torch.cat(targets)
        estimator.train()
        for chunk in torch.randperm(windows.shape[0], device=env.device).chunk(
            args.gradient_steps
        ):
            loss = estimator.loss(windows[chunk], targets[chunk])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        estimator.eval()
        if it % 50 == 0 or it == args.iterations - 1:
            print(
                f"it {it:4d} loss {loss.item():.4f} dagger {use_dagger} "
                f"{time.time() - start:.0f}s",
                flush=True,
            )
    env._backend.close()

    # Validation on a fresh seed, policy acting on the true mu.
    val_env, val_runner, _ = make_env(args, args.seed + 1000, args.num_envs)
    val_rollout = Rollout(val_env, val_runner, features, args.frames, args.stride)
    for _ in range(200):  # let episodes desynchronize from the reset
        val_rollout.step()
    report = evaluate(estimator, val_rollout, args.validation_steps)
    val_env._backend.close()

    default_dir = "state_estimator" + (
        "_switch" if args.friction_switch_probability > 0 else ""
    )
    out_dir = args.output or policy_path.parent / default_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report |= {
        "policy_checkpoint": str(policy_path.resolve()),
        "arguments": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "features": features,
    }
    estimator.save(
        out_dir / "estimator.pt", extra={"policy_checkpoint": str(policy_path)}
    )
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    mu = report["mu"]
    print(f"\nsaved {out_dir / 'estimator.pt'}")
    for name in ("ice_settled", "ground_settled"):
        s = mu[name]
        print(
            f"mu {name}: MAE {s['mae']:.3f}, "
            f"median rel err {s['median_rel_err']:.0%}, p90 {s['p90_rel_err']:.0%}"
        )
    for bucket, s in mu["ice_by_time_since_reset"].items():
        if s:
            print(
                f"  ice, {bucket} after reset: median rel err {s['median_rel_err']:.0%}"
            )
    for key in ("after_switch_to_ice_by_time", "after_switch_to_ground_by_time"):
        for bucket, s in mu.get(key, {}).items():
            if s:
                err = s["median_rel_err"]
                print(f"  {key[13:]} {bucket}: median rel err {err:.0%}")
    for axis, s in report["v"].items():
        print(
            f"v_{axis}: RMSE {s['rmse_m_s']:.3f} m/s (true std {s['true_std_m_s']:.2f})"
        )


if __name__ == "__main__":
    main()
