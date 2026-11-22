"""Prepare the estimator and deploy checkpoint from an oracle gait checkpoint.

Run in one MoLab runtime after ``go2osc_push_glide_deploy_oracle`` training:

  python scripts/prepare_go2osc_deploy.py pretrain --checkpoint .../model_500.pt
  python scripts/prepare_go2osc_deploy.py bootstrap --checkpoint .../model_500.pt \
      --estimator logs/go2osc_deploy_estimator/estimator.pt --run-name oracle_bootstrap
"""
import argparse
import copy
from pathlib import Path

import torch

import gym.envs  # noqa: F401
from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def build(task, device, num_envs, seed, estimator=None):
    cfg, train = copy.deepcopy(task_registry.get_cfgs(task))
    cfg.env.num_envs = num_envs
    cfg.seed = train.seed = seed
    if estimator is not None:
        cfg.state_estimator.init_path = str(Path(estimator).resolve())
    train.runner.device = device
    train.runner.resume = False
    train.logging.enable_local_saving = False
    train.log_dir = None
    task_registry.convert_frequencies_to_params(cfg, train)
    set_seed(seed)
    env = task_registry.make_env(task, cfg, device, True, "mujoco")
    runner = task_registry.make_alg_runner(env, train)
    return env, runner


def step(env, runner):
    runner.set_actions(runner.actor_cfg["actions"], runner.get_inference_actions(), runner.actor_cfg["disable_actions"])
    env.step()
    runner.reset_envs()


def pretrain(args):
    env, runner = build("go2osc_push_glide_deploy_oracle", args.device, args.num_envs, args.seed)
    try:
        runner.load(args.checkpoint, load_optimizer=False)
        runner.switch_to_eval()
        # Oracle actor stays frozen; only supervised estimator updates run.
        env.cfg.state_estimator.learn = True
        env.set_learning(True)
        for i in range(args.steps):
            step(env, runner)
            if (i + 1) % 200 == 0:
                print(f"[estimator] {i + 1}/{args.steps}, updates={env.estimator.updates}, loss={env.estimator.last_loss}", flush=True)
        env.set_learning(False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        env.estimator.model.save(args.output)
        print(f"wrote {args.output}")
    finally:
        env._backend.close()


def bootstrap(args):
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    env, runner = build("go2osc_push_glide_deploy", "cpu", 1, args.seed, args.estimator)
    try:
        # Oracle and deploy actors share the same 58-column shape; only the
        # semantic source of the four state columns changes.
        runner.alg.actor.load_state_dict(source["actor_state_dict"])
        runner.alg.critic.load_state_dict(source["critic_state_dict"])
        out = Path("logs/go2osc_push_glide_deploy") / args.run_name
        out.mkdir(parents=True, exist_ok=True)
        iteration = int(source["iter"])
        state = {
            "actor_state_dict": runner.alg.actor.state_dict(),
            "critic_state_dict": runner.alg.critic.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "critic_optimizer_state_dict": runner.alg.critic_optimizer.state_dict(),
            "iter": iteration,
            "env_state_dict": env.checkpoint_state(),
        }
        path = out / f"model_{iteration}.pt"
        torch.save(state, path)
        print(f"wrote {path}")
    finally:
        env._backend.close()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("pretrain")
    a.add_argument("--checkpoint", type=Path, required=True)
    a.add_argument("--output", type=Path, default=Path("logs/go2osc_deploy_estimator/estimator.pt"))
    a.add_argument("--device", default="cuda:0"); a.add_argument("--num_envs", type=int, default=4096)
    a.add_argument("--steps", type=int, default=3000); a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("bootstrap")
    b.add_argument("--checkpoint", type=Path, required=True); b.add_argument("--estimator", type=Path, required=True)
    b.add_argument("--run-name", default="oracle_bootstrap"); b.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    (pretrain if args.stage == "pretrain" else bootstrap)(args)


if __name__ == "__main__":
    main()
