from pathlib import Path
import random

import torch

from gym import GYM_ROOT_DIR
from gym.utils.helpers import select_model, set_seed
from gym.utils.original_cfg import original_cfg_source_dir, original_task_from_run
from gym.utils.task_registry import select_backend, task_registry

LOG_ROOT = Path(GYM_ROOT_DIR) / "logs"


def resolve_run_dir(task_name, run_name=None):
    """logs/<task_name>/<run_name>, or the most recent run with a checkpoint."""
    task_dir = LOG_ROOT / task_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"No logs for task '{task_name}': {task_dir} not found")
    if run_name is not None:
        run_dir = task_dir / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run '{run_name}' not found in {task_dir}")
        return run_dir
    runs = [d for d in task_dir.iterdir() if d.is_dir() and any(d.glob("model_*.pt"))]
    if not runs:
        raise FileNotFoundError(f"No runs with a model_*.pt checkpoint in {task_dir}")
    return max(runs, key=lambda d: d.stat().st_ctime)


# Returns the Actor of a run in logs/<task_name>/ (default: most recent), where
# task_name comes from deploy_cfg.custom_impl. The task class, env and train
# configs all come from that run's saved source, not from the repo's gym.envs;
# the only code taken from outside the log is the CustomConfig extension.
# Builds the training env, so only use this on the PC; the robot loads bundles.
def setup_actor(run_name=None, deploy_cfg=None):
    if deploy_cfg is None:
        raise ValueError("Must pass the DeployConfig built from the task's custom_impl")
    task_name = deploy_cfg.task_name
    run_dir = resolve_run_dir(task_name, run_name)
    if not any(run_dir.glob("model_*.pt")):
        raise FileNotFoundError(f"No model_*.pt checkpoint in {run_dir}")
    checkpoint = run_dir / select_model(str(run_dir), -1)

    # Raises OriginalCfgError if the run's snapshot does not declare task_name
    with original_task_from_run(task_name, run_dir) as (task_class, env_cfg, train_cfg):
        print(f"Loaded task '{task_name}' from: {original_cfg_source_dir(run_dir)}")

        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 99999
        if hasattr(env_cfg, "commands"):
            env_cfg.commands.resampling_time = 99999
        if hasattr(env_cfg, "push_robots"):
            env_cfg.push_robots.toggle = False
        if hasattr(env_cfg, "init_state") and hasattr(env_cfg.init_state, "reset_mode"):
            env_cfg.init_state.reset_mode = "reset_to_range"

        env_cfg.seed = random.randint(0, 10000)
        train_cfg.seed = random.randint(0, 10000)

        train_cfg.runner.device = "cpu"
        train_cfg.runner.resume = False  # the checkpoint is loaded below
        train_cfg.runner.experiment_name = task_name
        train_cfg.runner.load_run = run_dir.name
        train_cfg.logging.enable_local_saving = False

        task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
        task_registry.set_log_dir_name(train_cfg)
        env_cfg.seed = set_seed(env_cfg.seed)

        # Built inside the block so the backend also comes from the snapshot
        env = task_class(
            cfg=env_cfg,
            device="cpu",
            headless=True,
            backend=select_backend(env_cfg, "cpu"),
        )
        runner = task_registry.make_alg_runner(env, train_cfg)
        print(f"Loading model from: {checkpoint}")
        runner.load(str(checkpoint))
        runner.switch_to_eval()
        deploy_cfg.run_dir = str(run_dir)

    actor = runner.alg.actor
    if actor.num_obs != deploy_cfg.obs_dim():
        raise ValueError(
            f"{run_dir} actor takes {actor.num_obs} obs, but "
            f"{type(deploy_cfg.custom_impl).__name__}.obs_vector gives "
            f"{deploy_cfg.obs_dim()}"
        )
    return actor


class RLController:
    """Runs one policy. `policy` maps a scaled obs vector to raw actions.

    Pass the policy from policy_bundle.load_bundle(); without one, the actor is
    built from the training logs (PC-tethered workflow).
    """

    def __init__(self, cfg, policy=None):
        self.cfg = cfg
        self.policy = policy if policy is not None else setup_actor(None, cfg)
        self.action_scale = torch.tensor(
            getattr(cfg.DeployScaling, "dof_pos_target", 1.0)
        )

    # Returns torch.tensor(12): offsets from the static default posture in radians.
    def act(self, obs_vector):
        with torch.no_grad():
            return self.policy(obs_vector) * self.action_scale
