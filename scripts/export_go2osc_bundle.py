"""Export a go2osc_push_glide_deploy checkpoint as a bundle for the Go2.

Run on the PC, right after training (it builds the training environment):

    uv run scripts/export_go2osc_bundle.py --name push_glide_v1 \
        --checkpoint logs/go2osc_push_glide_deploy/<run>/model_4000.pt

Writes logs/policies/<name>/:

    policy.pt            the actor (TorchScript), its outputs clamped as
                         in training (the action limit, and the joint
                         limits when the task sets joint_target_margin)
    estimator.pt         the state estimator (TorchScript)
    go2osc_onboard.json  oscillator, speed-target and estimator settings
    deploy_meta.json     the staff runner's bundle description
    training_config.json the configuration the policy was trained with
    SHA256SUMS           hashes of every file above and of the checkpoint

Deploy it with ``python -m go2_deploy.deploy <nic> --bundle logs/policies/<name>``.
"""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch

import gym.envs  # noqa: F401
from gym.envs.base.domain_randomization import set_domain_randomization_range
from gym.envs.go2.go2osc_deploy import onboard_settings
from gym.utils.helpers import class_to_dict, set_seed
from gym.utils.task_registry import task_registry

TASK = "go2osc_push_glide_deploy"
ONBOARD_FILE = "go2osc_onboard.json"
ESTIMATOR_FILE = "estimator.pt"
CONFIG_FILE = "training_config.json"


class ClampedActor(torch.nn.Module):
    """The actor with its outputs held inside the joint-target bounds.

    The runner multiplies the output by the action scale and adds the
    default posture, so bounds in action units reproduce the simulator's
    clamp: the action limit, and the joint limits less the margin when the
    task sets control.joint_target_margin.
    """

    def __init__(self, actor, low, high):
        super().__init__()
        self.actor = actor
        self.num_obs = actor.num_obs
        self.num_actions = actor.num_actions
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    def forward(self, observations):
        actions = self.actor.act_inference(observations)
        return torch.maximum(torch.minimum(actions, self.high), self.low)


class EstimatorModule(torch.nn.Module):
    """History window -> (mu, vx, vy, vz) in physical units."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, window):
        return self.model(window) * self.model.target_scale


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(task, checkpoint):
    """(environment, runner) with the checkpoint loaded.

    The configuration stays as trained (the bundle records its sensing
    settings); only the mass variation is off, so that the robot's weight
    is the nominal one.
    """
    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            task, original_cfg=False, experiment_name=None, load_run=None
        )
    )
    env_cfg.env.num_envs = 1
    env_cfg.push_robots.toggle = False
    set_domain_randomization_range(env_cfg, "link_mass_scale_range", None)
    env_cfg.seed = 0
    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(0)
    env = task_registry.make_env(task, env_cfg, "cpu", True, "mujoco")
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.load(checkpoint, load_optimizer=False)
    runner.switch_to_eval()
    return env, runner


def export(env, runner, out_dir, checkpoint, task=TASK, provenance=None):
    """Write the bundle for a loaded (environment, runner). Returns its hashes."""
    from go2_deploy.deploy_config import DeployConfig
    from go2_deploy.policy_bundle import export_bundle
    from go2_deploy.utility.import_custom_impl import load_custom_impl

    share = getattr(env, "estimate_share", 1.0)
    if share < 1.0:
        raise ValueError(
            f"{checkpoint} was trained with the estimates at a share of "
            f"{share:.2f} (state_estimator.handover): the robot has only the "
            "estimates, so export a checkpoint from after the hand-over"
        )
    if getattr(env, "joint_target_progress", 1.0) < 1.0:
        raise ValueError(
            f"{checkpoint} was trained before the joint-target clamp had "
            "closed (control.joint_target_ramp): export a later checkpoint"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = onboard_settings(env)
    ranges = env.cfg.commands.ranges
    forward = getattr(env.cfg.commands, "forward_uniform_range", None)
    forward = list(forward) if forward else list(ranges.lin_vel_x)
    settings["command_range"] = [
        [forward[0], -ranges.lin_vel_y, -ranges.yaw_vel],
        [forward[1], ranges.lin_vel_y, ranges.yaw_vel],
    ]
    scale = torch.tensor(settings["action_scale"])
    low, high = (torch.tensor(bound) for bound in settings["target_offset_bounds"])
    actor = ClampedActor(copy.deepcopy(runner.alg.actor), low / scale, high / scale)

    cfg = DeployConfig(load_custom_impl(task))
    cfg.task_name = task
    cfg.obs_vector = list(runner.actor_cfg["obs"])
    cfg.kp = settings["stiffness"]
    cfg.kd = settings["damping"]
    cfg.default_dof_pos = torch.tensor(settings["default_dof_pos"])
    cfg.command_limits = {
        "lin_vel_x": max(abs(forward[0]), abs(forward[1])),
        "lin_vel_y": ranges.lin_vel_y,
        "yaw_vel": ranges.yaw_vel,
    }
    cfg.phase_frequency = 0.0  # unused: the oscillators set their own rate
    cfg.DeployScaling = type(
        "DeployScaling",
        (),
        {
            name: torch.as_tensor(env.scales[name], dtype=torch.float)
            for name in cfg.obs_vector + ["dof_pos_target"]
            if name in env.scales
        },
    )
    export_bundle(actor, cfg, str(out_dir), provenance)

    estimator = EstimatorModule(copy.deepcopy(env.estimator.model)).eval()
    with torch.no_grad():
        example = torch.zeros(1, env.estimator.model.net.num_inputs)
        traced = torch.jit.trace(estimator, example)
    torch.jit.save(traced, str(out_dir / ESTIMATOR_FILE))
    settings["estimator_sha256"] = sha256(out_dir / ESTIMATOR_FILE)
    settings["task"] = cfg.task_name
    settings["trained_without_action_smoothing"] = True
    (out_dir / ONBOARD_FILE).write_text(json.dumps(settings, indent=2))

    saved_config = Path(checkpoint).parent / "effective_config.json"
    if saved_config.exists():
        shutil.copyfile(saved_config, out_dir / CONFIG_FILE)
    else:
        # The configuration this export loaded the checkpoint with: the
        # code's, so export from the commit that trained (see provenance).
        configuration = {
            "task": task,
            "env_config": class_to_dict(env.cfg),
            "actor": runner.actor_cfg,
            "critic": runner.critic_cfg,
            "algorithm": runner.alg_cfg,
            "runner": runner.cfg,
        }
        (out_dir / CONFIG_FILE).write_text(
            json.dumps(configuration, indent=2, default=str)
        )
    hashes = {"checkpoint " + Path(checkpoint).name: sha256(checkpoint)}
    for path in sorted(out_dir.iterdir()):
        if path.name != "SHA256SUMS":
            hashes[path.name] = sha256(path)
    (out_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    )
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Bundle name")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--out_root", default=Path("logs") / "policies", type=Path)
    arguments = parser.parse_args()
    env, runner = build(arguments.task, arguments.checkpoint)
    try:
        provenance = {
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": sha256(arguments.checkpoint),
            "task": arguments.task,
            "iteration": int(runner.it),
            "git_commit": git_commit(),
        }
        out_dir = arguments.out_root / arguments.name
        hashes = export(
            env, runner, out_dir, arguments.checkpoint, arguments.task, provenance
        )
    finally:
        env._backend.close()
    print(f"Wrote {out_dir}")
    for name, digest in hashes.items():
        print(f"  {digest}  {name}")


if __name__ == "__main__":
    main()
