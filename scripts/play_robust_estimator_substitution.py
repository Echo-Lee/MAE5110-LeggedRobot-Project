"""Play the robust-carve actor with its privileged inputs replaced by estimates.

The actor checkpoint stays unchanged: its 66 observation columns are still in
the same order.  Only their sources change at inference time:

* foot_traction_obs     -> foot_traction_est (Jacobian / motor torque estimate)
* friction_estimate_obs -> contact_friction_hat
* base_lin_vel          -> base_lin_vel_hat

The oscillator clock also uses the previous control step's mu-hat.  This is a
causal plug-in test, not policy fine-tuning and not a new registered task.
"""

import argparse
import copy
from pathlib import Path

import torch

from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from gym.envs.go2.go2osc_est_config import Go2OscPushGlideEstCfg
from gym.utils.helpers import set_seed
from gym.utils.task_registry import select_backend, task_registry
from scripts.play import command_schedule, get_play_args, play, record


class RobustEstimatorSubstitutionEnv(Go2OscPushGlideEst):
    """Robust dynamics with the actor-facing simulator signals substituted."""

    def _set_actor_estimates(self):
        self.friction_estimate_obs.copy_(self.contact_friction_hat)

    def _advance_clock_speed(self, reset_mask=None):
        # The parent calls this before its estimator update.  Use the prior
        # estimate, exactly as a controller must; during construction, no
        # estimator exists yet and the ordinary reset path is retained.
        if not hasattr(self, "contact_friction_hat"):
            return super()._advance_clock_speed(reset_mask)
        saved = self.friction_estimate_obs.clone()
        self.friction_estimate_obs.copy_(self.contact_friction_hat)
        try:
            return super()._advance_clock_speed(reset_mask)
        finally:
            self.friction_estimate_obs.copy_(saved)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        # The actor and the next clock update both receive the newest estimate.
        self._set_actor_estimates()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        if hasattr(self, "contact_friction_hat"):
            self._set_actor_estimates()


def arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--estimator",
        type=Path,
        default=Path("logs/go2osc_push_glide_est/state_estimator_2800/estimator.pt"),
        help="Estimator-only export (HistoryStateEstimator format).",
    )
    extra, remaining = parser.parse_known_args()
    args = get_play_args(remaining)
    if args.task != "go2osc_push_glide_robust":
        parser.error("this script only supports --task go2osc_push_glide_robust")
    if not extra.estimator.is_file():
        parser.error(f"estimator file not found: {extra.estimator}")
    return args, extra


def make(args, estimator_path):
    # This mirrors scripts/play.py, except the environment class and the
    # actor's observation names.  Saved robust configs remain authoritative.
    from scripts.play import play_cfgs

    env_cfg, train_cfg = play_cfgs(args)
    env_cfg.state_estimator = copy.deepcopy(Go2OscPushGlideEstCfg.state_estimator)
    env_cfg.state_estimator.init_path = str(estimator_path.resolve())
    env_cfg.state_estimator.learn = False
    env_cfg.scaling.base_lin_acc = Go2OscPushGlideEstCfg.scaling.base_lin_acc
    env_cfg.scaling.contact_friction_hat = (
        Go2OscPushGlideEstCfg.scaling.contact_friction_hat
    )
    env_cfg.scaling.base_lin_vel_hat = Go2OscPushGlideEstCfg.scaling.base_lin_vel_hat

    replacement = {
        "foot_traction_obs": "foot_traction_est",
        "friction_estimate_obs": "contact_friction_hat",
        "base_lin_vel": "base_lin_vel_hat",
    }
    train_cfg.actor.obs = [replacement.get(name, name) for name in train_cfg.actor.obs]

    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)
    backend = select_backend(env_cfg, args.device, args.backend)
    env = RobustEstimatorSubstitutionEnv(
        cfg=env_cfg, device=args.device, headless=args.headless, backend=backend
    )
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner


def policy_step(env, runner, command):
    if command is not None:
        env.commands[:] = command
    runner.set_actions(
        runner.actor_cfg["actions"],
        runner.get_inference_actions(),
        runner.actor_cfg["disable_actions"],
    )
    env.step()


@torch.no_grad()
def main():
    args, extra = arguments()
    env, runner = make(args, extra.estimator)
    command_at = command_schedule(env, args)
    print(f"loaded robust actor with estimator substitution: {extra.estimator}")
    if args.video:
        # scripts.play.record expects its own policy_step; use the same loop
        # locally so replacement remains active.
        from gym.utils.interfaces.VideoRecorder import VideoRecorder

        recorder = VideoRecorder(env, args.video, overlay=args.overlay)
        for step in range(round(args.video_seconds / env.dt)):
            policy_step(env, runner, command_at(step))
            recorder.record_frame()
        recorder.close()
    else:
        # Install the same keyboard interface and command overlay as play.py.
        from gym.utils.interfaces.CommandVisualizer import CommandVisualizer
        from gym.utils.interfaces.MujocoKeyboardInterface import MujocoKeyboardInterface

        if args.keyboard and args.command is None:
            MujocoKeyboardInterface(env)
        CommandVisualizer(env)
        for step in range(10 * int(env.max_episode_length)):
            policy_step(env, runner, command_at(step))
            env.check_exit()


if __name__ == "__main__":
    main()
