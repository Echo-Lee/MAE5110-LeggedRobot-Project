"""go2osc_push_glide_robust_est: sliding stops and turn-rate tracking with
the learned state estimator.

The environment is go2osc_push_glide_robust: sliding stops with a commanded
turn rate, fast slides on slippery ground, friction-limited speed and carve
references, Poisson friction changes with per-foot splits, sensor errors and
actuation delay. The estimator of go2osc_push_glide_est runs inside it.

The robust actor reads three signals the real Go2 does not provide; each is
replaced in place by its deployable counterpart of the same size:

    foot_traction_obs     [8] simulator contact-force ratios
      -> foot_traction_est      Jacobian-transpose ratios from joint torques
    friction_estimate_obs [1] true mean foot friction with a synthetic
                              lag, gain and noise
      -> contact_friction_hat   mu-hat from the learned estimator
    base_lin_vel          [3] true body velocity plus noise
      -> base_lin_vel_hat       v-hat from the learned estimator

Order and sizes are unchanged, so a go2osc_push_glide_robust checkpoint
loads directly. The friction-limited speed clock (osc.clock_follows_speed_target)
also follows mu-hat instead of the synthetic estimate. Turn-rate tracking
reads the gyro (base_ang_vel), which the robot has. Rewards and the critic
keep the true values.

Fine-tune a robust policy (copy its run to logs/go2osc_push_glide_robust_est/
first) with a pretrained estimator:

    uv run scripts/train.py --task go2osc_push_glide_robust_est --resume \\
        --load_run <robust run> --max_iterations 500 --device cuda:0 --headless

with state_estimator.init_path set to the estimator file. A checkpoint of
this task already carries its estimator, so later resumes need no file.
"""

from gym.envs.go2.go2osc_config import (
    Go2OscPushGlideRobustCfg,
    Go2OscPushGlideRobustRunnerCfg,
)
from gym.envs.go2.go2osc_est_config import Go2OscPushGlideEstCfg

# Robust actor input -> deployable replacement of the same size and scale.
DEPLOYABLE_REPLACEMENTS = {
    "foot_traction_obs": "foot_traction_est",
    "friction_estimate_obs": "contact_friction_hat",
    "base_lin_vel": "base_lin_vel_hat",
}

ROBUST_EST_ACTOR_OBSERVATIONS = [
    DEPLOYABLE_REPLACEMENTS.get(name, name)
    for name in Go2OscPushGlideRobustRunnerCfg.actor.obs
]


class Go2OscPushGlideRobustEstCfg(Go2OscPushGlideRobustCfg):
    # No friction_switch: the robust task's friction_changes already redraw
    # the ground, per foot, and a whole-env switch would overwrite the split.

    class state_estimator(Go2OscPushGlideEstCfg.state_estimator):
        pass

    class scaling(Go2OscPushGlideRobustCfg.scaling):
        base_lin_acc = Go2OscPushGlideEstCfg.scaling.base_lin_acc
        contact_friction_hat = Go2OscPushGlideEstCfg.scaling.contact_friction_hat
        base_lin_vel_hat = Go2OscPushGlideEstCfg.scaling.base_lin_vel_hat


class Go2OscPushGlideRobustEstRunnerCfg(Go2OscPushGlideRobustRunnerCfg):
    class actor(Go2OscPushGlideRobustRunnerCfg.actor):
        obs = ROBUST_EST_ACTOR_OBSERVATIONS

        class noise(Go2OscPushGlideRobustRunnerCfg.actor.noise):
            # Joint-torque noise already reaches the estimator's copy of the
            # Jacobian ratios; the actor's copy gets the robust task's
            # traction noise. The estimates carry the estimator's own error.
            foot_traction_est = Go2OscPushGlideRobustRunnerCfg.actor.noise.foot_traction_obs

    class runner(Go2OscPushGlideRobustRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_robust_est"
