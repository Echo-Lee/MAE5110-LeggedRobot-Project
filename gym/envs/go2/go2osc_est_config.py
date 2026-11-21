"""go2osc_push_glide_est: the push-glide task with a learned state estimator.

The environment runs the estimator and publishes mu-hat and v-hat. The
actor is deployable: it reads the team's inputs without the privileged
foot_traction_obs (true contact-force ratios), then the estimates
(ESTIMATE_OBSERVATIONS). scripts/widen_go2osc_policy.py converts a
go2osc_push_glide checkpoint: the traction columns are removed (the same as
feeding their mean, about 0; test 2 showed the policy barely uses them), the
estimates get zero weights, and the pretrained estimator is stored in the
same checkpoint, so fine-tuning is a plain resume:

    uv run scripts/train.py --task go2osc_push_glide_est --resume \
        --load_run widened_posture1800 --max_iterations 500 --device cuda:0 --headless
"""

from gym.envs.go2.go2_config import BASE_HEIGHT_REF
from gym.envs.go2.go2osc_posture_config import Go2OscPushGlidePostureRunnerCfg
from gym.envs.go2.go2osc_config import (
    PHASE_OBSERVATIONS,
    Go2OscPushGlideCfg,
    Go2OscPushGlideRunnerCfg,
)

GRAVITY_M_S2 = 9.81
POSTURE_WEIGHTS = Go2OscPushGlidePostureRunnerCfg.critic.reward.weights

# Same envelope as the team's task. The three bins only give friction
# switches a "different surface" to jump to; weights proportional to bin
# width keep the overall draw uniform on [0.1, 1.0], as in the base task.
FRICTION_RANGE = [0.1, 1.0]
FRICTION_BINS = [[0.1, 0.3], [0.3, 0.6], [0.6, 1.0]]
FRICTION_BIN_WEIGHTS = [0.2, 0.3, 0.4]

# Signals the real Go2 provides: IMU (gyro, gravity direction, accelerometer),
# joint encoders, the previous joint targets, the operator's command and the
# controller's own oscillator phases, and the motor torques the Go2 reports.
# The Jacobian features turn torques and joint velocities into foot
# forces, traction ratios and foot velocities (go2osc_est.py); they cut mu-hat
# error from 0.161 to 0.151 (two seeds each) on the posture1800 policy.
ESTIMATOR_FEATURES = PHASE_OBSERVATIONS + [
    "base_lin_acc",
    "torques",
    "foot_force_est",
    "foot_traction_est",
    "foot_vel_rel_base",
]

# Estimates appended to the actor's inputs: mu-hat [1] and body-frame v-hat [3].
ESTIMATE_OBSERVATIONS = ["contact_friction_hat", "base_lin_vel_hat"]
# Actor inputs the real robot cannot provide; the deployable actor drops them.
PRIVILEGED_ACTOR_OBSERVATIONS = ["foot_traction_obs"]
# The simulator's true values the estimates stand for, same sizes and scales
# (oracle comparison only): mu [1] and body-frame v [3].
TRUE_STATE_OBSERVATIONS = ["contact_friction_obs", "base_lin_vel"]


class Go2OscPushGlideEstCfg(Go2OscPushGlideCfg):
    class domain_randomization(Go2OscPushGlideCfg.domain_randomization):
        class startup(Go2OscPushGlideCfg.domain_randomization.startup):
            contact_friction_range = FRICTION_RANGE
            contact_friction_bins = FRICTION_BINS
            contact_friction_bin_weights = FRICTION_BIN_WEIGHTS

        class episode(Go2OscPushGlideCfg.domain_randomization.episode):
            # A new surface every episode, not one per env for all of
            # training: the estimator needs many surfaces per env.
            resample_contact_friction = True

    class friction_switch:
        # Mid-episode surface change (Go2OscPushGlideEst): a fraction of
        # episodes jump once to a friction drawn from another bin, like
        # stepping from ice onto carpet. Teaches the estimator to re-estimate.
        probability = 0.3
        time_range = [1.0, 3.0]  # s after reset, uniform (episodes are 4 s)

    class state_estimator:
        # Initial estimator file (relative paths are from the repository
        # root). None starts untrained: valid for pretraining, or when a
        # go2osc_push_glide_est checkpoint supplies the weights.
        init_path = None
        features = ESTIMATOR_FEATURES
        # 25 frames, every 2nd control step: 0.5 s of history at 100 Hz.
        frames = 25
        stride = 2
        hidden_dims = [512, 256, 128]
        # Keep training on the true labels while PPO trains the policy.
        learn = True

        class sensor_noise:
            # Noise on the estimator's inputs only, in physical units. The
            # commanded joint targets, commands and oscillator phases are
            # known exactly and get none.
            enabled = True
            std = {
                "base_lin_acc": 0.5,  # m/s^2 (~0.05 g, IMU noise + vibration)
                "dof_pos_obs": 0.01,  # rad (encoder), as the actor's noise
                "dof_vel": 0.1,  # rad/s (differentiated encoder)
                "torques": 0.3,  # N m (estimated from motor current)
            }
            # Constant per episode and axis, uniform in +-value [m/s^2].
            accelerometer_bias = 0.2

        learning_rate = 3.0e-4
        samples_per_step = 256  # random envs kept per control step
        update_interval = 24  # control steps between updates
        gradient_steps = 4  # Adam steps per update

    class scaling(Go2OscPushGlideCfg.scaling):
        base_lin_acc = GRAVITY_M_S2  # accelerometer read in units of g
        contact_friction_hat = 1.0
        base_lin_vel_hat = BASE_HEIGHT_REF  # same scale as base_lin_vel


# The team's actor inputs in their order, without the privileged ones.
DEPLOYABLE_ACTOR_OBSERVATIONS = [
    name
    for name in Go2OscPushGlideRunnerCfg.actor.obs
    if name not in PRIVILEGED_ACTOR_OBSERVATIONS
]


class Go2OscPushGlideEstRunnerCfg(Go2OscPushGlideRunnerCfg):
    class actor(Go2OscPushGlideRunnerCfg.actor):
        # The deployable inputs, then the estimates. The critic keeps its
        # privileged inputs (training only).
        obs = DEPLOYABLE_ACTOR_OBSERVATIONS + ESTIMATE_OBSERVATIONS

    class critic(Go2OscPushGlideRunnerCfg.critic):
        class reward(Go2OscPushGlideRunnerCfg.critic.reward):
            class weights(Go2OscPushGlideRunnerCfg.critic.reward.weights):
                # Keep the joint-limit penalties the base policy was
                # fine-tuned with (go2osc_push_glide_posture), so stage 5
                # does not drift back onto the calf stops.
                dof_pos_target_limits = POSTURE_WEIGHTS.dof_pos_target_limits
                dof_pos_limits = POSTURE_WEIGHTS.dof_pos_limits

    class runner(Go2OscPushGlideRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_est"


# Stage 5 comparisons. Same environment as go2osc_push_glide_est (friction
# switches, sensor noise, the estimator running and learning); only the
# actor's inputs differ, so a difference in the fine-tuned policies comes
# from what the actor is told.


class Go2OscPushGlideBlindRunnerCfg(Go2OscPushGlideEstRunnerCfg):
    """B, blind: deployable inputs only, no friction or speed information."""

    class actor(Go2OscPushGlideEstRunnerCfg.actor):
        obs = list(DEPLOYABLE_ACTOR_OBSERVATIONS)

    class runner(Go2OscPushGlideEstRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_blind"


class Go2OscPushGlideOracleRunnerCfg(Go2OscPushGlideEstRunnerCfg):
    """Oracle: the true mu and v in place of the estimates (upper bound)."""

    class actor(Go2OscPushGlideEstRunnerCfg.actor):
        obs = DEPLOYABLE_ACTOR_OBSERVATIONS + TRUE_STATE_OBSERVATIONS

    class runner(Go2OscPushGlideEstRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_oracle"
