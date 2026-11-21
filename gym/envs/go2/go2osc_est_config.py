"""go2osc_push_glide_est: the push-glide task with a learned state estimator.

Environment runs the estimator and
publishes mu-hat and v-hat, but the actor and critic inputs are still the
team's, so a go2osc_push_glide checkpoint (e.g. the 1500-iteration main policy) loads
unchanged and the estimator can be trained and checked against it. Adding
mu-hat and v-hat to the actor is step 2.
"""

from gym.envs.go2.go2_config import BASE_HEIGHT_REF
from gym.envs.go2.go2osc_config import (
    PHASE_OBSERVATIONS,
    Go2OscPushGlideCfg,
    Go2OscPushGlideRunnerCfg,
)

GRAVITY_M_S2 = 9.81

# Same envelope as the team's task. The three bins only give friction
# switches a "different surface" to jump to; weights proportional to bin
# width keep the overall draw uniform on [0.1, 1.0], as in the base task.
FRICTION_RANGE = [0.1, 1.0]
FRICTION_BINS = [[0.1, 0.3], [0.3, 0.6], [0.6, 1.0]]
FRICTION_BIN_WEIGHTS = [0.2, 0.3, 0.4]

# Signals the real Go2 provides: IMU (gyro, gravity direction, accelerometer),
# joint encoders, the previous joint targets, the operator's command and the
# controller's own oscillator phases, and the motor torques the Go2 reports
# (stage 3: a small gain). No contact forces, no true velocity.
ESTIMATOR_FEATURES = PHASE_OBSERVATIONS + ["base_lin_acc", "torques"]


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


class Go2OscPushGlideEstRunnerCfg(Go2OscPushGlideRunnerCfg):
    # Actor and critic inputs unchanged until step 2.

    class runner(Go2OscPushGlideRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_est"
