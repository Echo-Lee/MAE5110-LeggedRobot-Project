"""Configuration for joint push-glide policy and state-estimator training.

The actor uses onboard IMU, encoder and torque signals, controller state,
and learned friction and body-velocity estimates. The critic and estimator
labels use simulator state during training. Estimator handover and joint-target
clamping progress with the training iteration.
"""

from gym.envs.go2.go2osc_onboard import (
    ACTOR_OBSERVATIONS,
    ObservationScaling,
)
from gym.envs.go2.go2osc_config import (
    Go2OscPushGlideRobustCfg,
    Go2OscPushGlideRobustRunnerCfg,
)
from gym.envs.go2.go2osc_config import PHASE_OBSERVATIONS

# Every actor input comes from the robot: IMU, joint encoders, the governed
# command, previous targets, oscillator phases, estimates and speed target.
# The shared module owns the observation order and scaling.
DEPLOY_ACTOR_OBSERVATIONS = ACTOR_OBSERVATIONS


class Go2OscPushGlideDeployCfg(Go2OscPushGlideRobustCfg):
    class control(Go2OscPushGlideRobustCfg.control):
        # Joint targets stay this far [rad] inside the joint limits: the
        # real Go2's limits are hard stops. The margin provides clearance
        # for dynamic tracking error as joints move toward their limits.
        joint_target_margin = 0.2

        class joint_target_ramp:
            # Training iterations over which the clamp closes, from
            # `slack` [rad] beyond the limits to joint_target_margin
            # inside them. The negative initial margin permits targets
            # beyond a limit during the simulation curriculum.
            start = 1000
            end = 1500
            slack = 0.4

    class command_governor:
        # The forward command the actor and the speed target get leads
        # v-hat by at most this [m/s]. Above osc.stop_threshold, or the
        # clocks never leave standing.
        margin = 0.5

    class reward_settings(Go2OscPushGlideRobustCfg.reward_settings):
        # Joint angles are charged once they come this close [rad] to a
        # joint limit: on the real Go2 the limits are hard stops.
        joint_limit_margin = 0.1

    class friction_switch:
        # Ground changes come from friction_changes (robust task); the
        # estimator task's own mid-episode switch is off.
        probability = 0.0
        time_range = [1.0, 3.0]

    class state_estimator:
        # Joint policy and estimator training starts without a checkpoint.
        init_path = None
        features = PHASE_OBSERVATIONS + [
            "base_lin_acc",
            "torques",
            "foot_force_est",
            "foot_traction_est",
            "foot_vel_rel_base",
        ]
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
        # Freeze input normalization after the initial estimator updates.
        normalization_updates = 300

        class handover:
            # Training iterations. Until `start` the actor and the speed
            # target read stand-ins for the estimates (simulated friction
            # and body velocity with noise), while the estimator learns
            # beside the policy. From `end` they read mu-hat and v-hat
            # only, as on the robot; between, a linear blend.
            start = 1500
            end = 2000
            # Half-width [m/s] of uniform noise on the stand-in velocity.
            velocity_noise = 0.1

    class sensing(Go2OscPushGlideRobustCfg.sensing):
        # The clocks' foot load comes from the joint torques, sampled once
        # per control step, with the gain, lag and noise below (inherited).
        foot_load_source = "joint_torque"

    class scaling(ObservationScaling, Go2OscPushGlideRobustCfg.scaling):
        pass


class Go2OscPushGlideDeployRunnerCfg(Go2OscPushGlideRobustRunnerCfg):
    class actor(Go2OscPushGlideRobustRunnerCfg.actor):
        obs = DEPLOY_ACTOR_OBSERVATIONS

        class noise(Go2OscPushGlideRobustRunnerCfg.actor.noise):
            # Half-widths of uniform noise on the measured inputs. The
            # estimates carry the estimator's own error, which comes from
            # the sensor noise on its inputs.
            base_ang_vel = 0.2  # [rad/s]
            projected_gravity = 0.05
            dof_pos_obs = 0.01  # [rad]
            dof_vel = 0.5  # [rad/s]

    class critic(Go2OscPushGlideRobustRunnerCfg.critic):
        class reward(Go2OscPushGlideRobustRunnerCfg.critic.reward):
            class weights(Go2OscPushGlideRobustRunnerCfg.critic.reward.weights):
                # Keep the joint angles off their stops: per joint 0 outside
                # the margin, 1 at the limit.
                joint_limit_margin = 2.0
                # Penalize requested targets beyond the applied clamp so
                # saturated outputs retain a learning signal.
                target_saturation = 1.0
                # The same cost again while the clocks stand. Walking uses
                # the clamp in stance; a standing robot that parks its knee
                # outputs beyond it stands on straight legs.
                standing_saturation = 4.0

    class runner(Go2OscPushGlideRobustRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_deploy"
