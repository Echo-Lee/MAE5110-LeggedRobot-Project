"""go2osc_push_glide_deploy: the robust push-glide task, ready for the robot.

The policy of ``go2osc_push_glide_robust`` read three signals a Go2 cannot
measure (true foot traction, a friction estimate derived from the true
friction, and the true body velocity), and its clocks read true foot forces.
This task replaces all of them with what the robot computes on board
(``go2_deploy/custom_implementation/go2osc_push_glide.py``) and trains the
state estimator together with the
policy, from scratch, in one run:

    uv run scripts/train.py --task go2osc_push_glide_deploy \
        --max_iterations 4000 --device cuda:0 --headless

Then check the joint limits and export the bundle for the robot:

    uv run scripts/check_go2_joint_limits.py --task go2osc_push_glide_deploy \
        --checkpoint logs/go2osc_push_glide_deploy/<run>/model_4000.pt
    uv run scripts/export_go2osc_bundle.py --name push_glide_v1 \
        --checkpoint logs/go2osc_push_glide_deploy/<run>/model_4000.pt
"""

from go2_deploy.custom_implementation.go2osc_push_glide import (
    ACTOR_OBSERVATIONS,
    Go2OscPushGlide,
)
from gym.envs.go2.go2osc_config import (
    Go2OscPushGlideRobustCfg,
    Go2OscPushGlideRobustRunnerCfg,
)
from gym.envs.go2.go2osc_est_config import (
    Go2OscPushGlideEstCfg,
)

# Every actor input comes from the robot: IMU, joint encoders, the governed
# command, previous targets, oscillator phases, estimates and speed target.
# The custom configuration owns the ordered declaration used by both paths.
DEPLOY_ACTOR_OBSERVATIONS = ACTOR_OBSERVATIONS


class Go2OscPushGlideDeployCfg(Go2OscPushGlideRobustCfg):
    class control(Go2OscPushGlideRobustCfg.control):
        # Joint targets stay this far [rad] inside the joint limits: the
        # real Go2's limits are hard stops. The exported policy carries
        # the same clamp. The margin provides clearance for dynamic
        # tracking error as joints move toward their limits.
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

    class state_estimator(Go2OscPushGlideEstCfg.state_estimator):
        # Trained from scratch beside the policy: no file to start from.
        init_path = None
        # Its input mean/std average over this many estimator updates
        # (one per update_interval control steps) before they freeze,
        # because the first states of an
        # untrained policy are not the states of a walking one.
        normalization_updates = 300

        class handover:
            # Training iterations. Until `start` the actor and the speed
            # target read stand-ins for the estimates (the simulated
            # friction estimate; the true body velocity with noise), as
            # go2osc_push_glide_robust did, while the estimator learns
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

    class scaling(Go2OscPushGlide.DeployScaling, Go2OscPushGlideRobustCfg.scaling):
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
