"""go2osc_push_glide_deploy: the robust push-glide task, ready for the robot.

The policy of ``go2osc_push_glide_robust`` read three signals a Go2 cannot
measure (true foot traction, a friction estimate derived from the true
friction, and the true body velocity), and its clocks read true foot forces.
This task replaces all of them with what the robot computes on board
(``go2osc_onboard.py``) and trains the state estimator together with the
policy, from scratch, in one run:

    uv run scripts/train.py --task go2osc_push_glide_deploy \
        --max_iterations 4000 --device cuda:0 --headless

Then check the joint limits and export the bundle for the robot:

    uv run scripts/check_go2_joint_limits.py --task go2osc_push_glide_deploy \
        --checkpoint logs/go2osc_push_glide_deploy/<run>/model_4000.pt
    uv run scripts/export_go2osc_bundle.py --name push_glide_v1 \
        --checkpoint logs/go2osc_push_glide_deploy/<run>/model_4000.pt
"""

from gym.envs.go2.go2osc_config import (
    PHASE_OBSERVATIONS,
    Go2OscPushGlideRobustCfg,
    Go2OscPushGlideRobustRunnerCfg,
)
from gym.envs.go2.go2osc_est_config import (
    ESTIMATE_OBSERVATIONS,
    Go2OscPushGlideEstCfg,
)
from gym.envs.go2.go2osc_posture_config import Go2OscPushGlidePostureRunnerCfg

POSTURE_WEIGHTS = Go2OscPushGlidePostureRunnerCfg.critic.reward.weights

# Every actor input comes from the robot: IMU, joint encoders, the command,
# the previous joint targets and the controller's own oscillator phases
# (PHASE_OBSERVATIONS), the estimator's mu-hat and v-hat, and the speed
# target v_t the controller integrates from the command and mu-hat.
DEPLOY_ACTOR_OBSERVATIONS = (
    PHASE_OBSERVATIONS + ESTIMATE_OBSERVATIONS + ["clock_speed_obs"]
)


class Go2OscPushGlideDeployCfg(Go2OscPushGlideRobustCfg):
    class control(Go2OscPushGlideRobustCfg.control):
        # Joint targets stay this far [rad] inside the joint limits: the
        # real Go2's limits are hard stops. The exported policy carries the
        # same clamp.
        joint_target_margin = 0.05

    class friction_switch:
        # Ground changes come from friction_changes (robust task); the
        # estimator task's own mid-episode switch is off.
        probability = 0.0
        time_range = [1.0, 3.0]

    class state_estimator(Go2OscPushGlideEstCfg.state_estimator):
        # Trained from scratch beside the policy: no file to start from.
        init_path = None
        # Its input mean/std average over this many updates (one per PPO
        # iteration) before they freeze, because the first states of an
        # untrained policy are not the states of a walking one.
        normalization_updates = 300

    class sensing(Go2OscPushGlideRobustCfg.sensing):
        # The clocks' foot load comes from the joint torques, sampled once
        # per control step, with the gain, lag and noise below (inherited).
        foot_load_source = "joint_torque"

    class scaling(Go2OscPushGlideRobustCfg.scaling):
        base_lin_acc = Go2OscPushGlideEstCfg.scaling.base_lin_acc
        contact_friction_hat = Go2OscPushGlideEstCfg.scaling.contact_friction_hat
        base_lin_vel_hat = Go2OscPushGlideEstCfg.scaling.base_lin_vel_hat


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
                # Keep the joints off their stops (go2osc_push_glide_posture):
                # a commanded angle beyond 90% of the joint range, and a
                # measured angle beyond the limit.
                dof_pos_target_limits = POSTURE_WEIGHTS.dof_pos_target_limits
                dof_pos_limits = POSTURE_WEIGHTS.dof_pos_limits

    class runner(Go2OscPushGlideRobustRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_deploy"
