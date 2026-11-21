"""go2osc_push_glide_posture: the push-glide task with joint-limit penalties.

The 1500-iteration push-glide policy keeps its calves on or past their joint
limits most of the time (MuJoCo limits are soft, so the joint sinks up to
~0.4 rad past the stop). The real Go2 has hard mechanical stops there. This
task only switches on the two existing penalties, so a policy can be
fine-tuned off the stops:

    dof_pos_target_limits  commanded angle beyond 90% of the joint range
                           (soft_dof_pos_limit), as a fraction of the range
    dof_pos_limits         actual angle beyond the hard limit [rad]

Fine-tune the 1500-iteration policy (copy it to
logs/go2osc_push_glide_posture/main1500/ first):

    uv run scripts/train.py --task go2osc_push_glide_posture --resume \\
        --load_run main1500 --max_iterations 300 --device cuda:0 --headless

Weights 5 / 10 for 300 iterations (policy saved as logs/go2osc_posture1800):
calves on the stop while walking at 1 m/s fell from 84% to 0.2%, speed at a
1 m/s command rose from 0.81-0.87 to 0.86-0.92 m/s, no falls. Weights 10 / 20
also cleared the stops but crouched the body ~9 cm lower. Standing still,
the calves still rest lightly on the stops (1.5 N m against 6.3 before).
"""

from gym.envs.go2.go2osc_config import Go2OscPushGlideCfg, Go2OscPushGlideRunnerCfg


class Go2OscPushGlidePostureCfg(Go2OscPushGlideCfg):
    pass


class Go2OscPushGlidePostureRunnerCfg(Go2OscPushGlideRunnerCfg):
    class critic(Go2OscPushGlideRunnerCfg.critic):
        class reward(Go2OscPushGlideRunnerCfg.critic.reward):
            class weights(Go2OscPushGlideRunnerCfg.critic.reward.weights):
                # At the 1500-iteration posture these cost about 1.1 per step
                # while walking, against about 3.8 for speed tracking.
                dof_pos_target_limits = 5.0
                dof_pos_limits = 10.0

    class runner(Go2OscPushGlideRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_posture"
