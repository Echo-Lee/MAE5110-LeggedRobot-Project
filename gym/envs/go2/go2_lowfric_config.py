from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg

# Ice-like and ordinary-ground friction. MuJoCo applies one coefficient to
# both robot and ground geoms per environment (see mujoco_backend_base).
ICE_FRICTION = [0.05, 0.2]
GROUND_FRICTION = [0.6, 1.2]


class Go2LowFricCfg(Go2Cfg):
    class env(Go2Cfg.env):
        # Long enough for slip, recovery, and several command resamples to
        # occur within one episode on low friction.
        episode_length_s = 10

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            contact_friction_range = [ICE_FRICTION[0], GROUND_FRICTION[1]]
            contact_friction_bins = [ICE_FRICTION, GROUND_FRICTION]
            contact_friction_bin_weights = [0.5, 0.5]

        class episode(Go2Cfg.domain_randomization.episode):
            resample_contact_friction = True

    class scaling(Go2Cfg.scaling):
        contact_friction = 1.0


class Go2LowFricRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        # Privileged teacher input: true friction. Replace with an estimate
        # for a deployable student policy.
        obs = Go2RunnerCfg.actor.obs + ["contact_friction"]

    class critic(Go2RunnerCfg.critic):
        obs = Go2RunnerCfg.critic.obs + ["contact_friction"]

        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                # Energy penalty. Run Nov14_22-13-21_ (no energy terms) used
                # ~1000 W mean mechanical power and ~200 Nm^2 mean squared
                # torque, so these start at about -1.0 and -0.3 per step
                # against a total reward of ~7. Power discourages wasted leg
                # motion (tapping, drift when commanded to stop); squared
                # torque is lowest when load is shared by all legs.
                power = 1.0e-3  # new term (default 0: absent)
                torques = 1.5e-3  # initial value: 5.0e-6 (inherited from go2)

    class algorithm(Go2RunnerCfg.algorithm):
        # Initial value: 0.01 (inherited from LeggedRobotRunnerCfg).
        entropy_coef = 0.002

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2_lowfric"
