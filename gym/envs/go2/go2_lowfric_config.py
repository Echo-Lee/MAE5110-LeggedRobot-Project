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

    class reward_settings(Go2Cfg.reward_settings):
        # Glide reward (Go2LowFric._reward_glide). The friction gate is
        # sigmoid((threshold - mu) / width): ~0.88-0.99 on the ice bin
        # (mu 0.05-0.2), ~0.003 or less on the ground bin (mu >= 0.6).
        glide_friction_threshold = 0.3
        glide_friction_width = 0.05
        # Commanded planar speed (m/s) above which gliding is rewarded; also
        # the body speed below which the reward is scaled down.
        glide_min_command = 0.3
        # Go2 foot collision sphere radius (resources/robots/go2/urdf/go2.urdf).
        foot_radius = 0.022
        # A foot earns full glide credit once it carries this fraction of body
        # weight (a quarter = an even share over four feet); lighter contact
        # earns proportionally less, so scuffing feet earn little.
        glide_full_load_fraction = 0.25
        # Seconds of continuous load before a foot earns full glide credit
        # (ramps up linearly), so taps and brief scuffs earn little.
        glide_min_contact_time = 0.1


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
                # Glide reward, in [0, 1] per step. Run Nov14_23-04-56_ (no
                # glide term) already scores 0.10 per step on ice and 0.00 on
                # ground with it, against tracking_lin_vel's weight of 4.0.
                glide = 1.0  # new term (default 0: absent)
                # Each foot touchdown on low friction costs 1 x weight. The
                # power-penalty run touched down ~32 times/s on ice (~0.32 per
                # step, so ~-0.16); the first glide run ~57 times/s.
                ice_touchdowns = 0.5  # new term (default 0: absent)

    class algorithm(Go2RunnerCfg.algorithm):
        # Initial value: 0.01 (inherited from LeggedRobotRunnerCfg).
        entropy_coef = 0.002

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2_lowfric"
