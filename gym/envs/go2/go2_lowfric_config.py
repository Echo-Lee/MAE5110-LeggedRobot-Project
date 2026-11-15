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

    class commands(Go2Cfg.commands):
        # Run-up-and-slide episodes (Go2LowFric._apply_slide_commands). On
        # ice, a fraction of episodes replace the random commands with: hold
        # forward speed v0 for slide_runup_time, then decay the command at
        # the rate pure sliding decelerates, v0 - slide_decel_scale*mu*g*t,
        # down to a stop for the rest of the episode. With the power penalty,
        # coasting on planted feet is the cheapest way to follow that decay.
        # Fraction of command resamples that become pure turn-in-place (zero
        # forward/sideways speed). Off: run 12's configuration is the stage-1
        # baseline. Trained 0.1 on every surface (runs 13, 13b) and on ground
        # only (run 14, Nov15_21-43-55_): every variant made the ice slide
        # fragile (robustness 0/9 vs 8/9 for runs 12/12b); run 14 also froze
        # on ice for many commands (walking grid 14/30).
        turn_in_place_probability = 0.0
        # Only envs with friction >= this get turn-in-place commands (the
        # ground bin starts at 0.6); ice keeps run 12's command mix.
        turn_in_place_min_friction = 0.6
        turn_in_place_min_yaw = 0.5  # rad/s; avoids near-zero "turns"
        slide_probability = 0.3  # of eligible (ice) episodes
        slide_max_friction = 0.2  # only where mu <= this
        slide_speed_range = [0.8, 1.3]  # v0 (m/s); mu=0.05 tops out ~1.3
        slide_runup_time = 4.0  # s; reaching v0 on mu=0.05 takes ~3 s
        slide_decel_scale = 1.0
        # Seconds at zero after the slide stops; then the slide ends and the
        # env gets normal random commands (turning in place included).
        slide_stop_hold_time = 0.5
        # Residual hold (run 12): during the slide phase, joint targets
        # (offsets from the standing pose) are multiplied by this, so the
        # policy keeps ~20% authority to balance around the standing pose.
        # Scripted full hold (scale 0) brakes at ~mu*g on mu <= 0.1 but fell
        # 88% at mu = 0.2, v0 = 1.2; the learned slide (scale 1) only
        # reached 0.66-0.85 mu*g.
        slide_action_scale = 0.2

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
        # slide_phase: 1 during a run-up-and-slide's slide, so the policy
        # can tell a slide from an ordinary command change.
        obs = Go2RunnerCfg.actor.obs + ["contact_friction", "slide_phase"]

    class critic(Go2RunnerCfg.critic):
        obs = Go2RunnerCfg.critic.obs + ["contact_friction", "slide_phase"]

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
                # Glide terms, off. Trained values: glide = 1.0 (runs
                # Nov15_04-33-19_, Nov15_05-26-38_) and ice_touchdowns = 0.5
                # (Nov15_05-26-38_). With isotropic friction a gliding foot
                # always brakes, so both runs were slower on ice than the
                # power-penalty run; v2 also favored tucking a leg (fewer
                # touchdowns). Kept available for later experiments.
                glide = 0.0
                ice_touchdowns = 0.0
                # Joint-velocity penalty during the slide phase only. Off
                # (run 11). Trained value 3.0 (run 10, Nov15_16-53-07_): it
                # froze the robot in a crouched, non-standing pose (pose dev
                # 0.24 -> 0.45 rad), barely improved braking, and stiffened
                # walking (25/30, top speed 2.74 -> 2.03 m/s at mu = 1).
                slide_still = 0.0
                # Standing-pose hold during the slide phase only. 3.0 as in
                # runs 9-10 and 12. Run 11 (Nov15_18-40-46_) tried 10.0: pose
                # got worse (0.24 -> 0.34 rad) and walking degraded (22/30),
                # so the slide is now constrained via slide_action_scale
                # instead of a heavier penalty.
                slide_pose = 3.0

    class algorithm(Go2RunnerCfg.algorithm):
        # Initial value: 0.01 (inherited from LeggedRobotRunnerCfg).
        entropy_coef = 0.002

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2_lowfric"
