"""Go2 adaptation of Zhang et al.'s decentralized oscillator ablations."""

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg

PAPER_OBSERVATIONS = [
    "base_ang_vel",
    "projected_gravity",
    "commands",
    "dof_pos_obs",
    "dof_vel",
    "dof_pos_target",
]
PHASE_OBSERVATIONS = PAPER_OBSERVATIONS + ["oscillator_obs"]


class Go2OscCfg(Go2Cfg):
    class env(Go2Cfg.env):
        episode_length_s = 4.0

    class control(Go2Cfg.control):
        ctrl_frequency = 100
        desired_sim_frequency = 500
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}

    class osc:
        observations = True
        reward = True
        coupling = True
        init_to = "random"
        init_w_offset = True
        timeout_physical_reset_probability = 0.75
        stop_threshold = 0.5
        omega_stop = 1.0
        omega_offset = 1.5
        omega_max = 4.0
        coupling_stop = 4.0
        coupling_moving = 1.0
        stance_offset = 1.0
        omega_noise_std = 0.25
        coupling_noise_std = 0.25
        process_noise_std = 0.25
        grf_threshold = 0.1

    class commands(Go2Cfg.commands):
        # Optional uniform forward commands; standing dropout still applies.
        forward_uniform_range = None
        # Total probability assigned equally to the two uniform-range endpoints.
        forward_boundary_probability = 0.0
        forward_centers = [-3.0, -1.0, 0.0, 1.0, 3.0]
        forward_noise_std = 1.0
        include_standing = True

    class push_robots(Go2Cfg.push_robots):
        toggle = True
        interval_s = 10
        max_push_vel_xy = 0.2
        push_box_dims = [0.0, 0.0, 0.0]

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            contact_friction_range = [0.1, 1.0]
            link_mass_scale_range = None

        class episode(Go2Cfg.domain_randomization.episode):
            scale_ranges = {}

    class asset(Go2Cfg.asset):
        class robot_layout(Go2Cfg.asset.robot_layout):
            dof_groups = {
                "hip_abduction": [
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                ]
            }

    class scaling(Go2Cfg.scaling):
        base_ang_vel = [0.3, 0.3, 0.1]
        dof_pos = [0.2, 0.3, 0.3] * 4
        dof_pos_obs = dof_pos
        dof_pos_target = dof_pos
        oscillator_obs = 1.0
        oscillators_vel = 1.0


class Go2OscRunnerCfg(Go2RunnerCfg):
    runner_class_name = "OscillatorRunner"

    class actor(Go2RunnerCfg.actor):
        frequency = 100
        hidden_dims = [256, 256, 128]
        activation = "elu"
        obs = PHASE_OBSERVATIONS
        normalize_obs = True

    class critic(Go2RunnerCfg.critic):
        normalize_obs = True
        hidden_dims = [256, 256, 128]
        activation = "elu"
        obs = ["base_height", "base_lin_vel"] + PHASE_OBSERVATIONS + ["oscillators_vel"]

        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                orientation = 1.0
                torques = 5.0e-7
                ang_vel_xy = 0.0
                min_base_height = 1.5
                action_rate = 0.01
                action_rate2 = 0.001
                cursorial = 0.25
                swing_grf = 5.0
                stance_grf = 5.0

            class termination_weight:
                termination = 0.15

    class algorithm(Go2RunnerCfg.algorithm):
        # Four complete passes: eight minibatches per pass, 32 updates each
        # for actor and critic. PPO2 optimizes them separately with unit
        # coefficient on the critic loss.
        rollout_size = 4096 * 32
        batch_size = rollout_size // 8
        max_gradient_steps = 4 * 8
        learning_rate = 1.0e-5
        clip_param = 0.2
        entropy_coef = 0.02
        max_grad_norm = 1.0
        use_clipped_value_loss = True
        schedule = "adaptive"
        desired_kl = 0.01
        lr_range = [1.0e-5, 1.0e-2]
        lr_ratio = 1.5
        discount_horizon = 1.0
        GAE_bootstrap_horizon = 2.0
        gamma = 0.99
        lam = 0.995

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2osc"
        max_iterations = 2000
        algorithm_class_name = "PPO2"


class Go2OscORC000Cfg(Go2OscCfg):
    class osc(Go2OscCfg.osc):
        observations = False
        reward = False
        coupling = False


class Go2OscORC000RunnerCfg(Go2OscRunnerCfg):
    class actor(Go2OscRunnerCfg.actor):
        obs = PAPER_OBSERVATIONS

    class critic(Go2OscRunnerCfg.critic):
        obs = ["base_height", "base_lin_vel"] + PAPER_OBSERVATIONS

        class reward(Go2OscRunnerCfg.critic.reward):
            class weights(Go2OscRunnerCfg.critic.reward.weights):
                swing_grf = 0.0
                stance_grf = 0.0


class Go2OscORC010Cfg(Go2OscORC000Cfg):
    class osc(Go2OscORC000Cfg.osc):
        reward = True


class Go2OscORC010RunnerCfg(Go2OscRunnerCfg):
    class actor(Go2OscRunnerCfg.actor):
        obs = PAPER_OBSERVATIONS

    class critic(Go2OscRunnerCfg.critic):
        obs = ["base_height", "base_lin_vel"] + PAPER_OBSERVATIONS


class Go2OscORC011Cfg(Go2OscORC010Cfg):
    class osc(Go2OscORC010Cfg.osc):
        coupling = True


class Go2OscORC011RunnerCfg(Go2OscORC010RunnerCfg):
    pass


class Go2OscORC100Cfg(Go2OscCfg):
    class osc(Go2OscCfg.osc):
        reward = False
        coupling = False


class Go2OscORC100RunnerCfg(Go2OscRunnerCfg):
    class critic(Go2OscRunnerCfg.critic):
        class reward(Go2OscRunnerCfg.critic.reward):
            class weights(Go2OscRunnerCfg.critic.reward.weights):
                swing_grf = 0.0
                stance_grf = 0.0


class Go2OscORC101Cfg(Go2OscORC100Cfg):
    class osc(Go2OscORC100Cfg.osc):
        coupling = True


class Go2OscORC101RunnerCfg(Go2OscORC100RunnerCfg):
    pass


class Go2OscORC110Cfg(Go2OscCfg):
    class osc(Go2OscCfg.osc):
        coupling = False


class Go2OscORC110RunnerCfg(Go2OscRunnerCfg):
    pass


class Go2OscORC111Cfg(Go2OscCfg):
    pass


class Go2OscORC111RunnerCfg(Go2OscRunnerCfg):
    pass


class Go2OscORC111GRFCfg(Go2OscORC111Cfg):
    class scaling(Go2OscORC111Cfg.scaling):
        foot_grf_obs = 1.0


class Go2OscORC111GRFRunnerCfg(Go2OscORC111RunnerCfg):
    class actor(Go2OscORC111RunnerCfg.actor):
        obs = PHASE_OBSERVATIONS + ["foot_grf_obs"]

    class critic(Go2OscORC111RunnerCfg.critic):
        obs = Go2OscORC111RunnerCfg.critic.obs + ["foot_grf_obs"]

    class runner(Go2OscORC111RunnerCfg.runner):
        experiment_name = "go2osc_orc111_grf"


class Go2OscORC111GRFDirectionalCfg(Go2OscORC111Cfg):
    class scaling(Go2OscORC111Cfg.scaling):
        foot_grf_angle = 1.0

    class osc(Go2OscORC111Cfg.osc):
        directional_gain = 1.0
        directional_min_load = 0.05
        directional_min_progress = 0.5
        directional_stance_gate = False
        directional_feedback_mode = "offset"
        moving_frequency_scale = 1.0

    class reward_settings(Go2OscORC111Cfg.reward_settings):
        stance_force_sliding_gain = 0.0
        stance_force_sliding_ratio_reference = 0.2
        stance_force_sliding_min_load = 0.05
        stance_force_sliding_epsilon = 0.0001
        velocity_tracking_tolerance = None
        velocity_tracking_reward = "gaussian"
        velocity_tracking_reference_speed = 1.0
        stance_foot_vertical_speed_reference = 0.1
        touchdown_vertical_speed_reference = 0.1
        soft_support_load_reference = 0.05
        sustained_support_duration = 0.1
        sliding_speed_reference = 0.2
        sliding_speed_reward = "saturated"
        sliding_tracking_width = 0.3
        sliding_speed_deadzone = 0.0
        sliding_contact_initial_fraction = 0.0
        sliding_contact_age_mode = "force"
        sliding_engagement_min_load = 0.05
        sliding_engagement_min_speed = 0.05
        sliding_engagement_max_gap = 0.01
        sliding_engagement_max_clearance = 0.001
        sliding_vertical_speed_reference = 0.05
        sliding_lateral_speed_reference = 0.1
        sliding_contact_ramp_duration = 0.02
        sliding_clearance_reference = 0.001
        sliding_body_match_width = None
        foot_clearance_height_reference = 0.01
        yaw_tracking_tolerance = None
        mechanical_power_reference_speed = 1.0


class Go2OscORC111GRFDirectionalRunnerCfg(Go2OscORC111RunnerCfg):
    class actor(Go2OscORC111RunnerCfg.actor):
        obs = PHASE_OBSERVATIONS + ["foot_grf_angle"]

    class critic(Go2OscORC111RunnerCfg.critic):
        obs = Go2OscORC111RunnerCfg.critic.obs + ["foot_grf_angle"]

        class reward(Go2OscORC111RunnerCfg.critic.reward):
            class weights(Go2OscORC111RunnerCfg.critic.reward.weights):
                mechanical_power = 0.0
                normalized_torques = 0.0
                normal_load_squared = 0.0
                normal_load_change_squared = 0.0
                stance_foot_vertical_velocity = 0.0
                foot_vertical_velocity = 0.0
                loaded_foot_fraction = 0.0
                soft_foot_support = 0.0
                sustained_foot_support = 0.0
                forward_sliding = 0.0
                foot_clearance = 0.0
                loaded_contact_switches = 0.0
                touchdown_vertical_velocity = 0.0
                joint_velocity_squared = 0.0

    class runner(Go2OscORC111RunnerCfg.runner):
        experiment_name = "go2osc_orc111_grf_directional"


class Go2OscPushGlideCfg(Go2OscORC111GRFDirectionalCfg):
    """Push/glide phase signals for mixed-friction training."""

    class commands(Go2OscORC111GRFDirectionalCfg.commands):
        # Continuous from rest, so no speed near the stop threshold is untrained.
        forward_uniform_range = [0.0, 1.5]
        # Standing dropout also yields straight and turn-in-place commands.
        include_standing = True

        class ranges(Go2OscORC111GRFDirectionalCfg.commands.ranges):
            lin_vel_x = [0.0, 1.5]
            # Steering that friction 0.1 can still hold at 1.5 m/s forward.
            lin_vel_y = 0.3  # max [m/s]
            yaw_vel = 0.5  # max [rad/s]

    class control(Go2OscORC111GRFDirectionalCfg.control):
        # Largest joint-target offset the motors receive, in action units.
        # Walking stays below 6; a falling robot used to command hundreds.
        action_limit = 10.0

    class scaling(Go2OscORC111GRFDirectionalCfg.scaling):
        foot_traction_obs = 1.0
        contact_friction_obs = 1.0

    class osc(Go2OscORC111GRFDirectionalCfg.osc):
        # The gated push signal replaces angle feedback.
        directional_gain = 0.0
        # Foot speed relative to the base [m/s] over which the gate switches.
        push_gate_speed = 0.1
        traction_ratio_limit = 2.0
        # Slow walking uses the moving clocks; only near-zero commands stand.
        stop_threshold = 0.2
        # A sidestep or turn above these also uses the moving clocks.
        sideways_stop_threshold = 0.1  # [m/s]
        turning_stop_threshold = 0.1  # [rad/s]

    class reward_settings(Go2OscORC111GRFDirectionalCfg.reward_settings):
        # Load per foot that push_load credits [body weights]: a two-foot share.
        push_load_cap = 0.5
        # -sin(phase) at which the push weight reaches 1; 1.0 is a plain sine.
        push_window_saturation = 0.5
        # Width of the Gaussian turning reward [rad/s]: half the command range,
        # so a small steady yaw-rate error is not free.
        yaw_tracking_tolerance = 0.25
        # Most that tilt_rate charges per step [rad^2/s^2]; 4 is 2 rad/s.
        tilt_rate_cap = 4.0
        # Slowest commanded speed [m/s] that cost_of_transport divides by.
        cost_of_transport_min_speed = 0.5
        # Most that cost_of_transport charges per step; walking is near 0.6.
        cost_of_transport_cap = 5.0


class Go2OscPushGlideRunnerCfg(Go2OscORC111GRFDirectionalRunnerCfg):
    class actor(Go2OscORC111GRFDirectionalRunnerCfg.actor):
        obs = PHASE_OBSERVATIONS + ["foot_traction_obs"]

    class critic(Go2OscORC111GRFDirectionalRunnerCfg.critic):
        obs = Go2OscORC111RunnerCfg.critic.obs + [
            "foot_traction_obs",
            "contact_friction_obs",
        ]

        class reward(Go2OscORC111GRFDirectionalRunnerCfg.critic.reward):
            class weights(Go2OscORC111GRFDirectionalRunnerCfg.critic.reward.weights):
                swing_grf = 0.0
                stance_grf = 0.0
                push_load = 5.0
                glide_drag = 5.0
                # Vertical base speed: separates a grounded gait from hopping.
                lin_vel_z = 2.0
                # Squared roll and pitch rate: damps the rocking of the trot.
                tilt_rate = 1.0
                # Joint power per weight and commanded speed; the Cot tasks use it.
                cost_of_transport = 0.0

    class runner(Go2OscORC111GRFDirectionalRunnerCfg.runner):
        experiment_name = "go2osc_push_glide"


class Go2OscPushGlideF075Cfg(Go2OscPushGlideCfg):
    class osc(Go2OscPushGlideCfg.osc):
        moving_frequency_scale = 0.75


class Go2OscPushGlideF075RunnerCfg(Go2OscPushGlideRunnerCfg):
    class runner(Go2OscPushGlideRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_f075"


class Go2OscPushGlideF050Cfg(Go2OscPushGlideCfg):
    class osc(Go2OscPushGlideCfg.osc):
        moving_frequency_scale = 0.5


class Go2OscPushGlideF050RunnerCfg(Go2OscPushGlideRunnerCfg):
    class runner(Go2OscPushGlideRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_f050"


class Go2OscPushGlideCotCfg(Go2OscPushGlideCfg):
    class reward_settings(Go2OscPushGlideCfg.reward_settings):
        # Hold the commanded speed [m/s], so that energy is saved by changing
        # the gait and not by walking slower.
        velocity_tracking_tolerance = 0.5


class Go2OscPushGlideCotRunnerCfg(Go2OscPushGlideRunnerCfg):
    class critic(Go2OscPushGlideRunnerCfg.critic):
        class reward(Go2OscPushGlideRunnerCfg.critic.reward):
            class weights(Go2OscPushGlideRunnerCfg.critic.reward.weights):
                cost_of_transport = 1.0

    class runner(Go2OscPushGlideRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_cot"


class Go2OscPushGlideCotMuCfg(Go2OscPushGlideCotCfg):
    pass


class Go2OscPushGlideCotMuRunnerCfg(Go2OscPushGlideCotRunnerCfg):
    class actor(Go2OscPushGlideCotRunnerCfg.actor):
        # The true friction coefficient: the best case for any estimator.
        obs = Go2OscPushGlideCotRunnerCfg.actor.obs + ["contact_friction_obs"]

    class runner(Go2OscPushGlideCotRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_cot_mu"


class Go2OscPushGlideRobustCfg(Go2OscPushGlideCotCfg):
    """Energy-priced walking in both directions, with what hardware adds.

    Sensor errors, a delay before targets reach the motors, varied mass and
    motor gains, harder pushes, and ground friction that changes during an
    episode and differs between feet. Every range below is an assumption to
    replace with measurements from the robot.
    """

    class env(Go2OscPushGlideCotCfg.env):
        # Two full commands per episode: the second starts from speed the
        # robot built itself, on whatever ground it is on by then.
        episode_length_s = 6.0

    class commands(Go2OscPushGlideCotCfg.commands):
        # Backward as well as forward [m/s]. Fast commands are worth chasing
        # on low friction because speed tracking is measured against a
        # friction-limited reference (reward_settings.speed_reference).
        forward_uniform_range = [-1.0, 2.0]
        # Sliding stops. A robot moving faster than sliding_stop_min_speed
        # [m/s] when it gets a new command (episode start or command change)
        # is, with this probability, told to stop instead: zero speed and a
        # turn rate drawn over the whole range.
        sliding_stop_probability = 0.5
        sliding_stop_min_speed = 0.5
        # The stop is held this long [s]; then a new command is drawn.
        sliding_stop_duration = 1.5
        # Friction budget for slowing down, as a fraction of friction times
        # gravity: the speed target falls at this rate and is tracked both
        # ways. At 0.9 a stop is a real slide, about 1.1 times the shortest
        # possible one. A carved turn shares the budget (reward_settings).
        # A range [low, high] is drawn per command.
        braking_range = [0.9, 0.9]

        class ranges(Go2OscPushGlideCotCfg.commands.ranges):
            lin_vel_x = [-1.0, 2.0]

    class osc(Go2OscPushGlideCotCfg.osc):
        # The clocks follow a forward speed target v_t that approaches the
        # command at the rates of the speed reference, from the friction
        # estimate (at least clock_speed_min_friction). The frequency is
        # 1.5 + |v_t| Hz, and the standing branch waits until |v_t| <= 0.2.
        clock_follows_speed_target = True
        clock_speed_min_friction = 0.02

    class init_state(Go2OscPushGlideCotCfg.init_state):
        # Episodes also begin moving backward.
        root_vel_range = [
            [-1.5, 2.5],  # x
            [0.0, 0.0],  # y
            [-0.05, 0.05],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(Go2OscPushGlideCotCfg.control):
        # Time from a joint target leaving the policy to the motors acting on
        # it [s]; one value per robot and episode.
        actuation_delay_range = [0.0, 0.02]

    class push_robots(Go2OscPushGlideCotCfg.push_robots):
        interval_s = 5
        max_push_vel_xy = 0.5

    class domain_randomization(Go2OscPushGlideCotCfg.domain_randomization):
        class startup(Go2OscPushGlideCotCfg.domain_randomization.startup):
            link_mass_scale_range = [0.9, 1.1]
            contact_friction_range = [0.05, 1.0]

        class episode(Go2OscPushGlideCotCfg.domain_randomization.episode):
            scale_ranges = {
                "p_gains": [0.8, 1.2],
                "d_gains": [0.8, 1.2],
            }

    class fast_slides:
        # Share of fresh episodes that start sliding along the heading at a
        # speed from speed_range [m/s] on ground of friction from
        # friction_range, told to stop with a turn rate (a sliding stop).
        probability = 0.2
        friction_range = [0.05, 0.3]
        speed_range = [1.5, 2.5]

    class falls:
        # A robot tilted further than this [deg] has fallen: the episode
        # ends as when the base touches the ground.
        max_tilt = 60.0

    class friction_changes:
        # Mean number of redraws of the ground friction per second.
        rate = 0.2
        # Share of draws in which each foot gets its own friction.
        split_probability = 0.3

    class sensing:
        # Friction estimate the actor receives: the mean friction under the
        # feet through a first-order lag [s], times a gain fixed for the
        # episode, plus noise at every control step.
        friction_estimate_gain_range = [0.7, 1.4]
        friction_estimate_noise_std = 0.05
        friction_estimate_time_constant = 0.5
        # Normal load the clocks receive: gain per foot and episode, noise at
        # every physics step [body weights].
        foot_load_gain_range = [0.8, 1.2]
        foot_load_noise_std = 0.05
        # Gain per foot and episode on the traction ratios the actor observes.
        foot_traction_gain_range = [0.7, 1.3]
        # First-order lag on both foot-force signals [s].
        foot_force_time_constant = 0.02

    class scaling(Go2OscPushGlideCotCfg.scaling):
        friction_estimate_obs = 1.0
        foot_friction_obs = 1.0
        clock_speed_obs = 3.0

    class reward_settings(Go2OscPushGlideCotCfg.reward_settings):
        # Charge energy per metre actually made along the command.
        cost_of_transport_speed = "travelled"
        # Measure speed tracking against a reference that moves toward the
        # command at these fractions of friction times gravity when speeding
        # up and when braking, at most the caps [m/s^2].
        speed_reference = True
        speed_reference_gain = 0.5
        speed_reference_brake = 0.9
        speed_reference_max_gain = 2.0
        speed_reference_max_brake = 5.0
        # Turning faster than commanded by more than this [rad/s] is
        # penalised (reward term yaw_overshoot).
        yaw_overshoot_tolerance = 0.25
        # Told zero planar speed and a turn rate, the robot carves: its path
        # turns at the commanded rate, capped by friction, and turning and
        # braking together use the braking budget (speed times turn rate
        # sideways, the rest to slow down). At least this share of the
        # budget always brakes.
        speed_reference_carve = True
        speed_reference_carve_min_brake = 0.5
        # Reward term standing, while the clocks stand: planar speed squared,
        # plus this weight times yaw rate squared, plus this weight per foot
        # carrying less than standing_lift_load body weights.
        standing_turn_weight = 0.5
        standing_lift_weight = 0.25
        standing_lift_load = 0.05


class Go2OscPushGlideRobustRunnerCfg(Go2OscPushGlideCotRunnerCfg):
    class actor(Go2OscPushGlideCotRunnerCfg.actor):
        # A friction estimate with errors, and the body velocity.
        obs = Go2OscPushGlideCotRunnerCfg.actor.obs + [
            "friction_estimate_obs",
            "base_lin_vel",
            "clock_speed_obs",
        ]

        class noise(Go2OscPushGlideCotRunnerCfg.actor.noise):
            # Half-widths of uniform noise, in the units of each signal.
            base_ang_vel = 0.2  # [rad/s]
            projected_gravity = 0.05
            dof_pos_obs = 0.01  # [rad]
            dof_vel = 0.5  # [rad/s]
            foot_traction_obs = 0.1
            base_lin_vel = 0.1  # [m/s]

    class critic(Go2OscPushGlideCotRunnerCfg.critic):
        # The true friction under each foot.
        obs = Go2OscPushGlideCotRunnerCfg.critic.obs + [
            "foot_friction_obs",
            "clock_speed_obs",
        ]

        class reward(Go2OscPushGlideCotRunnerCfg.critic.reward):
            class weights(Go2OscPushGlideCotRunnerCfg.critic.reward.weights):
                cost_of_transport = 3.0
                # As much as speed tracking, so that a turn commanded during a
                # sliding stop is worth following.
                tracking_ang_vel = 4.0
                # Spinning out: 3 rad/s against a 0.5 command costs 5 per second.
                yaw_overshoot = 1.0
                # Standing still while the clocks stand (see reward_settings).
                standing = 2.0

            class termination_weight(
                Go2OscPushGlideCotRunnerCfg.critic.reward.termination_weight
            ):
                # Applied once per fall, without time integration, on top of
                # the reward the fallen robot no longer collects.
                termination = 5.0

    class runner(Go2OscPushGlideCotRunnerCfg.runner):
        experiment_name = "go2osc_push_glide_robust"
