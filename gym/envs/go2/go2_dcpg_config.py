"""Configuration for the paper-faithful decentralized-oscillator Go2 task."""

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2DCPGCfg(Go2Cfg):
    class control(Go2Cfg.control):
        # Vanilla Go2 uses 100 Hz simulation.  This task deliberately matches
        # the paper's 100 Hz policy/PD rate and 500 Hz physics integration.
        desired_sim_frequency = 500

    class oscillator:
        # ORC bits are observation / reward / coupling.  Both supported modes
        # retain observation and reward; only feedback differs.
        orc_toggle = "111"

        # Independent uniform phase reset, followed by the source's optional
        # common circular shift.  No fixed inter-leg relation is specified.
        init_with_common_offset = True

        # Eq. (5) schedule, evaluated from abs(commanded forward velocity).
        stop_threshold = 0.5
        omega_stop = 1.0
        coupling_stop = 4.0
        offset_stop = 1.0
        omega_intercept = 1.5
        omega_max = 4.0
        coupling_moving = 1.0
        omega_min = 0.1

        # Official source samples these at the control rate (100 Hz).
        omega_noise_std = 0.25
        coupling_noise_std = 0.25
        # Official source samples this at each 500 Hz physics substep.
        process_noise_std = 0.25

    class reward_settings(Go2Cfg.reward_settings):
        # DCPG does not enable the existing Go2 energy regularizer.
        energy_weight = 0.0


class Go2DCPGRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        obs = Go2RunnerCfg.actor.obs + ["oscillator_obs"]

    class critic(Go2RunnerCfg.critic):
        obs = Go2RunnerCfg.critic.obs + ["oscillator_obs", "oscillators_vel"]

        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                # Disable the inherited Go2 energy term for this experiment.
                energy = 0.0
                # The official implementation uses both terms at this scale.
                swing_grf = 5.0
                stance_grf = 5.0

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2_dcpg"
