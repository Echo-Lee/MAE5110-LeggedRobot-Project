from gym.envs.go2.go2trot_config import Go2TrotCfg
import torch
from go2_deploy.utility.import_custom_impl import custom_impl


class DeployConfig:
    task_name = "go2osc_push_glide_robust"
    experiment_name = "go2osc_push_glide_robust_carve_4000"
    load_run = "Nov20_18-29-45_"
    checkpoint = 4000

    ctrl_freq = 100  # Hz

    kp = 30.0  # Stiffness constant
    kd = 2.0  # Damping constant

    # Apply exponential moving average to actions
    exp_moving_avg = False
    ema_smoothing_factor = 0.1

    obs_vector = custom_impl.CustomDeployConfig.obs_vector

    # Scale observations
    class DeployScaling(custom_impl.CustomDeployConfig.DeployScaling):
        pass

    # Set ranges of joystick commands
    command_limits = {
        "lin_vel_x": 2.0, "lin_vel_y": 0.3, "yaw_vel": 0.5,
    }

    # Specify size of observation vector. should not have to modify this
    obs_sizes = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "commands": 3,
        "dof_pos_obs": 12,
        "dof_vel": 12,
        "dof_accel": 12,
        "dof_pos_target": 12,
    }
    obs_sizes.update(custom_impl.CustomDeployConfig.custom_obs_sizes)

    # Canonical FL, FR, RL, RR order; hip, thigh, calf within each leg.
    default_dof_pos = torch.tensor(
        [
            angle
            for leg in ("FL", "FR", "RL", "RR")
            for angle in (
                Go2TrotCfg.init_state.default_joint_angles["hip_joint"],
                Go2TrotCfg.init_state.default_joint_angles[f"{leg}_thigh_joint"],
                Go2TrotCfg.init_state.default_joint_angles["calf_joint"],
            )
        ]
    )

    # s: how often terminal should print ctrl freq, instructions
    terminal_log_period = 2.0

    # Check for unsafe config
    def __init__(self):
        if self.ctrl_freq < 20:
            raise ValueError("ctrl_freq should be > 20")
        if self.kp <= 0:
            raise ValueError("kp should be > 0")
        if self.kd <= 0:
            raise ValueError("kd should be > 0")
        for obs in self.obs_vector:
            if obs not in self.obs_sizes:
                raise KeyError("observation " + obs + " not supported")
