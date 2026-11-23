import torch
import numpy as np
from go2_deploy.custom_implementation.custom_config import CustomConfig


class DeployConfig:
    # custom_impl: CustomConfig instance, from load_custom_impl(task_name)
    def __init__(self, custom_impl):
        if not isinstance(custom_impl, CustomConfig):
            raise TypeError(
                f"custom_impl must be a CustomConfig instance, got {custom_impl!r}"
            )
        self.custom_impl = custom_impl

        # Default parameters

        self.task_name = custom_impl.task_name

        self.ctrl_freq = custom_impl.ctrl_freq  # Hz

        self.kp = custom_impl.kp
        self.kd = custom_impl.kd

        # Apply exponential moving average to actions
        self.exp_moving_avg = custom_impl.exp_moving_avg
        self.ema_time_const = 0.095  # seconds
        self.ema_smoothing_factor = -np.expm1(
            -(1 / self.ctrl_freq) / self.ema_time_const
        )  # alpha in EMA

        self.obs_vector = list(self.custom_impl.obs_vector)

        # Scale observations
        class DeployScaling(self.custom_impl.DeployScaling):
            pass

        self.DeployScaling = DeployScaling

        # Set ranges of joystick commands
        self.command_limits = {
            "lin_vel_x": 2.0,  # m/s
            "lin_vel_y": 1.0,  # m/s
            "yaw_vel": 3.0,  # rad/s
        }

        # Specify size of observation vector pieces
        self.obs_sizes = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "commands": 3,
            "dof_pos_obs": 12,
            "dof_vel": 12,
            "dof_accel": 12,
            "dof_pos_target": 12,
        }

        self.obs_sizes.update(self.custom_impl.custom_obs_sizes)

        # Static default posture the actions are offsets from (canonical
        # FL, FR, RL, RR order; hip, thigh, calf within each leg)
        self.default_dof_pos = torch.tensor(
            self.custom_impl.default_dof_pos, dtype=torch.float
        )
        # Training run the policy came from; set by setup_actor()
        self.run_dir = None

        # s: how often terminal should print ctrl freq, instructions
        self.terminal_log_period = 2.0

        # Set by policy_bundle.load_bundle(): the bundle's deploy_meta.json
        self.bundle_meta = None

        self.validate()

    def obs_dim(self):
        return sum(self.obs_sizes[obs] for obs in self.obs_vector)

    # Check for unsafe config
    def validate(self):
        if self.ctrl_freq < 20:
            raise ValueError("ctrl_freq should be > 20")
        if self.kp <= 0:
            raise ValueError("kp should be > 0")
        if self.kd <= 0:
            raise ValueError("kd should be > 0")
        for obs in self.obs_vector:
            if obs not in self.obs_sizes:
                raise KeyError("observation " + obs + " not supported")
        if self.default_dof_pos.shape != (12,):
            raise ValueError(
                f"default_dof_pos should have 12 joint angles, "
                f"got shape {tuple(self.default_dof_pos.shape)}"
            )
