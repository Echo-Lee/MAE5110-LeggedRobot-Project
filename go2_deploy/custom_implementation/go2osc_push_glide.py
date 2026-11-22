"""On-board adapter for the 66-input robust-carve checkpoint.

It replaces simulator traction, friction, and base velocity with quantities
computed from Go2 encoders, torque estimates, and the IMU.  The actor itself
is unchanged.
"""

from pathlib import Path
import os

import torch

from gym.envs.go2.go2osc_config import PHASE_OBSERVATIONS, Go2OscPushGlideRobustCfg
from gym.envs.go2.go2osc_est_config import Go2OscPushGlideEstCfg
from gym.envs.go2.go2osc_onboard import Go2OscOnboard, ONBOARD_OBSERVATIONS, leg_geometry
from gym.utils.torch_quat import quat_rotate_inverse
from learning.modules.history_estimator import HistoryStateEstimator

UNITREE_TO_CANONICAL = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
ESTIMATOR = Path(os.environ.get(
    "GO2OSC_ESTIMATOR", "logs/go2osc_push_glide_est/state_estimator_2800/estimator.pt"
))


class CustomDeployConfig:
    # Same 66 columns and order as Go2OscPushGlideRobustRunnerCfg.actor.obs.
    obs_vector = PHASE_OBSERVATIONS + [
        "foot_traction_est", "contact_friction_hat", "base_lin_vel_hat",
        "clock_speed_obs",
    ]
    custom_obs_sizes = {**ONBOARD_OBSERVATIONS, "foot_traction_est": 8}

    class DeployScaling(Go2OscPushGlideRobustCfg.scaling):
        base_lin_acc = Go2OscPushGlideEstCfg.scaling.base_lin_acc
        contact_friction_hat = 1.0
        base_lin_vel_hat = Go2OscPushGlideEstCfg.scaling.base_lin_vel_hat


def _settings(model):
    cfg, osc, reward = Go2OscPushGlideRobustCfg, Go2OscPushGlideRobustCfg.osc, Go2OscPushGlideRobustCfg.reward_settings
    scales = {name: getattr(CustomDeployConfig.DeployScaling, name, 1.0) for name in model.config["features"]}
    return {
        "dt": 1.0 / 100.0, "substeps": 5, "leg_geometry": leg_geometry(),
        "default_dof_pos": [0.0, 0.66, -1.36] * 4, "weight": 15.7 * 9.81,
        "friction_clip": [0.02, 1.5], "load_time_constant": cfg.sensing.foot_force_time_constant,
        "estimator": {"features": model.config["features"], "frames": model.config["frames"], "stride": model.config["stride"], "scales": scales},
        "osc": {name: getattr(osc, name) for name in ("stop_threshold", "sideways_stop_threshold", "turning_stop_threshold", "omega_stop", "omega_offset", "omega_max", "coupling_stop", "coupling_moving", "stance_offset", "directional_min_load", "push_gate_speed")},
        "speed_target": {"min_friction": osc.clock_speed_min_friction, "gain": reward.speed_reference_gain, "brake": reward.speed_reference_brake, "max_gain": reward.speed_reference_max_gain, "max_brake": reward.speed_reference_max_brake, "carve": reward.speed_reference_carve, "carve_min_brake": reward.speed_reference_carve_min_brake},
    }


class _Estimator(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, window):
        mu, vel = self.model.predict(window)
        return torch.cat((mu.unsqueeze(1), vel), dim=1)


def get_custom_buffers_dict():
    model, _ = HistoryStateEstimator.load(ESTIMATOR)
    onboard = Go2OscOnboard(_settings(model), _Estimator(model).eval())
    return {"go2osc_onboard": onboard, "go2osc_observations": onboard.observations(), "go2osc_command": torch.zeros(3), "go2osc_running": False}


def _sensors(controller, message):
    motors = torch.tensor([[m.q, m.dq, m.tau_est] for m in message.motor_state[:12]]).index_select(0, UNITREE_TO_CANONICAL)
    quat = torch.tensor(message.imu_state.quaternion)[[1, 2, 3, 0]]
    gravity = quat_rotate_inverse(quat.unsqueeze(0), torch.tensor([[0., 0., -1.]]))
    return {"dof_pos": motors[:, 0].unsqueeze(0), "dof_vel": motors[:, 1].unsqueeze(0), "torques": motors[:, 2].unsqueeze(0), "base_ang_vel": torch.tensor(message.imu_state.gyroscope).unsqueeze(0), "projected_gravity": gravity, "base_lin_acc": torch.tensor(message.imu_state.accelerometer).unsqueeze(0), "dof_pos_target": controller.last_action[0].unsqueeze(0)}


def update_custom_buffers(controller, t):
    if controller.last_lowstate_msg is None: return
    command = controller.last_command.clamp(torch.tensor([-1., -.3, -.5]), torch.tensor([2., .3, .5])).unsqueeze(0)
    sensors = _sensors(controller, controller.last_lowstate_msg)
    running = controller._state.name == "CUSTOM_CTRL"
    controller.go2osc_observations = controller.go2osc_onboard.step(sensors, command) if running and controller.go2osc_running else controller.go2osc_onboard.reset(sensors, command)
    controller.go2osc_command, controller.go2osc_running = command[0], running


def get_custom_obs_pieces():
    return {
        "commands": lambda c, m: c.go2osc_command,
        "foot_traction_est": lambda c, m: c.go2osc_onboard.traction[0],
        **{name: (lambda c, m, name=name: c.go2osc_observations[name][0]) for name in ONBOARD_OBSERVATIONS},
    }
