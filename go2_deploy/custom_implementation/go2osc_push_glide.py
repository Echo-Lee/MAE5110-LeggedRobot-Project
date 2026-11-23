"""Staff-runner adapter for the push/glide policy and its on-board estimator.

This module reads LowState messages and controller buffers. It neither imports
the Unitree SDK nor sends commands; the staff runner owns robot communication.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch

from go2_deploy.custom_implementation.custom_config import CustomConfig
from gym.envs.go2.go2osc_deploy_config import (
    DEPLOY_ACTOR_OBSERVATIONS,
    Go2OscPushGlideDeployCfg,
)
from gym.envs.go2.go2osc_onboard import ONBOARD_OBSERVATIONS, Go2OscOnboard
from gym.utils.torch_quat import quat_rotate_inverse

BUNDLE_VARIABLE = "GO2OSC_BUNDLE"
ONBOARD_FILE = "go2osc_onboard.json"
ESTIMATOR_FILE = "estimator.pt"
UNITREE_TO_CANONICAL = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
STICK_DEADBAND = 0.05


def bundle_directory():
    """Use the runner's explicit bundle, or the adapter's environment override."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle", type=Path)
    arguments, _ = parser.parse_known_args(sys.argv[1:])
    directory = arguments.bundle or os.environ.get(BUNDLE_VARIABLE)
    if not directory:
        raise RuntimeError("go2osc_push_glide requires a policy bundle (--bundle)")
    return Path(directory).expanduser().resolve()


def load_onboard(directory):
    """Load the exported settings and their hash-checked estimator on the CPU."""
    directory = Path(directory)
    settings = json.loads((directory / ONBOARD_FILE).read_text())
    estimator_path = directory / ESTIMATOR_FILE
    digest = hashlib.sha256(estimator_path.read_bytes()).hexdigest()
    if digest != settings["estimator_sha256"]:
        raise ValueError(f"{estimator_path} does not match the exported estimator hash")
    estimator = torch.jit.load(str(estimator_path), map_location="cpu").eval()
    return Go2OscOnboard(settings, estimator), settings


def check_runner_settings(cfg, settings):
    """The runner's timing and applied actions must match the exported policy."""
    expected = 1.0 / float(settings["dt"])
    if not math.isclose(float(cfg.ctrl_freq), expected, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"go2osc_push_glide needs a {expected:g} Hz control loop")
    if cfg.exp_moving_avg:
        raise RuntimeError("go2osc_push_glide was trained without action smoothing")


def read_sensors(main_controller, lowstate_msg):
    """Physical-unit observations in canonical FL, FR, RL, RR joint order."""
    motors = torch.tensor(
        [[motor.q, motor.dq, motor.tau_est] for motor in lowstate_msg.motor_state[:12]],
        dtype=torch.float,
    ).index_select(0, UNITREE_TO_CANONICAL)
    imu = lowstate_msg.imu_state
    quaternion = torch.tensor(imu.quaternion, dtype=torch.float)[[1, 2, 3, 0]]
    gravity = quat_rotate_inverse(
        quaternion.unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]])
    )
    return {
        "dof_pos": motors[:, 0].unsqueeze(0),
        "dof_vel": motors[:, 1].unsqueeze(0),
        "torques": motors[:, 2].unsqueeze(0),
        "base_ang_vel": torch.tensor(imu.gyroscope, dtype=torch.float).unsqueeze(0),
        "projected_gravity": gravity,
        "base_lin_acc": torch.tensor(imu.accelerometer, dtype=torch.float).unsqueeze(0),
        "dof_pos_target": main_controller.last_action[0].detach().clone().unsqueeze(0),
    }


def read_command(main_controller):
    """Clip physical stick commands to the trained envelope, with a centre band."""
    remote = main_controller.remote_controller
    values = torch.tensor(
        [remote.lin_vel_x, remote.lin_vel_y, remote.yaw_vel], dtype=torch.float
    )
    limits = torch.tensor(
        [
            main_controller.cfg.command_limits[name]
            for name in ("lin_vel_x", "lin_vel_y", "yaw_vel")
        ],
        dtype=torch.float,
    )
    trained = torch.tensor(main_controller.go2osc_settings["command_range"])
    low, high = torch.maximum(trained[0], -limits), torch.minimum(trained[1], limits)
    values = torch.maximum(torch.minimum(values, high), low)
    return torch.where(values.abs() <= STICK_DEADBAND * limits, 0.0, values)


def standing_report(onboard, sensors):
    loads = onboard.load[0]
    accelerometer = sensors["base_lin_acc"][0].norm().item()
    velocity = [round(value, 3) for value in onboard.velocity[0].tolist()]
    return (
        "go2osc check | foot load FL FR RL RR "
        f"{[round(value, 3) for value in loads.tolist()]} "
        f"(sum {loads.sum().item():.2f}) | accelerometer {accelerometer:.2f} m/s^2 "
        f"| estimated speed {velocity} m/s, friction {onboard.friction[0, 0]:.3f}"
    )


class Go2OscPushGlide(CustomConfig):
    task_name = "go2osc_push_glide_deploy"
    ctrl_freq = 100
    kp = 20.0
    kd = 0.5
    exp_moving_avg = False
    obs_vector = list(DEPLOY_ACTOR_OBSERVATIONS)
    custom_obs_sizes = dict(ONBOARD_OBSERVATIONS)
    default_dof_pos = 4 * [
        Go2OscPushGlideDeployCfg.init_state.default_joint_angles[joint]
        for joint in ("hip", "thigh", "calf")
    ]

    class DeployScaling(Go2OscPushGlideDeployCfg.scaling):
        pass

    def get_custom_buffers_dict(self):
        onboard, settings = load_onboard(bundle_directory())
        return {
            "go2osc_onboard": onboard,
            "go2osc_settings": settings,
            "go2osc_observations": onboard.observations(),
            "go2osc_command": torch.zeros(3),
            "go2osc_running": False,
            "go2osc_checked": False,
        }

    def update_custom_buffers(self, main_controller, t):
        if not main_controller.go2osc_checked:
            check_runner_settings(main_controller.cfg, main_controller.go2osc_settings)
            main_controller.go2osc_checked = True
        message = main_controller.last_lowstate_msg
        if message is None:
            return
        sensors = read_sensors(main_controller, message)
        command = read_command(main_controller).unsqueeze(0)
        running = main_controller._state.name == "CUSTOM_CTRL"
        onboard = main_controller.go2osc_onboard
        if running and main_controller.go2osc_running:
            observations = onboard.step(sensors, command)
        else:
            observations = onboard.reset(sensors, command)
        if running and not main_controller.go2osc_running:
            print(standing_report(onboard, sensors), flush=True)
        main_controller.go2osc_observations = observations
        main_controller.go2osc_command = command[0].clone()
        main_controller.go2osc_running = running

    def get_custom_obs_pieces(self):
        pieces = {
            name: (
                lambda main_controller, lowstate_msg, name=name: (
                    main_controller.go2osc_observations[name][0]
                )
            )
            for name in ONBOARD_OBSERVATIONS
        }
        # The deadbanded command as the operator gave it. The actor reads
        # governed_commands, which the on-board module derives from it.
        pieces["commands"] = lambda main_controller, lowstate_msg: (
            main_controller.go2osc_command
        )
        return pieces
