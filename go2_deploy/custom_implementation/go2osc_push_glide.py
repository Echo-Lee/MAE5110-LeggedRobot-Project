"""Standalone push-glide observations and control state for the Go2 staff runner.

The actor consumes IMU and joint measurements, a governed command, previous
joint targets, four oscillator phases and learned friction/velocity estimates.
All observation arithmetic and state updates live here and use only PyTorch
and the runner's CustomConfig interface. Robot communication stays in the
staff runner; importing this module neither loads the simulator nor the SDK.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import torch

from go2_deploy.custom_implementation.custom_config import CustomConfig

# Actor input order: 58 scalars in canonical FL, FR, RL, RR joint order.
ACTOR_OBSERVATIONS = [
    "base_ang_vel",
    "projected_gravity",
    "governed_commands",
    "dof_pos_obs",
    "dof_vel",
    "dof_pos_target",
    "oscillator_obs",
    "contact_friction_hat",
    "base_lin_vel_hat",
    "clock_speed_obs",
]
ACTOR_OBSERVATION_SIZES = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "governed_commands": 3,
    "dof_pos_obs": 12,
    "dof_vel": 12,
    "dof_pos_target": 12,
    "oscillator_obs": 8,
    "contact_friction_hat": 1,
    "base_lin_vel_hat": 3,
    "clock_speed_obs": 1,
}


LEGS = ("FL", "FR", "RL", "RR")
GRAVITY_M_S2 = 9.81
# Damping of the Jacobian inverse [m], force clip [body weights] and
# traction-ratio clip: the values go2osc_est.py uses.
JACOBIAN_DAMPING = 0.01
FOOT_FORCE_CLIP = 3.0
TRACTION_RATIO_CLIP = 2.0
# Oscillator phases a controller starts from: the standing clocks' rest point.
STANDING_PHASE = 1.5 * math.pi
# Names of the policy inputs this module provides, and their sizes.
ONBOARD_OBSERVATIONS = {
    name: ACTOR_OBSERVATION_SIZES[name]
    for name in (
        "oscillator_obs",
        "contact_friction_hat",
        "base_lin_vel_hat",
        "clock_speed_obs",
        "governed_commands",
    )
}
GO2_URDF = Path(__file__).resolve().parents[2] / "resources/robots/go2/urdf/go2.urdf"


def leg_geometry(urdf_path=GO2_URDF):
    """Joint and foot offsets of the four legs, read from the URDF [m].

    ``{"hip": [4, 3], "thigh": [4, 3], "calf": [4, 3], "foot": [4, 3]}``:
    each joint's origin in its parent link, and the centre of the foot's
    collision sphere in the calf link. The hip turns about x, the thigh and
    calf about y; every joint frame is unrotated at zero angle.
    """
    root = ET.parse(urdf_path).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    links = {link.attrib["name"]: link for link in root.findall("link")}

    def origin(element):
        node = element.find("origin")
        if node is None:
            return [0.0, 0.0, 0.0]
        if any(abs(float(v)) > 1e-9 for v in node.attrib.get("rpy", "0 0 0").split()):
            raise ValueError("leg_geometry expects unrotated joint frames")
        return [float(v) for v in node.attrib.get("xyz", "0 0 0").split()]

    axes = {"hip": "1 0 0", "thigh": "0 1 0", "calf": "0 1 0"}
    geometry = {"hip": [], "thigh": [], "calf": [], "foot": []}
    for leg in LEGS:
        for part, axis in axes.items():
            joint = joints[f"{leg}_{part}_joint"]
            if joint.find("axis").attrib["xyz"].split() != axis.split():
                raise ValueError(f"unexpected axis for {leg}_{part}_joint")
            geometry[part].append(origin(joint))
        foot = origin(joints[f"{leg}_foot_joint"])
        sphere = origin(links[f"{leg}_foot"].find("collision"))
        geometry["foot"].append([a + b for a, b in zip(foot, sphere)])
    return geometry


def _rotate_x(angle, vector):
    cos, sin = angle.cos(), angle.sin()
    x, y, z = vector.unbind(dim=-1)
    return torch.stack((x, cos * y - sin * z, sin * y + cos * z), dim=-1)


def _rotate_y(angle, vector):
    cos, sin = angle.cos(), angle.sin()
    x, y, z = vector.unbind(dim=-1)
    return torch.stack((cos * x + sin * z, y, -sin * x + cos * z), dim=-1)


def leg_kinematics(dof_pos, geometry):
    """(J [N, 4, 3, 3], foot position [N, 4, 3]) in the base frame.

    ``dof_pos`` [N, 12] are absolute joint angles. J is the geometric
    Jacobian of the foot point: column j = axis_j x (foot - joint_j).
    """
    q = dof_pos.reshape(-1, 4, 3)
    hip, thigh, calf = q.unbind(dim=-1)
    tensor = {
        name: torch.as_tensor(value, dtype=q.dtype, device=q.device)
        for name, value in geometry.items()
    }
    knee = thigh + calf
    # Positions relative to the hip joint, built from the foot upward.
    shank = _rotate_y(knee, tensor["foot"].expand_as(q))
    below_thigh = _rotate_y(thigh, tensor["calf"].expand_as(q)) + shank
    below_hip = _rotate_x(hip, tensor["thigh"] + below_thigh)
    foot = tensor["hip"] + below_hip
    x_axis = torch.zeros_like(q)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(q)
    y_axis[..., 1] = 1.0
    pitch_axis = _rotate_x(hip, y_axis)
    columns = torch.stack(
        (
            torch.cross(x_axis, below_hip, dim=-1),
            torch.cross(pitch_axis, _rotate_x(hip, below_thigh), dim=-1),
            torch.cross(pitch_axis, _rotate_x(hip, shank), dim=-1),
        ),
        dim=-1,
    )
    return columns, foot


def leg_features(jacobian, lever, torques, dof_vel, gyro, weight, min_load):
    """Foot forces and velocities from joint torques and speeds.

    Returns, as go2osc_est.py computes them:
        force     [N, 12] ground force on each foot, F = -(J^T)^-1 tau
                          [body weights], clipped
        ratio     [N, 8]  tangential / normal force (x, y); 0 unloaded
        velocity  [N, 12] foot velocity relative to the base origin,
                          gyro x r + J qdot [m/s]
        stroke    [N, 4, 3] J qdot alone: the foot's velocity as the turning
                          base sees it [m/s]
    ``weight`` [N] is the robot's weight [N].
    """
    n = torques.shape[0]
    tau = torques.reshape(n, 4, 3).unsqueeze(-1)
    damping = JACOBIAN_DAMPING**2 * torch.eye(3, dtype=tau.dtype, device=tau.device)
    gram = jacobian.transpose(-1, -2) @ jacobian + damping
    force = -(jacobian @ torch.linalg.solve(gram, tau)).squeeze(-1)
    force = (force / weight.reshape(n, 1, 1)).clamp(-FOOT_FORCE_CLIP, FOOT_FORCE_CLIP)
    normal = force[..., 2]
    loaded = normal > min_load
    ratio = (force[..., :2] / normal.clamp(min=1e-6).unsqueeze(-1)).clamp(
        -TRACTION_RATIO_CLIP, TRACTION_RATIO_CLIP
    )
    ratio = torch.where(loaded.unsqueeze(-1), ratio, torch.zeros_like(ratio))
    stroke = (jacobian @ dof_vel.reshape(n, 4, 3).unsqueeze(-1)).squeeze(-1)
    omega = gyro.unsqueeze(1).expand(n, 4, 3)
    velocity = torch.cross(omega, lever, dim=-1) + stroke
    return force.reshape(n, 12), ratio.reshape(n, 8), velocity.reshape(n, 12), stroke


def clock_schedule(clock_speed, command, settings):
    """(omega, coupling, offset, moving, translating) of Equation (5).

    ``clock_speed`` [N, 1] is the speed target v_t, ``command`` [N, 3].
    The standing clocks run only when v_t is near zero and no sidestep or
    turn is commanded.
    """
    speed = clock_speed.abs()
    slow = speed <= settings["stop_threshold"]
    steering = command[:, 1:3].abs()
    slow = (
        slow
        & (steering[:, :1] <= settings["sideways_stop_threshold"])
        & (steering[:, 1:] <= settings["turning_stop_threshold"])
    )
    omega = torch.where(
        slow,
        torch.full_like(speed, settings["omega_stop"]),
        (settings["omega_offset"] + speed).clamp(max=settings["omega_max"]),
    )
    coupling = torch.where(
        slow,
        torch.full_like(speed, settings["coupling_stop"]),
        torch.full_like(speed, settings["coupling_moving"]),
    )
    offset = slow.to(speed.dtype) * settings["stance_offset"]
    translating = (speed > settings["stop_threshold"]) | (
        steering[:, :1] > settings["sideways_stop_threshold"]
    )
    return omega, coupling, offset, ~slow, translating


def push_signal(load, stroke, clock_speed, command, translating, gate_speed):
    """Load on each foot [N, 4], gated to feet stroking against the travel.

    ``stroke`` [N, 4, 3] is each foot's velocity seen from the base. A foot
    moving backward relative to commanded travel is pushing (gate near 1);
    one swinging forward is not (near 0). Without a travel direction (a
    turn in place, standing) the load passes ungated.
    """
    travel = torch.zeros_like(command)
    travel[:, :1] = clock_speed
    travel[:, 1] = command[:, 1]
    heading = travel / travel.norm(dim=1, keepdim=True).clamp(min=1.0e-6)
    along = (stroke * heading.unsqueeze(1)).sum(dim=-1)
    gate = torch.sigmoid(-along / gate_speed)
    return torch.where(translating, load * gate, load)


def carve(speed, budget, command, settings):
    """(turn rate to follow [rad/s], braking rate [m/s^2]) of a stop.

    Told zero planar speed and a turn rate, the robot carves: turning
    takes speed times turn rate of the braking budget, the rest slows it.
    """
    turn = command[:, 2]
    if not settings["carve"]:
        return turn, budget
    carving = (command[:, :2].abs().sum(dim=1) == 0.0) & (turn != 0.0)
    sideways_limit = budget * (1.0 - settings["carve_min_brake"] ** 2) ** 0.5
    cap = sideways_limit / speed.clamp(min=1.0e-6)
    capped = torch.maximum(torch.minimum(turn, cap), -cap)
    sideways = (speed * capped.abs()).clamp(max=sideways_limit)
    carved = (budget.square() - sideways.square()).clamp(min=0.0).sqrt()
    return torch.where(carving, capped, turn), torch.where(carving, carved, budget)


def govern_command(command, forward_speed, margin):
    """The command [N, 3] with its forward part kept near the forward speed.

    The policy cannot tell ice from grip while the robot stands, so a fast
    command from rest makes it push harder than ice allows. The forward
    command handed to the policy and to the speed target therefore leads
    the estimated forward speed ``forward_speed`` [N] by at most ``margin``
    [m/s], in the commanded direction. Slowing down is never limited.
    ``margin`` None returns the command itself.
    """
    governed = command.clone()
    if margin is None:
        return governed
    high = forward_speed.clamp(min=0.0) + margin
    low = forward_speed.clamp(max=0.0) - margin
    governed[:, 0] = torch.maximum(torch.minimum(command[:, 0], high), low)
    return governed


def advance_clock_speed(clock_speed, command, friction, dt, settings):
    """Move v_t [N, 1] one control step toward the forward command.

    Speeding up at ``gain`` times friction times gravity, slowing at the
    braking share of it less what a carved turn takes; each with a cap.
    """
    target = command[:, 0]
    speed = clock_speed[:, 0]
    limit = friction.clamp(min=settings["min_friction"]) * GRAVITY_M_S2
    speeding = (target.abs() > speed.abs()) & (target * speed >= 0.0)
    budget = (settings["brake"] * limit).clamp(max=settings["max_brake"])
    _, brake = carve(speed.abs(), budget, command, settings)
    rate = torch.where(
        speeding, (settings["gain"] * limit).clamp(max=settings["max_gain"]), brake
    )
    step = rate * dt
    change = torch.maximum(torch.minimum(target - speed, step), -step)
    return (speed + change).unsqueeze(1)


class Go2OscOnboard:
    """Oscillators, speed target and state estimator of N robots.

    ``settings`` is the plain dictionary ``onboard_settings`` in
    go2osc_deploy.py builds from the training configuration (saved in the
    policy bundle); ``estimator`` maps a history window [N, frames *
    features] to (mu, vx, vy, vz) [N, 4] in physical units.

    ``reset`` starts a run; ``step`` advances one control step. Both
    return the policy inputs this module owns (ONBOARD_OBSERVATIONS), in
    physical units.
    """

    def __init__(self, settings, estimator, num_robots=1, device="cpu"):
        self.settings = settings
        self.estimator = estimator
        self.num = num_robots
        self.device = torch.device(device)
        self.geometry = settings["leg_geometry"]
        self.dt = float(settings["dt"])
        self.substeps = int(settings["substeps"])
        layout = settings["estimator"]
        self.features = list(layout["features"])
        self.frames, self.stride = int(layout["frames"]), int(layout["stride"])
        self.feature_scales = {
            name: torch.tensor(scale, dtype=torch.float, device=self.device)
            for name, scale in layout["scales"].items()
        }
        self.default_dof_pos = torch.tensor(
            settings["default_dof_pos"], dtype=torch.float, device=self.device
        )
        self.weight = torch.full(
            (self.num,), float(settings["weight"]), device=self.device
        )
        n = self.num
        self.phase = torch.full((n, 4), STANDING_PHASE, device=self.device)
        self.clock_speed = torch.zeros(n, 1, device=self.device)
        self.friction = torch.zeros(n, 1, device=self.device)
        self.velocity = torch.zeros(n, 3, device=self.device)
        self.push = torch.zeros(n, 4, device=self.device)
        self.load = torch.zeros(n, 4, device=self.device)
        self.command = torch.zeros(n, 3, device=self.device)
        self.governed = torch.zeros(n, 3, device=self.device)
        self.command_margin = settings["speed_target"].get("command_margin")
        self.history = None

    # -- sensors -> features ------------------------------------------------

    def _measure(self, sensors, command):
        """Estimator frame [N, F] and the legs' (normal load, stroke)."""
        jacobian, lever = leg_kinematics(sensors["dof_pos"], self.geometry)
        force, ratio, velocity, stroke = leg_features(
            jacobian,
            lever,
            sensors["torques"],
            sensors["dof_vel"],
            sensors["base_ang_vel"],
            self.weight,
            self.settings["osc"]["directional_min_load"],
        )
        values = {
            "base_ang_vel": sensors["base_ang_vel"],
            "projected_gravity": sensors["projected_gravity"],
            "commands": command,
            "dof_pos_obs": sensors["dof_pos"] - self.default_dof_pos,
            "dof_vel": sensors["dof_vel"],
            "dof_pos_target": sensors["dof_pos_target"],
            "oscillator_obs": self._oscillator_obs(),
            "base_lin_acc": sensors["base_lin_acc"],
            "torques": sensors["torques"],
            "foot_force_est": force,
            "foot_traction_est": ratio,
            "foot_vel_rel_base": velocity,
        }
        pieces = []
        for name in self.features:
            value = values[name]
            if name in self.feature_scales:
                value = value / self.feature_scales[name]
            pieces.append(value)
        normal = force.reshape(self.num, 4, 3)[..., 2].clamp(0.0, 1.0)
        return torch.cat(pieces, dim=1), normal, stroke

    def _oscillator_obs(self):
        return torch.cat((self.phase.cos(), self.phase.sin()), dim=1)

    def _estimate(self):
        window = self.history[:, self.stride - 1 :: self.stride].reshape(self.num, -1)
        with torch.no_grad():
            out = self.estimator(window)
        low, high = self.settings["friction_clip"]
        self.friction = out[:, :1].clamp(low, high)
        self.velocity = out[:, 1:4]

    def _refresh_push(self, normal, stroke, command, restart=False):
        lag = self.settings["load_time_constant"]
        if restart:
            self.load = normal.clone()
        else:
            self.load = self.load + (self.dt / (lag + self.dt)) * (normal - self.load)
        _, _, _, _, translating = clock_schedule(
            self.clock_speed, command, self.settings["osc"]
        )
        self.push = push_signal(
            self.load.clamp(0.0, 1.0),
            stroke,
            self.clock_speed,
            command,
            translating,
            self.settings["osc"]["push_gate_speed"],
        )

    def observations(self):
        return {
            "oscillator_obs": self._oscillator_obs(),
            "contact_friction_hat": self.friction,
            "base_lin_vel_hat": self.velocity,
            "clock_speed_obs": self.clock_speed,
            "governed_commands": self.governed,
        }

    # -- run ------------------------------------------------------------------

    def _govern(self, command):
        self.governed = govern_command(
            command, self.velocity[:, 0], self.command_margin
        )

    def reset(self, sensors, command, phase=None, clock_speed=None):
        """Start from the robot's present state; returns the policy inputs.

        A robot starts at rest, so v_t starts at zero. ``clock_speed`` [N, 1]
        is for a simulated episode that begins already moving.
        """
        self.phase = (
            torch.full((self.num, 4), STANDING_PHASE, device=self.device)
            if phase is None
            else phase.clone()
        )
        self.command = command.clone()
        frame, normal, stroke = self._measure(sensors, command)
        length = self.frames * self.stride
        self.history = frame.unsqueeze(1).repeat(1, length, 1)
        self._estimate()
        # One repeated frame carries no motion, so v-hat cannot start v_t.
        self.clock_speed = (
            torch.zeros(self.num, 1, device=self.device)
            if clock_speed is None
            else clock_speed.clone()
        )
        self._govern(command)
        self._refresh_push(normal, stroke, command, restart=True)
        return self.observations()

    def step(self, sensors, command):
        """One control step: sensors and command now -> the policy inputs."""
        osc = self.settings["osc"]
        # The step that just ended ran on the schedule and push signal set
        # at its start.
        omega, coupling, offset, _, _ = clock_schedule(
            self.clock_speed, self.command, osc
        )
        dt = self.dt / self.substeps
        for _ in range(self.substeps):
            drift = omega - self.push * coupling * (self.phase.cos() + offset)
            self.phase = (self.phase + dt * 2 * math.pi * drift).remainder(2 * math.pi)
        self._govern(command)
        self.clock_speed = advance_clock_speed(
            self.clock_speed,
            self.governed,
            self.friction[:, 0],
            self.dt,
            self.settings["speed_target"],
        )
        frame, normal, stroke = self._measure(sensors, command)
        self.history = torch.cat((self.history[:, 1:], frame.unsqueeze(1)), dim=1)
        self._estimate()
        self._refresh_push(normal, stroke, command)
        self.command = command.clone()
        return self.observations()


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by the inverse of quaternion q. q is [x, y, z, w]."""
    q_w = q[..., 3:]  # [..., 1]
    q_vec = q[..., :3]  # [..., 3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


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
    obs_vector = ACTOR_OBSERVATIONS
    custom_obs_sizes = ACTOR_OBSERVATION_SIZES
    default_dof_pos = 4 * [0.0, 0.66, -1.36]

    class DeployScaling:
        # Observations are divided by these physical-unit scales. The actor's
        # joint-target outputs are multiplied by dof_pos_target exactly once.
        base_ang_vel = [0.3, 0.3, 0.1]
        projected_gravity = 1.0
        governed_commands = [3, 1, 3]
        dof_pos_obs = 4 * [0.2, 0.3, 0.3]
        dof_vel = 4 * [2.0, 2.0, 4.0]
        dof_pos_target = 4 * [0.2, 0.3, 0.3]
        oscillator_obs = 1.0
        contact_friction_hat = 1.0
        base_lin_vel_hat = 0.4
        clock_speed_obs = 3.0

        # Physical features also used by the estimator and legacy bundles.
        base_height = 0.3
        base_lin_acc = 9.81
        base_lin_vel = 0.4
        commands = [3, 1, 3]
        contact_friction_obs = 1.0
        dof_pos = 4 * [0.2, 0.3, 0.3]
        foot_friction_obs = 1.0
        foot_grf_angle = 1.0
        foot_traction_obs = 1.0
        friction_estimate_obs = 1.0
        oscillators_vel = 1.0
        tau_ff = 4 * [23.7, 23.7, 45.43]

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
