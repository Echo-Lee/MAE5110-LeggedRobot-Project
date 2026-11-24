"""Shared observation definitions and feedback math for push-glide training.

All tensors use canonical FL, FR, RL, RR order. Commands and state estimates
are expressed in physical units; observation scaling is applied by the runner.
"""

import torch

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

GRAVITY_M_S2 = 9.81


class ObservationScaling:
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

    # Physical features used by the estimator and privileged critic.
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
