"""Read-only check of the Go2 signals the push-glide policy depends on.

Sends nothing to the robot. With the robot standing under Unitree's own
controller, it reads LowState_ and prints, once a second, what the
on-board code makes of it:

    uv run --frozen --extra unitree_sdk python scripts/check_go2osc_signals.py \
        <nic> --bundle logs/policies/push_glide_v1

Standing still on four feet, expect about 0.25 of the weight on each foot
(sum near 1), an accelerometer near 9.81 m/s^2 and an estimated speed near
0. A sum far from 1 means the motors' torque estimate or the robot weight
in the bundle does not match the simulator.

Not run against a robot yet (no SDK where it was written); the report it
prints is covered by tests/unit_tests/test_go2osc_onboard.py.
"""

import argparse
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import torch

IMPLEMENTATION = (
    Path(__file__).resolve().parents[1]
    / "go2_deploy/custom_implementation/go2osc_push_glide.py"
)


def load_implementation():
    spec = importlib.util.spec_from_file_location("go2osc_push_glide", IMPLEMENTATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("interface", help="Robot network interface, e.g. eth0")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    arguments = parser.parse_args()

    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    implementation = load_implementation()
    onboard, _ = implementation.load_onboard(arguments.bundle)
    latest = {}

    def on_lowstate(message):
        latest["message"] = message

    ChannelFactoryInitialize(0, arguments.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(on_lowstate, 10)
    # read_sensors only needs the previous joint-target offsets from the
    # controller: zero here, since this script commands nothing.
    controller = SimpleNamespace(last_action=torch.zeros(2, 12))
    end = time.monotonic() + arguments.seconds
    while time.monotonic() < end:
        time.sleep(1.0)
        if "message" not in latest:
            print("waiting for rt/lowstate ...")
            continue
        sensors = implementation.read_sensors(controller, latest["message"])
        onboard.reset(sensors, torch.zeros(1, 3))
        print(implementation.standing_report(onboard, sensors))


if __name__ == "__main__":
    main()
