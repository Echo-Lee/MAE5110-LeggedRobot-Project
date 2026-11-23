"""The robot's observation implementation runs without the training simulator."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from tests.unit_tests import test_go2osc_onboard as onboard_tests

bundle = onboard_tests.bundle
REPOSITORY = Path(__file__).resolve().parents[2]
BLOCK_SIMULATORS = """\
import importlib.abc
import sys

blocked = {
    "gym", "gymnasium", "mujoco", "mujoco_warp", "warp", "vlearn",
    "isaacgym", "unitree_sdk2py", "cyclonedds",
}
attempted = []

class NoSimulatorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            attempted.append(fullname)
            raise ModuleNotFoundError(
                "simulator/SDK deliberately unavailable", name=fullname
            )

sys.meta_path.insert(0, NoSimulatorImports())
"""
CHECK_IMPORTS = """\
assert not attempted, attempted
assert not any(name.split(".")[0] in blocked for name in sys.modules)
"""


def _without_simulators(code, *arguments):
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY))
    environment.pop("GO2OSC_BUNDLE", None)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            BLOCK_SIMULATORS + textwrap.dedent(code) + CHECK_IMPORTS,
            *arguments,
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_task_import_and_discovery_need_neither_simulator_nor_bundle():
    result = _without_simulators(
        """
        from go2_deploy.custom_implementation.go2osc_push_glide import (
            ACTOR_OBSERVATIONS, Go2OscPushGlide,
        )
        from go2_deploy.custom_implementation.custom_config import CustomConfig
        from go2_deploy.deploy_config import DeployConfig
        from go2_deploy.utility.import_custom_impl import load_custom_impl

        implementation = load_custom_impl("go2osc_push_glide_deploy")
        assert isinstance(implementation, CustomConfig)
        assert implementation.obs_vector == ACTOR_OBSERVATIONS
        assert DeployConfig(implementation).obs_dim() == 58
        assert Go2OscPushGlide.task_name == implementation.task_name
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bundle_callbacks_produce_58_inputs_without_any_simulator(bundle):
    # The parent process creates genuine TorchScript files. The fresh child
    # cannot import the environment used to export them or any simulator.
    _, _, directory = bundle
    result = _without_simulators(
        """
        from types import SimpleNamespace
        import torch
        from go2_deploy.custom_implementation.go2osc_push_glide import (
            ACTOR_OBSERVATIONS, ONBOARD_OBSERVATIONS,
        )
        from go2_deploy.policy_bundle import load_bundle
        from go2_deploy.utility.deploy_utility import lowstate_to_obs
        from go2_deploy.utility.import_custom_impl import load_custom_impl

        torch.set_num_threads(1)
        directory = sys.argv[1]
        sys.argv = ["standalone-config-test", "--bundle", directory]
        implementation = load_custom_impl("go2osc_push_glide_deploy")
        policy, cfg = load_bundle(directory, implementation)
        assert cfg.obs_vector == ACTOR_OBSERVATIONS
        assert cfg.obs_dim() == 58
        remote = SimpleNamespace(lin_vel_x=0.0, lin_vel_y=0.0, yaw_vel=0.0)
        motor_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        posture = cfg.default_dof_pos[motor_order]
        message = SimpleNamespace(
            motor_state=[
                SimpleNamespace(q=float(angle), dq=0.0,
                                tau_est=5.0 if index % 3 == 2 else 0.0)
                for index, angle in enumerate(posture)
            ],
            imu_state=SimpleNamespace(
                quaternion=[1.0, 0.0, 0.0, 0.0],
                gyroscope=[0.0, 0.0, 0.0],
                accelerometer=[0.0, 0.0, 9.81],
            ),
        )
        controller = SimpleNamespace(
            cfg=cfg,
            obs_vec_size=cfg.obs_dim(),
            remote_controller=remote,
            last_action=torch.zeros(2, 12),
            last_command=torch.zeros(3),
            last_lowstate_msg=message,
            _state=SimpleNamespace(name="RECOVERY"),
        )
        for name, value in implementation.get_custom_buffers_dict().items():
            setattr(controller, name, value)
        implementation.update_custom_buffers(controller, 0.0)
        controller._state.name = "CUSTOM_CTRL"
        pieces = implementation.get_custom_obs_pieces()
        action_scale = torch.tensor(cfg.DeployScaling.dof_pos_target)
        for step in range(100):
            command = ((1.0, 0.0, 0.0) if step < 25 else
                       (-0.6, 0.0, 0.0) if step < 50 else
                       (0.0, 0.0, 0.4) if step < 75 else
                       (0.0, 0.0, 0.0))
            remote.lin_vel_x, remote.lin_vel_y, remote.yaw_vel = command
            controller.last_command[:] = torch.tensor(command)
            implementation.update_custom_buffers(controller, step / cfg.ctrl_freq)
            for name, size in ONBOARD_OBSERVATIONS.items():
                piece = pieces[name](controller, message)
                assert piece.shape == (size,), (name, piece.shape)
                assert torch.isfinite(piece).all(), (step, name)
            observation = lowstate_to_obs(controller, message)
            assert observation.shape == (58,)
            assert torch.isfinite(observation).all(), step
            with torch.no_grad():
                action = policy(observation)
            assert action.shape == (12,) and torch.isfinite(action).all()
            controller.last_action[0].copy_(action * action_scale)
        assert controller.go2osc_checked and controller.go2osc_running
        print("STANDALONE_CALLBACKS_OK: 100 steps, 58 inputs")
        """,
        str(directory),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STANDALONE_CALLBACKS_OK: 100 steps, 58 inputs" in result.stdout


def test_training_references_the_canonical_observation_order_and_scales(monkeypatch):
    from go2_deploy.custom_implementation.go2osc_push_glide import (
        ACTOR_OBSERVATIONS,
        Go2OscPushGlide,
    )
    from gym.envs.go2.go2osc_deploy_config import (
        DEPLOY_ACTOR_OBSERVATIONS,
        Go2OscPushGlideDeployCfg,
        Go2OscPushGlideDeployRunnerCfg,
    )

    assert Go2OscPushGlide.obs_vector is ACTOR_OBSERVATIONS
    assert DEPLOY_ACTOR_OBSERVATIONS is ACTOR_OBSERVATIONS
    assert Go2OscPushGlideDeployRunnerCfg.actor.obs is ACTOR_OBSERVATIONS
    scales = Go2OscPushGlide.DeployScaling
    training_scales = Go2OscPushGlideDeployCfg.scaling
    assert issubclass(training_scales, scales)
    for name in ACTOR_OBSERVATIONS + ["base_lin_acc"]:
        assert getattr(training_scales, name, 1.0) == getattr(scales, name, 1.0)
    # A copied value would silently drift if the deployment definition changed.
    for name in ("base_ang_vel", "governed_commands", "dof_pos_target"):
        sentinel = [2.0] * len(getattr(scales, name))
        monkeypatch.setattr(scales, name, sentinel)
        assert getattr(training_scales, name) is sentinel


@pytest.mark.parametrize("field", ["order", "size", "scale", "posture"])
def test_bundle_cannot_replace_the_canonical_observation_contract(bundle, field):
    from go2_deploy.custom_implementation.go2osc_push_glide import Go2OscPushGlide
    from go2_deploy.policy_bundle import config_from_meta

    _, _, directory = bundle
    metadata = json.loads((directory / "deploy_meta.json").read_text())
    implementation = Go2OscPushGlide()
    cfg = config_from_meta(metadata, implementation)
    assert cfg.custom_impl is implementation
    assert issubclass(cfg.DeployScaling, implementation.DeployScaling)
    if field == "order":
        metadata["obs_vector"][0:2] = metadata["obs_vector"][1::-1]
    elif field == "size":
        metadata["obs_sizes"]["base_ang_vel"] += 1
    elif field == "scale":
        metadata["scaling"]["dof_pos_target"][0] += 0.1
    else:
        metadata["default_dof_pos"][1] += 0.01
    with pytest.raises(ValueError, match=field):
        config_from_meta(metadata, implementation)
