"""The MuJoCo viewer camera can lock onto, and release, the robot's root body.

A stub stands in for the passive viewer, so no window opens.
"""

import contextlib
from types import SimpleNamespace

import mujoco
import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.utils.interfaces.MujocoKeyboardInterface import (
    CAMERA_FOLLOW_KEYCODE,
    KEYCODE_TO_ACTION,
    MujocoKeyboardInterface,
)


class _StubViewer:
    def __init__(self):
        self.cam = SimpleNamespace(
            type=mujoco.mjtCamera.mjCAMERA_FREE,
            trackbodyid=-1,
            distance=5.0,
            elevation=-45.0,
            azimuth=90.0,
        )
        self.syncs = 0

    def is_running(self):
        return True

    def sync(self):
        self.syncs += 1

    def lock(self):
        return contextlib.nullcontext()

    def close(self):
        pass


@pytest.fixture
def viewer(monkeypatch):
    import mujoco.viewer

    stub = _StubViewer()
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *a, **k: stub)
    monkeypatch.setattr(mujoco.viewer, "_MJPYTHON", object())
    return stub


def _go2_backend():
    from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2Cfg()
    cfg.env.num_envs = 2
    cfg.seed = 0
    task_registry.convert_frequencies_to_params(cfg, Go2RunnerCfg())
    backend = select_backend(cfg, "cpu", "mujoco")
    backend.setup(cfg, num_envs=2, device="cpu", task=None)
    return backend


def test_tracking_locks_onto_root_body_and_releases(viewer):
    backend = _go2_backend()
    try:
        assert backend.supports_camera_tracking
        backend.set_camera_tracking(True)
        backend.render(sync_frame_time=False)

        root = backend._mjm.jnt_bodyid[0]
        assert backend._mjm.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
        assert viewer.cam.trackbodyid == root
        assert viewer.cam.distance == backend.TRACKING_DISTANCE_M

        # Applied once: later frames must not undo the user's mouse orbit.
        viewer.cam.azimuth = 10.0
        backend.render(sync_frame_time=False)
        assert viewer.cam.azimuth == 10.0

        backend.set_camera_tracking(False)
        backend.render(sync_frame_time=False)
        assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE
        assert viewer.cam.trackbodyid == -1
        assert viewer.cam.azimuth == 10.0  # released where it was
    finally:
        backend.close()


def test_tracking_requested_before_viewer_exists_applies_on_first_render(viewer):
    backend = _go2_backend()
    try:
        backend.set_camera_tracking(True)
        assert backend._viewer is None
        backend.render(sync_frame_time=False)
        assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    finally:
        backend.close()


def test_fixed_base_robot_cannot_track():
    from tests.unit_tests.test_teleop_keyboard import _pendulum_cfg

    backend = MuJocoCPUBackend()
    backend.setup(_pendulum_cfg(False), num_envs=1, device="cpu", task=None)
    try:
        assert not backend.supports_camera_tracking
        with pytest.raises(ValueError, match="floating-base"):
            backend.set_camera_tracking(True)
    finally:
        backend.close()


class _TrackingBackend:
    def __init__(self, supported=True):
        self._viewer_key_callback = None
        self.supports_camera_tracking = supported
        self.camera_tracking = False
        self.requests = []

    def set_camera_tracking(self, enabled):
        self.requests.append(enabled)
        self.camera_tracking = enabled


def _env(backend):
    return SimpleNamespace(
        commands=torch.zeros(1, 3),
        cfg=SimpleNamespace(),
        _backend=backend,
    )


def test_f_key_toggles_camera_lock_without_touching_commands():
    assert CAMERA_FOLLOW_KEYCODE not in KEYCODE_TO_ACTION
    backend = _TrackingBackend()
    env = _env(backend)
    ui = MujocoKeyboardInterface(env)
    commands = env.commands.clone()  # teleop sets an initial command

    ui._on_key(CAMERA_FOLLOW_KEYCODE)
    ui._on_key(CAMERA_FOLLOW_KEYCODE)

    assert backend.requests == [True, False]
    assert torch.equal(env.commands, commands)


def test_f_key_on_fixed_base_is_a_no_op():
    backend = _TrackingBackend(supported=False)
    MujocoKeyboardInterface(_env(backend))._on_key(CAMERA_FOLLOW_KEYCODE)
    assert backend.requests == []
