"""Teleop shortcuts preserve viewer choices without changing native physics."""

import contextlib
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.utils.interfaces.MujocoKeyboardInterface import MujocoKeyboardInterface


KEYS = {ord(key) for key in "IJKLNMRF"}


class _Viewer:
    def __init__(self):
        self.opt = mujoco.MjvOption()
        flags = mujoco.MjvScene().flags.copy()
        self.user_scn = SimpleNamespace(flags=flags.copy())
        self.native_flags = flags.copy()
        self.previous_user_flags = flags.copy()
        self.cam = SimpleNamespace(
            type=mujoco.mjtCamera.mjCAMERA_FREE,
            trackbodyid=-1,
            distance=5.0,
            elevation=-45.0,
            azimuth=90.0,
        )
        self.before_sync = []
        self.on_next_sync = None
        self.close_count = 0

    def is_running(self):
        return True

    def lock(self):
        return contextlib.nullcontext()

    def close(self):
        self.close_count += 1

    def sync(self):
        self.before_sync.append(
            (self.opt.flags.copy(), self.user_scn.flags.copy(), self.opt.frame)
        )
        if self.on_next_sync is not None:
            callback, self.on_next_sync = self.on_next_sync, None
            callback()
        # MuJoCo's passive Sync picks changed user_scn flags over its internal
        # scene flags, then copies the native flags back to user_scn.
        changed = self.user_scn.flags != self.previous_user_flags
        self.native_flags[changed] = self.user_scn.flags[changed]
        self.user_scn.flags[:] = self.native_flags
        self.previous_user_flags[:] = self.user_scn.flags


def _backend():
    backend = MuJocoCPUBackend()
    backend._viewer = _Viewer()
    backend._has_free_joint = True
    backend._mjm = SimpleNamespace(jnt_bodyid=np.array([1]))
    return backend, backend._viewer


@pytest.mark.parametrize(
    "key,kind,index",
    [
        ("I", "vis", mujoco.mjtVisFlag.mjVIS_INERTIA),
        ("J", "vis", mujoco.mjtVisFlag.mjVIS_JOINT),
        ("N", "vis", mujoco.mjtVisFlag.mjVIS_ISLAND),
        ("M", "vis", mujoco.mjtVisFlag.mjVIS_COM),
        ("F", "vis", mujoco.mjtVisFlag.mjVIS_CONTACTFORCE),
        ("K", "render", mujoco.mjtRndFlag.mjRND_SKYBOX),
        ("L", "render", mujoco.mjtRndFlag.mjRND_ADDITIVE),
        ("R", "render", mujoco.mjtRndFlag.mjRND_REFLECTION),
    ],
)
def test_teleop_key_restores_its_visual_choice_and_native_render_flags(
    key, kind, index
):
    backend, viewer = _backend()
    backend.protect_viewer_shortcuts(KEYS)
    seen = []
    backend._viewer_key_callback = seen.append
    if kind == "vis":
        before = int(viewer.opt.flags[index])
        viewer.opt.flags[index] = not before
    else:
        before = int(viewer.native_flags[index])
        # The GUI changes the internal scene, not the public mirror. Restoring
        # only before sync would miss this pending native change.
        viewer.native_flags[index] = not before
    backend._dispatch_viewer_key(ord(key))
    backend.render(sync_frame_time=False)
    assert seen == [ord(key)]
    if kind == "vis":
        assert viewer.before_sync[0][0][index] == before
        assert viewer.opt.flags[index] == before
    else:
        assert viewer.before_sync[-1][1][index] == before
        assert viewer.native_flags[index] == before
        assert viewer.user_scn.flags[index] == before


def _teleop(backend):
    env = SimpleNamespace(
        _backend=backend,
        commands=torch.zeros(1, 3),
        cfg=SimpleNamespace(),
        max_episode_length_s=10.0,
        exit=False,
    )
    interface = MujocoKeyboardInterface(env)
    backend.protect_viewer_shortcuts(KEYS)
    return env, interface


def _update_teleop(interface):
    # The queued adapter applies events on the simulation thread.
    if hasattr(interface, "update"):
        interface.update()


def test_i_increases_forward_command_without_leaving_red_inertia_boxes():
    backend, viewer = _backend()
    env, interface = _teleop(backend)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 1
    backend._dispatch_viewer_key(ord("I"))
    assert not viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA]
    _update_teleop(interface)
    backend.render(sync_frame_time=False)
    torch.testing.assert_close(env.commands, torch.tensor([[1.5, 0.0, 0.0]]))
    assert not viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA]


def test_f_changes_camera_without_changing_frame_or_contact_force_choices():
    backend, viewer = _backend()
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    _, interface = _teleop(backend)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
    backend._dispatch_viewer_key(ord("F"))
    _update_teleop(interface)
    backend.render(sync_frame_time=False)
    assert backend.camera_tracking
    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert viewer.opt.frame == mujoco.mjtFrame.mjFRAME_BODY
    assert not viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]


def test_unprotected_frame_shortcuts_and_ui_frame_changes_remain_available():
    backend, viewer = _backend()
    backend.protect_viewer_shortcuts(KEYS)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_GEOM
    backend._dispatch_viewer_key(295)  # MuJoCo's native frame shortcut is F6.
    backend.render(sync_frame_time=False)
    assert viewer.opt.frame == mujoco.mjtFrame.mjFRAME_GEOM


def test_deliberate_viewer_ui_changes_become_the_next_teleop_baseline():
    backend, viewer = _backend()
    backend.protect_viewer_shortcuts(KEYS)
    inertia = mujoco.mjtVisFlag.mjVIS_INERTIA
    reflection = mujoco.mjtRndFlag.mjRND_REFLECTION
    viewer.opt.flags[inertia] = 1
    viewer.native_flags[reflection] = 0
    backend.render(sync_frame_time=False)
    assert viewer.opt.flags[inertia] == 1
    assert viewer.user_scn.flags[reflection] == 0
    assert len(viewer.before_sync) == 1  # No additional sync on idle frames.
    viewer.opt.flags[inertia] = 0
    backend._dispatch_viewer_key(ord("I"))
    backend.render(sync_frame_time=False)
    assert viewer.opt.flags[inertia] == 1
    assert viewer.native_flags[reflection] == 0


def test_native_option_change_arriving_during_sync_is_also_restored():
    backend, viewer = _backend()
    backend.protect_viewer_shortcuts(KEYS)
    inertia = mujoco.mjtVisFlag.mjVIS_INERTIA
    backend._dispatch_viewer_key(ord("I"))
    viewer.on_next_sync = lambda: viewer.opt.flags.__setitem__(inertia, 1)
    backend.render(sync_frame_time=False)
    assert viewer.before_sync[0][0][inertia] == 0
    assert viewer.before_sync[-1][0][inertia] == 0
    assert viewer.opt.flags[inertia] == 0


def test_protected_key_arriving_during_sync_keeps_the_original_baseline():
    backend, viewer = _backend()
    backend.protect_viewer_shortcuts(KEYS)
    inertia = mujoco.mjtVisFlag.mjVIS_INERTIA

    def press():
        viewer.opt.flags[inertia] = 1
        backend._dispatch_viewer_key(ord("I"))

    viewer.on_next_sync = press
    backend.render(sync_frame_time=False)
    assert viewer.opt.flags[inertia] == 0


@pytest.mark.parametrize("key", [256, 27])
@pytest.mark.parametrize("has_callback", [False, True])
def test_escape_records_exit_even_without_teleop_and_forwards_existing_callback(
    key, has_callback
):
    backend, viewer = _backend()
    seen = []
    if has_callback:
        backend._viewer_key_callback = seen.append
    backend._dispatch_viewer_key(key)
    assert backend._viewer_exit_requested
    assert seen == ([key] if has_callback else [])
    assert viewer.close_count == 0


def test_close_releases_the_passive_viewer_once_and_resets_frame_timing():
    backend, viewer = _backend()
    backend._render_deadline = 10.0
    backend._render_pending_time = 0.02
    backend.close()
    assert viewer.close_count == 1
    assert backend._viewer is None
    assert backend._render_deadline is None
    assert backend._render_pending_time == 0.0
    backend.close()
    assert viewer.close_count == 1
