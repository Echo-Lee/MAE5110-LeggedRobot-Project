"""Plain MuJoCo (mj_step) backend — no Warp dependency.

Works on any platform (Linux CPU, Mac Apple Silicon). Each environment owns an
MjData. Models are shared unless link-mass DR requires one compact model per
environment; physics runs in a Python loop and state is copied numpy→torch
after each step.
"""

import copy
import os
from queue import Empty, SimpleQueue
import time

import mujoco
import numpy as np
import torch

from gym import GYM_ROOT_DIR
from gym.envs.base.domain_randomization import (
    contact_friction_range,
    link_mass_scale_range,
)
from gym.envs.base.mujoco_backend_base import (
    MuJocoBackendBase,
    WXYZ_TO_XYZW,
    XYZW_TO_WXYZ,
)
from gym.utils.torch_quat import quat_apply, quat_rotate_inverse


class MuJocoCPUBackend(MuJocoBackendBase):
    """SimBackend backed by plain mujoco.mj_step."""

    def __init__(self) -> None:
        super().__init__()
        self._datas: list[mujoco.MjData] = []

        # State tensors (allocated in setup)
        self._dof_state_t: torch.Tensor = None  # [N, num_dof, 2]
        self._dof_pos_view: torch.Tensor = None  # [N, num_dof] view into above
        self._dof_vel_view: torch.Tensor = None  # [N, num_dof] view into above
        self._root_states_t: torch.Tensor = None  # [N, 13]
        self._rigid_body_states_t: torch.Tensor = None  # [N, num_bodies, 13]
        self._contact_forces_t: torch.Tensor = None  # [N, num_bodies, 3]

        # Viewer (created lazily on first render call)
        self._viewer = None
        self._show_ui = False  # overridden from cfg.viewer.show_ui in setup()
        self._viewer_key_callback = None
        self._viewer_overlay_fn = None  # called each render() before sync()
        self._viewer_exit_requested = False
        self._protected_shortcuts = {}
        self._shortcut_values = {}
        self._pending_shortcuts = SimpleQueue()
        self._deferred_shortcuts = set()
        # Root-tracking camera request; applied to the viewer on the next
        # render(), since the viewer is created lazily there.
        self._camera_tracking = False
        self._camera_tracking_dirty = False
        self._render_deadline = None
        self._render_pending_time = 0.0
        self._randomize_contact_friction = False
        self._randomize_link_mass = False
        self._models: list[mujoco.MjModel] | None = None
        # [num_envs, ngeom] sliding friction once any body has its own value.
        self._geom_friction_np: np.ndarray | None = None

    # ── State tensors ──────────────────────────────────────────────────────────

    @property
    def dof_state(self) -> torch.Tensor:
        return self._dof_state_t.view(self._num_envs * self._num_dof, 2)

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._dof_pos_view

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._dof_vel_view

    @property
    def root_states(self) -> torch.Tensor:
        return self._root_states_t

    @property
    def rigid_body_states(self) -> torch.Tensor:
        return self._rigid_body_states_t.view(self._num_envs * self._num_bodies, 13)

    @property
    def contact_forces(self) -> torch.Tensor:
        return self._contact_forces_t

    @property
    def contact_friction(self) -> torch.Tensor:
        return self._contact_friction_t

    def set_contact_friction(
        self,
        env_ids: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> None:
        ids, values = self._prepare_contact_friction_update(env_ids, coefficients)
        if ids.numel() == 0:
            return
        if not self._randomize_contact_friction:
            raise RuntimeError(
                "contact-friction randomization was not enabled before setup"
            )
        self._contact_friction_t[ids] = values
        if self._geom_friction_np is not None:
            self._geom_friction_np[ids.cpu().numpy()] = values.cpu().numpy()[:, None]
        if self._models is not None:
            for env_id, value in zip(ids.tolist(), values.tolist()):
                self._set_model_contact_friction(self._models[env_id], value)

    def set_contact_friction_by_body(
        self,
        env_ids: torch.Tensor,
        body_indices: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> None:
        ids, bodies, values = self._prepare_body_friction_update(
            env_ids, body_indices, coefficients
        )
        if ids.numel() == 0:
            return
        if not self._randomize_contact_friction:
            raise RuntimeError(
                "contact-friction randomization was not enabled before setup"
            )
        geoms, columns = self._geoms_by_body(bodies)
        if self._geom_friction_np is None:
            self._geom_friction_np = np.repeat(
                self._contact_friction_t.cpu().numpy().astype(np.float64)[:, None],
                self._mjm.ngeom,
                axis=1,
            )
        rows = ids.cpu().numpy()
        table = values.cpu().numpy().astype(np.float64)
        self._geom_friction_np[rows] = table.min(axis=1, keepdims=True)
        self._geom_friction_np[rows[:, None], geoms[None, :]] = table[:, columns]
        self._contact_friction_t[ids] = values.mean(dim=1)
        if self._models is not None:
            for env_id in rows.tolist():
                self._models[env_id].geom_friction[:, 0] = self._geom_friction_np[
                    env_id
                ]

    def set_link_mass_scale(
        self,
        env_ids: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        ids, values = self._prepare_link_mass_scale_update(env_ids, scales)
        if ids.numel() == 0:
            return
        if not self._randomize_link_mass:
            raise RuntimeError("link-mass randomization was not enabled before setup")

        masses = self._nominal_link_mass * values
        inertias = self._nominal_link_inertia * values.unsqueeze(-1)
        self._link_mass_t[ids] = masses
        self._link_inertia_t[ids] = inertias
        body_indices = self._canonical_to_native_body_np
        for row, env_id in enumerate(ids.tolist()):
            model = self._models[env_id]
            data = self._datas[env_id]
            model.body_mass[body_indices] = masses[row].cpu().numpy()
            model.body_inertia[body_indices] = inertias[row].cpu().numpy()

            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            warmstart = data.qacc_warmstart.copy()
            time = data.time
            mujoco.mj_setConst(model, data)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.time = time
            mujoco.mj_forward(model, data)
            data.qacc_warmstart[:] = warmstart

    def _activate_domain(self, env_id: int) -> None:
        """Activate one environment's parameters on the shared CPU model."""
        if not self._randomize_contact_friction or self._models is not None:
            return
        if self._geom_friction_np is not None:
            self._mjm.geom_friction[:, 0] = self._geom_friction_np[env_id]
            return
        self._set_model_contact_friction(
            self._mjm, float(self._contact_friction_t[env_id])
        )

    def _model_for_env(self, env_id: int) -> mujoco.MjModel:
        if self._models is None:
            return self._mjm
        return self._models[env_id]

    # ── World building ─────────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        self._viewer_exit_requested = False
        self._device = device
        self._num_envs = num_envs
        self._randomize_contact_friction = contact_friction_range(cfg) is not None
        self._randomize_link_mass = link_mass_scale_range(cfg) is not None
        headless = task.headless if task is not None else True
        # Off-screen rendering (video recording) is headless but still draws:
        # load visuals for it as for the interactive viewer.
        renders = not headless or getattr(cfg.asset, "render_offscreen", False)
        visual_model = getattr(cfg.asset, "mujoco_visual_model", None)
        visual_model_path = (
            visual_model.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
            if visual_model is not None and renders
            else None
        )
        if visual_model_path is not None and not os.path.isfile(visual_model_path):
            print(
                "WARNING: optional MuJoCo visual model is unavailable: "
                f"{visual_model_path}. Falling back to the URDF visuals."
            )
            visual_model_path = None

        mjm = self._load_model(
            cfg,
            discard_visual=self._randomize_link_mass and not renders,
            visual_model_path=visual_model_path,
        )
        self._configure_model(mjm, cfg, device)
        self._run_task_callbacks(mjm, task)
        self._initialize_link_properties(mjm, num_envs, device)

        viewer_cfg = getattr(cfg, "viewer", None)
        self._show_ui = bool(getattr(viewer_cfg, "show_ui", False))

        # Allocate PyTorch state tensors
        self._dof_state_t = torch.zeros(num_envs, self._num_dof, 2, device=device)
        self._dof_pos_view = self._dof_state_t[..., 0]
        self._dof_vel_view = self._dof_state_t[..., 1]
        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        self._root_states_t[:, 6] = 1.0  # identity quaternion (scalar-last, w=1)
        self._rigid_body_states_t = torch.zeros(
            num_envs, self._num_bodies, 13, device=device
        )
        self._rigid_body_states_t[:, :, 6] = 1.0
        self._contact_forces_t = torch.zeros(
            num_envs, self._num_bodies, 3, device=device
        )
        self._contact_friction_t = torch.full(
            (num_envs,), self._nominal_contact_friction, device=device
        )

        # Mass changes require environment-specific derived model constants.
        # A headless training model has already discarded visual assets, so
        # private copies stay compact; an off-screen recording keeps them
        # (it runs few environments). Other serial worlds share one model.
        if self._randomize_link_mass:
            self._models = [mjm] + [copy.copy(mjm) for _ in range(num_envs - 1)]
            self._datas = [mujoco.MjData(model) for model in self._models]
        else:
            self._datas = [mujoco.MjData(mjm) for _ in range(num_envs)]
        for i, data in enumerate(self._datas):
            self._activate_domain(i)
            model = self._model_for_env(i)
            mujoco.mj_forward(model, data)
            mujoco.mj_rnePostConstraint(model, data)
            self._copy_env_state_from_mujoco(i)

    # ── Per-step ───────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        torques_np = torques.cpu().numpy()[:, self._native_to_canonical_dof_np]
        off = self._qvel_offset
        for i, d in enumerate(self._datas):
            self._activate_domain(i)
            d.qfrc_applied[off:] = torques_np[i]
            mujoco.mj_step(self._model_for_env(i), d)
        self._sync_state_from_mujoco()
        # Count simulated duration once per batch, independent of world count.
        # A separate clock survives episode resets of individual MjData objects.
        self._render_pending_time += self._mjm.opt.timestep

    def _sync_state_from_mujoco(self) -> None:
        for i, d in enumerate(self._datas):
            self._activate_domain(i)
            # cfrc_ext is only populated with constraint/contact forces by
            # mj_rnePostConstraint; mj_step alone leaves it at zero.
            model = self._model_for_env(i)
            mujoco.mj_rnePostConstraint(model, d)
            # mj_step integrates qpos/qvel after computing body kinematics.
            # Refresh only the derived pose/velocity fields so every public
            # state tensor describes the completed step. A second mj_forward
            # would also recompute collision, constraints, and acceleration.
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)
            mujoco.mj_comVel(model, d)
            self._copy_env_state_from_mujoco(i)

    def _copy_env_state_from_mujoco(self, env_id: int) -> None:
        """Copy already-refreshed native state into one environment's live views.

        Reset's forward pass has current body kinematics; step refreshes those
        separately after integration. Both paths populate contact forces with
        rnePostConstraint before copying. Keeping this copy local lets partial
        resets preserve every unselected native state and public cache.
        """
        d = self._datas[env_id]
        model = self._model_for_env(env_id)
        dof_order = self._canonical_to_native_dof_np
        body_order = self._canonical_to_native_body_np
        self._dof_pos_view[env_id] = torch.from_numpy(
            d.qpos[self._qpos_offset :][dof_order].copy()
        )
        self._dof_vel_view[env_id] = torch.from_numpy(
            d.qvel[self._qvel_offset :][dof_order].copy()
        )
        self._contact_forces_t[env_id] = torch.from_numpy(
            d.cfrc_ext[body_order, 3:6].copy()
        )
        rbs = self._rigid_body_states_t[env_id]
        rbs[:, 0:3] = torch.from_numpy(d.xpos[body_order].copy())
        mj_quat = torch.from_numpy(d.xquat[body_order].copy())
        rbs[:, 3:7] = mj_quat[:, WXYZ_TO_XYZW]
        angular_velocity = d.cvel[body_order, 0:3]
        root_com = d.subtree_com[model.body_rootid[body_order]]
        body_offset = d.xpos[body_order] - root_com
        linear_velocity = d.cvel[body_order, 3:6] - np.cross(
            body_offset, angular_velocity
        )
        rbs[:, 7:10] = torch.from_numpy(linear_velocity.copy())
        rbs[:, 10:13] = torch.from_numpy(angular_velocity.copy())
        if self._has_free_joint:
            rs = self._root_states_t[env_id]
            rs[:3] = torch.from_numpy(d.qpos[:3].copy())
            mj_quat = torch.from_numpy(d.qpos[3:7].copy())
            rs[3:7] = mj_quat[WXYZ_TO_XYZW]
            rs[7:10] = torch.from_numpy(d.qvel[:3].copy())
            # Free-joint translation is world-frame, but angular qvel is
            # body-local. Public root velocities are both world-frame.
            rs[10:13] = quat_apply(
                mj_quat[WXYZ_TO_XYZW], torch.from_numpy(d.qvel[3:6].copy())
            )

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_state(self, reset_mask: torch.Tensor) -> None:
        qoff = self._qpos_offset
        voff = self._qvel_offset
        for i in reset_mask.nonzero(as_tuple=False).flatten().tolist():
            self._activate_domain(i)
            model = self._model_for_env(i)
            canonical_pos = self._clamp_dof_positions(self._dof_pos_view[i])
            self._dof_pos_view[i].copy_(canonical_pos)
            self._datas[i].qpos[qoff:] = canonical_pos.cpu().numpy()[
                self._native_to_canonical_dof_np
            ]
            self._datas[i].qvel[voff:] = (
                self._dof_vel_view[i].cpu().numpy()[self._native_to_canonical_dof_np]
            )
            if self._has_free_joint:
                rs = self._root_states_t[i].cpu()
                self._datas[i].qpos[:3] = rs[:3].numpy()
                self._datas[i].qpos[3:7] = rs[3:7][XYZW_TO_WXYZ].numpy()
                self._datas[i].qvel[:3] = rs[7:10].numpy()
                # Use the requested orientation, which may change in this reset.
                self._datas[i].qvel[3:6] = quat_rotate_inverse(
                    rs[3:7], rs[10:13]
                ).numpy()
            mujoco.mj_forward(model, self._datas[i])
            mujoco.mj_rnePostConstraint(model, self._datas[i])
            self._copy_env_state_from_mujoco(i)

    def set_all_root_states(self) -> None:
        if not self._has_free_joint:
            return
        for i in range(self._num_envs):
            self._activate_domain(i)
            model = self._model_for_env(i)
            rs = self._root_states_t[i].cpu()
            self._datas[i].qpos[:3] = rs[:3].numpy()
            self._datas[i].qpos[3:7] = rs[3:7][XYZW_TO_WXYZ].numpy()
            self._datas[i].qvel[:3] = rs[7:10].numpy()
            self._datas[i].qvel[3:6] = quat_rotate_inverse(rs[3:7], rs[10:13]).numpy()
            mujoco.mj_forward(model, self._datas[i])
            mujoco.mj_rnePostConstraint(model, self._datas[i])
            self._copy_env_state_from_mujoco(i)

    # ── Rendering ─────────────────────────────────────────────────────────────

    # Initial framing when tracking starts; the user can still orbit and zoom
    # around the tracked body with the mouse.
    TRACKING_DISTANCE_M = 1.5
    TRACKING_ELEVATION_DEG = -20.0
    TRACKING_AZIMUTH_DEG = 135.0

    def protect_viewer_shortcuts(self, keycodes) -> None:
        """Keep teleop presses from changing MuJoCo's visualization options.

        Only a pressed key's colliding fields are restored. Ordinary viewer UI
        edits remain available and become the baseline for subsequent presses.
        """
        keycodes = set(keycodes)
        shortcuts = {}
        for kind, table in (
            ("vis", mujoco.mjVISSTRING),
            ("render", mujoco.mjRNDSTRING),
        ):
            for index, (_, _, shortcut) in enumerate(table):
                if len(shortcut) == 1 and ord(shortcut.upper()) in keycodes:
                    shortcuts.setdefault(ord(shortcut.upper()), set()).add(
                        (kind, index)
                    )
        self._protected_shortcuts = shortcuts
        self._deferred_shortcuts.clear()
        self._pending_shortcuts = SimpleQueue()
        self._shortcut_values = {}
        if self._viewer is None:
            options, scene = mujoco.MjvOption(), mujoco.MjvScene()
            for fields in shortcuts.values():
                for kind, index in fields:
                    value = (
                        options.flags[index] if kind == "vis" else scene.flags[index]
                    )
                    self._shortcut_values[kind, index] = int(value)
        else:
            with self._viewer.lock():
                self._remember_shortcut_values(self._viewer)

    @staticmethod
    def _shortcut_value(viewer, field):
        kind, index = field
        if kind == "vis":
            return int(viewer.opt.flags[index])
        return int(viewer.user_scn.flags[index])

    def _remember_shortcut_values(self, viewer, exclude=()) -> None:
        for fields in self._protected_shortcuts.values():
            for field in fields:
                if field not in exclude:
                    self._shortcut_values[field] = self._shortcut_value(viewer, field)

    def _take_pending_shortcuts(self):
        fields, self._deferred_shortcuts = self._deferred_shortcuts, set()
        while True:
            try:
                keycode = self._pending_shortcuts.get_nowait()
            except Empty:
                return fields
            fields.update(self._protected_shortcuts.get(keycode, ()))

    def _restore_shortcut_values(self, viewer, fields) -> None:
        for field in fields:
            kind, index = field
            value = self._shortcut_values[field]
            if kind == "vis":
                viewer.opt.flags[index] = value
            else:
                viewer.user_scn.flags[index] = value

    def _sync_viewer(self, viewer) -> None:
        if not self._protected_shortcuts:
            viewer.sync()
            return
        with viewer.lock():
            changed = self._take_pending_shortcuts()
            self._restore_shortcut_values(viewer, changed)
        viewer.sync()
        with viewer.lock():
            changed.update(self._take_pending_shortcuts())
            self._restore_shortcut_values(viewer, changed)
        if changed:
            # Native render flags arrive via user_scn during sync. Synchronize
            # the restored values once more so they reach the native scene.
            viewer.sync()
        with viewer.lock():
            late = self._take_pending_shortcuts()
            self._deferred_shortcuts.update(late)
            self._remember_shortcut_values(viewer, exclude=late)

    def _dispatch_viewer_key(self, keycode) -> None:
        # MuJoCo forwards Escape after its own free-camera action. The main
        # simulation thread closes the viewer, never this UI callback thread.
        if keycode in (256, 27):
            self._viewer_exit_requested = True
        if keycode in self._protected_shortcuts:
            self._pending_shortcuts.put(keycode)
            if self._viewer is not None:
                # Native handling precedes this callback, and opt is shared.
                # Undo its visualization toggles before the GUI can draw them.
                with self._viewer.lock():
                    self._restore_shortcut_values(
                        self._viewer,
                        {
                            field
                            for field in self._protected_shortcuts[keycode]
                            if field[0] == "vis"
                        },
                    )
        if self._viewer_key_callback is not None:
            self._viewer_key_callback(keycode)

    @property
    def camera_tracking(self) -> bool:
        return self._camera_tracking

    @property
    def supports_camera_tracking(self) -> bool:
        """Only a floating-base robot has a root body to follow."""
        return self._has_free_joint

    def set_camera_tracking(self, enabled: bool) -> None:
        """Lock the viewer camera onto environment 0's floating base.

        The passive viewer draws ``self._datas[0]``, so environment 0 is the
        robot on screen. Disabling leaves the camera where it is, free.
        """
        if enabled and not self._has_free_joint:
            raise ValueError("camera tracking needs a floating-base robot")
        self._camera_tracking = enabled
        self._camera_tracking_dirty = True

    def _apply_camera_tracking(self, viewer) -> None:
        with viewer.lock():
            if self._camera_tracking:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                # The free joint is joint 0 and sits on the robot's root body.
                viewer.cam.trackbodyid = int(self._mjm.jnt_bodyid[0])
                viewer.cam.distance = self.TRACKING_DISTANCE_M
                viewer.cam.elevation = self.TRACKING_ELEVATION_DEG
                viewer.cam.azimuth = self.TRACKING_AZIMUTH_DEG
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.trackbodyid = -1
        self._camera_tracking_dirty = False

    def render(self, sync_frame_time: bool = True) -> None:
        import platform
        import mujoco.viewer

        self._activate_domain(0)
        if self._viewer is None:
            if platform.system() == "Darwin":
                import mujoco.viewer as _mjv

                if _mjv._MJPYTHON is None:
                    raise RuntimeError(
                        "MuJoCo passive viewer on macOS requires mjpython.\n"
                        "Run with: .venv/bin/mjpython scripts/train.py ...\n"
                        "Or use --headless to disable the viewer."
                    )

            # Indirect through a wrapper so callers can set _viewer_key_callback
            # after the viewer is up (LeggedRobot.__init__ calls reset()→step()
            # →_render() during construction, before any user interface installs).
            # Hidden panels keep the view clear but leave native shortcuts
            # active. The keyboard adapter registers its conflicting keys
            # with protect_viewer_shortcuts. --viewer_ui shows the panels.
            self._viewer = mujoco.viewer.launch_passive(
                self._mjm,
                self._datas[0],
                key_callback=self._dispatch_viewer_key,
                show_left_ui=self._show_ui,
                show_right_ui=self._show_ui,
            )
        if self._viewer.is_running():
            now = time.monotonic()
            if sync_frame_time and self._render_deadline is not None:
                # Include every physics substep since the previous render.
                # Rebase after slow frames rather than racing to catch up.
                self._render_deadline = max(
                    self._render_deadline + self._render_pending_time, now
                )
                remaining = self._render_deadline - now
                if remaining > 0:
                    time.sleep(remaining)
            else:
                self._render_deadline = now if sync_frame_time else None
            self._render_pending_time = 0.0
            if self._camera_tracking_dirty:
                self._apply_camera_tracking(self._viewer)
            if self._viewer_overlay_fn is not None:
                self._viewer_overlay_fn(self._viewer)
            self._sync_viewer(self._viewer)
        else:
            self._render_deadline = None
            self._render_pending_time = 0.0

    def close(self) -> None:
        """Close the passive viewer from the caller's thread, once."""
        viewer, self._viewer = self._viewer, None
        self._render_deadline = None
        self._render_pending_time = 0.0
        if viewer is not None:
            viewer.close()
