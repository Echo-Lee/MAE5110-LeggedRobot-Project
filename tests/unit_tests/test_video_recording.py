"""Playback recording: CLI options, overlay drawing, and an end-to-end MP4."""

import shutil
import subprocess

import numpy as np
import pytest
from PIL import ImageFont

from gym.utils.interfaces.VideoRecorder import (
    draw_overlay,
    foot_layout,
    frame_stride,
    overlay_lines,
)
from scripts.play import get_play_args, select_mujoco_gl


def test_video_defaults_to_one_headless_env():
    args = get_play_args(["--task", "go2osc_push_glide", "--video", "out.mp4"])
    assert args.headless is True
    assert args.num_envs == 1
    assert args.overlay is True
    plain = get_play_args(["--task", "go2osc_push_glide"])
    assert plain.num_envs == 32 and plain.headless is False


def test_friction_and_command_options_parse():
    args = get_play_args(
        [
            "--task",
            "go2osc_push_glide",
            "--friction",
            "0.1",
            "--command",
            "1",
            "0",
            "-0.5",
        ]
    )
    assert args.friction == pytest.approx(0.1)
    assert args.command == [1.0, 0.0, -0.5]
    with pytest.raises(SystemExit):
        get_play_args(["--task", "go2osc_push_glide", "--friction", "-0.1"])


def test_frame_stride_plays_back_in_real_time():
    assert frame_stride(0.01, 50) == 2
    assert frame_stride(0.01, 100) == 1
    assert frame_stride(0.02, 100) == 1  # never zero


def test_overlay_lines_show_command_against_actual():
    lines = overlay_lines(1.234, 0.1, [1.0, 0.0, -0.5], [0.85, 0.02, -0.4])
    assert lines[0] == "t  1.23 s    mu 0.100"
    assert "vx +1.00" in lines[1] and "yaw -0.50" in lines[1]
    assert "vx +0.85" in lines[2]


def test_draw_overlay_marks_loaded_feet_and_keeps_the_rest_of_the_frame():
    frame = np.full((240, 400, 3), 200, dtype=np.uint8)
    font = ImageFont.load_default(size=12)
    lines = overlay_lines(0.0, 0.5, [0, 0, 0], [0, 0, 0])
    labels = ("FL", "FR", "RL", "RR")
    both = draw_overlay(frame, lines, list(zip(labels, [1, 0, 0, 0])), font)
    none = draw_overlay(frame, lines, list(zip(labels, [0, 0, 0, 0])), font)
    assert both.shape == frame.shape and both.dtype == np.uint8
    assert np.array_equal(both[-20:, -20:], frame[-20:, -20:])  # untouched
    assert not np.array_equal(both, none)  # a loaded foot box is drawn lit


# Base-frame (x forward, y left) foot positions for each URDF's foot order.
GO2_FEET = [(0.19, 0.14), (0.19, -0.14), (-0.19, 0.14), (-0.19, -0.14)]  # FL FR RL RR
CHEETAH_FEET = [(0.19, -0.1), (0.19, 0.1), (-0.19, -0.1), (-0.19, 0.1)]  # RF LF RH LH


def test_foot_layout_labels_feet_from_geometry_not_urdf_order():
    assert foot_layout(GO2_FEET) == ([0, 1, 2, 3], ["FL", "FR", "RL", "RR"])
    # Mini Cheetah lists right before left: reorder so left/right stay true.
    assert foot_layout(CHEETAH_FEET) == ([1, 0, 3, 2], ["FL", "FR", "RL", "RR"])
    assert foot_layout([(0.0, -0.1), (0.0, 0.1)]) == ([1, 0], ["L", "R"])


@pytest.mark.parametrize("task", ["go2", "mini_cheetah"])
def test_recorder_foot_labels_match_the_robot(task):
    from gym.utils.interfaces.VideoRecorder import feet_in_base_frame
    from scripts.play import make_play_env, play_cfgs

    args = get_play_args(["--task", task, "--seed", "0", "--headless"])
    args.num_envs = 1
    env_cfg, _ = play_cfgs(args)
    env = make_play_env(args, env_cfg)
    try:
        order, labels = foot_layout(feet_in_base_frame(env))
        names = list(env.robot_layout.body_groups["feet"])
    finally:
        env._backend.close()
    assert labels == ["FL", "FR", "RL", "RR"]
    # Every label names the URDF foot it draws, whatever the naming scheme.
    aliases = {"FL": "lf", "FR": "rf", "RL": "lh", "RR": "rh"}
    for i, label in zip(order, labels):
        name = names[i].lower()
        assert name.startswith(label.lower()) or name.startswith(aliases[label])


@pytest.mark.parametrize(
    "platform, environ, expected",
    [
        ("linux", {}, "egl"),  # headless Linux (MoLab)
        ("linux", {"DISPLAY": ":0"}, None),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, None),
        ("darwin", {}, None),  # a macOS terminal has no DISPLAY
        ("win32", {}, None),
        ("linux", {"MUJOCO_GL": "osmesa"}, "osmesa"),  # user override kept
        ("darwin", {"MUJOCO_GL": "glfw"}, "glfw"),
    ],
)
def test_select_mujoco_gl_only_defaults_to_egl_on_headless_linux(
    platform, environ, expected
):
    args = get_play_args(["--task", "go2", "--video", "out.mp4"])
    select_mujoco_gl(args, environ, platform)
    assert environ.get("MUJOCO_GL") == expected


def test_select_mujoco_gl_leaves_viewer_playback_alone():
    environ = {}
    select_mujoco_gl(get_play_args(["--task", "go2"]), environ, "linux")
    assert "MUJOCO_GL" not in environ


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_cli_entry_point_selects_gl_before_setup(monkeypatch, platform):
    """main() is what `python scripts/play.py` runs; check its GL choice."""
    import scripts.play as play

    seen = {}

    def fake_setup(args):
        seen["gl"] = play.os.environ.get("MUJOCO_GL")
        raise SystemExit(0)

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(play.sys, "platform", platform)
    monkeypatch.setattr(play, "setup", fake_setup)
    with pytest.raises(SystemExit):
        play.main(["--task", "go2", "--video", "out.mp4"])
    assert seen["gl"] == ("egl" if platform == "linux" else None)


def test_cli_script_runs_without_display_or_gl(tmp_path):
    """The real entry point, as a subprocess with no DISPLAY or MUJOCO_GL."""
    import os
    import sys

    from gym import GYM_ROOT_DIR

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DISPLAY", "WAYLAND_DISPLAY", "MUJOCO_GL")
    }
    env["PYTHONPATH"] = GYM_ROOT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, os.path.join(GYM_ROOT_DIR, "scripts/play.py"), "--help"],
        env=env,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "--video" in result.stdout


def _play_env(task, extra):
    from scripts.play import make_play_env, play_cfgs

    args = get_play_args(["--task", task, "--seed", "7", *extra])
    args.num_envs = 2
    env_cfg, _ = play_cfgs(args)
    return env_cfg, make_play_env(args, env_cfg)


def test_video_playback_keeps_the_same_physics():
    """Same seed, with and without --video: identical masses and motion."""
    import torch

    plain_cfg, plain = _play_env("go2", ["--headless"])
    video_cfg, video = _play_env("go2", ["--video", "out.mp4"])
    try:
        assert video_cfg.domain_randomization.startup.link_mass_scale_range == (
            plain_cfg.domain_randomization.startup.link_mass_scale_range
        )
        assert plain_cfg.domain_randomization.startup.link_mass_scale_range
        assert video_cfg.asset.render_offscreen
        assert not plain_cfg.asset.render_offscreen
        # The recording keeps visuals that headless playback discards.
        assert (
            video._backend._model_for_env(0).ntex
            > plain._backend._model_for_env(0).ntex
        )
        torch.testing.assert_close(
            video._backend._link_mass_t, plain._backend._link_mass_t
        )
        nominal = plain._backend._nominal_link_mass.expand_as(
            plain._backend._link_mass_t
        )
        assert not torch.allclose(plain._backend._link_mass_t, nominal)  # randomized
        for i in range(2):
            a, b = plain._backend._model_for_env(i), video._backend._model_for_env(i)
            np.testing.assert_array_equal(a.body_mass, b.body_mass)
            np.testing.assert_array_equal(a.body_inertia, b.body_inertia)
        for _ in range(50):
            plain.step()
            video.step()
        torch.testing.assert_close(video.root_states, plain.root_states)
        torch.testing.assert_close(video.dof_state, plain.dof_state)
    finally:
        plain._backend.close()
        video._backend.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_record_writes_real_time_mp4(tmp_path):
    import torch

    from gym.envs.go2.go2 import Go2
    from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg
    from gym.utils.interfaces.VideoRecorder import VideoRecorder
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2Cfg()
    cfg.env.num_envs = 1
    cfg.seed = 0
    cfg.asset.render_offscreen = True  # as play.py --video
    task_registry.convert_frequencies_to_params(cfg, Go2RunnerCfg())
    env = Go2(cfg, "cpu", True, select_backend(cfg, "cpu", "mujoco"))
    path = tmp_path / "clip.mp4"
    try:
        try:
            recorder = VideoRecorder(env, path, width=320, height=180)
        except Exception as error:  # no OpenGL context on this machine
            pytest.skip(f"off-screen rendering unavailable: {error}")
        for step in range(20):
            env.step()
            recorder.capture(step)
        recorder.close()
    finally:
        env._backend.close()
    assert recorder.frames == 10  # 100 Hz control, 50 fps video
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "320,180,10"
    assert torch.isfinite(env.base_lin_vel).all()


def test_go2_uses_the_fetched_mesh_model():
    from gym.envs.go2.go2_config import Go2Cfg

    from scripts.fetch_go2_meshes import MANIFEST

    assert Go2Cfg.asset.mujoco_visual_model.endswith("go2/mjcf/go2.xml")
    assert "go2.xml" in MANIFEST and "LICENSE" in MANIFEST
    # One OBJ per material split of every URDF mesh stem.
    stems = {
        name.split("/")[1].rsplit("_", 1)[0]
        for name in MANIFEST
        if name.endswith(".obj")
    }
    assert {"base", "hip", "thigh", "thigh_mirror", "calf", "calf_mirror"} <= stems
