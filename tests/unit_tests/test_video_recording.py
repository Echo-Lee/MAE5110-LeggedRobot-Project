"""Playback recording: CLI options, overlay drawing, and an end-to-end MP4."""

import shutil
import subprocess

import numpy as np
import pytest
from PIL import ImageFont

from gym.utils.interfaces.VideoRecorder import (
    draw_overlay,
    frame_stride,
    overlay_lines,
)
from scripts.play import get_play_args


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
        get_play_args(["--task", "go2osc_push_glide", "--friction", "0"])


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
    both = draw_overlay(frame, lines, [True, False, False, False], font)
    none = draw_overlay(frame, lines, [False, False, False, False], font)
    assert both.shape == frame.shape and both.dtype == np.uint8
    assert np.array_equal(both[-20:, -20:], frame[-20:, -20:])  # untouched
    assert not np.array_equal(both, none)  # a loaded foot box is drawn lit


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
