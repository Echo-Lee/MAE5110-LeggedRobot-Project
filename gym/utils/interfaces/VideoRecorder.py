"""Off-screen MP4 recording of environment 0 with a camera that follows the robot.

Frames are rendered with MuJoCo's off-screen renderer from the CPU backend's
environment-0 MjData (the same robot the passive viewer shows) and piped to
ffmpeg as raw RGB, so no window is needed. Set MUJOCO_GL=egl on machines
without a display. Playback runs at real time: one frame every ``stride``
control steps, with the video frame rate set to match.

The optional overlay shows elapsed time, the friction coefficient, commanded
and actual velocity, and which feet carry load, so a video can be read next
to the evaluation plots.
"""

import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FOOT_LOAD_THRESHOLD_N = 5.0
_PANEL = (20, 20, 20, 170)
_TEXT = (240, 239, 236, 255)
_MUTED = (195, 194, 183, 255)
_FOOT_ON = (57, 135, 229, 255)
_FOOT_OFF = (80, 80, 78, 255)


def frame_stride(dt: float, fps: float) -> int:
    """Control steps between frames so the video plays at real time."""
    return max(1, round(1.0 / (fps * dt)))


def overlay_lines(time_s, friction, command, velocity, estimate=None) -> list[str]:
    """Text rows: time and friction, then commanded and actual vx, vy, yaw.

    ``estimate`` (mu-hat), when given, is shown next to the true friction.
    """

    def row(name, v):
        return f"{name}  vx {v[0]:+5.2f}  vy {v[1]:+5.2f}  yaw {v[2]:+5.2f}"

    return [
        f"t {time_s:5.2f} s    mu {friction:.3f}"
        + ("" if estimate is None else f"  (est {estimate:.3f})"),
        row("cmd", command),
        row("act", velocity),
    ]


def foot_layout(positions) -> tuple[list[int], list[str]]:
    """Order feet front-to-rear, left-to-right and label them from geometry.

    ``positions`` are the feet's (x, y[, z]) in the base frame (x forward,
    y left), in the robot's own foot order. Returns indices into that order,
    sorted FL, FR, RL, RR for a quadruped (L, R for a biped), and matching
    labels, so the overlay is right whatever order a URDF lists its feet in.
    """
    xy = np.asarray(positions, dtype=float)[:, :2]
    front = xy[:, 0] > xy[:, 0].mean()
    left = xy[:, 1] > xy[:, 1].mean()
    fore_aft = len(xy) > 2
    order = sorted(
        range(len(xy)), key=lambda i: (fore_aft and not front[i], not left[i])
    )
    labels = [
        ("" if not fore_aft else "F" if front[i] else "R") + ("L" if left[i] else "R")
        for i in order
    ]
    return order, labels


def draw_overlay(frame: np.ndarray, lines, feet, font) -> np.ndarray:
    """Return ``frame`` with a translucent text panel and foot-contact boxes.

    ``feet`` is a sequence of (label, loaded) pairs, drawn left to right.
    """
    image = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    pad, line_h = 10, int(font.size * 1.35)
    box = font.size + 4
    width = max(draw.textlength(line, font=font) for line in lines)
    width = max(width, len(feet) * (box * 2 + pad))
    height = line_h * len(lines) + box + 2 * pad + 6
    draw.rectangle((8, 8, 8 + width + 2 * pad, 8 + height), fill=_PANEL)
    y = 8 + pad
    for i, line in enumerate(lines):
        draw.text((8 + pad, y), line, font=font, fill=_TEXT if i else _MUTED)
        y += line_h
    x = 8 + pad
    for label, loaded in feet:
        draw.rectangle(
            (x, y + 2, x + box, y + 2 + box), fill=_FOOT_ON if loaded else _FOOT_OFF
        )
        draw.text((x + box + 4, y + 2), label, font=font, fill=_TEXT)
        x += box * 2 + pad
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


def feet_in_base_frame(env) -> np.ndarray:
    """Environment 0's foot positions relative to its base, in the base frame."""
    from gym.utils.torch_quat import quat_rotate_inverse

    feet = env._rigid_body_pos[0, env.feet_indices] - env.root_states[0, :3]
    quat = env.base_quat[0].expand(len(feet), 4)
    return quat_rotate_inverse(quat, feet).cpu().numpy()


class VideoRecorder:
    TRACKING_DISTANCE_M = 2.0
    TRACKING_ELEVATION_DEG = -20.0
    TRACKING_AZIMUTH_DEG = 135.0

    def __init__(
        self,
        env,
        path,
        *,
        fps=50.0,
        width=960,
        height=540,
        overlay=True,
    ):
        backend = env._backend
        if not hasattr(backend, "_datas"):
            raise ValueError("--video needs the MuJoCo CPU backend (--device cpu)")
        if not backend.supports_camera_tracking:
            raise ValueError("--video follows the robot base; needs a floating base")
        if width % 2 or height % 2:
            raise ValueError("video width and height must be even for H.264")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("--video needs ffmpeg on PATH")

        self.env = env
        self.overlay = overlay
        self.stride = frame_stride(env.dt, fps)
        self.fps = 1.0 / (self.stride * env.dt)
        self._data = backend._datas[0]
        model = backend._model_for_env(0)
        # The off-screen framebuffer defaults to 640x480; grow it to fit.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self._renderer = mujoco.Renderer(model, height, width)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._camera.trackbodyid = int(model.jnt_bodyid[0])  # free joint's body
        self._camera.distance = self.TRACKING_DISTANCE_M
        self._camera.elevation = self.TRACKING_ELEVATION_DEG
        self._camera.azimuth = self.TRACKING_AZIMUTH_DEG
        self._font = ImageFont.load_default(size=max(12, height // 30))
        self._foot_order, self._foot_labels = foot_layout(feet_in_base_frame(env))
        self._ffmpeg = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{self.fps:.6g}",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def capture(self, step: int) -> None:
        """Render after control step ``step`` (0-based) if a frame is due."""
        if step % self.stride:
            return
        self._renderer.update_scene(self._data, camera=self._camera)
        frame = self._renderer.render()
        if self.overlay:
            env = self.env
            lines = overlay_lines(
                (step + 1) * env.dt,
                env._backend.contact_friction[0].item(),
                env.commands[0].tolist(),
                [
                    env.base_lin_vel[0, 0].item(),
                    env.base_lin_vel[0, 1].item(),
                    env.base_ang_vel[0, 2].item(),
                ],
                self._mu_hat(),
            )
            loads = env.contact_forces[0, env.feet_indices, 2]
            loaded = (loads > FOOT_LOAD_THRESHOLD_N).tolist()
            feet = [
                (label, loaded[i])
                for i, label in zip(self._foot_order, self._foot_labels)
            ]
            frame = draw_overlay(frame, lines, feet, self._font)
        self._ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def _mu_hat(self):
        """Estimated friction shown next to mu, for tasks that publish one."""
        estimate = getattr(self.env, "contact_friction_hat", None)
        return None if estimate is None else estimate[0, 0].item()

    def close(self) -> None:
        self._renderer.close()
        self._ffmpeg.stdin.close()
        if self._ffmpeg.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with code {self._ffmpeg.returncode}")
