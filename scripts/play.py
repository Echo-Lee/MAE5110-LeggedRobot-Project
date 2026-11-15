"""Play back a policy trained with a supported Q2 physics backend.

Usage:
    uv run scripts/play.py --task pendulum
    uv run scripts/play.py --task mini_cheetah --num_envs 8
    uv run scripts/play.py --task mini_cheetah --load_run May08_12-34-56_ \
                           --checkpoint 1500

Supports MuJoCo CPU/Warp and optional VSim checkpoints.

Record a real-time MP4 of environment 0 instead of opening a viewer, on a
chosen surface and with a fixed command (MuJoCo CPU backend):
    uv run scripts/play.py --task go2_lowfric --load_run RUN --original_cfg \
        --friction 0.1 --command 1.0 0 0 --video logs/videos/ice.mp4
"""

import argparse
import os
import random

import torch

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def get_play_args(argv=None):
    parser = argparse.ArgumentParser(description="Play a policy trained with Q2")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="environments to simulate (default: 32, or 1 with --video)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Checkpoint root under logs/ (default: task config value)",
    )
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Run directory name under logs/<experiment_name>/ (default: latest)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Model iteration index (default: latest)",
    )
    parser.add_argument(
        "--original_cfg",
        action="store_true",
        help="Load environment and runner configs saved with the selected run.",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--backend",
        type=str,
        default="mujoco",
        choices=["mujoco", "vsim"],
        help="Physics backend (vsim is CUDA-only; start via "
        "`uv run --env-file .env.vsim ...`)",
    )
    parser.add_argument(
        "--viewer_ui",
        action="store_true",
        default=False,
        help="show the MuJoCo viewer's side panels (hidden by default: their "
        "shortcuts bind most letters to visualisation toggles, which fire "
        "alongside keyboard teleop)",
    )
    parser.add_argument(
        "--follow_camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lock the MuJoCo viewer camera onto the robot (environment 0); "
        "mouse still orbits and zooms around it, F toggles it during "
        "playback. Use --no-follow_camera for a fixed camera.",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="fix every environment's contact friction to this coefficient, "
        "replacing the task's friction randomization",
    )
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "YAW"),
        help="hold this velocity command (m/s, m/s, rad/s) for the whole "
        "playback; disables keyboard teleop",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="record environment 0 to this MP4 path instead of opening a "
        "viewer (MuJoCo CPU backend; needs ffmpeg)",
    )
    parser.add_argument(
        "--video_seconds",
        type=float,
        default=10.0,
        help="length of the recording in simulated seconds",
    )
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw time, friction, command vs actual velocity, and foot "
        "contacts on the video. Use --no-overlay for a clean video.",
    )
    parser.add_argument(
        "--keyboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keyboard teleop in the MuJoCo viewer (see "
        "gym/utils/interfaces/MujocoKeyboardInterface.py for key bindings). "
        "Use --no-keyboard to disable.",
    )
    args = parser.parse_args(argv)
    if args.num_envs is None:
        args.num_envs = 1 if args.video else 32
    if args.video:
        # A recording renders off-screen; no interactive viewer is opened.
        args.headless = True
    if args.friction is not None and args.friction <= 0:
        parser.error("--friction must be positive")
    if args.video_seconds <= 0:
        parser.error("--video_seconds must be positive")
    return args


def setup(args):
    # Register tasks (imports the task classes).
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = task_registry.get_cfgs(
        args.task,
        original_cfg=args.original_cfg,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
    )

    # Play-time overrides — small batch, long episodes, no pushes.
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = 50
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.show_ui = args.viewer_ui
    if args.video:
        # Per-environment mass randomization makes the CPU backend discard
        # visual assets when headless (no ground texture, skybox, or lights),
        # which hides sliding in a video. Record at nominal link masses.
        env_cfg.domain_randomization.startup.link_mass_scale_range = None
    if args.friction is not None:
        # Use the friction axis as a zero-width envelope: every environment
        # starts on exactly this coefficient and keeps it, and the backend
        # still allocates per-environment friction.
        startup = env_cfg.domain_randomization.startup
        startup.contact_friction_range = [args.friction, args.friction]
        startup.contact_friction_bins = None
        startup.contact_friction_bin_weights = None
        env_cfg.domain_randomization.episode.resample_contact_friction = False

    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    else:
        env_cfg.seed = random.randint(0, 10000)
        train_cfg.seed = env_cfg.seed

    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env(
        args.task,
        env_cfg,
        device=args.device,
        headless=args.headless,
        backend=args.backend,
    )

    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner


def policy_step(env, runner, command):
    if command is not None:
        # Re-pin every step: the task's own command sampler may fire.
        env.commands[:] = command
    runner.set_actions(
        runner.actor_cfg["actions"],
        runner.get_inference_actions(),
        runner.actor_cfg["disable_actions"],
    )
    env.step()


def play(env, runner, command=None):
    n_steps = 10 * int(env.max_episode_length)
    for _ in range(n_steps):
        policy_step(env, runner, command)
        env.check_exit()


def record(env, runner, args, command=None):
    from gym.utils.interfaces.VideoRecorder import VideoRecorder

    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    recorder = VideoRecorder(env, args.video, overlay=args.overlay)
    try:
        for step in range(round(args.video_seconds / env.dt)):
            policy_step(env, runner, command)
            recorder.capture(step)
    finally:
        recorder.close()
    print(
        f"wrote {args.video}: {recorder.frames} frames at {recorder.fps:g} fps, "
        f"friction {env._backend.contact_friction[0].item():.3f}, "
        "nominal link masses"
    )


if __name__ == "__main__":
    args = get_play_args()
    if args.video and "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
        # Headless machines (e.g. MoLab) need EGL for off-screen rendering.
        # MuJoCo reads this when first imported, which happens in setup().
        os.environ["MUJOCO_GL"] = "egl"
    with torch.no_grad():
        env, runner = setup(args)
        command = None
        if args.command is not None:
            if not hasattr(env, "commands"):
                raise SystemExit(f"--command: task {args.task} has no commands")
            command = torch.tensor(args.command, device=env.device)
        # The MuJoCo warp (GPU) backend is headless-only: render() is a no-op and
        # it has no passive viewer, so nothing draws.  Point the user at cpu.
        if (
            args.backend == "mujoco"
            and not args.headless
            and not hasattr(env._backend, "_viewer_overlay_fn")
        ):
            print(
                "WARNING: the MuJoCo warp (GPU) backend has no interactive viewer "
                "— nothing will render. Use --device cpu for playback."
            )
        if args.follow_camera and not args.headless:
            # Fixed-base tasks (pendulum, cartpole) have nothing to follow
            # and keep their configured camera silently.
            if not hasattr(env._backend, "set_camera_tracking"):
                print(
                    "WARNING: --follow_camera is only supported by the MuJoCo "
                    "CPU viewer; keeping the configured camera."
                )
            elif env._backend.supports_camera_tracking:
                env._backend.set_camera_tracking(True)
        if (
            args.keyboard
            and args.command is None
            and hasattr(env, "commands")
            and not args.headless
        ):
            # Each viewer has its own input model: MuJoCo dispatches key
            # events via a callback, vlearn polls key state per frame.
            if args.backend == "vsim":
                from gym.utils.interfaces.VsimKeyboardInterface import (
                    VsimKeyboardInterface,
                )

                VsimKeyboardInterface(env)
            else:
                from gym.utils.interfaces.MujocoKeyboardInterface import (
                    MujocoKeyboardInterface,
                )

                MujocoKeyboardInterface(env)
        if hasattr(env, "commands") and not args.headless:
            # Same indicators, different drawing primitives per viewer.
            if args.backend == "vsim":
                from gym.utils.interfaces.VsimCommandVisualizer import (
                    VsimCommandVisualizer,
                )

                VsimCommandVisualizer(env)
            else:
                from gym.utils.interfaces.CommandVisualizer import CommandVisualizer

                CommandVisualizer(env)
        if args.video:
            record(env, runner, args, command)
        else:
            play(env, runner, command)
