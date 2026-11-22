"""Record the included Go2 policy accelerating and receiving a zero command."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gym.envs  # noqa: E402, F401
from gym.envs.base.domain_randomization import (  # noqa: E402
    set_domain_randomization_range,
)
from gym.utils.helpers import set_seed  # noqa: E402
from gym.utils.task_registry import task_registry  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dir", type=Path, default=ROOT / "policies/go2osc_push_glide"
    )
    parser.add_argument("--friction", type=float, default=0.05)
    parser.add_argument("--speed", type=float, default=2.5)
    parser.add_argument("--switch-time", type=float, default=14.0)
    parser.add_argument("--duration", type=float, default=26.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ramp-start-speed", type=float)
    parser.add_argument("--ramp-start-time", type=float, default=6.0)
    parser.add_argument("--ramp-end-time", type=float, default=12.0)
    parser.add_argument("--no-video", action="store_true", help="Save motion data only")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/go2osc/command_stop.mp4",
    )
    args = parser.parse_args()
    numeric = (args.friction, args.speed, args.switch_time, args.duration)
    if not all(math.isfinite(x) for x in numeric):
        parser.error("Friction, speed and times must be finite")
    if args.friction < 0 or args.speed < 0:
        parser.error("Friction and forward speed must be nonnegative")
    if not 0 < args.switch_time < args.duration:
        parser.error("Times must satisfy 0 < switch-time < duration")
    if args.ramp_start_speed is not None:
        if not math.isfinite(args.ramp_start_speed) or args.ramp_start_speed < 0:
            parser.error("Ramp starting speed must be finite and nonnegative")
        if not 0 <= args.ramp_start_time < args.ramp_end_time <= args.switch_time:
            parser.error("Ramp must finish before the zero-command switch")
    if args.output.suffix != ".mp4":
        parser.error("Output must have an .mp4 extension")
    if not args.no_video and shutil.which("ffmpeg") is None:
        parser.error("Video recording requires ffmpeg on PATH")
    return args


def restore_config(target, saved):
    """Restore nested config classes, replacing dictionary-valued settings."""
    for name, value in saved.items():
        current = getattr(target, name, None)
        if isinstance(value, dict) and hasattr(current, "__dict__"):
            restore_config(current, value)
        else:
            setattr(target, name, copy.deepcopy(value))


def load_policy(args):
    metadata = json.loads((args.policy_dir / "policy.json").read_text())
    saved = json.loads((args.policy_dir / metadata["configuration"]).read_text())
    checkpoint = args.policy_dir / metadata["checkpoint"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("Checkpoint SHA256 does not match policy.json")
    task = metadata["task"]
    if saved["schema_version"] != 1 or saved["task"] != task:
        raise ValueError("Unsupported or mismatched saved policy configuration")
    cfg, training = copy.deepcopy(task_registry.get_cfgs(task))
    restore_config(cfg, saved["env_config"])
    restore_config(training, saved["runner_config"])
    # This checkpoint predates the optional steering-specific clock schedule.
    for name in ("sideways_stop_threshold", "turning_stop_threshold"):
        setattr(cfg.osc, name, saved["env_config"]["osc"].get(name))
    training.critic.reward.weights.tilt_rate = saved["runner_config"]["critic"][
        "reward"
    ]["weights"].get("tilt_rate", 0.0)
    cfg.env.num_envs = 1
    cfg.env.episode_length_s = args.duration + 5.0
    cfg.commands.resampling_time = args.duration + 10.0
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.init_state.lin_vel = cfg.init_state.ang_vel = [0.0, 0.0, 0.0]
    cfg.push_robots.toggle = False
    # Real Go2 meshes when installed (scripts/fetch_go2_meshes.py); the
    # backend falls back to collision shapes if they are missing. The saved
    # configuration predates them, so set them here.
    cfg.asset.mujoco_visual_model = "{GYM_ROOT_DIR}/resources/robots/go2/mjcf/go2.xml"
    cfg.asset.render_offscreen = True
    cfg.terrain.static_friction = cfg.terrain.dynamic_friction = args.friction
    set_domain_randomization_range(
        cfg, "contact_friction_range", [args.friction, args.friction]
    )
    cfg.seed = training.seed = args.seed
    training.runner.device = "cpu"
    training.runner.resume = False
    training.logging.enable_local_saving = False
    training.log_dir = None
    task_registry.convert_frequencies_to_params(cfg, training)
    set_seed(args.seed)
    env = task_registry.make_env(task, cfg, "cpu", True, "mujoco")
    try:
        runner = task_registry.make_alg_runner(env, training)
        runner.load(checkpoint, load_optimizer=False)
        runner.switch_to_eval()
        set_seed(args.seed)
        # Reset once without the implicit physics step in the public reset API.
        env._explicit_reset = True
        try:
            env._reset_idx(torch.ones(1, dtype=torch.bool))
        finally:
            env._explicit_reset = False
        env._reset_buffers()
        np.testing.assert_allclose(env.root_states[:, 7:13].numpy(), 0)
        np.testing.assert_allclose(env._backend.contact_friction.numpy(), args.friction)
        np.testing.assert_allclose(env.contact_friction_obs.numpy(), args.friction)
        return env, runner, metadata
    except Exception:
        env._backend.close()
        raise


def grid(scene, center):
    center = np.round(center / 0.1) * 0.1
    for axis in (0, 1):
        for offset in range(-16, 17):
            if scene.ngeom >= scene.maxgeom:
                return
            start = np.array([center[0], center[1], 0.001])
            end = start.copy()
            start[axis] -= 1.6
            end[axis] += 1.6
            start[1 - axis] += offset * 0.1
            end[1 - axis] += offset * 0.1
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0.6, 0.65, 0.7, 1.0]),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                1.4 if offset % 5 == 0 else 0.6,
                start,
                end,
            )
            scene.ngeom += 1


def main():
    args = arguments()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env, runner, metadata = load_policy(args)
    renderer = writer = encoder_log = None
    rows = []
    failure_time = None
    width, height, fps = 1280, 720, 50
    stride = round(1.0 / fps / env.dt)
    try:
        if not args.no_video:
            from PIL import Image, ImageDraw, ImageFont

            fonts = [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            font_path = next((p for p in fonts if Path(p).is_file()), None)

            def font(size):
                return (
                    ImageFont.truetype(font_path, size)
                    if font_path
                    else ImageFont.load_default(size=size)
                )

            large, medium, small = font(27), font(25), font(21)
            model = env._backend._model_for_env(0)
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            model.vis.headlight.active = 1
            model.vis.headlight.ambient[:] = 0.6
            model.vis.headlight.diffuse[:] = 0.8
            meshes = model.nmesh > 0  # real Go2 visuals were loaded
            if not meshes:
                model.geom_rgba[model.geom_bodyid != 0] = [0.25, 0.43, 0.63, 1]
            model.geom_rgba[model.geom_bodyid == 0] = [0.75, 0.78, 0.81, 1]
            renderer = mujoco.Renderer(model, height=height, width=width)
            option = mujoco.MjvOption()
            option.geomgroup[:] = 1
            if meshes:
                # Keep the meshes' own colors; hide the robot's collision
                # shapes (group 3), which would otherwise overlap them.
                option.geomgroup[3] = 0
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 1.8, 115, -20
            encoder_log = args.output.with_suffix(".encoder.log").open("w")
            writer = subprocess.Popen(
                [
                    shutil.which("ffmpeg"),
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
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(args.output),
                ],
                stdin=subprocess.PIPE,
                stderr=encoder_log,
            )

        with torch.inference_mode():
            for step in range(round(args.duration / env.dt)):
                start = step * env.dt
                command = args.speed
                if args.ramp_start_speed is not None:
                    fraction = np.clip(
                        (start - args.ramp_start_time)
                        / (args.ramp_end_time - args.ramp_start_time),
                        0,
                        1,
                    )
                    command = float(
                        args.ramp_start_speed
                        + fraction * (args.speed - args.ramp_start_speed)
                    )
                if start >= args.switch_time - 1e-9:
                    command = 0.0
                env.commands[:] = torch.tensor([command, 0.0, 0.0])
                runner.set_actions(
                    runner.actor_cfg["actions"],
                    runner.get_inference_actions(),
                    runner.actor_cfg["disable_actions"],
                )
                env.step()
                when = (step + 1) * env.dt
                root = env.root_states[0].numpy().copy()
                tilt = math.degrees(
                    math.acos(float((-env.projected_gravity[0, 2]).clamp(-1, 1)))
                )
                rows.append(
                    [
                        when,
                        command,
                        *root,
                        float(env.base_lin_vel[0, 0]),
                        tilt,
                        bool(env.terminated[0]),
                        bool(env.timed_out[0]),
                    ]
                )
                if env.terminated.any() and failure_time is None:
                    failure_time = when
                    print(f"Termination contact at {when:.2f} s; recording continues.")
                if env.timed_out.any():
                    raise RuntimeError("Unexpected episode timeout")
                # No physical reset, phase reset or substituted control at the switch.
                if writer is not None and step % stride == stride - 1:
                    camera.lookat[:] = root[:3]
                    camera.lookat[2] = 0.25
                    renderer.update_scene(
                        env._backend._datas[0], camera=camera, scene_option=option
                    )
                    grid(renderer.scene, root[:2])
                    canvas = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0, 0, width, 106), fill=(17, 23, 32))
                    draw.text(
                        (22, 12),
                        f"Go2 | Policy: checkpoint {metadata['iteration']}"
                        f" | Ground friction: {args.friction:g}",
                        font=large,
                        fill="white",
                    )
                    stage = "BUILDING SPEED" if command else "ZERO VELOCITY COMMAND"
                    if failure_time is not None:
                        stage = "TERMINATION CONTACT DETECTED"
                    draw.text(
                        (22, 55),
                        f"{stage} | t = {when:.2f} s | command = {command:.2f} m/s",
                        font=medium,
                        fill=(255, 199, 90) if command == 0 else (95, 225, 175),
                    )
                    draw.rectangle((0, height - 90, width, height), fill=(17, 23, 32))
                    draw.text(
                        (22, height - 80),
                        f"Measured forward speed: {float(env.base_lin_vel[0, 0]):.2f}"
                        f" m/s | planar speed: {np.linalg.norm(root[7:9]):.2f} m/s",
                        font=medium,
                        fill="white",
                    )
                    draw.text(
                        (22, height - 42),
                        f"Command becomes zero at "
                        f"{args.switch_time:.1f} s | Same policy throughout",
                        font=small,
                        fill=(205, 215, 225),
                    )
                    writer.stdin.write(np.asarray(canvas).tobytes())
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("Video encoding failed; inspect the encoder log")
            writer = None
        trace = np.asarray(rows)
        np.savez_compressed(args.output.with_suffix(".npz"), control_trace=trace)
        before = trace[round(args.switch_time / env.dt) - 1]
        report = {
            "checkpoint": metadata["checkpoint"],
            "checkpoint_sha256": metadata["sha256"],
            "friction": args.friction,
            "seed": args.seed,
            "control_dt_s": env.dt,
            "command_forward_m_s": args.speed,
            "command_switch_s": args.switch_time,
            "command_after_switch": [0, 0, 0],
            "duration_s": args.duration,
            "ramp_start_speed_m_s": args.ramp_start_speed,
            "ramp_start_time_s": args.ramp_start_time
            if args.ramp_start_speed is not None
            else None,
            "ramp_end_time_s": args.ramp_end_time
            if args.ramp_start_speed is not None
            else None,
            "initial_velocity": "zero",
            "oscillator_noise": "saved configuration",
            "actor": "deterministic mean, clean observations",
            "failure_time_s": failure_time,
            "planar_speed_at_switch_m_s": float(np.linalg.norm(before[9:11])),
            "net_planar_displacement_after_switch_m": float(
                np.linalg.norm(trace[-1, 2:4] - before[2:4])
            ),
            "final_planar_speed_m_s": float(np.linalg.norm(trace[-1, 9:11])),
            "trace_columns": [
                "time_s",
                "command_vx_m_s",
                "root_x",
                "root_y",
                "root_z",
                "quat_x",
                "quat_y",
                "quat_z",
                "quat_w",
                "world_vx",
                "world_vy",
                "world_vz",
                "world_wx",
                "world_wy",
                "world_wz",
                "body_vx",
                "tilt_deg",
                "terminated",
                "timed_out",
            ],
        }
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        if writer is not None:
            writer.stdin.close()
            writer.wait()
        if renderer is not None:
            renderer.close()
        if encoder_log is not None:
            encoder_log.close()
        env._backend.close()


if __name__ == "__main__":
    main()
