# Changes: low-friction Go2 work

Local review file (gitignored). Baseline: commit `6283dc9`. Branch: `ti237CU/dev`.
Everything below is committed as of `03666c4`, except the **run 8** changes marked *(uncommitted)*.

| Commit | Summary |
|---|---|
| `d775cd0` | Low/high-friction training task, friction evaluation |
| `a06c68b` | Camera lock in the viewer |
| `32aaeef` | Lower entropy coefficient |
| `d8d7b9b` | Power (energy) reward |
| `78b0fd5` | Glide reward v1, video recording, matched-speed CoT evaluation |
| `1facc95` | Glide reward v2 (closes the foot-scuffing loophole) |
| `03666c4` | Glide off; run-up-and-slide episodes and slide evaluation |
| *(uncommitted)* | Run 8: slide ends after a 0.5 s hold, `slide_phase` observation, `slide_still` reward (off); legend placement in `plot_friction_eval.py` |

Run tests (`env -u PYTHONPATH` keeps the ROS Python path out of the venv):

```bash
env -u PYTHONPATH uv run --frozen python -m pytest -q
env -u PYTHONPATH uv run --frozen python -m pytest gym learning -q
```

## 1. Training task

| File | Commits | What it does |
|---|---|---|
| `gym/envs/go2/go2_lowfric.py` | `d775cd0`, `78b0fd5`, `1facc95`, `03666c4` | `Go2LowFric` task. `contact_friction` observation (true μ, privileged). Foot contact timer and touchdown flags. `_reward_glide` (v2: slide × load share × persistence) and `_reward_ice_touchdowns`. Run-up-and-slide episodes: `slide_speed_profile()`, `_sample_slide_episodes()`, `_apply_slide_commands()`. *(uncommitted)* slide ends after `slide_stop_hold_time` and random commands resume; `slide_phase` observation + `slide_phase_profile()`; `_reward_slide_still` (joint velocity penalty, slide phase only). |
| `gym/envs/go2/go2_lowfric_config.py` | all except `a06c68b` | Friction bins ice [0.05, 0.2] / ground [0.6, 1.2] redrawn each episode; 10 s episodes; μ in actor and critic inputs; `entropy_coef` 0.002 (was 0.01); `power` 1e-3, `torques` 1.5e-3 (was 5e-6); `glide` and `ice_touchdowns` **0** (trained values 1.0 / 0.5 in comments); glide settings; `commands.slide_*` (30% of ice episodes, v0 0.8–1.3 m/s, 4 s run-up, decay at μg). *(uncommitted)* `slide_stop_hold_time` 0.5 s; `slide_phase` in actor and critic inputs; `slide_still` weight 0. |
| `gym/envs/__init__.py` | `d775cd0` | Registers task `go2_lowfric`. |
| `gym/envs/base/domain_randomization.py` | `d775cd0` | Friction bins + weights, per-episode friction resampling (no GPU sync), bin validation. Off by default for other tasks. |
| `gym/envs/base/legged_robot.py` | `d8d7b9b` | `_reward_power`: −Σ\|τ·q̇\| (W). Active only where weighted. |

## 2. Evaluation

| File | Commits | What it does |
|---|---|---|
| `gym/utils/locomotion_metrics.py` | `d775cd0`, `78b0fd5`, `03666c4` | Pure metric functions: `summarize_locomotion` (falls, tracking, slip, CoT, touchdowns, duty, torque saturation), `contact_point_velocity` (rolling ≠ slipping), `replica_speed_and_cot` + `cot_at_speed` (CoT at exact actual speed), `summarize_slide` (slide distance/decel vs friction physics, feet down, touchdowns, energy). |
| `scripts/evaluate_go2_friction.py` | `d775cd0`, `03666c4` | 7 μ × 6 commands × replicas; JSON; `--check` grades 30 cases; `--misreport_friction` tests reliance on μ. |
| `scripts/plot_friction_eval.py` | `d775cd0` | Metric-vs-μ panels for several reports + Markdown table. *(uncommitted)* legend below the panels. |
| `scripts/evaluate_cot_vs_friction.py` | `78b0fd5`, `03666c4` | Matched-speed CoT: sweeps commanded speeds, interpolates CoT at 0.5 / 1.0 / 1.5 m/s actual speed; top speed per μ. |
| `scripts/plot_cot_vs_friction.py` | `78b0fd5` | One panel per target speed + top-speed panel; `--positive_work`. |
| `scripts/evaluate_slide.py` | `03666c4` | Run-up-and-slide from rest: 5 μ × 2 speeds × 8 replicas; distance vs v²/(2μg), decel vs μg, all-feet-down fraction, touchdowns/s, energy/m. *(uncommitted)* sets `slide_phase` for policies that observe it; `--hold`. |

## 3. Playback and video

| File | Commits | What it does |
|---|---|---|
| `gym/envs/base/mujoco_cpu_backend.py` | `a06c68b` | Camera follows env 0's base (`set_camera_tracking`). |
| `gym/utils/interfaces/MujocoKeyboardInterface.py` | `a06c68b` | `F` toggles the camera lock. |
| `gym/utils/interfaces/VideoRecorder.py` | `78b0fd5` | Off-screen render → ffmpeg MP4, tracking camera, overlay (t, μ, command vs actual, loaded feet). |
| `scripts/play.py` | `a06c68b`, `78b0fd5`, `03666c4` | `--follow_camera`, `--friction`, `--command`, `--slide` / `--slide_runup`, `--video` / `--video_seconds` / `--overlay`. Random slide episodes off in playback. *(uncommitted)* `--slide` also sets `slide_phase`; `--slide_hold`. |

## 4. Tests

| File | Commits | Covers |
|---|---|---|
| `gym/envs/base/test_domain_randomization.py` | `d775cd0` | Bins, weighted sampling, per-episode resampling, overrides. |
| `tests/unit_tests/test_domain_randomization.py` | `d775cd0` | Config helper with new friction fields. |
| `tests/unit_tests/test_go2_lowfric.py` | `d775cd0`, `03666c4` | μ observation; slide profile, ice-only sampling, profile through resamples, old configs never slide. *(uncommitted)* phase profile, phase on only while sliding, slide ends and commands resume, run 7 configs keep holding, `slide_still`, play schedule. |
| `gym/utils/test_locomotion_metrics.py` | `d775cd0`, `78b0fd5`, `03666c4` | Metric math incl. CoT interpolation and slide physics (coasting ≈ 1, stepping, half braking ≈ 2). |
| `tests/unit_tests/test_power_reward.py` | `d8d7b9b` | Power reward zero point, units, actuated joints. |
| `tests/unit_tests/test_mujoco_camera_tracking.py` | `a06c68b` | Camera lock/release, fixed base refused, `F` key. |
| `tests/unit_tests/test_glide_reward.py` | `78b0fd5`, `1facc95`, `03666c4` | Glide v2 and touchdown penalty (21 tests); weights now 0. |
| `tests/unit_tests/test_video_recording.py` | `78b0fd5` | Video CLI options, overlay, end-to-end MP4 (ffprobe). |

## 5. Docs and config

| File | Commits | What it does |
|---|---|---|
| `genAI_skills/README_MUJOCO.md` | `d775cd0`, `a06c68b`, `78b0fd5`, `03666c4` | Task table row; friction eval, camera lock, playback flags, matched-speed CoT, slide episodes and evaluation. |
| `.gitignore` | `78b0fd5`, `03666c4` | Ignores `changes.md`, `designNotebook.md`. |

## 6. Created, then removed

- **`notebooks/molab_lowfric.py`**: MoLab train/evaluate/plot notebook. Never committed; removed on your side.
- **Stand-still command sampling** (`commands.stand_still_probability`): removed on request, to revisit later.

## Local outputs (gitignored)

| Path | Contents |
|---|---|
| `logs/go2/`, `logs/go2_lowfric/` | Unpacked training runs |
| `logs/eval/*.json`, `*.png` | Evaluation reports and plots |
| `logs/videos/*.mp4` | Rollout videos |
