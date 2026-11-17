# Changes: low-friction Go2 work

Local review file (gitignored). Baseline: commit `6283dc9`. Branch: `ti237CU/dev`.
Latest commit: `50df29c` (revert to run 12 config). Rows marked *(uncommitted)* are on disk only; *(local)* = untracked outputs in `logs/`.

> The branch trains the frozen **stage-1 baseline (run 12)** config again since `50df29c`.

| Commit | Summary |
|---|---|
| `d775cd0` | Low/high-friction training task, friction evaluation |
| `a06c68b` | Camera lock in the viewer |
| `32aaeef` | Lower entropy coefficient |
| `d8d7b9b` | Power (energy) reward |
| `78b0fd5` | Glide reward v1, video recording, matched-speed CoT evaluation |
| `1facc95` | Glide reward v2 (closes the foot-scuffing loophole) |
| `03666c4` | Glide off; run-up-and-slide episodes and slide evaluation |
| `0401678` | Run 8: slide ends after a 0.5 s hold, `slide_phase` observation, `slide_still` reward (off); legend placement in `plot_friction_eval.py` |
| `2f3bac0` | Run 9: `slide_pose` reward (weight 3); slide metrics `min_foot_loaded_fraction`, `pose_rmse_rad` |
| `f62a2a5` | Run 10: `slide_still` weight 0 → 3; slide steps count only after ≥ 50 ms in the air (`count_steps`), contact chatter reported as `contact_breaks_per_s` |
| `78b0b02` | Run 11: `slide_still` 3 → 0, `slide_pose` 3 → 10 |
| `2885576` | Run 12: residual hold (`commands.slide_action_scale` 0.2, `Go2LowFric._apply_slide_action_scale` in `_pre_decimation_step`); `slide_pose` back to 3 |
| `a742102` | `scripts/train.py` saves `seed.json` (seed + whether from `--seed` or random) in each run folder; test `tests/unit_tests/test_train_seed.py` |
| `a742102` | Run 13: `commands.turn_in_place_probability` 0.1 + `turn_in_place_min_yaw` 0.5 (`Go2LowFric._resample_commands`); 4 tests |
| `3ace272` | Run 14: `turn_in_place_min_friction` 0.6 (ground-only turns, true μ); robustness suite `scripts/evaluate_slide_robustness.py` (9 scenarios, grading, `--regrade`); `scripts/evaluate_slide.py` refactored (`make_slide_env`, `SlidePerturbations`, `trial_seconds`); per-replica slide start in `summarize_slide`; tests `test_slide_robustness.py` |
| `50df29c` | **Revert to run 12 config**: `turn_in_place_probability` 0.1 → 0 (option and code kept; runs 13, 13b, 14 all broke the slide); slide metric `no_runup_rate` + "no run-up" grading; notebook Stage 1 result |
| *(uncommitted)* | **State estimator (stage 2a):** `learning/modules/history_estimator.py` (`ProprioHistory`, `HistoryStateEstimator` predicting μ̂ and v̂, `EstimatorInTheLoop`), `scripts/train_state_estimator.py` (online supervised training with the frozen policy, DAgger phase, validation report); tests `tests/unit_tests/test_history_estimator.py` (moved from `learning/modules/`) |
| *(uncommitted)* | `--state_estimator` (policy reads μ̂) in `evaluate_slide.py`, `evaluate_slide_robustness.py`, `evaluate_go2_friction.py`, `play.py`; video overlay shows μ̂; eval reports store Path arguments as strings |
| *(uncommitted)* | **Friction switches (stage 2b):** `friction_switch` config (probability 0 for policy training, `time_range` [2, 8] s); `Go2LowFric._sample_friction_switches` / `_apply_friction_switch` / `schedule_friction_switch`, `time_since_friction_change` buffer (one switch per episode to the other bin, never in slide episodes); `scripts/evaluate_friction_switch.py` (6 transitions × 4 commands, falls, tracking before/transient/adapted, μ̂ settling); `train_state_estimator.py --friction_switch_probability` (validation by time since switch, output `state_estimator_switch/`); `EstimatorInTheLoop(feed_policy=False)` and `--estimator_observe_only`; 4 new tests |
| *(uncommitted)* | **Student fine-tune (stage 2c):** task `go2_lowfric_est`: `gym/envs/go2/go2_lowfric_est.py` (`Go2LowFricEst`: in-env estimator, `contact_friction_hat` / `base_lin_vel_hat`, `set_learning`, `checkpoint_state`, `load_checkpoint_state`), `gym/envs/go2/go2_lowfric_est_config.py` (actor μ → μ̂ in place, switches p 0.3, estimator settings), registered in `gym/envs/__init__.py`; `ConcurrentEstimator` in `history_estimator.py` (online training under inference mode; out-of-place history updates); `OnPolicyRunner` optional env hooks (learning toggle, env state saved in checkpoints); eval scripts and `play.py` turn random switches off; `evaluate_slide.py` noise and μ̂ records for the in-env estimator; overlay shows the task's own μ̂; `EstimatorInTheLoop` refuses tasks with their own estimator; tests `tests/unit_tests/test_go2_lowfric_est.py` (5) + 2 in `test_history_estimator.py` |
| *(uncommitted)* | **Continuous friction (run 16):** `go2_lowfric_est` bins ice / middle (0.2–0.6) / ground, weights 0.4 / 0.25 / 0.35 (`MIDDLE_FRICTION`); `go2_lowfric` keeps two bins (run 12 baseline); friction switches go to any other bin, weighted by bin weights; 2 new tests |
| *(uncommitted)* | **Eval fix + latch (stage 2d, run 17):** `Go2LowFric.external_slide_phase` (set by `evaluate_slide.py` and `play.py --slide`, so the in-env estimator sees the scripted slide phase); `state_estimator.latch_in_slide` in `Go2LowFricEst` (μ̂ held from slide start to phase end; on in the config, off for run 15/16 snapshots); eval reports, switch eval and overlay record the policy's μ̂ input (`contact_friction_hat`); 2 new tests |
| *(uncommitted)* | **Accelerometer (stage 2e, runs 18a/18b):** `Go2LowFric.base_lin_acc` (IMU specific force, body frame, g units; `_post_physics_step`, reset handling); `LEGACY_FEATURES` / `DEFAULT_FEATURES` (+ `base_lin_acc`), `expand_estimator_inputs` (zero-weight widening), `ConcurrentEstimator.set_model` and first-batch normalization of new inputs; `Go2LowFricEst` widens older estimators and stores `estimator_config` in checkpoints; config `features` comment; 5 new tests |
| *(local)* | Runs 18a `Nov15_23-24-23_est_accel_latch/`, 18b `Nov15_23-24-48_est_accel_nolatch/` (`model_3500.pt`); reports `robustness_run18{a,b}.json`, `lowfric_run18{a,b}.json`, `switch_run18{a,b}.json`; re-recorded videos `run15_slide_mu0.1_v1.2.mp4`, `run16_slide_mu0.2_v1.2.mp4` |
| *(local)* | Run 19 control `Nov16_00-28-02_est_noaccel_control/`, run 18a seed 2 `Nov16_00-28-29_est_accel_latch_seed2/`; reports `{robustness,lowfric,switch}_run19_control.json`, `..._run18a_seed2.json`; videos `run18a_slide_mu{0.1,0.2}_v1.2.mp4`, `run15_slide_mu0.2_v1.2.mp4` |
| *(local)* | Run 17 `logs/go2_lowfric/Nov15_22-56-19_est_latch/`; reports `robustness_run{15,16}_phasefix.json`, `robustness_run16_latch_noretrain.json`, `robustness_run17.json`, `lowfric_run17.json`, `switch_run17.json` |
| *(local)* | Run 16 `logs/go2_lowfric/Nov15_22-14-34_est_continuous/` (`model_2500.pt`); reports `switch_run16.json`, `robustness_run16.json`, `lowfric_run16.json` |
| *(local)* | Run 15 `logs/go2_lowfric/Nov15_21-34-07_est_finetune/` (`model_2000.pt` holds policy + estimator); reports `switch_run15.json`, `robustness_run15.json`, `lowfric_run15.json`; video `logs/videos/stage1/run15_slide_mu0.1_v1.2.mp4` |
| *(local)* | Estimator `logs/go2_lowfric/Nov15_19-11-13_/state_estimator_switch/`; reports `switch_run12_{truemu,estimator,estimator_observe,estimator_switch}.json`, `robustness_run12_estimator_switch.json`, `lowfric_slide12_estimator_switch.json` |
| *(local)* | Estimator `logs/go2_lowfric/Nov15_19-11-13_/state_estimator/` (`estimator.pt`, `validation.json`); reports `robustness_run12_estimator.json`, `lowfric_slide12_estimator.json`; video `run12_estimator_slide_mu0.1_v1.2.mp4` |
| *(local)* | Stage 1 frozen on run 12 (`Nov15_19-11-13_`): videos `logs/videos/stage1/`, figures `fig6_slide_progression.png`, `fig7_robustness_run12.png`; `robustness_run12.json` re-graded |

Run tests (`env -u PYTHONPATH` keeps the ROS Python path out of the venv):

```bash
env -u PYTHONPATH uv run --frozen python -m pytest -q
env -u PYTHONPATH uv run --frozen python -m pytest gym learning -q
```

## 1. Training task

| File | Commits | What it does |
|---|---|---|
| `gym/envs/go2/go2_lowfric.py` | `d775cd0`, `78b0fd5`, `1facc95`, `03666c4` | `Go2LowFric` task. `contact_friction` observation (true μ, privileged). Foot contact timer and touchdown flags. `_reward_glide` (v2: slide × load share × persistence) and `_reward_ice_touchdowns`. Run-up-and-slide episodes: `slide_speed_profile()`, `_sample_slide_episodes()`, `_apply_slide_commands()`; slide ends after `slide_stop_hold_time` and random commands resume; `slide_phase` observation + `slide_phase_profile()`; `_reward_slide_still`, `_reward_slide_pose`; residual hold `_apply_slide_action_scale()`; turn-in-place sampling in `_resample_commands()` (off). |
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
