# Commanded Slide: To Do

Goal: a commanded, controlled slide on slippery ground with heading tracking, built on the learned state estimator (`go2osc_push_glide_est`, branch `ti237CU/state-estimator`).

## Physics to design around

While all feet slide (kinetic friction μ), with centre-of-mass height h:

| Quantity | Formula |
|---|---|
| Deceleration | a = μg |
| Stopping distance | d = v² / (2μg) |
| Slide duration | t = v / (μg) |
| Forward pitching moment while braking | ≈ m·μg·h |

## Status (2026-11-21)

- [x] Synced `ti237CU/slide` with `main` 77f6b4b (PR #4 brought sliding stops, fast slides, carve/turn-rate tracking: `go2osc_push_glide_robust`). Old staged work saved as branch `ti237CU/slide-staged-backup`.
- [x] New task `go2osc_push_glide_robust_est` (`gym/envs/go2/go2osc_robust_est*.py`): robust ground + in-env estimator. Actor swaps in place `foot_traction_obs -> foot_traction_est`, `friction_estimate_obs -> contact_friction_hat`, `base_lin_vel -> base_lin_vel_hat` (same sizes, so robust checkpoints load). Speed clock brakes/carves with mu-hat. Est friction switch off (robust per-foot redraws instead). Tests: `tests/unit_tests/test_go2osc_robust_est.py`.
- [ ] Get / train a `go2osc_push_glide_robust` checkpoint (none locally).
- [ ] Pretrain the estimator on the robust policy (`pretrain_go2osc_estimator.py` is hard-wired to `go2osc_push_glide_est` and a `go2osc_push_glide` policy; needs a `--task` option).
- [ ] Check v-hat during whole-body slides: with all feet sliding, joint/Jacobian signals carry no speed information, and the 0.5 s window may not hold the run-up. Teleported fast slides make this worst-case.

## To do

- [ ] **Estimate or probe μ before the slide**, to decide whether sliding is feasible (a braced pose tips over on grippy ground) and to plan the run-up speed from the stopping distance.
  - Candidate: a scripted probe. One planted foot pushes sideways with ramped force (τ = JᵀF); μ = tangential / normal force at slip onset. The Jacobian forces are accurate when standing (correlation 0.94).
- [ ] **Set posture and lean from μ** (braking moment ≈ m·μg·h), and adjust them from the measured deceleration during the slide.
  - Known issue from dev: at μ 0.2 the FR leg is overloaded and RL nearly lifts.
- [ ] **Scale yaw corrections by μ.** Steering authority comes from foot friction, so it is proportional to μ. Per-foot Jacobian forces (`foot_force_est`, `foot_traction_est`) show which feet are steering.
  - First validate the Jacobian forces during real slides, against simulator contact forces (as in `results.md`, figure 4).
- [ ] **μ during the slide:** latching μ̂ at the slide's start (as on dev, `latch_in_slide`) is reasonable. Measuring it live from deceleration (accelerometer: a = μg) and from sliding-foot force ratios is better.

## Background

- `ti237CU/dev`: the earlier slide work on `go2_lowfric` (slide phase, `latch_in_slide`, slide evaluation). With the accelerometer, slide success went from 5/9 to 9/9.
- `results.md` on `ti237CU/state-estimator`: estimator accuracy, and Jacobian force validation.
