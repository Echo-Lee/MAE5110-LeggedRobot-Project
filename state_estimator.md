# Learned State Estimator for `go2osc_push_glide`

## Goal

Give the Go2 a walking policy that adapts to the ground's friction coefficient (μ), tracks commanded body velocity, and performs a controlled slide on slippery surfaces while tracking a desired heading. It should rely only on signals the real robot has.

This needs a learned estimator of μ and body velocity, because neither can be measured directly on the robot:
- The foot sensors give only a rough vertical load.
- Leg odometry is wrong whenever the feet slip.

## Scope

**This PR**
- A learned estimator of μ and body velocity (v̂ₓ, v̂_y, v̂_z) from a short history of onboard sensor readings, including the accelerometer.
- Feeding μ̂ and v̂ to the team's push-glide policy and fine-tuning it.

**Next PR**
- The commanded controlled slide with heading tracking, built on this estimator.

## Estimator inputs and outputs

| | Signals | Notes |
|---|---|---|
| **Inputs** | Joint positions and velocities, IMU angular velocity, projected gravity, accelerometer (`base_lin_acc`), commands, previous actions | 0.5 s history. The accelerometer senses braking and drift directly (on dev it raised slide success from 5/9 to 9/9). |
| **Outputs** | μ̂, v̂ₓ, v̂_y, v̂_z | v̂_y is needed for heading tracking during the slide: direction of travel = atan2(v_y, vₓ). |
| **Not estimated** | Yaw rate, heading | The gyroscope measures yaw rate directly (already an actor input). Heading comes from integrating it over short spans. |

The actor receives μ̂ and v̂. The accelerometer feeds only the estimator.

## Implemented

Files to bring over from `ti237CU/dev`:

| Component | Purpose |
|---|---|
| Estimator network (`learning/modules/history_estimator.py`) | MLP over a 0.5 s sensor history; predicts μ̂ and body velocity |
| Runner hooks (`learning/runners/on_policy_runner.py`) | Trains the estimator alongside PPO; saves it in checkpoints |
| Friction randomization (`gym/envs/base/domain_randomization.py`) | Friction ranges and mid-episode friction switches |
| Unit tests | Estimator and domain-randomization tests |

## Plan

1. **Env subclass** (`go2osc_est.py`): add the accelerometer reading and friction switches; publish μ̂ and v̂.
2. **Config**: the team's actor inputs plus μ̂ and v̂.
3. **Task registration**: register `go2osc_push_glide_est`.
4. **Pretrain the estimator** on rollouts from the base policy.
5. **Widen the actor's inputs** (zero-initialized weights, including observation normalization).
6. **Fine-tune** the policy with the estimator in the loop.
7. **Evaluate** against the base policy:
   - pitch, speed tracking, falls, stop test
   - μ̂ and v̂ error
8. **Tests** for the new env and task.

## Decisions
- **Starting checkpoint:** `main`'s policy trained for 1500 iterations (`logs/go2osc_main1500/go2osc_push_glide/`): flat posture (pitch ≈ −0.5°), no falls, but about 0.82 m/s at a 1 m/s command. Stages 1–4 used the `508282d` retrain, because `main` at 500 iterations leaned about 10° and fell on ice; longer training fixed that, and the temporary restore of the `508282d` policy code was reverted.
- **Velocity target:** all three linear components (vₓ, v_y, v_z).
- **Accelerometer:** an estimator input, not an actor input.
- **Privileged traction input** (`foot_traction_obs`): Give Actor ground truth mu, give actor no mu, and then give actor estimated mu to verify results.

## To Do
- **Evaluation tools:** create unit tests to prove implementation and eval scripts for benchmarking.
- **Sensor noise:** model accelerometer noise and bias.
- **Covariance**
- **Target-cap the coefficient of friction**
- **1 sec history**

## Steps

| Stage | What | Result | Status |
|---|---|---|---|
| 1 | `go2osc_est.py`: accelerometer, friction switches, estimator outputs | Baseline estimator runs | **Done**: task `go2osc_push_glide_est` registered, 13 unit tests; the `508282d` checkpoint loads and walks unchanged |
| 2 | Test 1 with the baseline inputs | μ̂ and v̂ error on this gait | **Done**: v̂ₓ error 0.065 m/s (vs 0.46 guessing the average); μ̂ error 0.15 (vs 0.23), squashed toward the middle |
| 3 | Add **raw joint torques** as an estimator input (sensors the robot has) | Do torques help? | **Done**: small gain; μ̂ error 0.142 (vs 0.152), v̂ unchanged |
| 4 | Add **Jacobian foot forces and ratios** (F = (Jᵀ)⁻¹τ per stance leg) | Does the physics add anything beyond raw torques? | **Done**: no gain over torques (μ̂ error 0.142 both); forces accurate standing; walking, most stance samples tracked (median error 0.025 body weights) but impacts missed |
| 5 | Choose the inputs, then widen the actor and fine-tune | The three-way traction comparison | |

