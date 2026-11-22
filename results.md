# State Estimator Results (stages 1–4)

Results for the learned friction (μ) and body-velocity (v) estimator on the team's push-glide policy. Plan and status: [state_estimator.md](state_estimator.md).

**Setup:**
- **Policy:** `main`'s push-glide policy trained for 1500 iterations, frozen (it only walks the robots).
- **Simulation:** 4096 robots on GPU, μ uniform in 0.1–1.0 every episode, a mid-episode surface switch in 30% of episodes, random speed and turn commands.
- **Estimator:** an MLP over 0.5 s of sensor history, outputting μ̂, v̂ₓ, v̂_y and v̂_z. Pretrained for 3000 steps, then evaluated on fresh episodes with learning off.
- **Sensors:** clean in stages 2–4, so inputs can be compared without noise. The estimator used for stage 5 adds sensor noise (last section).
- An earlier version of these results on the `508282d` policy is in the [appendix](#appendix-results-on-the-508282d-policy); the conclusions are the same.

## Base policy

| Policy | Mean pitch (walking) | Speed at a 1 m/s command | Stop test, μ 0.05 |
|---|---|---|---|
| Published | ≈ −3° | 0.95 m/s | 0/4 falls |
| Retrain at `508282d` | ≈ −2° | 0.97–1.00 m/s | 0/4 falls |
| `main`, 500 iterations | ≈ −10° | 0.73–0.80 m/s | 1/4 falls |
| **`main`, 1500 iterations (used here)** | **≈ −0.5°** | **0.81–0.85 m/s** | **0/4 falls** |

At 500 iterations `main` leaned and fell; 1500 iterations fixed that. The 1500-iteration policy walks flat and never fell, but:
- **It settles at about 0.82 m/s for a 1 m/s command on every surface.** That gap is a target for stage 5, where v̂ gives the policy its actual speed.
- **Its calves rest on their joint limits about 90% of the time,** standing and walking: up to 0.4 rad past the −0.84 rad limit. The `508282d` policy did this 18% of the time at 1 m/s. On the real Go2 these are mechanical stops, so a gait that leans on them may stress the joints or be blocked by the robot's software limits.

## Test 2: does the policy use its privileged traction input?

![Traction ablation](docs/figures/test2_traction_ablation_main1500.png)

Zeroing `foot_traction_obs` (true contact-force ratios, which the real robot can't measure) leaves speed, pitch and the stop test essentially unchanged on every friction level, with no falls in 48 runs. The policy barely relies on it, so the deployable actor drops it (stage 5). The policy also walks with about the same gait on every surface.

## Figure 1: Estimator accuracy

![Estimator accuracy](docs/figures/fig1_estimator_accuracy_main1500.png)

Error of each estimator compared with always guessing the average:
- **Velocity is learned well:** forward error 0.056–0.061 m/s, against 0.38 m/s for guessing.
- **Friction is learned only partly:** μ̂ error 0.132–0.136, against 0.23 for guessing; on slippery ground (μ < 0.4), 0.127–0.129 against 0.29.
- **Inputs barely matter:** joint torques (stage 3) and Jacobian foot-force features (stage 4) change μ̂ error by at most 0.004.

## Figure 2: μ̂ against true μ

![μ calibration](docs/figures/fig2_mu_calibration_main1500.png)

μ̂ follows μ (correlation 0.77–0.79) but is squashed toward the middle: ice (μ ≈ 0.15) reads about 0.22 and grippy ground (μ ≈ 0.95) about 0.80, with ±0.2 of scatter (shaded band). This is the signature of a weakly observable quantity. Friction only shows when a foot is near slipping, and the push-glide gait rarely slips. Above the grip the gait actually uses, μ = 0.6 and μ = 1.0 look the same.

## Figure 3: Recovery after a surface switch

![Switch recovery](docs/figures/fig3_switch_recovery_main1500.png)

After the ground changes friction, μ̂ error falls from about 0.35 to about 0.14 within roughly 0.5 s, the length of the estimator's sensor history. All three input sets recover at the same rate, so recovery is limited by the history window, not by the inputs.

## Figure 4: Feature checks against simulator truth

![Feature validation](docs/figures/fig4_feature_validation_main1500.png)

- **Accelerometer (left):** reads about 0 g in free fall right after spawning, then about 1 g while walking, as a real IMU would.
- **Jacobian foot forces, F = −(Jᵀ)⁻¹τ (middle, right):** fail on this policy (correlation about 0). The calves rest on their joint limits, so the joint stops carry torque that the motors report but the ground doesn't explain.
- On the `508282d` policy, whose calves rarely touch the limits, the same forces were accurate standing (correlation 0.94) and tracked most stance samples while walking ([appendix](#appendix-results-on-the-508282d-policy)). The method is correct; this policy's posture breaks its assumption.

## Sensor noise

The estimator for stage 5 is trained with realistic sensor noise:
- accelerometer: 0.5 m/s², plus a per-episode bias of up to ±0.2 m/s²
- joint angles: 0.01 rad
- joint velocities: 0.1 rad/s
- torques: 0.3 N·m

Same policy and inputs (stage 3), with and without noise:

| | Clean sensors | **Noisy sensors (stage 5 estimator)** |
|---|---|---|
| μ̂ error | 0.134 | 0.137 |
| μ̂–μ correlation | 0.77 | 0.76 |
| v̂ error x / y / z (m/s) | 0.061 / 0.029 / 0.017 | 0.057 / 0.026 / 0.016 |

Noise costs almost nothing. The 0.5 s history averages out per-reading noise, and the accelerometer bias is constant within an episode, so the estimator learns around it. Run: `logs/go2osc_push_glide_est/pretrain_main1500_noise/`.

## Summary

| Stage | Inputs | μ̂ error | v̂ₓ error (m/s) |
|---|---|---|---|
| Guess the average | – | 0.229 | 0.38 |
| 2 | IMU (gyro, gravity, accelerometer), joints, targets, commands, oscillators | 0.136 | 0.059 |
| 3 | + joint torques | 0.134 | 0.061 |
| 4 | + Jacobian foot forces, traction ratios, foot velocities | 0.132 | 0.056 |
| Stage 5 estimator | Stage 3 inputs, noisy sensors | 0.137 | 0.057 |

**Conclusions:**
- v̂ is accurate and ready for the policy.
- μ̂ is limited by how little this gait reveals about friction, not by the sensors, the noise or the network.
- The next gains should come from the fine-tune (stage 5), where the policy can use μ̂ and v̂, and the estimator keeps training on the states the new policy visits.

Run folders: `logs/go2osc_push_glide_est/main1500_stage{2,3,4}/` and `pretrain_main1500_noise/`. Regenerate the figures with `POLICY_SET=main1500 python logs/eval/go2osc/plot_estimator_stages.py`.

## New base policy: off the joint stops (`posture1800`)

The 1500-iteration policy rested its calves on their joint stops, so it was fine-tuned for 300 more iterations with the two existing joint-limit penalties switched on (`go2osc_push_glide_posture`, weights 5 / 10). A stronger version (10 / 20) also cleared the stops but crouched about 9 cm lower.

| | `main1500` | **`posture1800`** |
|---|---|---|
| Calves on the stop, walking 1 m/s | 84% | **0.2%** |
| Calves on the stop, standing | 100% (6.3 N·m) | 89% (1.5 N·m) |
| Speed at a 1 m/s command | 0.81–0.87 m/s | **0.86–0.92 m/s** |
| Falls (walk + stop tests, 24 runs) | 0 | 0 |

![Feature validation, posture1800](docs/figures/fig4_feature_validation_posture1800.png)

Off the stops, the Jacobian foot forces work again while walking: correlation 0.63, median error 0.014 body weights, 96% of stance samples within ±0.1. Standing they still overestimate the rear legs, whose calves rest lightly on the stops.

**Estimator on the new base** (noisy sensors; the Jacobian features are computed from the same noisy torques and joint velocities the estimator sees):

| Inputs | μ̂ error, seed 0 / 1 | μ̂–μ correlation | v̂ₓ error (m/s) |
|---|---|---|---|
| IMU, joints, torques (as before) | 0.160 / 0.162 | 0.66 / 0.65 | 0.060 / 0.060 |
| **+ Jacobian features (now the default)** | **0.154 / 0.148** | **0.68 / 0.69** | 0.057 / 0.061 |

μ̂ is less accurate than on `main1500` (0.137). The new gait seems to reveal less about friction. Velocity is unchanged. Stage 5 starts from this base with the seed-1 Jacobian estimator (`pretrain_posture1800_noise_jacobian_s1`).

## Appendix: results on the `508282d` policy

The first version of stages 1–4, run before `main` was retrained for longer. Same setup and clean sensors.

| Stage | μ̂ error | μ̂–μ correlation | v̂ₓ error (m/s) |
|---|---|---|---|
| Guess the average | 0.228 | – | 0.46 |
| 2 | 0.152 | 0.72 | 0.065 |
| 3 | 0.142 | 0.74 | 0.070 |
| 4 | 0.142 | 0.74 | 0.069 |

Figures: [accuracy](docs/figures/fig1_estimator_accuracy.png), [calibration](docs/figures/fig2_mu_calibration.png), [switch recovery](docs/figures/fig3_switch_recovery.png), [feature checks](docs/figures/fig4_feature_validation.png), [test 2](docs/figures/test2_traction_ablation.png).

On this policy the Jacobian foot forces were accurate standing (correlation 0.94; total support 0.91 of body weight, low because leg weight is ignored). Walking, they tracked most stance samples: median error 0.025 body weights, 73% within ±0.1, missing touchdown impacts (2% of samples) and a branch dominated by leg dynamics.
