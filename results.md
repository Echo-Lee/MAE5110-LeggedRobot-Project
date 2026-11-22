# State Estimator Results (stages 1–4)

Results for the learned friction (μ) and body-velocity (v) estimator on the team's `go2osc_push_glide` policy. Plan and status: [state_estimator.md](state_estimator.md).

**Setup for all results:**
- **Policy:** the retrained push-glide policy at commit `508282d`, frozen (it only walks the robots).
- **Simulation:** 4096 robots on GPU, μ uniform in 0.1–1.0 every episode, a mid-episode surface switch in 30% of episodes, random speed and turn commands.
- **Estimator:** an MLP over 0.5 s of sensor history, outputting μ̂, v̂ₓ, v̂_y and v̂_z. Pretrained for 3000 steps, then evaluated on fresh episodes with learning off.
- **Sensors:** clean (no noise) in these runs.

## Base policy check

Why stages 1–4 used `508282d`: a 500-iteration retrain on `main` leaned about 10° and fell on ice, while a retrain at `508282d` reproduces the published policy. Training `main` for 1500 iterations fixed this, so **stage 5 builds on the 1500-iteration `main` policy**.

| Policy | Mean pitch (walking) | Speed at a 1 m/s command | Stop test, μ 0.05 |
|---|---|---|---|
| Published | ≈ −3° | 0.95 m/s | 0/4 falls |
| Retrain at `508282d` | ≈ −2° | 0.97–1.00 m/s | 0/4 falls |
| Retrain at `main`, 500 iterations | ≈ −10° | 0.73–0.80 m/s | 1/4 falls |
| **`main`, 1500 iterations (stage 5 base)** | **≈ −0.5°** | **0.81–0.84 m/s** (0.85 on μ 0.05) | **0/4 falls** |

## Test 2: does the policy use its privileged traction input?

![Traction ablation](docs/figures/test2_traction_ablation.png)

Zeroing `foot_traction_obs` (true contact-force ratios, which the real robot can't measure) leaves speed, pitch and the stop test unchanged across all friction levels, with no falls in 48 runs. The policy barely relies on it, so it can be removed from the actor with little risk. The policy also walks with roughly one gait on every surface. Its clearest weakness is speed tracking on ice (0.80 m/s at a 1 m/s command on μ 0.05).

## Figure 1: Estimator accuracy

![Estimator accuracy](docs/figures/fig1_estimator_accuracy.png)

Error of each estimator compared with always guessing the average.
- **Velocity is learned well:** forward error 0.065–0.070 m/s, against 0.46 m/s for guessing.
- **Friction is learned only partly:** μ̂ error 0.142–0.152, against 0.23 for guessing.
- **Inputs barely matter:** adding joint torques (stage 3) helps slightly; adding Jacobian foot-force features (stage 4) adds nothing more.

## Figure 2: μ̂ against true μ

![μ calibration](docs/figures/fig2_mu_calibration.png)

μ̂ follows μ (correlation 0.74) but is squashed toward the middle: ice (μ ≈ 0.15) reads about 0.24 and grippy ground (μ ≈ 0.95) about 0.78, with ±0.2 of scatter (shaded band). This is the signature of a weakly observable quantity. Friction only shows when a foot is near slipping, and the push-glide gait rarely slips. Above the grip the gait actually uses, μ = 0.6 and μ = 1.0 look identical.

## Figure 3: Recovery after a surface switch

![Switch recovery](docs/figures/fig3_switch_recovery.png)

After the ground changes friction, μ̂ error falls from about 0.34 to its steady level of about 0.14 within roughly 0.5 s, the length of the estimator's sensor history. Recovery is limited by the history window, not by the inputs: all three stages recover at the same rate.

## Figure 4: Feature checks against simulator truth

![Feature validation](docs/figures/fig4_feature_validation.png)

- **Accelerometer (left):** reads about 0 g in free fall right after spawning, then about 1 g while walking, as a real IMU would.
- **Jacobian foot forces, F = −(Jᵀ)⁻¹τ, standing (middle):** accurate, correlation 0.94. Slightly low because the legs' own weight is ignored.
- **Jacobian foot forces, walking (right):** track most stance samples (median error 0.025 body weights, 73% within ±0.1). They miss touchdown impacts (2% of samples, up to 6 body weights) and a branch where leg dynamics dominate.

## Summary

| Stage | Inputs | μ̂ error | v̂ₓ error (m/s) |
|---|---|---|---|
| Guess the average | – | 0.228 | 0.46 |
| 2 | IMU (gyro, gravity, accelerometer), joints, targets, commands, oscillators | 0.152 | 0.065 |
| 3 | + joint torques | 0.142 | 0.070 |
| 4 | + Jacobian foot forces, traction ratios, foot velocities | 0.142 | 0.069 |

**Conclusions:**
- v̂ is accurate and ready for the policy.
- μ̂ is limited by how little this gait reveals about friction, not by the sensors or the network.
- The next gains should come from the fine-tune (stage 5), where the policy can use μ̂ and v̂, and where the estimator keeps training on the states the new policy visits.

The 1500-iteration `main` policy walks flatter and never fell, but settles at about 0.82 m/s for a 1 m/s command on every surface. That speed-tracking gap is a target for stage 5, where v̂ gives the policy its actual speed.

Run folders: `logs/go2osc_push_glide_est/pretrain_*`. Regenerate the figures with `logs/eval/go2osc/plot_estimator_stages.py`.
