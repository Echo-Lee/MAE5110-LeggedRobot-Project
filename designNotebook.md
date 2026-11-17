# Design notebook: Go2 on low-friction surfaces

Engineering notebook to record the process of training, assumptions, edits, and observations. TLDR: the overall design process of the PPO policy and sim-to-real transfer.

**Goal:** train Go2 to walk with normal gaits on high-friction surfaces and slide on lower-friction surfaces. Possibly add a controlled stop, e.g. sticking the legs out in front, or a yaw motion where a leg sticks outward to create a torque.

## Setup and assumptions

- **Sim:** MuJoCo (CPU for viewing, Warp on GPU for training); PPO (`PPO2`), 1500 iterations for the later runs. Actor MLP 256-256-128 outputs joint position targets; PD control turns them into torques.
- **Friction:** one Coulomb coefficient μ per environment, the same in every direction, constant within an episode. Training draws μ from ice [0.05, 0.2] or ground [0.6, 1.2] (50/50) every episode.
- **Teacher policy:** the actor is given the true μ (privileged). Not deployable as is: a real robot needs a μ estimate (planned student/estimator stage).
- **Rewards** (inherited from `go2`): velocity tracking dominates (4.0 + 2.0); body level and height; action smoothness. No foot or gait terms.
- **Physics limit:** the feet can push at most μ·m·g, so acceleration is capped at μ·g (≈ 0.5 m/s² on μ = 0.05). This caps top speed and how fast commands can change on ice.

## Evaluation tools

- **Friction grid** (`evaluate_go2_friction.py`): 7 μ × 6 commands; falls, tracking, slip ratio, CoT, touchdowns per meter; 30 graded cases (ice and ground). `--misreport_friction` tells the policy the wrong μ, to test its reliance on it.
- **Matched-speed CoT** (`evaluate_cot_vs_friction.py`): CoT at exact *actual* speeds. Policies don't hold commanded speed, so a fixed command isn't a fixed speed.
- **Slide** (`evaluate_slide.py`): run-up then a μg-decay command; distance and deceleration vs friction physics, feet down, touchdowns, energy.
- **Slide robustness** (`evaluate_slide_robustness.py`): the slide evaluation under 9 scenarios (nominal, randomized robot, random trigger time, entry speeds 0.5/1.0/1.5, sideways + turning entry, mid-slide push, sensor noise, ±30% wrong μ, combined) on μ 0.05–0.2 plus untrained μ 0.3. Pass = no falls, mean decel ≥ 0.75 μg, distance ratio ≤ 1.5, decel within 10% of nominal; push scenarios graded on falls only (the kick adds speed after entry speed is measured, which biases the ratios).
- **Per-foot check and videos** (`play.py --friction --command/--slide --video`): show *how* it moves, e.g. tucked legs and tapping, which the averages hide.

## Experiment log

| # | Run | Change | Result |
|---|---|---|---|
| 0 | `go2` `Nov14_20-30-38_` (550 it) | Baseline, no ice training | No falls even on ice; stands still; uses 4 feet. Taps 12–15 Hz, drags feet; CoT 3.3–6.9. |
| 1 | `Nov14_20-02-07_` (550 it) | Ice/ground bins, μ observation | No falls; uses μ. Lopsided diagonal stance, drift when stopped, worse tracking than `go2`. |
| 2 | `Nov14_20-59-28_` (1500 it) | Longer training | Action noise *rose* 1.0 → 1.91; 3-legged gait; ground CoT 8.4. |
| 3 | `Nov14_22-13-21_` | `entropy_coef` 0.01 → 0.002 | Noise settled (0.36); best ground tracking so far; still lopsided, tapping. |
| 4 | **`Nov14_23-04-56_`** (best until run 8) | + power penalty | **29/30**; CoT 1.1–1.4; 150–220 W (was ~1000); all 4 feet, trot-like diagonal pairs emerged. |
| 5 | `Nov15_04-33-19_` | + glide reward v1 | Exploited: light-load foot scuffing; slower; habit spread to ground. 13/30. |
| 6 | `Nov15_05-26-38_` | Glide v2 (load share, persistence, touchdown penalty) | Loophole closed, but glide ignored (flat 0.016); tucked a leg; ice top speed halved. 18/30. |
| 7 | `Nov15_06-10-02_` | Glide off; run-up-and-slide episodes | No coasting: still steps (~48 touchdowns/s), slides 2.3–4× the pure-slide distance. Lost turn-in-place; 14/30. Gains: all 4 feet, ~5 steps/s/foot, less drift. |
| 8 | **`Nov15_06-56-01_`** | Slide ends after 0.5 s hold; `slide_phase` observation | **28/30**, turning back; CoT 0.95–1.31 (lowest yet). Slide: 1.5–2.2× pure-slide distance, 0.61–0.78 μg, 15–35% less energy, but brakes by stepping (all-4-feet 2–6%). |
| 9 | **`Nov15_07-35-27_`** (new best) | + `slide_pose` (hold standing pose while sliding, weight 3) | 28/30; drift when stopped ≈ 0. Slide: pose dev 0.43 → 0.24 rad, least-used foot 3–13% → 42–83%, all 4 feet down up to 61%, energy/m ↓. Rear-left leg also used when **walking** now. Decel still 0.66–0.85 μg. |
| 10 | `Nov15_16-53-07_` | + `slide_still` weight 3 | Joint speed in slide halved (1.34 → 0.65 rad/s), energy/m ↓, decel 0.62–0.82 μg (≈ run 9). But froze in a non-standing pose (dev 0.24 → 0.45 rad) and walking got stiffer: 25/30, top speed 2.74 → 2.03 m/s at μ = 1, ground CoT ↑. Run 9 stays best overall. |
| 11 | `Nov15_18-40-46_` | `slide_still` 3 → 0; `slide_pose` 3 → 10 | Slide decel 0.75–0.87 μg (slightly ↑) but pose dev 0.32–0.36 rad (worse than run 9) and more steps; walking much worse: 22/30, CoT at 1 m/s 1.95–3.29, ground turning broken. Run 9 stays best. |
| 12 | **`Nov15_19-11-13_`** (best slide) | Run 9 weights + residual hold (slide-phase targets × 0.2) | Best learned slide: distance 1.27–1.45× ideal, decel 0.65–0.87 μg, all 4 feet down 45–65%, pose dev 0.18–0.20, no falls at any μ. Walking 25/30 (turn-in-place weaker); CoT at 1 m/s 0.82 on ice (lowest yet); top speed 2.32 m/s at μ = 1. |
| 12b | `Nov15_19-31-15_` | Run 12 repeated (different random seed) | Slide reproduces (decel 0.80 vs 0.78 μg mean, no falls). Walking: 19/30 (run 12: 25); turn-in-place fails in both seeds; top speed on ice 1.72 vs 1.06 m/s. |
| 13 | `Nov15_20-06-14_` (seed 7263, random) | Run 12 + 10% pure turn-in-place commands | **Turning fixed**: 28/30, turn-in-place 0.11–0.28 rad/s, all turn cases pass; CoT at 1 m/s 0.74–1.24 (best yet). Slide good on μ ≤ 0.1 (like run 12) but **worse at μ ≥ 0.15**: decel 0.38–0.41 μg and 50% falls at μ = 0.2, v0 = 1.2. One seed only. |
| 13b | `Nov15_20-55-40_` (seed 4939, random) | Run 13 repeated | Robustness suite 0/9 (falls at μ = 0.15–0.2 up to 75%); walking 19/30, turn-in-place fails on ground (0.54–0.58 rad/s). Confirms run 13's config makes the slide fragile and does not reliably fix turning. |
| 14 | `Nov15_21-43-55_` (seed 8241, random) | Run 12 + 10% turn-in-place on ground only | **Failed**: robustness 0/9 (falls up to 75%), often no run-up on ice; walking 14/30, fails ordinary walking at μ 0.1–0.2 and doesn't turn on ice (≈1 rad/s error); ground turning 0.30–0.32. Reverted: `turn_in_place_probability` = 0. |
| 15 | **`Nov15_21-34-07_est_finetune`** (seed 1; deployable) | Run 12 fine-tuned 500 it on task `go2_lowfric_est`: actor reads μ̂ from a concurrently trained estimator, friction switches on (p 0.3) | **First policy without privileged μ**: robustness 9/9 *as first scored; 2/9 after the slide-phase evaluation fix (Stage 2d)*, walking 29/30, switch eval 24/24, tracking error ~0.05 m/s. Slide decel 1.01 / 0.91 / 0.79 / 0.43 μg at μ 0.05 / 0.1 / 0.15 / 0.2 (weak at 0.2). μ̂ ≈ 0.1–0.2 on most surfaces: one cautious gait everywhere (Stage 2c). |
| 16 | **`Nov15_22-14-34_est_continuous`** (seed 1; deployable) | Run 15 + 500 it with continuous friction (bins ice 0.05–0.2 / middle 0.2–0.6 / ground 0.6–1.2, weights 0.4 / 0.25 / 0.35) | **Middle friction fixed**: 2 m/s tracking at μ 0.3 / 0.5 0.43 / 0.46 → 0.16 / 0.13; sideways at mid μ better. μ̂ meaningful again (μ 0.8 → 0.93; ice → ground switches re-estimated). Walking 29/30, switches 24/24, robustness 8/9 as first scored, *3/9 after the evaluation fix (Stage 2d)*. **μ 0.2 braking not fixed** (0.45 μg): μ̂ is wrong *during* the slide (μ 0.1 → ~0.6). |
| 17 | **`Nov15_22-56-19_est_latch`** (seed 1; deployable) | Run 16 + 500 it with μ̂ latched at slide start (`latch_in_slide`) | Robustness **7/9** (corrected eval; run 16: 3/9): μ 0.2 braking 0.39 → 0.54 μg, distance 2.99 → 1.93×; falls at μ 0.2 in `entry_speeds` (2/8) and `combined` (1/8). μ̂ at μ 0.2 during run-up 0.11 → 0.38–0.47. Walking 29/30, switches 24/24. |
| 18a | **`Nov15_23-24-23_est_accel_latch`** (seed 1; **best deployable**) | Run 17 + 500 it; estimator gets the IMU accelerometer (widened with zero weights), latch on | **Robustness 9/9, no falls in any scenario** (first full pass under the corrected eval; teacher run 12: 8/9). Walking 29/30, switches 24/24, 2 m/s RMSE at μ 0.3 / 0.5 / 1: 0.09 / 0.14 / 0.10. μ̂ at μ 0.2 0.17–0.21 (run 17: 0.38–0.47). μ 0.2 braking still 0.37–0.65 μg (teacher 0.74). |
| 18b | `Nov15_23-24-48_est_accel_nolatch` (seed 1) | As 18a, latch off | Robustness 3/9: falls at μ 0.2 (62% in *entry_speeds*, 38% *push*) and μ 0.15 nominal (1/8). μ̂ at μ 0.2 most accurate yet (0.24–0.26), but braking harder on it tips the robot. The latch stays. |
| 18a-s2 | `Nov16_00-28-29_est_accel_latch_seed2` (seed 2) | Run 18a repeated (different seed) | Robustness 8/9: one fall in 8 at μ 0.2 (*combined*; plus one at untrained μ 0.3). μ̂ at μ 0.2 0.15–0.16. Walking 29/30, switches 24/24. **18a's result reproduces.** |
| 19 | `Nov16_00-28-02_est_noaccel_control` (seed 1) | Control: run 17 + 500 it, latch on, **no accelerometer** | Robustness 5/9: falls at μ 0.2 (38% *entry_speeds*, 12% *push*), 1/8 at μ 0.15 (*combined*), *messy_entry* decel. μ̂ at μ 0.2 0.34–0.53 (over-estimate, as run 17). **More iterations alone do not give 18a's gain.** |

### Result figures

Each run keeps the same color in every figure. Data: `logs/eval/*.json`.

| Figure | Shows |
|---|---|
| [fig1](logs/eval/figures/fig1_friction_all_runs.png) | Friction grid for runs 0–6: tracking, slip, CoT, touchdowns vs μ |
| [fig2](logs/eval/figures/fig2_mu_reliance.png) | Run 4 with the true μ vs told μ = 1 |
| [fig3](logs/eval/figures/fig3_cot_matched_speed.png) | Matched-speed CoT vs μ and top speed per μ |
| [fig4](logs/eval/figures/fig4_rollout_timeline_power.png) | Run 4, one full training-style episode on ice and on ground |
| [fig5](logs/eval/figures/fig5_training_curves.png) | Training curves: action noise, power penalty, glide reward |
| [fig6](logs/eval/figures/fig6_slide_progression.png) | Commanded slide-stop across training stages (runs 4, 7, 9, 12) |
| [fig7](logs/eval/figures/fig7_robustness_run12.png) | Run 12 slide robustness: braking per scenario × friction |

![Friction grid, runs 0-6](logs/eval/figures/fig1_friction_all_runs.png)
*Fig. 1. Run 4 (+power, yellow) has the lowest tracking error, slip and CoT at almost every μ; glide v1 (pink) shows the scuffing exploit (slip and touchdowns spike on ice).*

## Problems and solutions

1. **Initial low-friction training** (runs 1–3; baseline `go2` on ice: `logs/videos/go2_ice_mu0.1_fwd1.mp4`)
   - *Problem:* a lopsided "scoot and kick" gait (diagonal pair + one kicking leg). It looked like skateboarding, but the same posture appeared on ground, so it wasn't friction-driven. Nothing in the rewards forbids it.
   - *Problem:* the 0.01 entropy bonus made exploration noise grow on this harder task, so training stalled.
   - *Solution:* `entropy_coef` 0.002. Noise settles and reward keeps improving, but the gait problems remain, so they come from the rewards.
   - *Lesson:* MoLab only uses new code after rerunning the Setup cell (run `Nov14_21-55-37_` silently used the old config). Verify config values before training.

   ![Training curves](logs/eval/figures/fig5_training_curves.png)
   *Fig. 5. Left: with entropy 0.01 action noise grows (runs 1–2); with 0.002 it settles (runs 3–6). Middle: the power penalty drops from −1.4 to about −0.3 (≈ 300 W). Right: glide v1 is exploited early; glide v2 stays flat near 0.016 (ignored).*

2. **Power penalty** (run 4)
   - *Idea:* tracking/posture rewards ignore energy (torque weight ≈ 0), so wasteful gaits were free. Penalize mechanical power to favor efficient, possibly sliding, motion.
   - *Result:* the single biggest improvement. 3–5× lower CoT, all four legs used, a trot-like gait emerged without being prescribed, best tracking.
   - *Observation:* with the waste gone, matched-speed CoT vs μ is nearly flat with a mild U: minimum around μ 0.3–0.5, higher on ice (slip) and at μ > 1 (scuffing/impacts). Not monotonic even at fixed speed.

   ![Matched-speed CoT](logs/eval/figures/fig3_cot_matched_speed.png)
   *Fig. 3. CoT at matched actual speed: run 4 (yellow) ≈ 1–1.9 across μ vs ≈ 3.5–11 for runs without the power penalty (0, 1, 3). Bottom right: top speed rises with μ for every run, ~1.3–1.6 m/s at μ = 0.05. Gaps = speed never reached on that surface.*

   ![Rollout timeline](logs/eval/figures/fig4_rollout_timeline_power.png)
   *Fig. 4. Run 4 in full 10 s training-style episodes. On ice (left), reversing direction takes ~4.5 s because acceleration is capped at μg; on ground (right) it settles in ~0.5 s. Foot rows show the stepping is fast and irregular.*

3. **Slide reward from foot force and contact time** (runs 5–6)
   - *v1:* rewarded loaded feet sliding with the body. *Problem:* exploited by dragging lightly loaded feet in rapid taps (5 N threshold). Slower, and spread to ground walking.
   - *v2:* weight each foot by load share (¼ body weight) × contact persistence (0.1 s) + ice touchdown penalty. *Problem:* loophole closed, but the policy mostly ignored it and tucked a leg: fewer touchdowns and more load per foot both pay.
   - *Root cause:* with friction equal in every direction, a sliding foot always brakes at μ·N. Gliding can never be faster or cheaper than stepping, so rewarding it only trades away performance.
   - *Decision:* glide weights → 0 (code kept). Continuous skating would need direction-dependent anisotropic friction like a ice skate.
   - *Evidence:* Fig. 1 (glide v1 slip/touchdown spikes on ice), Fig. 3 (glide runs reach lower top speeds on ice than run 4), Fig. 5 (glide reward flat).

4. **Run-up then slide** (run 7, `Nov15_06-10-02_`)
   - *Idea:* instead of continuous gliding, a maneuver where friction does the work. On ice episodes, hold speed v0 for 4 s, then command a decay at exactly μ·g, the deceleration of sliding on all four feet.
   - *Why it should work:* coasting on planted feet costs ~0 joint power; stepping to slow down costs energy. With the power penalty, coasting becomes the cheapest way to follow the command. No new reward needed.
   - *Success criteria:* distance and decel ratios → 1, all four feet down during the slide, touchdowns/s ↓, energy/m ↓, ground grid unchanged (~29/30).
   - *Baseline, run 4 before slide training* (`logs/eval/slide_power.json`, video `logs/videos/power_slide_mu0.1_v1.2.mp4`):

     | μ | Entry speed | Distance (pure-slide prediction) | Decel ÷ μg | All 4 feet down | Touchdowns/s |
     |---|---|---|---|---|---|
     | 0.05 | 0.91 | 1.89 m (0.84) | 0.56 | 0% | 34 |
     | 0.1 | 1.04 | 1.12 m (0.56) | 0.59 | 0% | 34 |
     | 0.2 | 1.10 | 0.68 m (0.31) | 0.53 | 0% | 32 |

   - *Result, run 7 after slide training* (`logs/eval/slide_run7.json`, video `logs/videos/run7_slide_mu0.1_v1.2.mp4`), v0 = 1.2:

     | μ | Entry speed | Distance (pure-slide prediction) | Decel ÷ μg | All 4 feet down | Touchdowns/s |
     |---|---|---|---|---|---|
     | 0.05 | 0.67 | 1.83 m (0.46) | 0.33 | 30% | 44 |
     | 0.1 | 0.89 | 1.10 m (0.40) | 0.46 | 27% | 48 |
     | 0.2 | 1.10 | 0.72 m (0.31) | 0.54 | 22% | 49 |

   - *Problem:* no coasting. The policy still steps through the slide and slows at only 0.33–0.54 μg; the run-up is also slower. Holding a fixed posture would decelerate at exactly μg and track the command best, but the policy never found it: a slowly decaying command looks like any other command change, and nothing tells it a slide has started.
   - *Problem (side effect):* turn-in-place broke (yaw error ≈ 1 rad/s, so it doesn't turn). After stopping, slide episodes command all zeros for the rest of the episode, so "zero forward/sideways command" became mostly "stand still" in training (~7% of samples vs ~1% turn-in-place). Top speeds and CoT also got worse (CoT at 1 m/s 1.3–2.0 vs 1.1–1.3).
   - *Gains:* all four feet carry load, stepping slowed to ~5 touchdowns/s/foot on ground (run 4: ~8–10), less drift when stopped.
   - *Next:* (a) after the slide stops, return to normal random commands (fixes the turn regression); (b) tell the policy a slide is happening, e.g. a slide-phase flag/time in the observation, and if needed a slide-phase "hold still" term (penalize joint velocity while sliding).

5. **Slide phase observation** (run 8, `Nov15_06-56-01_`)
   - *Changes:* slide ends 0.5 s after the stop and random commands resume (zero-command training samples 7% → 0.8%); `slide_phase` flag in actor and critic inputs.
   - *Result, slide* (`logs/eval/slide_run8.json`, video `logs/videos/run8_slide_mu0.1_v1.2.mp4`), v0 = 1.2:

     | μ | Entry speed | Distance (pure-slide prediction) | Decel ÷ μg | All 4 feet down | Touchdowns/s |
     |---|---|---|---|---|---|
     | 0.05 | 0.79 | 1.13 m (0.64) | 0.69 | 5% | 36 |
     | 0.1 | 1.13 | 0.99 m (0.65) | 0.76 | 6% | 49 |
     | 0.2 | 1.29 | 0.81 m (0.43) | 0.67 | 3% | 65 |

   - *Observation:* the flag is used: the slide is now distinct and much better (distance 1.5–2.2× vs 2.0–4×, decel up to 0.78 μg, energy/m 134–159 vs 160–240). But it **brakes by rapid stepping**, not coasting (touchdowns rise with μ).
   - *Fix confirmed:* turn-in-place yaw error 0.15–0.32 rad/s (run 7: ~1.0); friction grid 28/30; matched-speed CoT 0.95–1.31, the lowest of all runs.
   - *Next (run 9):* turn on `slide_still`. In run 8's slides it scores −0.31 per step (unweighted) vs +3.5 for tracking, so weight ~2 makes holding still worth ~0.6 per step. Freezing also tracks the μg-decay command best, so the two terms agree.

6. **Hold the standing pose while sliding** (run 9, `Nov15_07-35-27_`)
   - *Observation (run 8):* the rear-left leg is held up (thigh ~0.73 rad off the pose): 0% load walking on ice, 8% in the slide. The slide is on **three splayed legs** (hips 0.33–0.48 rad out; RMS pose deviation 0.43 rad).
   - *Physics:* on flat ground with isotropic friction, total braking ≤ μ·m·g however the legs are placed, and standing still on sliding feet already reaches it. Splayed feet slide partly sideways, so their friction partly cancels left/right, which fits decel of only 0.67–0.78 μg. Sticking legs out does not brake harder here (no snow to dig into, no edges).
   - *Change:* `slide_pose` = −mean (q − q_standing)² during the slide phase only, weight 3 (≈ −0.56 per step for run 8's slides vs ≈ 3.5 for tracking). Chosen instead of `slide_still` because holding a fixed pose also implies holding still. `slide_still` stays 0 so run 9 has one change.
   - *New slide metrics:* least-used foot's load fraction (run 8: 3–13%) and RMS pose deviation (run 8: 0.41–0.44 rad).
   - *Result, slide* (`logs/eval/slide_run9.json`, video `logs/videos/run9_slide_mu0.1_v1.2.mp4`), v0 = 1.2:

     | μ | Distance ratio (run 8) | Decel ÷ μg (run 8) | All 4 feet down (run 8) | Least-used foot (run 8) | Pose dev rad (run 8) | Energy J/m (run 8) |
     |---|---|---|---|---|---|---|
     | 0.05 | 1.75 (1.78) | 0.66 (0.70) | 56% (6%) | 79% (8%) | 0.24 (0.44) | 135 (157) |
     | 0.1 | 1.52 (1.51) | 0.79 (0.76) | 41% (6%) | 70% (9%) | 0.24 (0.44) | 111 (134) |
     | 0.2 | 1.77 (1.91) | 0.69 (0.67) | 23% (4%) | 42% (10%) | 0.24 (0.42) | 117 (136) |

   - *Observation:* the pose hold worked: legs stay near the standing pose and all four feet carry load. Unexpected bonus: the rear-left leg is used while **walking** too (load 0.41–0.51 on every foot, ~6 touchdowns/s/foot on ground), so no separate walking fix is needed for now.
   - *Observation:* "touchdowns/s" in the slide rose (43–77), but 91% of the gaps last ≤ 20 ms (median 10 ms = one control step) and feet lift ≤ 4 cm: contact **chatter** from feet bouncing under load, not steps. The touchdown metric should ignore gaps shorter than ~50 ms.
   - *Normal walking:* 28/30; drift when commanded to stop ≈ 0.00–0.01 m/s (best yet); top speed 1.23 m/s at μ = 0.05, 2.74 at μ = 1 (best yet); CoT at 1 m/s 1.02–1.21. Turning in place on ice is weaker (0.47 rad/s error at μ = 0.05).
   - *Still open:* decel 0.66–0.85 μg, not μg; residual joint motion ~1.2 rad/s and uneven foot loads in the slide. Next: `slide_still` (damp joint motion), and a touchdown metric with a minimum air time.

7. **Damp joint motion while sliding** (run 10, `Nov15_16-53-07_`)
   - *Metric fix first:* slide touchdowns now count only landings after ≥ 50 ms in the air (`touchdowns_per_s`); every reloading is reported separately (`contact_breaks_per_s`). Re-evaluated run 9: **0.7–5 real steps/s** (fewest on the slickest ice) vs 43–78 contact breaks/s. Run 9 already mostly stands in the slide; the remaining gap to μg is bouncing and joint motion, not stepping.
   - *Change:* `slide_still` 0 → 3. Run 9 scores −0.183 per step in its slides (≈ −0.55 weighted) vs ≈ 3.8 for tracking.
   - *Success:* decel ratio and distance ratio → 1, contact breaks/s ↓, joint speed in the slide ↓ (run 9: ~1.2 rad/s), walking grid unchanged (28/30).
   - *Result, slide* (`logs/eval/slide_run10.json`, video `logs/videos/run10_slide_mu0.1_v1.2.mp4`), v0 = 1.2:

     | μ | Distance ratio (run 9) | Decel ÷ μg (run 9) | Steps/s (run 9) | Chatter/s (run 9) | Pose dev rad (run 9) | Energy J/m (run 9) |
     |---|---|---|---|---|---|---|
     | 0.05 | 1.38 (1.71) | 0.82 (0.67) | 0.8 (0.7) | 49 (49) | 0.50 (0.24) | 85 (135) |
     | 0.1 | 1.37 (1.51) | 0.80 (0.79) | 2.1 (2.5) | 71 (66) | 0.48 (0.24) | 81 (111) |
     | 0.2 | 1.67 (1.78) | 0.62 (0.69) | 5.6 (5.1) | 62 (78) | 0.43 (0.24) | 136 (117) |

   - *Walking:* 25/30 (run 9: 28); top speed 1.03 / 2.03 m/s at μ = 0.05 / 1 (run 9: 1.23 / 2.74); CoT at 1 m/s 1.07–1.44 (run 9: 1.02–1.21). Turning in place on ice improved (0.18 vs 0.47 rad/s). The stillness penalty made the whole gait stiffer, not just the slide.
   - *Why decel is still < μg* (force analysis at μ = 0.1, while moving): braking force = 0.74 μN (run 9) / 0.76 μN (run 10) even though 97–99% of loaded feet slide and carry the full weight. Slip is only partly along the travel direction (0.72 / 0.86) and the legs still move.
   - *Simulator check:* the robot **frozen in its standing pose** with no policy, launched at 1.2 m/s, decelerates at **0.95–1.01 μg** (μ 0.05–0.5). MuJoCo gives full sliding friction; the gap is the policy. Freezing in the standing pose is the ideal slide.
   - *Conclusion:* `slide_still` froze the robot, but in a crouched, non-standing pose, and leaked stiffness into walking. The target is specifically "freeze in the standing pose".

8. **Pull harder toward the standing pose** (run 11, `Nov15_18-40-46_`)
   - *Change:* `slide_still` 3 → 0 (it caused the crouch and the stiff walking); `slide_pose` 3 → 10. Run 9 scores −0.060 per step unweighted in its slides, so ≈ −0.6 at weight 10, vs ≈ 3.8 for tracking.
   - *Success:* pose dev well below run 9's 0.24 rad, decel → 0.95 μg (the frozen-pose sim result), distance ratio → 1, walking back to run 9 (28/30, top speed 2.74 m/s at μ = 1).
   - *Result* (`logs/eval/slide_run11.json`, video `logs/videos/run11_slide_mu0.1_v1.2.mp4`): decel 0.75–0.87 μg, distance ratio 1.33–1.63 (slightly better than run 9), but pose dev **0.32–0.36 rad** (run 9: 0.24) and 1.5–8 real steps/s. Walking: 22/30, matched-speed CoT at 1 m/s 1.95–3.29 (run 9: 1.02–1.21), top speed 2.13 m/s at μ = 1, turn-in-place on ground broken (0.74 rad/s). A heavier pose penalty did not buy a better pose and damaged walking.
   - *Scripted test (option b, no training):* run 9's policy for the run-up, then the PD controller holds the standing pose (targets 0) during the slide phase:

     | μ | v0 | Learned (run 9): dist ratio / decel ÷ μg | **Scripted hold**: dist ratio / decel ÷ μg | Scripted: fell |
     |---|---|---|---|---|
     | 0.05 | 0.8 | 1.53 / 0.81 | **1.26 / 0.92** | 0% |
     | 0.05 | 1.2 | 1.69 / 0.67 | **1.16 / 1.00** | 0% |
     | 0.1 | 0.8 | 1.55 / 0.82 | **1.16 / 0.95** | 0% |
     | 0.1 | 1.2 | 1.54 / 0.79 | **1.09 / 1.01** | 0% |
     | 0.2 | 0.8 | 1.81 / 0.66 | 1.57 / 0.77 | 0% |
     | 0.2 | 1.2 | 1.77 / 0.69 | 2.94 / 0.31 | **88%** |

   - *Insight:* a rigid standing pose brakes at ≈ μg on real ice (μ ≤ 0.1), confirming the sim test. But at μ = 0.2 the braking force (2 m/s²) acts at the feet, below the center of mass, and pitches a rigid robot forward onto its nose. Braking harder needs active balance (leaning back), which explains why the learned policies settle at ~0.7–0.85 μg: they trade braking for staying upright.

9. **Residual hold** (run 12, `Nov15_19-11-13_`)
   - *Change:* revert `slide_pose` to 3 (run 9's weights); new `commands.slide_action_scale = 0.2`. During the slide phase the joint targets (offsets from the standing pose) are multiplied by 0.2, applied inside the env step before the PD loop. The policy keeps ~20% authority (≈ ±0.25 rad at the thigh): enough to balance and lean back, not to splay or step. Constrains the action space instead of adding penalties, which stalled in runs 10–11.
   - *Preview (run 9's policy, not trained for the hold):*

     | μ | v0 | Dist ratio | Decel ÷ μg | All 4 feet | Pose dev (rad) | Fell |
     |---|---|---|---|---|---|---|
     | 0.05 | 1.2 | 1.11 | 1.06 | 62% | 0.21 | 0% |
     | 0.1 | 1.2 | 1.25 | 0.84 | 50% | 0.23 | 0% |
     | 0.2 | 1.2 | 1.79 | 0.50 | 8% | 0.24 | 0% |

   - *Success:* keep ≈ μg on μ ≤ 0.1, raise braking at μ = 0.2 without falls, walking back to run 9 (28/30).
   - *Result, slide* (`logs/eval/slide_run12.json`, video `logs/videos/run12_slide_mu0.1_v1.2.mp4`), v0 = 1.2:

     | μ | Dist ratio (run 9) | Decel ÷ μg (run 9) | All 4 feet (run 9) | Least-used foot (run 9) | Pose dev rad (run 9) | Energy J/m (run 9) | Fell |
     |---|---|---|---|---|---|---|---|
     | 0.05 | **1.30** (1.71) | **0.84** (0.67) | **63%** (56%) | **86%** (80%) | **0.20** (0.24) | **84** (135) | 0% |
     | 0.1 | **1.36** (1.51) | 0.77 (0.79) | **56%** (42%) | **82%** (71%) | **0.20** (0.24) | **82** (111) | 0% |
     | 0.2 | **1.27** (1.78) | 0.65 (0.69) | **45%** (25%) | **72%** (43%) | **0.19** (0.24) | 124 (117) | 0% |

   - *Observation:* the best learned slide so far on every slide metric except decel at μ ≥ 0.1, with no falls anywhere (the scripted hold fell 88% at μ = 0.2). But the trained policy brakes *less* on ice than the untrained preview (run 9 policy + hold: 0.84–1.08 μg at μ ≤ 0.1); training used the 20% authority partly for something other than braking.
   - *Walking:* 25/30 (run 9: 28). Turn-in-place weaker (0.36–0.67 rad/s; three yaw failures). Matched-speed CoT at 1 m/s 0.82–1.32, the lowest on ice of any run; top speed 1.06 / 2.32 m/s at μ = 0.05 / 1 (run 9: 1.23 / 2.74). All four legs used, but FR + RL carry more (0.5–0.63) than FL + RR (0.22–0.32).
   - *Second seed (run 12b, `Nov15_19-31-15_`, same config):*

     | Metric | Run 12 | Run 12b | Seed-to-seed difference |
     |---|---|---|---|
     | Slide decel ÷ μg (mean of 10 cases) | 0.78 | 0.80 | per case mean 0.07, max 0.18 |
     | Slide distance ratio (mean) | 1.35 | 1.42 | per case mean 0.19, max 0.35 |
     | All 4 feet down in slide (mean) | 56% | 53% | per case mean 3% |
     | Pose dev in slide (mean, rad) | 0.19 | 0.29 | 0.10 |
     | Friction grid passed | 25/30 | 19/30 | 6 cases |
     | Turn-in-place failures | 3 | 4 | both fail it |
     | Top speed μ = 0.05 / 1 (m/s) | 1.06 / 2.32 | 1.72 / 2.34 | 0.66 on ice |
     | CoT at 1 m/s, μ = 0.1 / 1 | 0.82 / 1.32 | 0.81 / 1.80 | |

   - *Conclusions:* (1) the residual-hold **slide is reproducible** (means within ~0.02–0.07, no falls in either seed). (2) The **turning loss is systematic**, not seed noise: both seeds fail turn-in-place. (3) **Walking scores vary a lot between seeds** (19 vs 25 of 30; ice top speed 1.06 vs 1.72), so earlier single-run walking comparisons (e.g. run 9's 28 vs run 12's 25) are within noise. Per-case slide differences up to ~0.2 are also noise.
   - *Lesson:* compare configurations on per-friction means over cases, and on ≥ 2 seeds for walking claims. `train.py` now saves `seed.json` in every run folder (`--seed` to fix it).
   - *Visual check:* the run-up-and-slide videos look good: the robot runs up, plants all four feet near the standing pose, and slides to a stop. The remaining issues are in the numbers (braking ~0.8 μg, turning), not visible failures.

10. **Turn-in-place commands** (run 13, `Nov15_20-06-14_`, seed 7263)
   - *Change:* `commands.turn_in_place_probability = 0.1`: that fraction of command resamples become pure turns (forward/sideways 0, |yaw| ≥ `turn_in_place_min_yaw` = 0.5 rad/s). Applies at reset, the 3 s resamples, and when a slide ends; slide episodes still override their own commands. Pure-turn training samples 1.3% → 10.4%.
   - *Why:* both residual-hold seeds fail turn-in-place; pure turns were ~1% of commands and slide episodes add long straight, zero-yaw stretches.
   - *Plan:* train with `--seed 1` and `--seed 2`; compare seed means against runs 12/12b. (Only one run came back, random seed 7263, now recorded in its `seed.json`.)
   - *Result, walking:* 28/30 (runs 12/12b: 25/19); turn-in-place yaw error 0.28 / 0.24 / 0.16 / 0.11 / 0.12 rad/s at μ = 0.05 / 0.1 / 0.2 / 0.8 / 1 (run 12: 0.56 / 0.50 / 0.36 / 0.11 / 0.67); only forward_fast on ice fails (physically limited). CoT at 1 m/s 0.74–1.24, the lowest yet. Top speed 1.10 / 2.21 m/s at μ = 0.05 / 1.
   - *Result, slide* (`logs/eval/slide_run13.json`), means over the 10 cases vs run 12 / 12b: distance 1.54 (1.35 / 1.42), decel 0.69 μg (0.78 / 0.80), all 4 feet 42% (56% / 53%), energy/m 171 (103 / 132). By friction: μ ≤ 0.1 is as good as run 12 (distance 1.31–1.42, decel 0.70–0.85, 0% falls); μ = 0.15–0.2 is worse: decel 0.38–0.71 and **50% falls at μ = 0.2, v0 = 1.2** (runs 12/12b: 0%). Videos: `logs/videos/run13_slide_mu0.1_v1.2.mp4` (good), `logs/videos/run13_slide_mu0.2_v1.2.mp4` (fall).
   - *Interpretation:* turning is fixed. The μ = 0.2 slide regression could be seed variance (one run) or a real trade-off; μ = 0.2 is the hardest slide case (most braking torque, the edge of the slide training range) and was already the weakest in runs 12/12b.

11. **Robustness suite** (stage 1 hardening)
   - *Built:* `scripts/evaluate_slide_robustness.py` (see Evaluation tools) on a refactored `evaluate_slide.py` (`SlidePerturbations`: random run-up, entry sideways/turn, push, sensor noise, μ estimate error, randomized robot; per-replica slide start in `summarize_slide`). Refactor check: run 13's nominal slide reproduces on ice within 0.02; μ 0.15–0.2 cases differ more because behavior there is borderline and sensitive to the initial pose.
   - *Results* (scenarios passed / 9; `logs/eval/robustness_run*.json`):

     | Run | Passed | Falls (any scenario, μ ≤ 0.2) | Notes |
     |---|---|---|---|
     | 12 (`Nov15_19-11-13_`) | 8/9 | 12% in *combined* at μ = 0.1 only | no falls even at untrained μ = 0.3 |
     | 12b (`Nov15_19-31-15_`) | 8/9 | none | fails *entry_speeds* on distance (1.60 at v0 = 1.5) |
     | 13 (`Nov15_20-06-14_`) | 0/9 | 12–100% at μ = 0.15–0.2 in every scenario | slide at μ ≥ 0.15 breaks down |

   - *Conclusions:* (1) Run 12's configuration gives a **robust slide on both seeds**: no falls under robot randomization, random trigger times, entry speeds 0.5–1.5, messy entry, pushes, sensor noise, or a ±30% wrong μ, and braking stays at 0.77–0.89 μg in all non-push scenarios. (2) Pushes reduce braking (0.63–0.70 μg) but cause no falls on 12b. (3) Run 13 (turn-in-place commands) is fragile at μ ≥ 0.15 in every scenario, while both run-12 seeds are robust: likely the turn-in-place change, not seed noise, but a second run-13 seed would confirm.
   - *Second seed of run 13 (13b, `Nov15_20-55-40_`):* robustness 0/9 again (worst fall 75% at μ ≤ 0.2, nominal decel 0.72 μg); walking 19/30 with turn-in-place failing on ground (0.54 / 0.58 rad/s at μ = 0.8 / 1).

     | Config | Seed | Robustness passed | Worst fall μ ≤ 0.2 | Walking grid | Turn-in-place yaw err μ 0.05 / 1 |
     |---|---|---|---|---|---|
     | Run 12 (no turn-in-place) | a | 8/9 | 12% | 25/30 | 0.56 / 0.67 |
     | Run 12 | b | 8/9 | 0% | 19/30 | 0.80 / 0.48 |
     | Run 13 (10% turn-in-place) | a | 0/9 | 100% | 28/30 | 0.28 / 0.12 |
     | Run 13 | b | 0/9 | 75% | 19/30 | 0.34 / 0.58 |

   - *Conclusions:* (1) **Turn-in-place commands make the slide fragile** on both seeds (0/9 vs 8/9), so it is the config, not the seed. (2) They **do not reliably fix turning**: seed a passes every turn case, seed b still fails on ground. (3) The walking grid swings 19–28/30 between seeds of the same config, so walking needs ≥ 2 seeds (or more replicas) to judge.
   - *Decision:* run 12's configuration (no turn-in-place) is the stage-1 baseline: a robust slide on both seeds. Turning in place is a separate, secondary weakness (not part of the slide-stop goal).

12. **Ground-only turn-in-place** (run 14, `Nov15_21-43-55_`, seed 8241)
   - *Change:* `turn_in_place_probability` 0.1 with `turn_in_place_min_friction` 0.6: only envs on the ground bin (true μ ≥ 0.6) get pure turn commands. Training mix: ice 1.2% pure turns (run 12: 1.3%), ground 10.7%.
   - *Hypothesis:* turn practice *on ice* is what destabilized the ice slide in runs 13/13b; ground-only keeps the ice-side training identical to run 12 while targeting 12b's ground turning failures. Does not address turn-in-place on ice.
   - *Pass:* robustness suite ≥ 8/9 on both seeds; turn-in-place yaw error at μ 0.8/1 clearly below 12/12b (0.11–0.67) on both seeds; walking grid not worse than 12/12b (19–25/30). If the slide still breaks, turning practice conflicts with the slide policy-wide and run 12 stays the baseline.
   - *Metric fix:* some slide distance ratios came out absurd (3,000–145,000): replicas that never ran up (entry speed ≈ 0) divided by a predicted distance ≈ 0. `summarize_slide` now counts entries below 0.2 m/s as `no_runup_rate` (excluded from the ratios) and the suite grades "no run-up" as a failure. Runs 12–13b had no such cases.
   - *Result (one seed):* robustness 0/9 (falls up to 75% at μ ≤ 0.2; no run-up at all for sideways/turning entries at μ ≥ 0.1). Walking 14/30: fails forward and sideways walking at μ 0.1–0.2 and does not turn on ice (yaw error ≈ 1.0 rad/s, i.e. stands still); ground turning 0.30–0.32 (better than 12, not than 13). Top speed 1.21 / 2.74 m/s at μ = 0.05 / 1.
   - *Conclusion:* worse than both run-12 seeds on every robustness and walking measure except ground top speed. Three runs of turn-in-place (13, 13b, 14) all broke the slide, so a second seed of run 14 is not worth it. Turn-in-place practice and the robust slide conflict in this setup.
   - *Decision:* `turn_in_place_probability` back to 0 (the `turn_in_place_min_friction` option stays). **Stage 1 baseline = run 12's configuration**; turning in place stays a known limitation.

## Observations

- **The policy uses μ:** told μ = 1 on ice, slip and CoT rise sharply. The best policy depends on it less, so a state estimator should be implemented.

  ![Reliance on mu](logs/eval/figures/fig2_mu_reliance.png)
  *Fig. 2. Run 4 told μ = 1 while on ice: slip 0.15 → 0.52 and CoT 1.37 → 2.18 at μ = 0.05; on ground the two curves meet.*
- **Ice speed limit:** top speed on μ = 0.05 is ~1.3–1.6 m/s for every policy; 2 m/s is physically out of reach.
- **Infeasible training commands:** about half the random commands on ice ask for speed changes faster than μ·g allows, a source of noise in training.
- **Survival:** 98–99.6% across full training-style episodes for the best policy (Fig. 4).
- **Averages hide gait problems:** the per-foot load and touchdown checks and the videos found every lopsided gait.

## Stage 1 result (frozen)

**Final checkpoint:** run 12, `logs/go2_lowfric/Nov15_19-11-13_/model_1500.pt` (config committed in `2885576`: power penalty, entropy 0.002, ice/ground friction bins, run-up-and-slide episodes with slide-phase flag, `slide_pose` 3, residual hold `slide_action_scale` 0.2, no turn-in-place). Chosen over run 12b (same config, other seed) for its walking and its slide video.

**What it does:** one policy, two modes. Normal walking/running on any friction (trot-like, all four legs, ≈0.8–1.3 CoT at 1 m/s); and, when the slide-phase flag is set on ice, a slide-stop: feet planted near the standing pose, small balance corrections only, stopping by friction.

| Measure | Run 12 | Baseline before slide work (run 4) |
|---|---|---|
| Slide braking (μ 0.05–0.2, mean) | 0.78 μg (0.73–0.86) | 0.55 μg (walks to a stop) |
| Stopping distance ÷ ideal | 1.35 | 2.0–2.3 |
| All 4 feet down during slide | 47–65% | 0% |
| Real steps during slide | 1.2–4 /s | ~33 /s |
| Slide robustness suite | 8/9 (no falls except 1/8 in *combined* at μ 0.1) | – |
| Walking grid | 25/30 | 29/30 |
| Matched-speed CoT at 1 m/s | 0.82–1.32 | 1.08–1.29 |

![Slide progression](logs/eval/figures/fig6_slide_progression.png)
*Fig. 6. Braking, stopping distance and feet-down time across the stages of slide training. Walking policy (yellow) brakes at ~0.55 μg by stepping; slide episodes alone (pink) were worse; the standing-pose reward (green) and the residual hold (blue, final) bring braking to ~0.75–0.86 μg with stopping distance ~1.35× ideal at every μ.*

![Run 12 robustness](logs/eval/figures/fig7_robustness_run12.png)
*Fig. 7. Run 12 under 9 scenarios: braking stays 0.73–0.98 μg except with a push (0.60–0.69, no falls) and the combined worst case (0.54–0.64, one fall in 8 at μ 0.1). μ 0.3 was never trained for slides: braking 0.29–0.68, no falls.*

**Videos** (`logs/videos/stage1/`): `run12_slide_mu0.05_v1.2`, `run12_slide_mu0.1_v1.2` (preferred), `run12_slide_mu0.2_v1.2`, `run12_slide_mu0.1_v0.8`, `run12_walk_ice_mu0.05_fwd1`, `run12_walk_ground_mu1_fwd1`, `run12_turn_ground_mu1`.

**Known limitations**
- Braking ≈ 0.78 μg, not μg: a rigid standing pose reaches μg on μ ≤ 0.1 but tips over at μ 0.2; the learned slide trades braking for balance.
- Turning in place is weak (0.5–0.7 rad/s error on some surfaces). Three attempts to train it (runs 13, 13b, 14) broke the slide.
- Walking quality varies a lot between seeds (run 12b: 19/30); the slide does not (12b: 8/9, no falls).
- Slide is commanded (slide-phase flag) and uses the true μ: a deployable version needs a μ estimate (Roadmap).
- Flat ground, isotropic friction, slide trained only for μ ≤ 0.2.

**Reproduce:** `scripts/evaluate_slide.py`, `scripts/evaluate_slide_robustness.py`, `scripts/evaluate_go2_friction.py`, `scripts/evaluate_cot_vs_friction.py` with `--task go2_lowfric --load_run Nov15_19-11-13_ --original_cfg`; videos with `scripts/play.py ... --friction MU --slide V0 --video PATH`.

**Repository state:** commit `3ace272` holds run 14's (rejected) config. The pending commit sets `turn_in_place_probability` back to 0, which restores run 12's training behavior exactly (the turn-in-place step returns before drawing any random numbers when off), plus the no-run-up metric. Evaluations of run 12 use `--original_cfg`, so they are unaffected either way.

## Stage 2a: friction and velocity estimator (deployable μ)

- *Why:* the actor reads the true μ, which no robot measures, so the policy is not deployable. Only the actor's μ must go; the critic and rewards may keep privileged information (training only).
- *Built:* `learning/modules/history_estimator.py` (`ProprioHistory`, `HistoryStateEstimator`, `EstimatorInTheLoop`), `scripts/train_state_estimator.py`; `--state_estimator` in `evaluate_slide.py`, `evaluate_slide_robustness.py`, `evaluate_go2_friction.py`, `play.py` (overlay shows μ and μ̂).
- *Estimator:* MLP (512-256-128) on a 0.5 s history (25 frames, every 2nd step) of IMU angular velocity, gravity direction, joint positions/velocities, previous joint targets, commands and the slide flag (1,150 inputs, true μ excluded). Outputs μ̂ and body velocity v̂ (x, y, z). Supervised MSE against simulator truth, trained online while the frozen run-12 policy drives the training mix; after 300 of 600 iterations half the envs act on μ̂ (DAgger-style) so the estimator also sees the policy's own estimate-driven states. 10 min on GPU. Saved to `logs/go2_lowfric/Nov15_19-11-13_/state_estimator/estimator.pt`.
- *Validation* (fresh seed, policy on true μ):

  | | μ̂ median rel. error | p90 | MAE |
  |---|---|---|---|
  | Ice, ≥ 1 s after reset | 14% | 47% | 0.022 |
  | Ground, ≥ 1 s after reset | 3% | 8% | 0.032 |
  | Ice, settling: 0–0.25 / 0.25–0.5 / 0.5–1 s | 45% / 17% / 12% | | |

  v̂ RMSE: x 0.15, y 0.12, z 0.06 m/s (true spread 0.90 / 0.42 / 0.25).
- *Policy on μ̂ (run 12, estimator in the loop):* slide robustness **7/9** (true μ: 8/9): nominal braking 0.78 μg and distance 1.33× (identical to true μ); one fall in 8 at μ 0.05 under *randomized* and 2/8 at μ 0.1 under *combined* (true μ: 1/8 there). Walking grid **24/30** (true μ: 25/30); turn-in-place on ice now passes, μ 0.8 turning and a μ 1 sideways case fail. Video: `logs/videos/stage1/run12_estimator_slide_mu0.1_v1.2.mp4`.
- *Observation:* during the commanded slide maneuver μ̂ is compressed toward ~0.1–0.14 (μ 0.05 → 0.09–0.11, μ 0.2 → 0.13), far less accurate than in the training mix, likely because a held standing pose reveals little about friction and evaluations start from rest. The policy still slides almost identically, consistent with its tolerance of ±30% μ error: within the ice range it needs only "this is ice".
- *Next (done in 2b):* friction patches (μ changing mid-episode) to train and test switching; optionally fine-tune the policy with μ̂ in the loop; feed v̂ to the actor if it helps on ice.

## Stage 2b: friction switches (μ changes mid-episode)

- *Why:* a deployed robot walks from carpet onto ice; μ̂ (and the gait) must switch on the new surface, not only after a reset.
- *Built:* `friction_switch` config and `Go2LowFric` switching (one switch per episode to the other bin, at 2–8 s, never in slide episodes; off for policy training), `time_since_friction_change`; `scripts/evaluate_friction_switch.py` (start at rest, fixed command, μ jumps at 4 s; 6 transitions incl. 2 no-change controls × 4 commands × 4 replicas; falls, velocity RMSE before / first second after / adapted, μ̂ settling within 30%); `train_state_estimator.py --friction_switch_probability`; `--estimator_observe_only` (policy keeps true μ, μ̂ only recorded).
- *Result 1, run 12 on true μ:* switches cost nothing: 0 falls, tracking after the switch like the controls; 19/24 graded, all 5 failures are `turn_left` yaw tracking (the known turn-in-place weakness, also in the controls).
- *Problem: closed-loop fixed point.* With the stage-2a estimator in the loop: 0 falls, 19/24, but μ̂ barely moves after ice → ground (0.1 → 0.8: μ̂ 0.24 at 0.5 s, 0.3 at 2 s), and even with no switch at μ 0.8 μ̂ sat at 0.22–0.4 for slow/sideways/turn commands. Observe-only, the same estimator tracks every switch within 0.5 s (0.76–0.95 after ice → ground, 0.07–0.15 after ground → ice) and the controls are accurate. *Cause:* the policy changes its gait with the μ it is told, and the estimator partly reads the gait; a cautious "ice gait" on carpet does not slip, so a low μ̂ confirms itself. μ only shows through slip (the observability limit in the roadmap).
- *Solution:* retrain the estimator with switches (probability 0.5) and more closed-loop data (DAgger from iteration 150, 75% of envs) → `state_estimator_switch/estimator.pt`. Validation: after a switch to ground 24% / 9% / 4% median error at 0–0.25 / 0.25–0.5 / 0.5–1 s; to ice 176% / 38% / 19%; settled ice 18% (stage 2a: 14%), ground 4%.
- *Result 2, closed loop with the switch estimator:*

  | | true μ | estimator 2a | **switch estimator** |
  |---|---|---|---|
  | Switch eval (graded) | 19/24 | 19/24 | **24/24** |
  | Falls in switch eval | 0 | 0 | 0 |
  | μ̂ 0.5 s after 0.05 → 1, forward | – | 0.23 | **0.99** |
  | μ̂ 0.5 s after 1 → 0.05, forward | – | 0.53 | **0.13** |
  | Walking grid | 25/30 | 24/30 | **28/30** |
  | Slide robustness | 8/9 | 7/9 | 6/9 |

  Ground → ice is re-estimated within 0.25–0.5 s for every command; ice → ground within 0.25–0.5 s while walking forward. Sideways and turn-in-place on new carpet stay "icy" (μ̂ ~0.2 for > 1 s, 0.45–0.6 at 2 s): the residual fixed point, harmless here (no falls, tracking passes).
- *Slide trade-off:* nominal slide unchanged (0.78 μg, 1.36×). Failures: `entry_speeds` (1/8 never ran up at μ 0.05, v0 0.5; distance 1.70× at μ 0.2), `push` (1/8 falls at μ 0.1), `combined` (2/8 at μ 0.1, as with estimator 2a). Each is 1–2 robots of 8, near run-to-run noise, but consistent with the slightly worse ice accuracy.
- *Observation:* an estimator trained for a frozen privileged policy hits a ceiling: the policy never learned to act on an imperfect μ̂, and the estimator can only learn from slip the policy chooses to cause. The usual fix is to adapt the policy to the estimate.
- *Next (not started):* fine-tune run 12 with the switch estimator in the loop (policy reads μ̂, friction switches on during training), then re-run all three evaluations; optionally add a small "probe" incentive (or v̂ input) so slow gaits reveal μ.

## Stage 2c: fine-tune the policy on μ̂ (deployable student, run 15)

- *Why:* stage 2b showed an estimator trained around a frozen privileged policy hits a ceiling: the policy never learned to act on an imperfect μ̂ (closed-loop fixed point, extra slide falls).
- *Built:* task `go2_lowfric_est` (`Go2LowFricEst`, `go2_lowfric_est_config.py`). The env runs the estimator every control step (`ConcurrentEstimator`) and publishes `contact_friction_hat` (clipped μ̂), which takes `contact_friction`'s slot in the actor input, so run 12's checkpoint loads unchanged; the critic, rewards, commands and physics keep the true μ. During PPO the estimator keeps training on true labels (256 random envs per step, 4 Adam steps every 24 steps, lr 3e-4), so it follows the changing policy; it starts from the stage-2b switch estimator. Friction switches on (p 0.3). `OnPolicyRunner` gained optional env hooks: `set_learning` (estimator trains only inside `learn()`), and `checkpoint_state` / `load_checkpoint_state` (estimator weights are stored inside every `model_*.pt`, so a checkpoint always replays with its own estimator; no `--state_estimator` needed, and passing one is refused).
- *Run:* `uv run scripts/train.py --task go2_lowfric_est --resume --load_run Nov15_19-11-13_ --max_iterations 500 --seed 1 --device cuda:0 --headless` (local GPU, ~10 min) → `Nov15_21-34-07_est_finetune/model_2000.pt`. Reward 5.9 → 7.4 over the 500 iterations; mean episode time at the 10 s limit from ~iteration 12 on (the shorter early episodes come from the randomized starting episode counters).
- *Result:*

  | | run 12, true μ | run 12 + switch estimator | **run 15** |
  |---|---|---|---|
  | Slide robustness | 8/9 | 6/9 | **9/9** as first scored; **2/9** after the evaluation fix (Stage 2d) |
  | Walking grid | 25/30 | 28/30 | **29/30** (only μ 0.05 at 2 m/s fails) |
  | Switch eval | 19/24 | 24/24 | **24/24**, tracking ~0.05 m/s (was 0.1–0.3) |
  | Slide decel μ 0.05 / 0.1 / 0.15 / 0.2 (μg) | 0.87 / 0.77 / 0.75 / 0.74 | – | **1.01 / 0.91 / 0.79 / 0.43** |
  | 2 m/s tracking RMSE at μ 0.3 / 0.5 / 1 | 0.37 / 0.24 / 0.07 | 0.22 / 0.07 / 0.08 | 0.43 / 0.46 / 0.11 |

  No falls in any slide scenario (push and combined included). Video: `logs/videos/stage1/run15_slide_mu0.1_v1.2.mp4`.
- *Observation: one cautious gait.* μ̂ reads ≈ 0.1–0.2 on most surfaces, ground included (μ 0.8 → 0.13–0.27 for forward / turning; sideways walking is the exception, 0.6–0.9). The fine-tuned policy found a gait that works everywhere without knowing μ precisely; that gait rarely slips, so the estimator has little to read. Consequences: excellent tracking and robustness on ice and ground, but braking at μ 0.2 and fast running at intermediate μ (0.3–0.5) got worse, the regions where "assume ice" is wrong. μ̂ is no longer a trustworthy friction readout for this policy, only an internal signal.
- *Run 16 (continuous friction, from run 15, 500 it):* adding a middle bin (0.2–0.6) fixed what "assume ice" broke at intermediate friction and restored μ̂ as a friction readout:

  | | run 15 | **run 16** |
  |---|---|---|
  | 2 m/s tracking RMSE at μ 0.3 / 0.5 (m/s) | 0.43 / 0.46 | **0.16 / 0.13** |
  | Sideways (0.5 m/s) RMSE at μ 0.3 / 0.5 | 0.18 / 0.19 | **0.11 / 0.09** |
  | μ̂ 2 s after 0.1 → 0.8 (forward) / no switch at 0.8 (forward) | 0.17 / 0.14 | **0.75 / 0.93** |
  | Walking grid / switch eval | 29/30, 24/24 | 29/30, 24/24 |
  | Slide robustness (as first scored → after the Stage 2d fix) | 9/9 → 2/9 | 8/9 → 3/9 |
  | Slide decel μ 0.05 / 0.1 / 0.15 / 0.2 (μg) | 1.01 / 0.91 / 0.79 / 0.43 | 0.98 / 0.96 / 0.74 / 0.45 |

- *Remaining problem: μ̂ during the slide.* Median μ̂ over the slide trials: μ 0.1 → 0.56–0.61, μ 0.2 → 0.29–0.41 (run 16). In the residual-hold pose the feet barely move relative to the body, so the history carries little friction information (and the slide phase is a small share of training data). On μ ≤ 0.1 a stiff hold brakes at ~μg regardless; at μ 0.2 a stiff hold tips over (scripted hold: 88% falls), so the teacher (run 12) braked well there only because it was told μ. The student, not knowing, stops gently (0.45 μg) and never falls.
- *Candidate fixes (not started):* (a) **latch μ̂ at slide start**: the surface does not change during a slide episode and μ̂ is accurate while running up on it, so hold the pre-slide estimate through the slide (estimator and actor input); (b) v̂ to the actor (deceleration is directly μg on all-feet sliding, so measured deceleration reveals μ); (c) more slide-phase weight in the estimator's training samples.
- *Earlier next steps:* if braking at μ 0.2 or speed at intermediate friction matters, (a) longer fine-tuning or continuous μ (not two bins) so the middle range is trained, (b) feed v̂ (body speed) to the actor, (c) the same pipeline from scratch (estimator and policy trained together) as a comparison. Deployment: export actor + estimator together (both in `model_2000.pt`).

## Stage 2d: evaluation fix and μ̂ latch in the slide (run 17)

- *Evaluation bug (found while building the latch):* `evaluate_slide.py` and `play.py --slide` set `slide_phase` from outside, but `Go2LowFric._apply_slide_commands` zeroed it inside `step()` whenever the env was not running its own slide episode. The policy read the scripted phase (set after the step), but the in-env estimator of `go2_lowfric_est` read it inside the step, i.e. "not sliding" throughout every evaluated slide. Training was unaffected. Fix: `external_slide_phase` flag (scripts that drive the phase set it; the env then leaves the phase alone). The stage-2a/2b external estimator (`EstimatorInTheLoop`) read the phase correctly and is unaffected.
- *Re-scored:* run 15 robustness 9/9 → **2/9**, run 16 8/9 → **3/9** (failures: stopping distance at μ 0.2, 2.2–3× ideal; one fall in 8 at μ 0.2 in *combined*). With the correct phase, μ̂ over the slide trials is compressed toward 0.1: μ 0.05 → 0.07–0.09, μ 0.2 → 0.11. The Stage 2c conclusions about walking, middle friction and switches stand (those evaluations never set the phase).
- *Option 1, latch:* `state_estimator.latch_in_slide` holds the actor's μ̂ at its slide-start value until the slide phase ends (estimator keeps running and training underneath). On run 16's checkpoint without retraining it changed nothing (3/9): the latched value is ~0.11 at μ 0.2, because μ̂ was already ~0.11 during the run-up. *Cause:* running straight at ~1 m/s never needs more than ~μ 0.1, so the feet do not slip and μ 0.1 vs 0.2 is not observable from proprioception.
- *Run 17 (run 16 + 500 it, latch on):* robustness **7/9**; μ 0.2 braking 0.54 μg (run 16: 0.39), distance 1.93× (2.99×); μ̂ at μ 0.2 rose to 0.38–0.47. Walking 29/30, switches 24/24. Likely mechanism: with the latch, only the run-up estimate matters for the slide, so policy and estimator, trained together, found a run-up that reveals friction better (now over-estimating μ 0.2 a little). New failures: falls at μ 0.2 in *entry_speeds* (2/8) and *combined* (1/8), where braking harder tips the robot (the teacher's known edge, cf. scripted hold 88% falls at μ 0.2).
- *Next (not started):* add the IMU **accelerometer** (body linear acceleration, which a real Go2 IMU provides) to the estimator inputs. While all feet slide, deceleration = μg, so μ becomes directly observable *during* the slide, instead of inferred from a run-up; the latch could then be dropped. v̂ to the actor after that if still needed.

## Stage 2e: IMU accelerometer for the estimator (runs 18a/18b)

- *Why:* a Go2's IMU also measures linear acceleration, which the estimator did not use. While all feet slide, deceleration = μg, so μ becomes observable in the slide; the gait-based inputs could not separate μ 0.1 from 0.2.
- *Built:* `Go2LowFric.base_lin_acc` (specific force = a − g in the body frame, units of g, finite difference of base velocity per physics step, restarted at resets); `DEFAULT_FEATURES` = `LEGACY_FEATURES` + `base_lin_acc`; `expand_estimator_inputs` widens a trained estimator with zero first-layer weights for new inputs (checked live on run 17: identical μ̂ to 6e-7), new inputs normalized on the first training batch; checkpoints now store `estimator_config` (older ones are read as `LEGACY_FEATURES`). Accelerometer has no sensor noise yet.
- *Runs:* both from run 17, 500 it, trained in parallel: **18a** latch on, **18b** latch off (accelerometer replacing the latch).
- *Result:* 18a passes all 9 slide scenarios with no falls, keeps walking 29/30 and switches 24/24; 18b falls at μ 0.2. In both, μ̂ at μ 0.2 moved toward the truth (0.17–0.26 vs 0.38–0.47 in run 17). With the latch the policy acts on the run-up estimate; without it, the more accurate in-slide μ̂ leads it to brake harder at μ 0.2, where a hard stance tips (the teacher's edge too).
- *Caveats:* one seed each; 18a differs from run 17 by the accelerometer *and* 500 more iterations (no run 17 + 500 it control); μ 0.2 braking (0.37–0.69 μg) is still below the teacher's 0.74, it passes on the μ 0.05–0.2 averages.
- *Attribution (runs 19, 18a-s2):* the control (run 17 + 500 it without the accelerometer) scored 5/9 with 6 fall-instances and kept over-estimating μ 0.2 (0.34–0.53); 18a's second seed scored 8/9 with 2 (one at untrained μ 0.3) and μ̂ 0.15–0.16. So the accelerometer, not the extra training, fixes μ̂ at μ 0.2 and removes most falls. Note the accelerometer runs slightly *under*-estimate μ 0.2, which also means gentler braking; part of the fall reduction may come from that.
- *What the accelerometer did not fix:* μ 0.2 braking is ~0.5 μg in all four runs (17, 19, 18a, 18a-s2) regardless of μ̂ accuracy, with all four feet down only 17–24% of the slide. Per-leg diagnostic (runs 12, 15, 18a; v0 1.2): in every run, the teacher included, FR carries the most load and RR the least (left/right asymmetry learned early, never penalized); at μ 0.2 the students shift ~50% of the load onto FR and nearly lift RL (loaded 30% of the slide; teacher 84%). The braking limit at μ 0.2 is now posture, not estimation.
- *Next (not started):* (1) per-foot load balance in the slide evaluation; (2) scripted test of braced slide poses (front feet forward) at μ 0.1–0.3, adopt as the residual-hold / slide_pose target if one brakes near μg without tipping; (3) left/right mirror symmetry in training; then accelerometer noise/bias and the slide range (μ ≤ 0.25–0.3).
- *Earlier next-step list:* accelerometer noise for realism; then the slide range (μ ≤ 0.25–0.3) or torque (`tau_est`) as the next sensor.

## Roadmap

Each stage changes one big thing (behavior → physics → structure), so results stay attributable.

1. **Finish controlled sliding** (privileged μ) — **done, frozen on run 12** (see Stage 1 result)
   - Turn-in-place experiments closed (runs 13, 13b, 14 all broke the slide). Stage 1 baseline: run 12's configuration (runs 12 / 12b). Next: freeze stage 1 (final checkpoint choice, videos, figures, summary).
   - *Done when:* slide brakes at ≥ 0.9 μg and walking stays ~28/30.
   - Extensions from the goal: controlled stop variants (yaw/spin to reorient while sliding). On flat isotropic ice, legs out cannot brake harder than μg.

2. **Skate-like feet: anisotropic friction, physics first**
   - Foot contacts with low friction along a "blade" direction and high friction across it. Hard part: the blade direction must follow each foot's heading (check how MuJoCo orients contact friction frames for the foot geometry).
   - *Validate with no policy* (like the frozen-pose test): launched forward, the frozen robot should glide far (decel ≈ μ_along·g); launched sideways, it should stop quickly. Don't train until this holds.
   - Note: anisotropy comes from the feet (blades/wheels), so it applies on every surface; a skating Go2 is effectively a different robot, not a reaction to slipping.

3. **Skating without oscillators** (baseline)
   - Start from the `go2_lowfric` config on the new feet; turn the glide reward back on (gliding now pays physically).
   - Evaluate with matched-speed CoT, top speed, slide/glide metrics, videos; compare against walking on ice.

4. **Decentralized oscillators (DCPG) only if needed**
   - Add a `go2_skate_osc` variant (per-leg oscillators with ground-force feedback, policy outputs residuals + frequency/coupling modulation, oscillators held in stance during the slide phase) if stage 3 shows:
     - no regular push-glide stroke (fast irregular stepping instead);
     - no CoT or top-speed gain over walking on ice despite the physics allowing it;
     - persistent asymmetry (tucked leg, one side doing all pushing);
     - unstable results across runs/seeds.
   - Compare against the stage 3 baseline on the same evaluations.

**Deployable version (μ estimation)** — **best: run 18a (Stages 2a–2e)**, policy reads only μ̂; slide 9/9, walking 29/30, switches 24/24; independent of stages 2–4.
- *Approach:* an explicit **neural-network state estimator** trained by supervised learning in simulation: input = a short history (~0.5 s) of what a real Go2 measures (joint positions/velocities, previous joint targets, IMU angular velocity and gravity direction, commands, slide flag); output = μ̂ (and base velocity v̂, which the actor also lacks); label = the true μ (and v) from the simulator. The run-12 policy then reads μ̂ instead of μ.
- *Existing code:* `learning/algorithms/SE.py` (`StateEstimator`: MSE regression, rollout storage, export) and `learning/modules/state_estimator.py` (MLP) exist but are not wired into any runner yet; history inputs are new.
- *Steps:* (1) record (history, μ, v) data with the frozen run-12 policy on randomized μ and train the estimator offline; check μ̂ accuracy by friction and how fast it settles; (2) run the policy on μ̂ through the existing evaluations (`--misreport_friction` already showed ±30% μ error is tolerated); (3) if the policy needs it, fine-tune or train jointly (estimator updated alongside PPO, policy sees μ̂ during training).
- *Observability limit:* μ only shows when feet slip. On grippy ground the estimator can only say "μ is high enough", which is fine because the policy barely depends on μ there (Fig. 2).
- *Friction patches:* train and test with μ changing mid-episode (ice → carpet → ice) so the estimate, and the gait, switch on contact with a new surface. **Done (Stage 2b):** time-based switches; estimator retrained with switches re-estimates in 0.25–0.5 s; closed-loop fixed point for slow gaits remains.
- *Alternatives (not chosen first):* teacher–student distillation of the whole policy (RMA-style latent instead of μ̂), or a recurrent policy with no explicit estimate. The explicit μ̂ keeps run 12 unchanged and is easy to inspect.

**Other open items**
- Continuous log-uniform μ instead of two bins; friction-aware commands on ice.
- Slide phase ends on a schedule, not on the robot's actual stop; consider a speed-based end (needs a velocity estimate).
- Revisit: stand-still episodes.


Prompt
implement option 1 and the braced pose test
Update changes.md as I pushed current work to github
What sensors are available?
Currently are we on continuous mu?
Bump up training to sliding to mu 0.3?
Implement accel noise and bias
How can we identify the best coefficient to start sliding?
Left/right mirror symmetry in training: removes the lopsidedness in walking too.
Add posture hypothesis to designNotebook.md