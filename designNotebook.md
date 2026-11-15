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
| 10 | pending | + `slide_still` weight 3 (damp joint motion while sliding) | Target: decel → μg, less chatter. Run 9 baseline: 0.7–5 real steps/s, 43–78 contact breaks/s in the slide. |

### Result figures

Each run keeps the same color in every figure. Data: `logs/eval/*.json`.

| Figure | Shows |
|---|---|
| [fig1](logs/eval/figures/fig1_friction_all_runs.png) | Friction grid for runs 0–6: tracking, slip, CoT, touchdowns vs μ |
| [fig2](logs/eval/figures/fig2_mu_reliance.png) | Run 4 with the true μ vs told μ = 1 |
| [fig3](logs/eval/figures/fig3_cot_matched_speed.png) | Matched-speed CoT vs μ and top speed per μ |
| [fig4](logs/eval/figures/fig4_rollout_timeline_power.png) | Run 4, one full training-style episode on ice and on ground |
| [fig5](logs/eval/figures/fig5_training_curves.png) | Training curves: action noise, power penalty, glide reward |

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

7. **Damp joint motion while sliding** (run 10, pending)
   - *Metric fix first:* slide touchdowns now count only landings after ≥ 50 ms in the air (`touchdowns_per_s`); every reloading is reported separately (`contact_breaks_per_s`). Re-evaluated run 9: **0.7–5 real steps/s** (fewest on the slickest ice) vs 43–78 contact breaks/s. Run 9 already mostly stands in the slide; the remaining gap to μg is bouncing and joint motion, not stepping.
   - *Change:* `slide_still` 0 → 3. Run 9 scores −0.183 per step in its slides (≈ −0.55 weighted) vs ≈ 3.8 for tracking.
   - *Success:* decel ratio and distance ratio → 1, contact breaks/s ↓, joint speed in the slide ↓ (run 9: ~1.2 rad/s), walking grid unchanged (28/30).

## Observations

- **The policy uses μ:** told μ = 1 on ice, slip and CoT rise sharply. The best policy depends on it less, so a state estimator should be implemented.

  ![Reliance on mu](logs/eval/figures/fig2_mu_reliance.png)
  *Fig. 2. Run 4 told μ = 1 while on ice: slip 0.15 → 0.52 and CoT 1.37 → 2.18 at μ = 0.05; on ground the two curves meet.*
- **Ice speed limit:** top speed on μ = 0.05 is ~1.3–1.6 m/s for every policy; 2 m/s is physically out of reach.
- **Infeasible training commands:** about half the random commands on ice ask for speed changes faster than μ·g allows, a source of noise in training.
- **Survival:** 98–99.6% across full training-style episodes for the best policy (Fig. 4).
- **Averages hide gait problems:** the per-foot load and touchdown checks and the videos found every lopsided gait.

## Open items and next steps

- Train and evaluate run 10 (`slide_still` = 3).
- Controlled stop from the goal (legs out in front, yaw torque): build on the slide phase once coasting works.
- Sim-to-real:
  - μ / body-velocity estimator (student policy); the actor currently gets no linear velocity.
  - Continuous log-uniform μ instead of two bins.
  - Friction-aware commands on ice.
- Revisit: stand-still episodes and decentralized phase oscillators (fix 8 Hz stepping).
- Try and implement anisotropic friction model to get a "skating gait"
