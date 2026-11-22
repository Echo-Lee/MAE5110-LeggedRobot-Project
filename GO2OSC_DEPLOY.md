# Sliding gaits on the real Go2: `go2osc_push_glide_deploy`

This is the hardware path for the team's push-glide policy. It answers the
four points the course staff raised before the hardware test.

| Staff point | Status |
|---|---|
| 1. One policy, published with SHA256 and saved config | **Not yet trained.** No earlier checkpoint is deployable (see below). Train with the command in [Train](#train); the export step writes the hashes and the saved config. |
| 2. Every actor input from the robot's sensors | Done in `go2osc_push_glide_deploy`; checked by tests. |
| 3. Joint limits | The current policy fails the check. The new task clamps and penalises; `scripts/check_go2_joint_limits.py` checks a trained policy. |
| 4. Oscillators and estimator inside the deploy runner | Staff runner `test-11-19` merged; `go2_deploy/custom_implementation/go2osc_push_glide.py` added; tested against the simulator. |

## Why no existing checkpoint can go on the robot

The latest policy (`go2osc_push_glide_robust`, carve run, iteration 4000):

- **Inputs the robot lacks.** Its actor reads `foot_traction_obs` (true
  contact-force ratios), `friction_estimate_obs` (derived from the true
  friction) and `base_lin_vel` (true body velocity). Its clocks read true
  foot forces.
- **Joint limits.** Its calf targets sit past the upper stop (−0.838 rad)
  17–32% of the time, reaching −0.14 rad, and the measured calf angle passes
  the stop by up to 0.04 rad. MuJoCo's stop is soft; the real one is not.
  With targets clamped inside the limits, the same policy stays upright but
  slows down (2.0 m/s command: 1.91 → 1.74 m/s on friction 1.0, 1.81 → 1.41
  on 0.1), so it has to be retrained with the clamp.

## What the new task changes

`go2osc_push_glide_deploy` is the robust task (friction-limited speed target,
carved sliding stops, hardware variation) with:

| | Robust task | Deploy task |
|---|---|---|
| Actor inputs | 66, three of them privileged | 58: IMU, joints, command, previous targets, oscillator phases, μ̂, v̂, speed target v_t |
| Friction and velocity | true values with noise | state estimator (μ̂, v̂), trained in the same run as the policy |
| Speed target v_t | from the true-friction estimate | from μ̂; starts an episode at v̂ |
| Clock feedback | true foot force, every physics step | joint-torque foot load F = −(Jᵀ)⁻¹τ and foot stroke J q̇, once per control step |
| Joint targets | ±10 action units | also kept 0.05 rad inside the joint limits |
| Rewards | – | + `dof_pos_target_limits` 5, `dof_pos_limits` 10 |

The estimator reads only: gyroscope, gravity direction, accelerometer, joint
angles, speeds and torques, the command, the previous targets and the
oscillator phases (0.5 s of history). The critic keeps the simulator's true
values; it is not deployed.

Swapping the clock feedback did not change the gait of the current policy:
with clocks fed from joint torques it walks and stops as before (speeds
within 0.03 m/s and stop times within 0.2 s of the true-force run on
friction 0.05–1.0, 60/60 upright). With sensor noise and hardware variation
on as well, it also stayed upright 60/60.

## Train

One run trains the policy and the estimator together, from scratch:

```sh
uv sync --frozen --extra gpu
PYTHONPATH=. uv run --frozen --extra gpu python scripts/train.py \
    --task go2osc_push_glide_deploy --backend mujoco --device cuda:0 \
    --num_envs 4096 --max_iterations 4000 --save_interval 50 --seed 0 \
    --experiment_name go2osc_push_glide_deploy --headless --disable_wandb
```

The policy and estimator train together from scratch. Evaluate the saved
checkpoint before exporting it for a hardware test.

### Numerical guards

The estimator and actor normalizer are shared across environments. Invalid
sensor-derived values must be contained before they reach those shared
modules.

| Signal or update | Handling |
|---|---|
| Estimator input frames | Replace non-finite entries with zero; bound scaled inputs to ±1e4 |
| Estimator training labels | Drop samples with non-finite labels |
| Estimator gradients | Skip non-finite gradient steps |
| Derived foot forces and velocities | Replace non-finite entries with zero |
| Primary simulator state | Stop with the affected environment, signal, command, friction and episode time |

Guard counters are reported alongside estimator losses when they change.
Use these diagnostics to locate invalid inputs; sanitization does not
establish that the simulator or learned policy is behaving correctly.

## Check and export

```sh
# Joint limits: exit code 0 and "inside the limits" are required.
PYTHONPATH=. uv run --frozen python scripts/check_go2_joint_limits.py \
    --task go2osc_push_glide_deploy \
    --checkpoint logs/go2osc_push_glide_deploy/<run>/model_4000.pt

# Bundle for the robot, with hashes and the saved configuration.
PYTHONPATH=. uv run --frozen python scripts/export_go2osc_bundle.py \
    --name push_glide_v1 \
    --checkpoint logs/go2osc_push_glide_deploy/<run>/model_4000.pt
```

`logs/policies/push_glide_v1/` then holds `policy.pt`, `estimator.pt`,
`go2osc_onboard.json`, `deploy_meta.json`, `training_config.json` and
`SHA256SUMS` (which also lists the checkpoint's hash). To publish:

```sh
cd logs/policies && tar czf push_glide_v1.tar.gz push_glide_v1
gh release create push-glide-v1 push_glide_v1.tar.gz \
    ../go2osc_push_glide_deploy/<run>/model_4000.pt \
    --title "go2osc_push_glide_deploy, iteration 4000" \
    --notes-file push_glide_v1/SHA256SUMS
```

## Deploy

```sh
uv run --frozen --extra unitree_sdk python -m go2_deploy.deploy <nic> \
    --bundle logs/policies/push_glide_v1
```

or onboard, with the staff script:
`scripts/deploy_go2_remote.py --bundle logs/policies/push_glide_v1 --host ...`.

What the bundle fixes, so that nothing has to be typed on the day:

- **Gains:** kp 20, kd 0.5 (the training values).
- **Posture:** hip 0, thigh 0.66, calf −1.36 rad.
- **Sticks:** forward −1.0 to 2.0 m/s, sideways ±0.3 m/s, turn ±0.5 rad/s.
  A stick within 5% of centre reads zero, so a released stick is the exact
  zero command of a sliding stop.
- **Joint targets:** the exported policy's outputs are clamped 0.05 rad
  inside every joint limit.

The implementation refuses to start if the runner would break what the
policy was trained with:

- no bundle (it carries the estimator and the clamp);
- a control rate other than 100 Hz;
- action smoothing on. `go2_deploy/deploy_config.py` now sets
  `exp_moving_avg = False`; the staff default (0.095 s) would lag every
  joint target by about a tenth of a second, which the policy never saw.

The runner loads `go2osc_push_glide.py` by default; set
`GO2_CUSTOM_IMPL=go2trot_base.py` for the staff's trot template.

### Before pressing A (policy on)

With the robot standing under Unitree's own controller, read the signals
without commanding anything:

```sh
uv run --frozen --extra unitree_sdk python scripts/check_go2osc_signals.py \
    <nic> --bundle logs/policies/push_glide_v1
```

It prints one line a second (the deploy run prints the same line once,
when the policy takes over):

```
go2osc check | foot load FL FR RL RR [0.25, 0.24, 0.26, 0.25] (sum 1.00) | accelerometer 9.81 m/s^2 | estimated speed [0.0, 0.0, 0.0] m/s, friction 0.6
```

Standing still on four feet, expect about a quarter of the weight on each
foot (sum near 1), an accelerometer near 9.81 and a speed near 0. A sum far
from 1 means the torque readings or the weight in `go2osc_onboard.json` do
not match this robot. This script has not been run on a robot: it was
checked with a stand-in for the SDK only.

## Files

| File | Role |
|---|---|
| `gym/envs/go2/go2osc_deploy.py`, `go2osc_deploy_config.py` | The task |
| `gym/envs/go2/go2osc_onboard.py` | What the robot computes: leg kinematics, foot load, oscillators, speed target, estimator history |
| `go2_deploy/custom_implementation/go2osc_push_glide.py` | Staff-runner adapter: LowState → those computations → observation pieces |
| `scripts/export_go2osc_bundle.py` | Checkpoint → bundle |
| `scripts/check_go2_joint_limits.py` | Joint-limit check for any Go2 task |
| `scripts/check_go2osc_signals.py` | Read-only signal check on the robot |
| `tests/unit_tests/test_go2osc_onboard.py` | Robot-side code against the simulator |

`test_go2osc_onboard.py` steps the simulator and the robot-side code from
the same sensor readings: oscillator phases, v_t, μ̂, v̂ and the push signal
agree to 1e-5 over 160 steps of walking, turning, sidestepping and carved
stops, and the full 58-value observation built by the deploy file from a
fake LowState message matches the training observation.

## Not verified

- **Training.** The task runs (CPU smoke run, 3 iterations) but no policy
  has been trained with it. How well μ̂ tracks friction, and how the policy
  performs with it, are unknown until the run above finishes.
- **The real signals.** `tau_est`, the accelerometer and the gyroscope are
  assumed to follow the simulator's conventions (torque about the joint
  axis, specific force in the body frame). The signal check above is the
  first test of that.
- **Robot mass.** Foot loads are in body weights of the URDF's 16.3 kg.
- **`tests/unit_tests/test_go2_deploy_obs.py`** is skipped: it targets the
  earlier Go2Trot runner and 7 of its tests fail on staff branch
  `test-11-19` itself.
