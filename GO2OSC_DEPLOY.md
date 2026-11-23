# Go2 push-glide training and deployment

`go2osc_push_glide_deploy` combines push/glide oscillators with a learned
friction and body-velocity estimator. Software checks cover the sensor
interfaces and controller execution; physical sensor conventions and robot
behavior require hardware validation.

| Component | Interface |
|---|---|
| Policy artifact | Publish the actor, estimator, saved settings, SHA256 manifest and exact source commit as a release. |
| Actor inputs | 58 values from IMU, joints, operator commands and controller state, including learned friction and velocity estimates. |
| Joint limits | Final joint targets are clamped 0.2 rad inside each limit. Check measured angles separately with `scripts/check_go2_joint_limits.py`. |
| Deployment runner | `go2_deploy/custom_implementation/go2osc_push_glide.py` runs the oscillators and estimator within the staff runner interface. |

## Select a policy

The robust task uses privileged simulator inputs and has 66 actor inputs;
its checkpoint is a training initializer, not a deploy-task bundle. Use a
checkpoint trained with the deploy task and export its matching estimator.

An export may apply controller settings that differ from the saved training
configuration. Its provenance must identify those overrides. Evaluate the
resulting controller separately and label an unevaluated or software-only
release as experimental. A target clamp alone does not establish that
measured joint angles remain inside the limits.

## Controller interfaces

`go2osc_push_glide_deploy` is the robust task (friction-limited speed target,
carved sliding stops, hardware variation) with:

| | Robust task | Deploy task |
|---|---|---|
| Actor inputs | 66, three of them privileged | 58: IMU, joints, governed command, previous targets, oscillator phases, μ̂, v̂, speed target v_t |
| Friction and velocity | true values with noise | state estimator (μ̂, v̂), trained in the same run as the policy |
| Speed target v_t | from the true-friction estimate, toward the command | from μ̂, toward the governed command; starts an episode at the body's forward speed (zero on the robot, which starts at rest) |
| Command governor | – | the forward command handed to the actor and to v_t is kept within `command_governor.margin` = 0.5 m/s of v̂ in the direction of travel; slowing down is never limited; sideways and turn commands pass unchanged |
| Clock feedback | true foot force, every physics step | joint-torque foot load F = −(Jᵀ)⁻¹τ and foot stroke J q̇, once per control step |
| Hand-over | – | until iteration 1500 the actor and v_t read the robust task's stand-ins (simulated friction estimate, true velocity with noise); blended to μ̂ and v̂ by iteration 2000 |
| Joint targets | ±10 action units | also kept 0.2 rad inside the joint limits; the clamp closes over iterations 1000 to 1500 |
| Rewards | – | + `joint_limit_margin` 2 (angles within 0.1 rad of a limit), `target_saturation` 1, `standing_saturation` 4 (the same cost, counted only while the clocks stand) |

The estimator reads only: gyroscope, gravity direction, accelerometer, joint
angles, speeds and torques, the raw command, the previous targets and the
oscillator phases (0.5 s of history). The critic keeps the simulator's true
values; it is not deployed.

## Train

The estimator trains beside the policy in the same run. Two schedules
follow the training iteration, one change at a time
(`control.joint_target_ramp`, then `state_estimator.handover`):

| Training iteration | Joint-target clamp | Actor's friction and velocity inputs, and v_t's friction |
|---|---|---|
| up to 1000 | open: 0.4 rad beyond the limits | stand-ins, as in the robust task: the simulated friction estimate and the true body velocity with ±0.1 m/s noise |
| 1000 to 1500 | closes linearly to 0.2 rad inside | stand-ins; the estimator learns beside the policy |
| 1500 to 2000 | closed | linear blend to μ̂ and v̂ |
| from 2000 | closed | μ̂ and v̂ only, as on the robot |

The log prints `[go2osc schedule] iteration N: joint targets kept ... the
limits; the estimates' share is ...` at each quarter of either schedule.
The export refuses a checkpoint from before iteration 2000. An environment
that is not told an iteration (the robot, a fresh evaluation) has the clamp
closed and reads the estimates.

**Fine-tune the robust policy (recommended).** The converter maps the carve
run's policy onto the deploy task's inputs (μ̂ and v̂ take the columns of
the signals they replace; `foot_traction_obs` is dropped) and labels it
iteration 1000, so the clamp closes and the estimator learns over the first
500 iterations, on a gait that already walks:

```sh
uv sync --frozen --extra gpu
PYTHONPATH=. uv run --frozen python scripts/convert_go2osc_robust_to_deploy.py \
    --checkpoint logs/go2osc_push_glide_robust_carve_4000/Nov20_18-29-45_/model_4000.pt \
    --experiment_name go2osc_push_glide_deploy_finetune
PYTHONPATH=. uv run --frozen --extra gpu python scripts/train.py \
    --task go2osc_push_glide_deploy --backend mujoco --device cuda:0 \
    --num_envs 4096 --resume --load_run from_robust --max_iterations 2000 \
    --save_interval 50 --seed 0 \
    --experiment_name go2osc_push_glide_deploy_finetune --headless
```

It ends at training iteration 3000 (`model_3000.pt`). The charts of a
resumed run count from 1, so the clamp closes over logged iterations 1 to
500 and the hand-over shows at 500 to 1000.

**From scratch.** The same task without `--resume`, `--max_iterations 4000`.

Evaluate the saved checkpoint before exporting it for a hardware test.

### Command governor

On low friction the body cannot follow a large forward command at once.
The governor limits how far the command seen by the actor and by v_t
leads the estimated speed:

    governed = clamp(command, min(v̂, 0) − 0.5, max(v̂, 0) + 0.5)   [m/s]

The estimator, the reward reference and the critic keep the raw command.
The standalone custom configuration applies the same rule on the robot
(`speed_target.command_margin` in `go2osc_onboard.json`).

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

### Joint limits

The real joint limits are hard stops. A target beyond one drives an
unloaded joint into its stop, so the joint **targets** are clamped
`control.joint_target_margin` = 0.2 rad inside every limit, and the
exported policy carries the same clamp. The clamp costs knee torque near
full extension (τ = kp (target − angle)), so `control.joint_target_ramp`
closes it during training for the policy to adapt.

- `joint_limit_margin` (weight 2.0, `reward_settings.joint_limit_margin`
  0.1 rad). Per joint, with d the distance of the angle into the margin
  before a limit: cost d / 0.1, so 0 outside the margin, 1 on the stop and
  more beyond it. Summed across joints.
- `target_saturation` (weight 1.0). Penalizes the difference between
  requested and applied joint targets, in action units, with a Huber cost
  summed across joints (quadratic below one unit, linear above). It keeps
  an output that the clamp cuts off from drifting further out.
- `standing_saturation` (weight 4.0). The same cost, counted only while
  the clocks stand (|v_t| ≤ 0.2 m/s, no sideways or turn command). Walking
  needs the clamp on the calves for part of each stride; standing does
  not, and outputs held beyond the clamp there leave the stance unsteady.

## Check and export

```sh
# Joint limits: exit code 0 and "inside the limits" are required, for the
# measured angles and the commanded targets (--angle_margin 0.05 demands
# clearance for the angles as well).
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
    --repo Echo-Lee/MAE5110-LeggedRobot-Project --target <matching-commit> \
    --title "go2osc_push_glide_deploy, iteration 4000" \
    --notes-file push_glide_v1/SHA256SUMS
```

## Deploy

```sh
uv run --frozen --extra unitree_sdk python -m go2_deploy.deploy <nic> \
    --task go2osc_push_glide_deploy \
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
- **Joint targets:** the exported policy's outputs are clamped 0.2 rad
  inside every joint limit, as in training.

The implementation refuses to start if the runner would break what the
policy was trained with:

- no bundle (it carries the estimator and the clamp);
- a control rate other than 100 Hz;
- action smoothing on. `Go2OscPushGlide` sets
  `exp_moving_avg = False`; the generic default (0.095 s) would lag every
  joint target by about a tenth of a second, which the policy never saw.

The wireless runner discovers `Go2OscPushGlide`, a `CustomConfig` subclass,
by its `task_name = "go2osc_push_glide_deploy"`. Its instance supplies the
custom buffers and observation callbacks. The class sets 100 Hz, kp 20,
kd 0.5 and no smoothing. Its observation computations, state buffers,
ordered input names, dimensions, scales and posture are defined directly
in `go2_deploy/custom_implementation/go2osc_push_glide.py`; this file has no
environment or simulator imports. Training imports this observation
declaration and scaling class. Bundle metadata must match the declaration
and is checked before inference. Use `--task go2trot` for the staff template.

For policy review, provide `logs/<task>/<run>/model_<iteration>.pt` and the
same run's `files/gym/envs/go2/` source snapshot. Preserve the original
training snapshot when an experiment directory differs from the task name.
The raw log folder supports reconstruction; this task requires the complete
exported bundle for deployment because it includes the estimator, settings
and final output clamp.

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
| `go2_deploy/custom_implementation/go2osc_push_glide.py` | Observation definitions, leg kinematics, foot load, oscillators, speed target, estimator buffers and staff-runner callbacks; standalone PyTorch code |
| `gym/envs/go2/go2osc_onboard.py` | Compatibility imports of the custom configuration's functions |
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

- **Training.** Forward locomotion, friction estimation and controlled
  sliding must be evaluated on checkpoints trained with the current rewards
  and target bounds.
- **Checkpoint behavior.** Passing the hand-over and clamp schedules does
  not establish locomotion quality; evaluate the selected checkpoint with
  its exported controller settings.
- **The real signals.** `tau_est`, the accelerometer and the gyroscope are
  assumed to follow the simulator's conventions (torque about the joint
  axis, specific force in the body frame). The signal check above is the
  first test of that.
- **Robot mass.** Foot loads are in body weights of the URDF's 16.3 kg.
