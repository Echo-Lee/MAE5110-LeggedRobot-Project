# Installation

Clone the repo:

```git clone https://github.com/LampLighterLab/BigRedGym.git```

Then, enter the BigRedGym folder.

Create the venv using uv:

```uv sync --frozen```

Note: you will need to use the ```--frozen``` flag every time you run a python script in this repo when the optional VSim wheel or Unitree SDK checkout is absent. example: ```uv run --frozen scripts/train.py --task=go2trot  --device=cuda:0 --headless --max_iterations=300```

## VS Code debugger

After `uv sync --frozen`, open the repository folder in VS Code with the Microsoft
Python and Python Debugger extensions installed. Select **Train Go2 (viewer)** in
Run and Debug, then press **F5**.

The shared [.vscode/launch.json](.vscode/launch.json) selects `.venv/bin/python`
on Linux, `.venv/Scripts/python.exe` on Windows, and `.venv/bin/mjpython` on macOS.
Edit the single `args` list to change the task, device, environment count, or
other CLI options. macOS users who encounter the library-loading error should
apply the workaround below first.

## macOS

The [MuJoCo passive viewer](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)
requires `mjpython` on macOS. Start with fewer environments than the Go2 trot
default of 4096 when running on CPU:

```sh
.venv/bin/mjpython scripts/train.py --task=go2trot --device=cpu --num_envs=16
```

If `mjpython` fails with `Library not loaded: @executable_path/../lib/libpython3.11.dylib`,
it is looking for uv's Python library relative to `.venv`. For this error, link
the existing library into `.venv/lib`, then rerun the command above:

```sh
ln -s "$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')/lib/libpython3.11.dylib" .venv/lib/libpython3.11.dylib
```

This workaround is local to the virtual environment; repeat it if you recreate
`.venv` and encounter the error again.

For training without a viewer, use regular Python with `--headless`:

```sh
uv run --frozen scripts/train.py --task=go2trot --device=cpu --num_envs=16 --headless
```

# Training

To train, run ```scripts/train.py```:

```
uv run --frozen scripts/train.py --task=go2trot
```

Mandatory arguments:
```
--task: the name of your environment (go2, go2trot, mini_cheetah, etc)
```

Optional arguments:
```
--device: cpu or cuda:0 (nvidia gpu)
--backend: mujoco or vsim (most likely you will only use mujoco)
--num_envs: how many environments to run in parallel in training
--max_iterations: how many training iterations to run (typically 300-500 is sufficient for the go2)
--headless: disable the viewer window during training
--save_interval: override checkpoint interval in learning iterations
--seed: random seed for rng
--batch_size: override batch size from cfg
--experiment_name: Experiment folder under logs/ (default: task config value)
--run_name: Run label appended to the timestamp in the log folder and W&B name
--resume: Resume optimizer and model state from an existing run
--load_run: Run directory under logs/<experiment_name>/ (default: latest).
--checkpoint: Checkpoint iteration to resume (default: latest).
--original_cfg: Load environment and runner configs saved with the selected run.
```

For example, `--task go2trot --run_name tracking_trial` saves to
`logs/go2trot/<timestamp>_tracking_trial/` and uses `<timestamp>_tracking_trial`
as the W&B display name. Use `--experiment_name` to change the parent folder.

# Testing

To test your policy in simulation, run ```scripts/play.py```:

```
uv run --frozen scripts/play.py --task=go2trot
```

Mandatory arguments:
```
--task: the name of your environment (go2, go2trot, mini_cheetah, etc)
```

Optional arguments:
```
--device: cpu or cuda:0 (nvidia gpu)
--backend: mujoco or vsim (most likely you will only use mujoco)
--experiment_name: Override experiment_name (log dir is logs/<experiment_name>/...)
--load_run: Run directory under logs/<experiment_name>/ (default: latest).
--checkpoint: Checkpoint iteration to resume (default: latest).
--original_cfg: Load environment and runner configs saved with the selected run.
```

Go2Trot policies observe sine/cosine gait phase and phase frequency. To resume or
play an older checkpoint trained without these inputs, add `--original_cfg` to
use the configuration saved with that run.

The automated regression suite runs with `uv run --frozen python -m pytest -q`.
See [limitations and validation scope](genAI_skills/README_MUJOCO.md#limitations-and-validation-scope)
for what the retained tests cover.

# Note on AI-generated files
Some of the code has been worked on by an AI agent, in the case where code segments have been heavily edited by AI, the AI-generated files in [genAI_skills](genAI_skills/) may be of some value in understanding the code, especially [README_MUJOCO.md](genAI_skills/README_MUJOCO.md).

# Go2 push/glide policy

`go2osc_push_glide` trains a Go2 policy with decentralized push/glide phase
oscillators in MuJoCo. The actor has 61 inputs, including eight signed
tangential-to-normal foot-force ratios. The critic has 70 inputs, including
the applied friction coefficient as privileged information.

Training uses flat ground with friction sampled uniformly from `[0.1, 1.0]`
per environment at startup. Moving forward commands are sampled uniformly
from `[0.0, 1.5]` m/s, with standing and turning samples. Lateral and yaw
commands are bounded by 0.3 m/s and 0.5 rad/s; commands resample every three
seconds.

The moving oscillator schedule is active above a forward command of 0.2 m/s,
a lateral command of 0.1 m/s, or a yaw command of 0.1 rad/s, using absolute
values. A pure turn uses ungated normal load; translation gates load along
the requested travel direction. Training also penalizes squared roll and
pitch rates with weight 1.0, capped at 4.0 rad²/s².

## Train from scratch

Run from the repository root with Python 3.11. GPU training requires a
compatible NVIDIA driver and CUDA-capable GPU. The repository includes
the Go2 physics asset.

```sh
uv sync --frozen --extra gpu
uv run --frozen --extra gpu python scripts/train.py \
    --task go2osc_push_glide --backend mujoco --device cuda:0 \
    --num_envs 4096 --max_iterations 500 --seed 0 \
    --headless --disable_wandb
```

Checkpoints are saved under `logs/go2osc_push_glide/`. For a small CPU run,
use `--device cpu --num_envs 16`; full training is intended for a GPU.

## Play the included policy

The trained checkpoint is [policies/go2osc_push_glide/model_500.pt](policies/go2osc_push_glide/model_500.pt).
Its configuration and training metadata are recorded in
[policy.json](policies/go2osc_push_glide/policy.json).
This checkpoint's saved configuration uses the forward-only oscillator
schedule and has no tilt-rate reward. `scripts/play.py` uses current task
defaults, including the steering schedule; the recorder below restores
the checkpoint's saved configuration.
Install the standard dependencies with `uv sync --frozen`, then run one of
these commands from the repository root.

macOS requires `mjpython` for the viewer:

```sh
uv run --frozen mjpython scripts/play.py \
    --task go2osc_push_glide --backend mujoco --device cpu --num_envs 1 \
    --experiment_name "$PWD/policies" --load_run go2osc_push_glide \
    --checkpoint 500 --seed 123
```

On Linux, use Python:

```sh
uv run --frozen python scripts/play.py \
    --task go2osc_push_glide --backend mujoco --device cpu --num_envs 1 \
    --experiment_name "$PWD/policies" --load_run go2osc_push_glide \
    --checkpoint 500 --seed 123
```

Playback starts at 1 m/s forward. Press **I/K** to adjust forward velocity,
**J/L** for lateral velocity, **N/M** for yaw, **R** to reset, and **Esc** to
quit. Releasing a key keeps the selected command. Playback samples friction
from the policy's training range.

## Record acceleration and a zero-command stop

Install `ffmpeg` (`brew install ffmpeg` on macOS or `sudo apt install ffmpeg`
on Ubuntu). The recorder loads the included checkpoint and its saved
configuration, fixes ground friction, starts from rest, and keeps the same
policy active when the velocity command becomes zero.

```sh
uv run --frozen mjpython scripts/record_go2osc_stop.py \
    --friction 0.05 --speed 2.5 --switch-time 14 --duration 26
```

On Linux, replace `mjpython` with `python`. On a Linux machine without a
display, set `MUJOCO_GL=egl` before the command. The video and its JSON and
NumPy motion records are saved under `artifacts/go2osc/command_stop.*`.
Use `--output artifacts/go2osc/my_stop.mp4` to choose another filename, or
`--no-video` to save motion data without rendering or requiring `ffmpeg`.

`--speed` specifies the commanded forward velocity; the overlay also shows
the measured speed. Friction 0.05 and a 2.5 m/s command are outside this
checkpoint's training ranges. No controller or reward changes are applied
during recording. For a gradual command increase, add
`--ramp-start-speed 1.5 --ramp-start-time 6 --ramp-end-time 12`.
