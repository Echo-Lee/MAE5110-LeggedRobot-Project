# From a trained policy to the Go2

```
train ──> logs/<task>/<run>/ ──> CustomConfig ──> export bundle ──> send to the Go2 over the cable ──> run (wired or wireless)
```

At deploy time only two things are read: your run in `logs/<task>/` (its own
saved copy of `gym/envs`, never the repo's) and your `CustomConfig` subclass in
`go2_deploy/custom_implementation/`. The robot itself only reads the bundle and
that subclass.

First time? Do the one-time setup in [README_DEPLOY.md](README_DEPLOY.md#setup)
(SDK, CycloneDDS, the PC's static IP `192.168.123.99/24` on the cable), and the
Jetson setup in [Onboard deployment](README_DEPLOY.md#onboard-deployment-go2-edu-wireless).

The examples below train a run called `trot_v1` and deploy it as the bundle `trot_v1`.

## 1. Train

```bash
uv run --frozen scripts/train.py --task go2trot --headless --run_name trot_v1
```

This writes `logs/go2trot/<Mon DD_HH-MM-SS>_trot_v1/`, e.g.
`logs/go2trot/Nov22_18-21-49_trot_v1/`. Don't pass `--experiment_name`: deploy
looks for runs in `logs/<task>/`. To see your runs:

```bash
ls logs/go2trot/
```

## 2. Describe the policy in a `CustomConfig`

Each task has exactly one subclass of `CustomConfig` in
`go2_deploy/custom_implementation/`, found by its `task_name`. For go2trot that
is `go2trot_base.py`; for a new task, copy it. Take every value from **your
run's snapshot**, `logs/<task>/<run>/files/gym/envs/`, not the repo:

| Attribute | Where it comes from in the run's config |
|---|---|
| `task_name` | the task you trained, e.g. `"go2trot"` |
| `obs_vector` | the runner config's `actor.obs`, **same order** |
| `custom_obs_sizes` | size of each obs not built in (built in: `base_ang_vel`, `projected_gravity`, `commands`, `dof_pos_obs`, `dof_vel`, `dof_accel`, `dof_pos_target`) |
| `DeployScaling` | the env config's `scaling`, including what it inherits |
| `default_dof_pos` | `init_state.default_joint_angles` as 12 values: FL, FR, RL, RR; hip, thigh, calf in each |

And three methods: `get_custom_buffers_dict()`, `update_custom_buffers(main_controller, t)`
and `get_custom_obs_pieces()`. Anything missing is a `TypeError`.

```bash
# The values to copy, for go2trot
grep -n -A12 "class actor" logs/go2trot/Nov22_18-21-49_trot_v1/files/gym/envs/go2/go2trot_config.py
grep -n -A10 "class scaling\|default_joint_angles" logs/go2trot/Nov22_18-21-49_trot_v1/files/gym/envs/go2/go2*_config.py

# Check your subclass loads (prints its class name, e.g. Go2TrotBase)
uv run --frozen python -c "from go2_deploy.utility.import_custom_impl import load_custom_impl; print(type(load_custom_impl('go2trot')).__name__)"
```

## 3. Export a bundle

```bash
# Newest run with a checkpoint
uv run --frozen python scripts/export_go2_policy.py --task go2trot --name trot_v1

# A specific run
uv run --frozen python scripts/export_go2_policy.py --task go2trot --name trot_v1 --run_name Nov22_18-21-49_trot_v1
```

It should print:

```
Loaded task 'go2trot' from: .../logs/go2trot/Nov22_18-21-49_trot_v1/files/gym/envs
Loading model from: .../logs/go2trot/Nov22_18-21-49_trot_v1/model_500.pt
Wrote logs/policies/trot_v1/deploy_meta.json
```

It fails if the run doesn't declare the task or the actor's input size doesn't
match `obs_vector`. Check that `provenance.run_dir` is your run and that
`default_dof_pos` and `scaling` are your values:

```bash
python3 -m json.tool logs/policies/trot_v1/deploy_meta.json
```

## 4. Send it to the Go2 and run it

Both ways start the same: the PC is on the ethernet cable (the robot is at
`192.168.123.18`), and `deploy_go2_remote.py` copies your **working tree,
uncommitted changes included**, to `~/BigRedGym` on the Jetson (everything
except `logs/`, `.git/`, `.venv/`), then the bundle to
`logs/policies/trot_v1/`. The policy always runs on the Jetson.

Before every run: area clear, someone on the remote. **`X` is the e-stop.**

```bash
ping -c 2 192.168.123.18   # the cable is up
```

### Wired: run it with the cable plugged in

Robot standing. This syncs, then starts the controller in the tmux session
`go2_deploy` on the Jetson and attaches your terminal to it, so you get the
script's output and the keyboard controls:

```bash
uv run --frozen python scripts/deploy_go2_remote.py --bundle logs/policies/trot_v1 --host unitree@192.168.123.18
```

Detach with `Ctrl+b d` (the run keeps going); rerun the same command to
reattach. To see what it would run without doing it, add `--dry_run`.

<details><summary>Tethered alternative: policy runs on the PC</summary>

Needs the SDK and CycloneDDS on the PC. Uses the newest run in `logs/<task>/`
directly (no bundle, no `--run_name`). The first argument is the PC's cable
interface: the one with `192.168.123.99` in `ip -br addr`, e.g. `eno1`:

```bash
export CYCLONEDDS_HOME="$HOME/.local/opt/cyclonedds-0.10.2"
uv run --frozen --extra unitree_sdk python -m go2_deploy.deploy eno1 --task go2trot
```
</details>

### Wireless: install over the cable, unplug, start from the remote

1. **Install over the cable.** This syncs the repo and bundle, makes it
   `logs/policies/current` on the Jetson and (re)installs the `go2_launcher`
   service, which runs on every boot and listens to the remote. sudo asks for
   the Jetson's password (default `123`):

   ```bash
   uv run --frozen python scripts/deploy_go2_remote.py --bundle logs/policies/trot_v1 --host unitree@192.168.123.18 --install
   ```

   It ends with `Installed trot_v1: unplug, then hold L1 + R1 to start it.`
   It doesn't start the policy.

2. **Unplug the cable.**
3. **Start:** robot lying down, area clear, **hold `L1 + R1` for 1 s**. The
   launcher starts `go2_deploy` with `logs/policies/current` in the tmux
   session `go2_deploy`. About 20 s later the robot stands up: that's the sign
   it worked. Then run it from the remote (step 5).
4. **Stop:** **hold `L2 + R2` for 1 s.** It runs the same e-stop as `X` (damp,
   lie down, stand back up) and exits, leaving the robot standing under
   Unitree's controller. `L1 + R1` starts a fresh run.

To switch policy: stop with `L2 + R2`, plug the cable back in, and `--install`
the new bundle:

```bash
uv run --frozen python scripts/deploy_go2_remote.py --bundle logs/policies/trot_v2 --host unitree@192.168.123.18 --install
```

Checking on it (with the cable plugged back in):

```bash
ssh -t unitree@192.168.123.18 tmux attach -t go2_deploy        # the run's output and keyboard controls (Ctrl+b d to detach)
ssh unitree@192.168.123.18 tail -n 20 BigRedGym/logs/deploy/launcher.log   # every L1 + R1 / L2 + R2 press
ssh -t unitree@192.168.123.18 journalctl -u go2_launcher -f   # the launcher service itself
```

## 5. Controls

| | Remote | Keyboard (wired, in tmux) |
|---|---|---|
| Run the policy (`CUSTOM_CTRL`) | `A` | `c` |
| Hold current pose (`INTERMEDIATE`) | `B` | `q` |
| **E-stop** (damp, lie down, stand up) | **`X`** | **Enter** |
| kp ±10% / kd ±10% | `Up`/`Down` / `Right`/`Left` | `i`/`k` / `l`/`j` |
| Toggle ABXY (to use Unitree's own combos) | `F1` | |
| Start / stop the script (hold 1 s, after `--install`) | `L1 + R1` / `L2 + R2` | |

The script starts in `RECOVERY` (Unitree's controller); an e-stop returns there
by itself. To end a wired run: return to `RECOVERY`, lie the robot down, then
`Ctrl+C` in tmux. To end a wireless run: `L2 + R2`.

At startup the script prints `Loaded policy bundle logs/policies/trot_v1` and
`Using custom implementation Go2TrotBase` (your class).

## 6. Get the logs

Over the cable, after either kind of run:

```bash
uv run --frozen python scripts/deploy_go2_remote.py --host unitree@192.168.123.18 --pull_logs
ls logs/deploy/
```

Each run in `logs/deploy/<timestamp>/` has the CSVs, `meta.json` (task, bundle
metadata), `terminal.log` and `exit.json` (why it ended). To replay a run's
observations through the actor on the PC:

```bash
uv run --frozen python go2_deploy/utility/replay_actor.py 2026-11-22_18-40-12 --task go2trot --run-name Nov22_18-21-49_trot_v1 --compare
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| rsync/ssh hangs with no output | The PC can't reach the robot. Check the cable and that `ip -br addr` shows `192.168.123.99` on it, then `ping -c 2 192.168.123.18`. |
| `uv` on the Jetson fails fetching `setuptools`/pypi | The Jetson has no internet. Share the PC's over the cable: [RUNNING.md](RUNNING.md#ssh-into-the-go2-edu-jetson-factory-defaults), with `wlan0` = your Wi-Fi interface and `<LAN>` = your cable interface (`ip -br addr`). Not persistent across reboots. |
| Wired deploy just shows an old shell prompt | The previous tmux session is still open. `ssh unitree@192.168.123.18 tmux kill-session -t go2_deploy`, then rerun. |
| `L1 + R1` does nothing | Check `launcher.log` (above): it ignores the press if a run is already going or there's no bundle at `logs/policies/current`. Was `--install` run? |
| `No CustomConfig subclass with task_name ...` / `Multiple custom implementations ...` | Exactly one file in `custom_implementation/` must set that `task_name`. |
| `actor takes N obs, but ... obs_vector gives M` | `obs_vector` or `custom_obs_sizes` doesn't match the run's `actor.obs`. |
| Robot shakes or stands oddly in `CUSTOM_CTRL` | Press `X`. Check `obs_vector` order, `DeployScaling` and `default_dof_pos` against the run's snapshot. Order and pose are not checked automatically. |
