# Running BigRedGym

## Setup (PC)
```bash
uv sync --frozen                 # sim + training deps (Python 3.11 is managed by uv)
uv run --frozen pytest           # default unit-test gate (no SDK, no GPU)
```

## Go2 hardware extras (PC or Jetson)
```bash
uv run --frozen scripts/fetch_unitree_sdk.py      # pinned SDK -> thirdparty/
# build CycloneDDS 0.10.2 into ~/.local/opt/cyclonedds-0.10.2 (see README_DEPLOY.md)
export CYCLONEDDS_HOME="$HOME/.local/opt/cyclonedds-0.10.2"
uv sync --frozen --extra unitree_sdk
uv run --frozen --extra unitree_sdk pytest -m unitree   # SDK tests
```
On the Jetson, add `--no-dev` to `uv sync`/`uv run`. The dev group pulls in
`mujoco-warp` → `warp-lang`, which has no wheel for the Jetson (aarch64, glibc 2.31).
pytest is in that group too, so skip the SDK tests there.

## Deploying to the Go2
Step by step, wired and wireless: [DEPLOY.md](DEPLOY.md).
- **Tethered (policy on PC):** `uv run --extra unitree_sdk python -m go2_deploy.deploy <nic> --task go2trot`
  (the actor is built from the latest run in `logs/`).
- **Onboard (policy on the Jetson, sent over the cable):**
  ```bash
  uv run --frozen python scripts/export_go2_policy.py --task go2trot --name trot_v1 [--run_name <run>]
  # wired: sync and run now, attached in tmux
  uv run --frozen python scripts/deploy_go2_remote.py --bundle logs/policies/trot_v1 --host unitree@192.168.123.18
  # wireless: install, unplug, then hold L1 + R1 on the remote to start (L2 + R2 stops)
  uv run --frozen python scripts/deploy_go2_remote.py --bundle logs/policies/trot_v1 --host unitree@192.168.123.18 --install
  uv run --frozen python scripts/deploy_go2_remote.py --host unitree@192.168.123.18 --pull_logs
  ```
  The full procedure, RC controls and known risks are in `README_DEPLOY.md`.

## SSH into the Go2 EDU Jetson (factory defaults)
- Cable: plug the PC into the robot's ethernet port and give it a static `192.168.123.99/24` address (see README_DEPLOY.md).
- `ssh unitree@192.168.123.18`. The default password is `123`, the same password works for `sudo`, and `unitree` is a sudoer. These are Unitree's published defaults; check them on your unit. `192.168.123.161` is the motion MCU, which has no shell.
- Optionally run `ssh-copy-id unitree@192.168.123.18` so later logins don't ask for a password.
- **The Jetson has no internet by default**, and `uv`, the wheels and the CycloneDDS clone all need it. Share the PC's connection over the cable (only the PC authenticates to eduroam):
  ```bash
  # PC (WAN=wlan0, LAN=the ethernet NIC to the robot)
  sudo sysctl -w net.ipv4.ip_forward=1
  sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
  sudo iptables -A FORWARD -i <LAN> -o wlan0 -j ACCEPT
  sudo iptables -A FORWARD -i wlan0 -o <LAN> -m state --state RELATED,ESTABLISHED -j ACCEPT
  # Jetson
  sudo ip route replace default via 192.168.123.99
  echo "nameserver 1.1.1.1" | sudo tee /etc/resolv.conf   # temporary
  ping -c1 pypi.org
  ```
  None of this survives a reboot on either side.
- Don't `apt upgrade` the Jetson: it can break JetPack and Unitree's services. Use `apt install` for specific packages (`tmux`, `cmake`, `build-essential`) and let `uv` handle Python.

## Gotchas
- `CYCLONEDDS_HOME` must be exported for `uv sync`/`uv run` with the `unitree_sdk` extra.
- `Actor.export()` also writes ONNX and fails without `onnxscript`; the bundle
  exporter writes TorchScript only, which is all the robot needs.
- eduroam can't link PC and robot: send the bundle over the cable, and run untethered with `--install` + `L1 + R1`.
- The onboard run lives in tmux session `go2_deploy`; an SSH drop doesn't stop it.
  The RC's X is the e-stop, not the PC.
- Bundles (`logs/policies/`) and `*.pt` are gitignored; move them with the deploy script.

## Go2 push/glide training and playback

`go2osc_push_glide_deploy` trains a push/glide policy and friction/body-velocity
estimator together from scratch. The actor has 58 inputs and 12 joint-target
outputs; the training critic has 75 inputs. Ground friction spans `[0.05, 1.0]`
and forward commands span `[-1.0, 2.0]` m/s, including standing and turning.
Control runs at 100 Hz and physics at 500 Hz.
This task uses MuJoCo CPU or MuJoCo Warp (`--backend mujoco`).

For Linux/NVIDIA training, verify `nvidia-smi`, then:

```sh
uv sync --frozen --extra gpu
PYTHONPATH=. uv run --frozen --extra gpu python scripts/train.py \
    --task go2osc_push_glide_deploy --backend mujoco --device cuda:0 \
    --num_envs 4096 --max_iterations 4000 --save_interval 50 --seed 0 \
    --experiment_name go2osc_push_glide_deploy --headless --disable_wandb
```

For a small CPU check, omit `--extra gpu` and use
`--device cpu --num_envs 16 --max_iterations 2`.
No initial policy or estimator checkpoint is required. The joint-target clamp
closes over iterations 1000–1500; learned estimates replace simulator stand-ins
over iterations 1500–2000.

Checkpoints and source/configuration snapshots are saved below
`logs/go2osc_push_glide_deploy/<run>/`. Replace `RUN` with a run directory:

```sh
PYTHONPATH=. uv run --frozen mjpython scripts/play.py \
    --task go2osc_push_glide_deploy --backend mujoco --device cpu --num_envs 1 \
    --experiment_name go2osc_push_glide_deploy --load_run RUN --checkpoint 4000 \
    --friction 0.05 --basic-visuals --seed 123 --keyboard
```

On Linux replace `mjpython` with `python`. I/K adjust forward velocity, J/L
lateral velocity and N/M yaw rate. R resets while retaining the command;
Esc exits. For a fixed command, replace `--keyboard` with
`--command 1.0 0.0 0.0`.

Playback uses the current task configuration. Archived snapshots require their
matching release code when their configuration imports modules absent from the
current checkout. Use `--original_cfg` only for compatible saved configurations.
