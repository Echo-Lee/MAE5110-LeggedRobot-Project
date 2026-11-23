Updated 11/22/2026

# Setup BRG + Unitree SDK + CycloneDDS on PC/host for the first time

To deploy a policy on the robot from your computer, you need to install the additional libraries ```cyclonedds``` and ```unitree-sdk2py```. To do this, run the following in the terminal:

```bash
uv run --frozen scripts/fetch_unitree_sdk.py
```

Then run this:

```bash
mkdir -p "$HOME/.local/src"
git clone --depth 1 --branch 0.10.2 \
    https://github.com/eclipse-cyclonedds/cyclonedds.git \
    "$HOME/.local/src/cyclonedds-0.10.2"
export CYCLONEDDS_HOME="$HOME/.local/opt/cyclonedds-0.10.2"
cmake -S "$HOME/.local/src/cyclonedds-0.10.2" \
    -B "$HOME/.local/src/cyclonedds-0.10.2/build" \
    -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build "$HOME/.local/src/cyclonedds-0.10.2/build" \
    --config RelWithDebInfo --target install --parallel 4
```

Then run this:

```bash
uv sync --frozen --extra unitree_sdk
```

The output should be something along the lines of:

```
Built unitree-sdk2py @ <location>
Built cyclonedds==0.10.2
...
```

Now, we need to setup the ethernet connection.

- Connect an ethernet cable to the robot and to your device.
- Go to Settings -> Network -> click the settings icon under Wired -> IPv4
- Switch IPv4 Method to Manual
- Under Addresses, in the first row, type `192.168.123.99` under Addresses, and `255.255.255.0` under Netmask
- Click Apply
- Turn on the connection under Wired

# Connecting to the Go2 for the first time

SSH instructions:

- Cable: plug the PC into the robot's ethernet port and give it a static `192.168.123.99/24` address (see README_DEPLOY.md).
- `ssh unitree@192.168.123.18`. The default password is `123`, the same password works for `sudo`, and `unitree` is a sudoer. `192.168.123.161` is the motion MCU, which has no shell.
- Optionally run `ssh-copy-id unitree@192.168.123.18` so later logins don't ask for a password.
- **The Jetson has no internet by default**, and `uv`, the wheels and the CycloneDDS clone all need it. Share the PC's connection over the cable (only the PC authenticates to eduroam):
  ```bash
  # Run this on the host PC (WAN=wlan0, LAN=the ethernet NIC to the robot)
  sudo sysctl -w net.ipv4.ip_forward=1
  sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
  sudo iptables -A FORWARD -i <LAN> -o wlan0 -j ACCEPT
  sudo iptables -A FORWARD -i wlan0 -o <LAN> -m state --state RELATED,ESTABLISHED -j ACCEPT
  # Run this on Jetson (via ssh)
  sudo ip route replace default via 192.168.123.99
  echo "nameserver 1.1.1.1" | sudo tee /etc/resolv.conf   # temporary
  ping -c1 pypi.org
  ```
  None of this survives a reboot on either side.
- Don't `apt upgrade` the Jetson: it can break JetPack and Unitree's services. Use `apt install` for specific packages (`tmux`, `cmake`, `build-essential`) and let `uv` handle Python.

Install `uv` and `tmux`, `uv python install 3.11`
   (JetPack ships 3.8), build CycloneDDS 0.10.2 as in Setup above. The first
   deploy syncs the repo; `uv run` then installs the locked aarch64 wheels.

# Export a policy on PC, then run it wirelessly

1. **Export a policy bundle on the PC:**
   With the implementation of the `CustomConfig` interface in `go2_deploy/custom_implementation/`, and the log directory from training in `logs/`, `uv run --frozen python scripts/export_go2_policy.py --task go2trot --name trot_v1 [--run_name <run>]`
   writes `logs/policies/trot_v1/` with `policy.pt` and `deploy_meta.json`.

2. **Over the cable, install the bundle:**
   `uv run --frozen python scripts/deploy_go2_remote.py --bundle logs/policies/trot_v1 --host unitree@192.168.123.18 --install`

3. Unplug. With the robot lying down in a clear area, **hold L1 + R1 for 1 s**.
   The launcher (`go2_deploy/launcher.py`) only listens to the remote, and
   starts `go2_deploy.deploy` in the tmux session `go2_deploy`. About 20 s
   later the robot stands up: that's the sign it worked.

4. **Hold L2 + R2 for 1 s to stop.**

5. **Get logs:** Plug in the ethernet cable to the Go2, then `uv run --frozen python scripts/deploy_go2_remote.py --host unitree@192.168.123.18 --pull_logs`.
   Each run's `meta.json` includes the full bundle metadata. Each run directory
   also has `terminal.log` (everything the script printed, timestamped per
   line) and `exit.json` (why and when it ended: L2 + R2 or `tmux kill-session`,
   Ctrl+C, or a crash with its traceback, plus the state and uptime). No
   `exit.json` means it was SIGKILLed or lost power.
   `logs/deploy/launcher.log` lists every L1 + R1 / L2 + R2 press and launcher
   boot.
