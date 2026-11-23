import csv
import json
import os
import queue
import time
from datetime import datetime
from operator import attrgetter

from go2_deploy.utility.thread import RecurrentThread

# Buttons parsed by UnitreeRemoteController, logged as one column each
REMOTE_BUTTONS = [
    "L1",
    "L2",
    "R1",
    "R2",
    "A",
    "B",
    "X",
    "Y",
    "Up",
    "Down",
    "Left",
    "Right",
    "Select",
    "Start",
    "F1",
    "F3",
]

# Rows are dropped rather than blocking a real-time thread. 20k rows is ~30 s of
# lowstate at 500 Hz, far more headroom than the writer thread should ever need.
QUEUE_MAXSIZE = 20000

FLUSH_INTERVAL = 1.0

# Array sizes come from the Unitree Go2 DDS IDL (unitree_go.msg.dds_).
# LowState_.motor_state has 20 slots but the Go2 only populates the first 12
# (3 per leg). Lower this to 12 to drop the always-zero slots and shrink the file.
LOWSTATE_MOTOR_SLOTS = 20
BMS_CELLS = 15
PATH_POINTS = 10

# Scalar fields of each nested DDS struct. The same tuple builds both the CSV
# header and the row, so the two cannot drift apart.
MOTOR_FIELDS = (
    "mode",
    "q",
    "dq",
    "ddq",
    "tau_est",
    "q_raw",
    "dq_raw",
    "ddq_raw",
    "temperature",
    "lost",
)
BMS_FIELDS = (
    "version_high",
    "version_low",
    "status",
    "soc",
    "current",
    "cycle",
)
PATH_POINT_FIELDS = ("t_from_start", "x", "y", "yaw", "vx", "vy", "vyaw")

_get_motor = attrgetter(*MOTOR_FIELDS)
_get_bms = attrgetter(*BMS_FIELDS)
_get_path_point = attrgetter(*PATH_POINT_FIELDS)


def _numbered(prefix, count):
    return [f"{prefix}_{i}" for i in range(count)]


# Each *_names / *_values pair below must list columns in the same order.


def _imu_names(prefix):
    return (
        [f"{prefix}_quat_{c}" for c in "wxyz"]
        + [f"{prefix}_gyro_{c}" for c in "xyz"]
        + [f"{prefix}_accel_{c}" for c in "xyz"]
        + [f"{prefix}_rpy_{c}" for c in "rpy"]
        + [f"{prefix}_temperature"]
    )


def _imu_values(imu):
    return [
        *imu.quaternion,
        *imu.gyroscope,
        *imu.accelerometer,
        *imu.rpy,
        imu.temperature,
    ]


def _motor_names(i):
    return [f"motor{i}_{f}" for f in MOTOR_FIELDS] + _numbered(f"motor{i}_reserve", 2)


def _motor_values(motor):
    return [*_get_motor(motor), *motor.reserve]


def _bms_names():
    return (
        [f"bms_{f}" for f in BMS_FIELDS]
        + _numbered("bms_bq_ntc", 2)
        + _numbered("bms_mcu_ntc", 2)
        + _numbered("bms_cell_vol", BMS_CELLS)
    )


def _bms_values(bms):
    return [*_get_bms(bms), *bms.bq_ntc, *bms.mcu_ntc, *bms.cell_vol[:BMS_CELLS]]


def _path_point_names():
    names = []
    for i in range(PATH_POINTS):
        names += [f"path_point{i}_{f}" for f in PATH_POINT_FIELDS]
    return names


def _path_point_values(path_points):
    values = []
    for p in path_points[:PATH_POINTS]:
        values += _get_path_point(p)
    return values


def new_run_dir(log_root="logs/deploy"):
    """Create and return logs/deploy/<YYYY-mm-dd_HH-MM-SS>."""
    run_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class _Stream:
    """One open CSV file plus its writer."""

    def __init__(self, path, header):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
        self.width = len(header)


class CSVLogger:
    """Streams Unitree SDK messages, observations, and policy actions to CSV.

    Producers (the DDS callbacks and the control loop) only extract plain floats
    and hand a row to a queue; a single background thread does all file I/O.

    lowstate.csv and sportmodestate.csv carry every field of the Go2's
    LowState_ and SportModeState_ DDS messages (IMU, all motor slots, battery
    management system, foot forces, remote controller, path points, ...).
    """

    def __init__(self, main_controller, log_root="logs/deploy", run_dir=None):
        self.main_controller = main_controller
        cfg = main_controller.cfg

        start_wall = datetime.now()
        self.start_monotonic = time.monotonic()
        # deploy.py creates the run dir up front so terminal.log covers startup
        self.run_dir = run_dir if run_dir is not None else new_run_dir(log_root)

        # Observation columns follow cfg.obs_vector so the header tracks any
        # config change instead of going stale.
        self.obs_columns = []
        for obs in cfg.obs_vector:
            self.obs_columns += _numbered("obs_" + obs, cfg.obs_sizes[obs])

        self.streams = {
            "lowstate": _Stream(
                os.path.join(self.run_dir, "lowstate.csv"), self._lowstate_header()
            ),
            "sportmodestate": _Stream(
                os.path.join(self.run_dir, "sportmodestate.csv"),
                self._sportmodestate_header(),
            ),
            "control": _Stream(
                os.path.join(self.run_dir, "control.csv"), self._control_header()
            ),
        }

        # Monotonic timestamps are only meaningful next to their wall-clock origin
        with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "start_wall_clock": start_wall.isoformat(),
                    "start_monotonic": self.start_monotonic,
                    "task_name": cfg.task_name,
                    "ctrl_freq": cfg.ctrl_freq,
                    "kp": cfg.kp,
                    "kd": cfg.kd,
                    "exp_moving_avg": cfg.exp_moving_avg,
                    "ema_smoothing_factor": cfg.ema_smoothing_factor,
                    "obs_vector": list(cfg.obs_vector),
                    # Full policy bundle metadata (scaling, hash, provenance),
                    # or None when the actor was built from training logs.
                    "policy_bundle": cfg.bundle_meta,
                },
                f,
                indent=2,
            )

        self.queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.dropped_rows = 0
        self.malformed_rows = 0
        self._warned_streams = set()
        self._last_flush_time = time.monotonic()
        self._closed = False

        self.writer_thread = RecurrentThread(interval=0.0, target=self._writer_loop)
        self.writer_thread.Start()

    # Headers

    def _lowstate_header(self):
        header = (
            ["t", "state", "tick", "crc"]
            + _numbered("head", 2)
            + ["level_flag", "frame_reserve"]
            + _numbered("sn", 2)
            + _numbered("version", 2)
            + ["bandwidth"]
            + _imu_names("imu")
        )
        for i in range(LOWSTATE_MOTOR_SLOTS):
            header += _motor_names(i)
        header += _numbered("foot_force", 4)
        header += _numbered("foot_force_est", 4)
        header += _bms_names()
        header += [
            "bit_flag",
            "adc_reel",
            "temperature_ntc1",
            "temperature_ntc2",
            "power_v",
            "power_a",
        ]
        header += _numbered("fan_frequency", 4)
        # Raw 40-byte wireless_remote payload, hex encoded, so anything the
        # parsed columns below do not capture can still be recovered offline.
        header += ["reserve", "wireless_remote_hex"]
        header += ["Lx", "Ly", "Rx", "Ry", "lin_vel_x", "lin_vel_y", "yaw_vel"]
        header += REMOTE_BUTTONS
        return header

    def _sportmodestate_header(self):
        return (
            [
                "t",
                "state",
                "stamp_sec",
                "stamp_nanosec",
                "error_code",
            ]
            + _imu_names("imu")
            + [
                "mode",
                "progress",
                "gait_type",
                "foot_raise_height",
                "position_x",
                "position_y",
                "position_z",
                "body_height",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "yaw_speed",
            ]
            + _numbered("range_obstacle", 4)
            + _numbered("foot_force", 4)
            + _numbered("foot_position_body", 12)
            + _numbered("foot_speed_body", 12)
            + _path_point_names()
        )

    def _control_header(self):
        # action_* is the raw actor output for this step; smoothed_action_* is
        # last_action[0] after the exponential moving average, i.e. what
        # actually became the lowcmd. The two are equal when
        # cfg.exp_moving_avg is off.
        return (
            ["t", "state", "phase"]
            + self.obs_columns
            + _numbered("action", 12)
            + _numbered("smoothed_action", 12)
            + _numbered("lowcmd_q", 12)
            + ["kp", "kd", "kp_mult", "kd_mult"]
        )

    # Producers -- called from the DDS callbacks and the control loop

    def log_lowstate(self, t, msg, remote_controller):
        row = [
            t,
            self.main_controller._state.name,
            msg.tick,
            msg.crc,
            *msg.head,
            msg.level_flag,
            msg.frame_reserve,
            *msg.sn,
            *msg.version,
            msg.bandwidth,
            *_imu_values(msg.imu_state),
        ]
        for i in range(LOWSTATE_MOTOR_SLOTS):
            row += _motor_values(msg.motor_state[i])
        row += [
            *msg.foot_force,
            *msg.foot_force_est,
            *_bms_values(msg.bms_state),
            msg.bit_flag,
            msg.adc_reel,
            msg.temperature_ntc1,
            msg.temperature_ntc2,
            msg.power_v,
            msg.power_a,
            *msg.fan_frequency,
            msg.reserve,
            bytes(msg.wireless_remote).hex(),
        ]
        row += [
            remote_controller.Lx,
            remote_controller.Ly,
            remote_controller.Rx,
            remote_controller.Ry,
            remote_controller.lin_vel_x,
            remote_controller.lin_vel_y,
            remote_controller.yaw_vel,
        ]
        row += [getattr(remote_controller, button) for button in REMOTE_BUTTONS]
        self._enqueue("lowstate", row)

    def log_sportmodestate(self, t, msg):
        row = [
            t,
            self.main_controller._state.name,
            msg.stamp.sec,
            msg.stamp.nanosec,
            msg.error_code,
            *_imu_values(msg.imu_state),
            msg.mode,
            msg.progress,
            msg.gait_type,
            msg.foot_raise_height,
            *msg.position,
            msg.body_height,
            *msg.velocity,
            msg.yaw_speed,
            *msg.range_obstacle,
            *msg.foot_force,
            *msg.foot_position_body,
            *msg.foot_speed_body,
            *_path_point_values(msg.path_point),
        ]
        self._enqueue("sportmodestate", row)

    def log_control(self, t, obs, action, smoothed_action, lowcmd):
        """Log one control step.

        `action` is the raw actor output; `smoothed_action` is last_action[0]
        after the exponential moving average over the 2x12 action buffer, which
        is the vector the lowcmd was built from.
        """
        cfg = self.main_controller.cfg
        row = [
            t,
            self.main_controller._state.name,
            getattr(self.main_controller, "phase", ""),
        ]
        row += obs.tolist() if obs is not None else [""] * len(self.obs_columns)
        # _act() returns None on the emergency path; leave those columns blank
        row += action.tolist() if action is not None else [""] * 12
        row += smoothed_action.tolist() if smoothed_action is not None else [""] * 12
        row += [lowcmd.motor_cmd[i].q for i in range(12)]
        row += [
            cfg.kp,
            cfg.kd,
            self.main_controller.kp_mult,
            self.main_controller.kd_mult,
        ]
        self._enqueue("control", row)

    def _enqueue(self, stream_name, row):
        if self._closed:
            return
        # A row whose width disagrees with its header would silently shift every
        # later column, so drop it and say so once per stream.
        if len(row) != self.streams[stream_name].width:
            self.malformed_rows += 1
            if stream_name not in self._warned_streams:
                self._warned_streams.add(stream_name)
                print(
                    f"[CSVLogger] {stream_name}: row has {len(row)} columns, "
                    f"header has {self.streams[stream_name].width}; dropping"
                )
            return
        try:
            self.queue.put_nowait((stream_name, row))
        except queue.Full:
            self.dropped_rows += 1

    # Consumer

    def _writer_loop(self):
        try:
            stream_name, row = self.queue.get(timeout=0.1)
        except queue.Empty:
            self._flush_if_due()
            return
        self.streams[stream_name].writer.writerow(row)
        self._flush_if_due()

    def _flush_if_due(self):
        t = time.monotonic()
        if t < self._last_flush_time + FLUSH_INTERVAL:
            return
        for stream in self.streams.values():
            stream.file.flush()
        self._last_flush_time = t

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.writer_thread.Wait(timeout=1.0)
        # Drain whatever the writer thread did not get to
        while True:
            try:
                stream_name, row = self.queue.get_nowait()
            except queue.Empty:
                break
            self.streams[stream_name].writer.writerow(row)
        for stream in self.streams.values():
            stream.file.flush()
            stream.file.close()
