"""Manual playback commands remain selected until the next input event."""

from types import SimpleNamespace
import pytest
import torch
from gym.utils.interfaces.teleop_bindings import TeleopCommands
from gym.utils.interfaces.MujocoKeyboardInterface import MujocoKeyboardInterface


class FakeEnv:
    """Minimal env surface the interfaces touch."""

    def __init__(self, num_envs=2):
        self.commands = torch.zeros(num_envs, 3)
        self.timed_out = torch.zeros(num_envs, dtype=torch.bool)
        self.max_episode_length_s = 10.0
        self.cfg = type("cfg", (), {})()
        self._backend = SimpleNamespace(_viewer_key_callback=None)
        self.exit = False
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


@pytest.mark.parametrize("keycode", [256, 27])
def test_mujoco_escape_requests_exit_without_changing_commands(keycode):
    env = FakeEnv()
    ui = MujocoKeyboardInterface(env)
    before = env.commands.clone()
    ui._on_key(keycode)
    assert env.exit
    torch.testing.assert_close(env.commands, before)


@pytest.mark.parametrize(
    "period,expected",
    [(3.0, 9999.0), (51.0, 9999.0), (9999.0, 9999.0), (20000.0, 20000.0)],
)
def test_manual_controls_keep_command_sampling_outside_the_playback_horizon(
    period, expected
):
    env = FakeEnv()
    env.max_episode_length_s = 50.0
    env.cfg.commands = SimpleNamespace(resampling_time=period)
    TeleopCommands(env)
    assert env.cfg.commands.resampling_time == expected
    assert env.cfg.commands.resampling_time > 10 * env.max_episode_length_s


def test_reset_keeps_each_manual_command_when_the_environment_resamples_it():
    env = FakeEnv()
    controls = TeleopCommands(env)
    selected = torch.tensor([[1.5, 0.2, -0.4], [-0.5, -0.2, 0.4]])
    env.commands.copy_(selected)
    original_tensor = env.commands

    def reset_with_new_commands():
        env.reset_count += 1
        env.commands.fill_(3.0)

    env.reset = reset_with_new_commands
    controls.apply("reset")
    assert env.reset_count == 1
    assert env.timed_out.all()
    assert env.commands is original_tensor
    torch.testing.assert_close(env.commands, selected)
