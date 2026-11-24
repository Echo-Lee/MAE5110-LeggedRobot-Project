"""Playback exits when its window closes and always releases the viewer."""

from types import SimpleNamespace

import pytest

from scripts import play


@pytest.mark.parametrize("reason", ["escape", "backend_escape", "window_close"])
def test_playback_stops_after_quit_or_window_close(monkeypatch, reason):
    viewer = SimpleNamespace(running=True)
    viewer.is_running = lambda: viewer.running
    backend = SimpleNamespace(_viewer=viewer, _viewer_exit_requested=False)
    env = SimpleNamespace(exit=False, _backend=backend, max_episode_length=100)
    calls = []
    env.check_exit = lambda: None

    def step(*args):
        calls.append(1)
        if reason == "escape":
            env.exit = True
        elif reason == "backend_escape":
            backend._viewer_exit_requested = True
        else:
            viewer.running = False

    monkeypatch.setattr(play, "policy_step", step)
    play.play(env, None, lambda step: None)
    assert len(calls) == 1


def test_already_closed_viewer_does_not_take_another_step(monkeypatch):
    backend = SimpleNamespace(_viewer=SimpleNamespace(is_running=lambda: False))
    env = SimpleNamespace(exit=False, _backend=backend, max_episode_length=100)
    monkeypatch.setattr(
        play, "policy_step", lambda *args: pytest.fail("stepped a closed viewer")
    )
    play.play(env, None, lambda step: None)


@pytest.mark.parametrize("interrupted", [False, True])
def test_main_closes_backend_on_return_or_keyboard_interrupt(monkeypatch, interrupted):
    closed = []
    backend = SimpleNamespace(close=lambda: closed.append(True))
    env = SimpleNamespace(_backend=backend)
    monkeypatch.setattr(play, "setup", lambda args: (env, None))

    def run(*args):
        if interrupted:
            raise KeyboardInterrupt

    monkeypatch.setattr(play, "play", run)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            play.main(["--task", "go2", "--headless"])
    else:
        play.main(["--task", "go2", "--headless"])
    assert closed == [True]
