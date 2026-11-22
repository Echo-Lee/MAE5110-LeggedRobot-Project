"""Playback friction overrides reach native contacts after saved-config loading."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gym.envs.go2.go2osc_config import Go2OscPushGlideCfg, Go2OscPushGlideRunnerCfg
from scripts import play


def playback_args(friction=None):
    argv = [
        "--task=go2osc_push_glide",
        "--original_cfg",
        "--experiment_name=go2osc_push_glide",
        "--load_run=saved_run",
        "--checkpoint=550",
        "--num_envs=2",
        "--seed=42",
        "--headless",
    ]
    if friction is not None:
        argv.append(f"--friction={friction}")
    return play.get_play_args(argv)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.1"])
def test_invalid_friction_is_rejected_before_loading_a_policy(value, capsys):
    with pytest.raises(SystemExit) as exc:
        play.get_play_args(["--task=go2osc_push_glide", f"--friction={value}"])
    assert exc.value.code == 2
    assert "friction" in capsys.readouterr().err.lower()


def test_friction_override_is_optional_and_accepts_zero():
    assert playback_args().friction is None
    assert playback_args(0).friction == 0.0
    assert playback_args(0.4).friction == 0.4


@pytest.mark.parametrize(
    "task", ["go2osc_push_glide", "go2osc_push_glide_robust", "go2osc_push_glide_est"]
)
def test_fixed_friction_video_with_basic_visuals_survives_resets(task):
    args = play.get_play_args(
        [
            "--task",
            task,
            "--friction",
            "0.05",
            "--video",
            "unused.mp4",
            "--basic-visuals",
            "--num_envs",
            "2",
            "--seed",
            "42",
        ]
    )
    cfg, _ = play.play_cfgs(args)
    assert cfg.asset.mujoco_visual_model is None
    assert cfg.asset.render_offscreen
    startup = cfg.domain_randomization.startup
    assert startup.contact_friction_range == [0.05, 0.05]
    assert startup.contact_friction_bins is None
    assert startup.contact_friction_bin_weights is None
    assert not cfg.domain_randomization.episode.resample_contact_friction
    env = play.make_play_env(args, cfg)
    backend = env._backend
    try:
        for _ in range(2):
            env.reset()
            for _ in range(5):
                env.step()
            torch.testing.assert_close(backend.contact_friction, torch.full((2,), 0.05))
            if hasattr(env, "foot_friction_obs"):
                torch.testing.assert_close(
                    env.foot_friction_obs, torch.full_like(env.foot_friction_obs, 0.05)
                )
            for env_id in range(2):
                backend._activate_domain(env_id)
                model = backend._model_for_env(env_id)
                collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
                np.testing.assert_allclose(model.geom_friction[collision, 0], 0.05)
    finally:
        backend.close()


@pytest.mark.parametrize("friction", [None, 0.4])
def test_saved_go2osc_native_contacts_use_playback_friction_through_reset(
    monkeypatch, friction
):
    saved_cfg = Go2OscPushGlideCfg()
    # Existing checkpoint configurations retain their original training range.
    saved_cfg.domain_randomization.startup.contact_friction_range = [0.4, 1.0]
    saved_runner_cfg = Go2OscPushGlideRunnerCfg()
    runner = SimpleNamespace(switch_to_eval=lambda: None)

    def get_cfgs(task, **kwargs):
        assert task == "go2osc_push_glide"
        assert kwargs == {
            "original_cfg": True,
            "experiment_name": "go2osc_push_glide",
            "load_run": "saved_run",
        }
        return deepcopy((saved_cfg, saved_runner_cfg))

    monkeypatch.setattr(play.task_registry, "get_cfgs", get_cfgs)
    monkeypatch.setattr(
        play.task_registry, "make_alg_runner", lambda env, train_cfg: runner
    )
    env, actual_runner = play.setup(playback_args(friction))
    backend = env._backend
    try:
        assert actual_runner is runner
        coefficients = backend.contact_friction.clone()
        if friction is None:
            assert env.cfg.domain_randomization.startup.contact_friction_range == [
                0.4,
                1.0,
            ]
            assert env.cfg.terrain.static_friction == saved_cfg.terrain.static_friction
            assert (
                env.cfg.terrain.dynamic_friction == saved_cfg.terrain.dynamic_friction
            )
            assert torch.all((coefficients >= 0.4) & (coefficients <= 1.0))
            assert coefficients.unique().numel() == 2
        else:
            assert env.cfg.domain_randomization.startup.contact_friction_range == [
                0.4,
                0.4,
            ]
            assert env.cfg.terrain.static_friction == 0.4
            assert env.cfg.terrain.dynamic_friction == 0.4
            torch.testing.assert_close(coefficients, torch.full((2,), 0.4))

        def assert_native_contacts():
            torch.testing.assert_close(backend.contact_friction, coefficients)
            for env_id, coefficient in enumerate(coefficients.tolist()):
                # Serial CPU worlds activate their parameters on a shared model.
                backend._activate_domain(env_id)
                model = backend._model_for_env(env_id)
                collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
                assert np.any(collision & (model.geom_bodyid == 0))
                assert np.any(collision & (model.geom_bodyid != 0))
                np.testing.assert_allclose(
                    model.geom_friction[collision, 0], coefficient
                )
                if model.npair:
                    np.testing.assert_allclose(model.pair_friction[:, :2], coefficient)

        assert_native_contacts()
        env.reset()
        env.step()
        assert_native_contacts()
        assert saved_cfg.domain_randomization.startup.contact_friction_range == [
            0.4,
            1.0,
        ]
    finally:
        backend.close()
