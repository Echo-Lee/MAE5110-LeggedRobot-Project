"""Playback can opt out of optional visual donors without changing physics."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2osc_config import Go2OscPushGlideCfg, Go2OscPushGlideRunnerCfg
from scripts import play


def playback_args(basic_visuals=False):
    argv = [
        "--task=go2osc_push_glide",
        "--original_cfg",
        "--experiment_name=go2osc_push_glide",
        "--load_run=saved_run",
        "--checkpoint=550",
        "--num_envs=1",
        "--seed=42",
    ]
    if basic_visuals:
        argv.append("--basic-visuals")
    return play.get_play_args(argv)


def stub_saved_config(monkeypatch, cfg, make_env):
    train_cfg = Go2OscPushGlideRunnerCfg()
    runner = SimpleNamespace(switch_to_eval=lambda: None)

    def get_cfgs(task_name, **kwargs):
        assert task_name == "go2osc_push_glide"
        assert kwargs == {
            "original_cfg": True,
            "experiment_name": "go2osc_push_glide",
            "load_run": "saved_run",
        }
        return cfg, train_cfg

    monkeypatch.setattr(play.task_registry, "get_cfgs", get_cfgs)
    monkeypatch.setattr(play.task_registry, "make_env", make_env)
    monkeypatch.setattr(
        play.task_registry, "make_alg_runner", lambda env, train_cfg: runner
    )
    return runner


def test_basic_visuals_is_explicit_and_mujoco_only(capsys):
    assert not play.get_play_args(["--task=go2osc_push_glide"]).basic_visuals
    assert playback_args(basic_visuals=True).basic_visuals
    with pytest.raises(SystemExit) as exc:
        play.get_play_args(
            ["--task=go2osc_push_glide", "--backend=vsim", "--basic-visuals"]
        )
    assert exc.value.code == 2
    assert "--basic-visuals requires --backend mujoco" in capsys.readouterr().err


@pytest.mark.parametrize("basic_visuals", [False, True])
def test_saved_visual_donor_overridden_only_when_requested(
    monkeypatch, tmp_path, basic_visuals
):
    cfg = Go2OscPushGlideCfg()
    donor_path = str(tmp_path / "missing_go2.xml")
    cfg.asset.mujoco_visual_model = donor_path
    received = []
    env = object()

    def make_env(task_name, env_cfg, **kwargs):
        assert task_name == "go2osc_push_glide"
        assert kwargs == {"device": "cpu", "headless": False, "backend": "mujoco"}
        received.append(env_cfg.asset.mujoco_visual_model)
        return env

    runner = stub_saved_config(monkeypatch, cfg, make_env)
    assert play.setup(playback_args(basic_visuals)) == (env, runner)
    assert received == [None if basic_visuals else donor_path]


def test_basic_visuals_compiles_nonheadless_go2_and_preserves_physics(
    monkeypatch, tmp_path
):
    cfg = Go2OscPushGlideCfg()
    cfg.asset.mujoco_visual_model = str(tmp_path / "missing_go2.xml")
    reference_cfg = deepcopy(cfg)
    cpu_backend = MuJocoCPUBackend()

    def make_env(task_name, env_cfg, *, device, headless, backend):
        assert task_name == "go2osc_push_glide" and backend == "mujoco"
        assert not headless
        cpu_backend.setup(
            env_cfg,
            num_envs=1,
            device=device,
            task=SimpleNamespace(headless=False),
        )
        return cpu_backend

    stub_saved_config(monkeypatch, cfg, make_env)
    backend, _ = play.setup(playback_args(basic_visuals=True))
    assert backend._viewer is None  # setup compiles the visible model, no window.
    model = backend._mjm
    robot_collision = (model.geom_bodyid != 0) & (
        (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    )
    assert robot_collision.any()
    assert np.all(model.geom_group[robot_collision] == 0)

    # The existing headless path skips the same missing visual donor. Compare
    # native physics and a short controlled rollout with the visible basic model.
    play.task_registry.convert_frequencies_to_params(
        reference_cfg, Go2OscPushGlideRunnerCfg()
    )
    reference = MuJocoCPUBackend()
    reference.setup(
        reference_cfg, num_envs=1, device="cpu", task=SimpleNamespace(headless=True)
    )
    assert reference._viewer is None
    assert backend.dof_names == reference.dof_names
    assert backend.body_names == reference.body_names
    for field in (
        "body_mass",
        "body_inertia",
        "jnt_type",
        "jnt_range",
        "dof_damping",
        "geom_type",
        "geom_size",
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "geom_solref",
    ):
        np.testing.assert_array_equal(
            getattr(model, field), getattr(reference._mjm, field)
        )

    for instance in (backend, reference):
        instance.root_states[:, 2] = 0.34
        instance.dof_pos[:] = torch.tensor([0.0, 0.75, -1.5] * 4)
        instance.reset_state(torch.ones(1, dtype=torch.bool))
        for _ in range(10):
            instance.step(torch.zeros_like(instance.dof_pos))
    torch.testing.assert_close(backend.root_states, reference.root_states)
    torch.testing.assert_close(backend.dof_state, reference.dof_state)
    torch.testing.assert_close(backend.contact_forces, reference.contact_forces)
