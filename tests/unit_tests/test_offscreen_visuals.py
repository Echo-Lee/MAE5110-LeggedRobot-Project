"""The CPU backend loads the Go2 visual model for off-screen rendering."""

from pathlib import Path

import pytest

from gym import GYM_ROOT_DIR

MESHES = Path(GYM_ROOT_DIR) / "resources/robots/go2/mjcf/go2.xml"


def _model(render_offscreen):
    from gym.envs.go2.go2 import Go2
    from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2Cfg()
    cfg.env.num_envs = 1
    cfg.seed = 0
    cfg.asset.render_offscreen = render_offscreen
    task_registry.convert_frequencies_to_params(cfg, Go2RunnerCfg())
    env = Go2(cfg, "cpu", True, select_backend(cfg, "cpu", "mujoco"))
    model = env._backend._model_for_env(0)
    env._backend.close()
    return model


@pytest.mark.skipif(not MESHES.is_file(), reason="run scripts/fetch_go2_meshes.py")
def test_headless_training_skips_meshes_but_recording_loads_them():
    assert _model(render_offscreen=False).nmesh == 0  # training stays lean
    recorded = _model(render_offscreen=True)
    assert recorded.nmesh > 0
    # Meshes are display-only: physics geoms are unchanged.
    mesh_geoms = recorded.geom_type == 7
    assert (recorded.geom_contype[mesh_geoms] == 0).all()
    assert (recorded.geom_conaffinity[mesh_geoms] == 0).all()
