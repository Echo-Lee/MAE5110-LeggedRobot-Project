import re
import shutil
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

import gym.envs as current_envs
from gym.utils import original_cfg
from gym.utils.original_cfg import (
    OriginalCfgError,
    load_original_cfgs_from_run,
    original_cfg_source_dir,
    original_task_from_run,
)
from gym.utils.logging_and_saving.local_code_save_helper import (
    save_original_cfg_to_logs,
)
from gym.utils.task_registry import TaskRegistry
from scripts.play import get_play_args
from scripts.train import get_train_args


def _write_saved_config_snapshot(run_dir):
    saved_envs = run_dir / "files" / "gym" / "envs"
    config_dir = saved_envs / "demo"
    config_dir.mkdir(parents=True)

    (saved_envs / "__init__.py").write_text(
        """
class_dict = {"CurrentTask": ".demo.saved_task"}
config_dict = {"SavedEnvCfg": ".demo.saved_config"}
runner_config_dict = {"SavedRunnerCfg": ".demo.saved_config"}
task_dict = {"demo": ["CurrentTask", "SavedEnvCfg", "SavedRunnerCfg"]}

raise RuntimeError("the saved registry module must not be executed")
"""
    )
    (config_dir / "parent_config.py").write_text(
        """
class SavedParentCfg:
    inherited_value = "from saved parent"
"""
    )
    (config_dir / "saved_task.py").write_text(
        'class CurrentTask:\n    source = "saved task class"\n'
    )
    (config_dir / "saved_config.py").write_text(
        """
from gym.envs.demo.parent_config import SavedParentCfg


class SavedEnvCfg(SavedParentCfg):
    source = "saved environment config"


class SavedRunnerCfg:
    class runner:
        experiment_name = "value inside snapshot"
        load_run = -1

    def __init__(self):
        self.runner = self.runner()
"""
    )


def test_saved_configs_load_complete_inheritance_without_leaking_modules(tmp_path):
    run_dir = tmp_path / "run_a"
    _write_saved_config_snapshot(run_dir)
    current_envs_module = sys.modules["gym.envs"]

    env_cfg, train_cfg = load_original_cfgs_from_run("demo", run_dir)

    assert env_cfg.source == "saved environment config"
    assert env_cfg.inherited_value == "from saved parent"
    assert train_cfg.runner.experiment_name == "value inside snapshot"
    assert sys.modules["gym.envs"] is current_envs_module
    assert sys.modules["gym.envs"] is current_envs
    assert "gym.envs.demo.saved_config" not in sys.modules
    assert "gym.envs.demo.parent_config" not in sys.modules


def test_saved_task_context_uses_snapshot_class_and_restores_imports(tmp_path):
    run_dir = tmp_path / "run_a"
    _write_saved_config_snapshot(run_dir)
    before = sys.modules["gym.envs"]
    with original_task_from_run("demo", run_dir) as (task, env_cfg, train_cfg):
        assert task.source == "saved task class"
        assert env_cfg.inherited_value == "from saved parent"
        assert train_cfg.runner.experiment_name == "value inside snapshot"
        assert sys.modules["gym.envs"] is not before
        assert "gym.envs.demo.saved_task" in sys.modules
    assert sys.modules["gym.envs"] is before
    assert "gym.envs.demo.saved_task" not in sys.modules


def test_saved_task_context_reports_missing_class_source(tmp_path):
    run_dir = tmp_path / "run_a"
    _write_saved_config_snapshot(run_dir)
    (run_dir / "files/gym/envs/demo/saved_task.py").unlink()
    with pytest.raises(OriginalCfgError, match="saved_task"):
        with original_task_from_run("demo", run_dir):
            pytest.fail("missing task class source must not load")


def test_registry_resolves_and_pins_the_original_config_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "logs" / "demo_experiment" / "run_a"
    _write_saved_config_snapshot(run_dir)
    monkeypatch.setattr(original_cfg, "GYM_ROOT_DIR", str(tmp_path))

    registry = TaskRegistry()
    current_train_cfg = SimpleNamespace(
        runner=SimpleNamespace(experiment_name="demo_experiment", load_run=-1)
    )
    registry.register(
        "demo",
        object,
        SimpleNamespace(source="current environment config"),
        current_train_cfg,
    )

    env_cfg, train_cfg = registry.get_cfgs("demo", original_cfg=True)

    assert env_cfg.source == "saved environment config"
    assert train_cfg.runner.experiment_name == "demo_experiment"
    assert train_cfg.runner.load_run == "run_a"
    assert train_cfg._original_cfg_source_dir == str(
        run_dir.resolve() / "files" / "gym" / "envs"
    )


def test_carried_original_config_takes_precedence(tmp_path):
    run_dir = tmp_path / "run_a"
    _write_saved_config_snapshot(run_dir)
    current_snapshot = run_dir / "files" / "gym" / "envs"
    carried_snapshot = run_dir / "files" / "original_cfg" / "gym" / "envs"
    carried_snapshot.parent.mkdir(parents=True)
    current_snapshot.rename(carried_snapshot)
    current_snapshot.mkdir(parents=True)

    assert original_cfg_source_dir(run_dir) == carried_snapshot.resolve()
    env_cfg, _ = load_original_cfgs_from_run("demo", run_dir)
    assert env_cfg.source == "saved environment config"


def test_original_config_snapshot_is_carried_without_bytecode(tmp_path):
    source_dir = tmp_path / "source_envs"
    cache_dir = source_dir / "demo" / "__pycache__"
    cache_dir.mkdir(parents=True)
    (source_dir / "__init__.py").write_text("task_dict = {}\n")
    (cache_dir / "saved_config.pyc").write_bytes(b"generated bytecode")

    log_dir = tmp_path / "new_run"
    save_original_cfg_to_logs(log_dir, source_dir)

    carried_dir = log_dir / "files" / "original_cfg" / "gym" / "envs"
    assert (carried_dir / "__init__.py").is_file()
    assert not (carried_dir / "demo" / "__pycache__").exists()


def test_original_config_reports_a_missing_source_snapshot(tmp_path):
    with pytest.raises(OriginalCfgError, match="Saved config manifest not found"):
        load_original_cfgs_from_run("demo", tmp_path / "run_without_files")


def test_train_and_play_accept_original_cfg_flag():
    train_args = get_train_args(["--task", "pendulum", "--original_cfg"])
    play_args = get_play_args(["--task", "pendulum", "--original_cfg"])

    assert train_args.original_cfg is True
    assert play_args.original_cfg is True


def test_run_saved_before_friction_bins_loads_and_constructs(tmp_path):
    """A pre-bins snapshot (--original_cfg) builds and steps the current task."""
    current = Path(current_envs.__file__).parent
    saved_envs = tmp_path / "run_a" / "files" / "gym" / "envs"
    shutil.copytree(current, saved_envs, ignore=shutil.ignore_patterns("__pycache__"))
    old_dr = saved_envs / "base" / "domain_randomization.py"
    text = re.sub(
        r"\n\s*(contact_friction_bins|contact_friction_bin_weights"
        r"|resample_contact_friction) = (None|False)",
        "",
        old_dr.read_text(),
    )
    old_dr.write_text(text)

    env_cfg, train_cfg = load_original_cfgs_from_run(
        "go2osc_push_glide", tmp_path / "run_a"
    )
    startup = env_cfg.domain_randomization.startup
    assert not hasattr(startup, "contact_friction_bins")
    assert not hasattr(
        env_cfg.domain_randomization.episode, "resample_contact_friction"
    )

    env_cfg.env.num_envs = 2
    env_cfg.seed = train_cfg.seed = 0
    train_cfg.runner.device = "cpu"
    from gym.utils.task_registry import task_registry

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = task_registry.make_env(
            "go2osc_push_glide", env_cfg, device="cpu", headless=True
        )
    try:
        env.step()
    finally:
        env._backend.close()
