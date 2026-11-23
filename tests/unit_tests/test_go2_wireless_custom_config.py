"""Wireless deployment selects and dispatches a task's CustomConfig instance."""

import inspect
import json
import math
from types import SimpleNamespace

import pytest
import torch

from go2_deploy.custom_implementation.custom_config import CustomConfig
from go2_deploy.custom_implementation.go2osc_push_glide import Go2OscPushGlide
from go2_deploy.deploy_config import DeployConfig
from go2_deploy.utility import deploy_utility, import_custom_impl
from tests.unit_tests import test_go2osc_onboard as onboard_tests
from tests.unit_tests.test_deploy_optional_sdk import _without_sdk

bundle = onboard_tests.bundle


class _MarkerConfig(CustomConfig):
    task_name = "marker"
    obs_vector = ["marker"]
    custom_obs_sizes = {"marker": 1}
    default_dof_pos = [0.0] * 12

    class DeployScaling:
        marker = 2.0

    def __init__(self, marker=0.0):
        self.marker = marker

    def get_custom_buffers_dict(self):
        return {}

    def update_custom_buffers(self, main_controller, t):
        pass

    def get_custom_obs_pieces(self):
        return {
            "marker": lambda controller, message: torch.tensor([self.marker]),
        }


def _write_implementation(directory, filename, class_name, task_name):
    (directory / filename).write_text(
        "from go2_deploy.custom_implementation.custom_config import CustomConfig\n"
        f"class {class_name}(CustomConfig):\n"
        f"    task_name = {task_name!r}\n"
        "    obs_vector = []\n"
        "    custom_obs_sizes = {}\n"
        "    default_dof_pos = [0.0] * 12\n"
        "    class DeployScaling: pass\n"
        "    def get_custom_buffers_dict(self): return {}\n"
        "    def update_custom_buffers(self, controller, t): pass\n"
        "    def get_custom_obs_pieces(self): return {}\n"
    )


def test_custom_config_rejects_missing_class_attributes():
    with pytest.raises(TypeError, match="obs_vector"):
        type("MissingAttributes", (CustomConfig,), {"task_name": "incomplete"})


def test_custom_config_rejects_a_mistyped_required_attribute():
    with pytest.raises(TypeError, match="default_dof_pos.*list"):
        type("WrongPostureType", (_MarkerConfig,), {"default_dof_pos": (0.0,) * 12})


def test_custom_config_requires_the_instance_callbacks():
    attributes = {
        name: getattr(_MarkerConfig, name) for name in CustomConfig.REQUIRED_ATTRIBUTES
    }
    incomplete = type("MissingCallbacks", (CustomConfig,), attributes)
    with pytest.raises(TypeError, match="abstract"):
        incomplete()


def test_loader_selects_the_push_glide_instance_without_loading_a_bundle():
    implementation = import_custom_impl.load_custom_impl("go2osc_push_glide_deploy")
    assert isinstance(implementation, CustomConfig)
    # Discovery imports modules under their file names; the interface identity
    # and declared task matter rather than a separately imported class object.
    assert type(implementation).__name__ == Go2OscPushGlide.__name__
    assert implementation.task_name == "go2osc_push_glide_deploy"
    assert inspect.ismethod(implementation.get_custom_buffers_dict)
    assert inspect.ismethod(implementation.update_custom_buffers)
    assert inspect.ismethod(implementation.get_custom_obs_pieces)
    cfg = DeployConfig(implementation)
    assert cfg.custom_impl is implementation
    assert cfg.obs_dim() == 58
    assert "governed_commands" in cfg.obs_vector
    torch.testing.assert_close(
        cfg.default_dof_pos, torch.tensor([0.0, 0.66, -1.36] * 4)
    )
    assert (cfg.ctrl_freq, cfg.kp, cfg.kd, cfg.exp_moving_avg) == (
        100,
        20.0,
        0.5,
        False,
    )


def test_go2trot_keeps_the_staff_control_defaults():
    cfg = DeployConfig(import_custom_impl.load_custom_impl("go2trot"))
    assert (cfg.ctrl_freq, cfg.kp, cfg.kd, cfg.exp_moving_avg) == (
        100,
        30.0,
        2.0,
        True,
    )


@pytest.mark.parametrize("task_name", [None, ""])
def test_loader_rejects_an_empty_task_name(task_name):
    with pytest.raises(ValueError, match="task name"):
        import_custom_impl.load_custom_impl(task_name)


def test_loader_finds_a_task_despite_an_unrelated_broken_file(monkeypatch, tmp_path):
    _write_implementation(tmp_path, "valid.py", "ValidTask", "valid_task")
    (tmp_path / "broken.py").write_text("raise RuntimeError('unrelated failure')\n")
    (tmp_path / "custom_config.py").write_text("raise AssertionError('not a task')\n")
    monkeypatch.setattr(import_custom_impl, "CUSTOM_IMPL_DIR", tmp_path)
    implementation = import_custom_impl.load_custom_impl("valid_task")
    assert isinstance(implementation, CustomConfig)
    assert implementation.task_name == "valid_task"


def test_loader_reports_failed_imports_when_a_task_is_missing(monkeypatch, tmp_path):
    (tmp_path / "broken.py").write_text("raise RuntimeError('unrelated failure')\n")
    monkeypatch.setattr(import_custom_impl, "CUSTOM_IMPL_DIR", tmp_path)
    with pytest.raises(LookupError) as error:
        import_custom_impl.load_custom_impl("missing_task")
    assert "missing_task" in str(error.value)
    assert "broken.py" in str(error.value)
    assert "unrelated failure" in str(error.value)


def test_loader_rejects_two_classes_for_the_same_task(monkeypatch, tmp_path):
    _write_implementation(tmp_path, "first.py", "First", "duplicated_task")
    _write_implementation(tmp_path, "second.py", "Second", "duplicated_task")
    monkeypatch.setattr(import_custom_impl, "CUSTOM_IMPL_DIR", tmp_path)
    with pytest.raises(ValueError, match="Multiple.*duplicated_task"):
        import_custom_impl.load_custom_impl("duplicated_task")


def test_loader_does_not_treat_an_imported_class_as_a_local_task(monkeypatch, tmp_path):
    (tmp_path / "imported.py").write_text(
        "from go2_deploy.custom_implementation.go2osc_push_glide "
        "import Go2OscPushGlide\n"
    )
    monkeypatch.setattr(import_custom_impl, "CUSTOM_IMPL_DIR", tmp_path)
    with pytest.raises(LookupError, match="go2osc_push_glide_deploy"):
        import_custom_impl.load_custom_impl("go2osc_push_glide_deploy")


def test_observation_dispatch_uses_each_controllers_own_instance():
    controllers = [
        SimpleNamespace(cfg=DeployConfig(_MarkerConfig(marker)), obs_vec_size=1)
        for marker in (3.0, 8.0)
    ]
    for controller, expected in zip(controllers, (1.5, 4.0)):
        torch.testing.assert_close(
            deploy_utility.lowstate_to_obs(controller, None), torch.tensor([expected])
        )


def test_bundle_restores_the_task_instance_and_matching_control_settings(bundle):
    from go2_deploy.policy_bundle import load_bundle

    _, _, directory = bundle
    policy, cfg = load_bundle(directory)
    assert isinstance(cfg.custom_impl, CustomConfig)
    assert cfg.custom_impl.task_name == "go2osc_push_glide_deploy"
    assert (cfg.ctrl_freq, cfg.kp, cfg.kd, cfg.exp_moving_avg) == (
        100,
        20.0,
        0.5,
        False,
    )
    assert cfg.obs_dim() == 58
    assert torch.isfinite(policy(torch.zeros(58))).all()


def test_bundle_preserves_its_recorded_runtime_settings(bundle):
    from go2_deploy.policy_bundle import config_from_meta

    _, _, directory = bundle
    metadata = json.loads((directory / "deploy_meta.json").read_text())
    metadata.update(
        ctrl_freq=50,
        exp_moving_avg=True,
        ema_time_const=0.2,
        kp=18.0,
        kd=0.7,
    )
    cfg = config_from_meta(metadata)
    assert (cfg.ctrl_freq, cfg.kp, cfg.kd, cfg.exp_moving_avg) == (
        50,
        18.0,
        0.7,
        True,
    )
    assert cfg.ema_time_const == 0.2
    assert cfg.ema_smoothing_factor == pytest.approx(-math.expm1(-1.0 / (50 * 0.2)))
    assert cfg.custom_impl.ctrl_freq == 100
    assert not cfg.custom_impl.exp_moving_avg


def test_older_bundle_uses_the_tasks_explicit_timing_and_filtering(bundle):
    from go2_deploy.policy_bundle import config_from_meta

    _, _, directory = bundle
    metadata = json.loads((directory / "deploy_meta.json").read_text())
    for name in ("ctrl_freq", "exp_moving_avg", "ema_time_const"):
        metadata.pop(name)
    cfg = config_from_meta(metadata)
    assert (cfg.ctrl_freq, cfg.kp, cfg.kd, cfg.exp_moving_avg) == (
        100,
        20.0,
        0.5,
        False,
    )


def test_bundle_rejects_an_implementation_for_another_task(bundle):
    from go2_deploy.policy_bundle import load_bundle

    _, _, directory = bundle
    implementation = import_custom_impl.load_custom_impl("go2trot")
    with pytest.raises(ValueError, match="bundle.*go2osc_push_glide_deploy.*go2trot"):
        load_bundle(directory, implementation)


def test_task_discovery_and_config_construction_do_not_import_the_sdk():
    result = _without_sdk("""
        from go2_deploy.custom_implementation.custom_config import CustomConfig
        from go2_deploy.deploy_config import DeployConfig
        from go2_deploy.utility.import_custom_impl import load_custom_impl

        implementation = load_custom_impl("go2osc_push_glide_deploy")
        assert isinstance(implementation, CustomConfig)
        assert DeployConfig(implementation).obs_dim() == 58
        assert "unitree_sdk2py" not in sys.modules
        assert "cyclonedds" not in sys.modules
    """)
    assert result.returncode == 0, result.stderr
