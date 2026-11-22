"""scripts/train.py --estimator_init: the in-env estimator's starting file."""

from types import SimpleNamespace

import pytest

from scripts.train import apply_estimator_init, get_train_args


def _cfg():
    return SimpleNamespace(state_estimator=SimpleNamespace(init_path=None))


def test_flag_defaults_to_the_config_file():
    args = get_train_args(["--task", "go2osc_push_glide_robust_est"])
    cfg = _cfg()
    apply_estimator_init(args, cfg)
    assert args.estimator_init is None
    assert cfg.state_estimator.init_path is None


def test_flag_sets_an_absolute_init_path(tmp_path, monkeypatch):
    (tmp_path / "estimator.pt").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    args = get_train_args(["--task", "t", "--estimator_init", "estimator.pt"])
    cfg = _cfg()
    apply_estimator_init(args, cfg)
    assert cfg.state_estimator.init_path == str(tmp_path / "estimator.pt")


def test_missing_file_fails_before_the_environment_is_built(tmp_path):
    args = get_train_args(["--task", "t", "--estimator_init", str(tmp_path / "no.pt")])
    with pytest.raises(FileNotFoundError):
        apply_estimator_init(args, _cfg())


def test_task_without_an_estimator_rejects_the_flag(tmp_path):
    (tmp_path / "estimator.pt").write_bytes(b"")
    args = get_train_args(
        ["--task", "go2osc_push_glide", "--estimator_init", str(tmp_path / "estimator.pt")]
    )
    with pytest.raises(ValueError, match="no in-env state estimator"):
        apply_estimator_init(args, SimpleNamespace())
