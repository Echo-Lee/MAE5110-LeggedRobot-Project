"""Isolate invalid sensor features and estimator samples during training.

Derived non-finite values must not poison shared model parameters or input
statistics. Invalid simulator states must stop with a useful diagnostic.
"""

import pytest
import torch

from gym.envs.go2.go2osc_deploy import Go2OscPushGlideDeploy
from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from gym.utils.task_registry import task_registry
from learning.modules.history_estimator import (
    FEATURE_LIMIT,
    ConcurrentEstimator,
    HistoryStateEstimator,
)
from tests.unit_tests.test_go2osc_onboard import NUM_ENVS, _make

ROBOT = 1


def _estimator(num_envs=8):
    torch.manual_seed(0)
    model = HistoryStateEstimator(
        features=["a"], feature_dims=[3], frames=2, stride=1, hidden_dims=[8]
    )
    estimator = ConcurrentEstimator(
        model, num_envs, "cpu", samples_per_step=8, update_interval=2, gradient_steps=2
    )
    estimator.reset(torch.ones(num_envs, dtype=torch.bool), torch.zeros(num_envs, 3))
    return estimator


def _weights(estimator):
    return torch.cat([value.flatten() for value in estimator.model.parameters()])


def test_estimator_zeroes_nonfinite_inputs_and_bounds_absurd_ones():
    estimator = _estimator()
    features = torch.ones(8, 3)
    features[0, 0] = float("nan")
    features[1, 1] = float("inf")
    features[2, 2] = -1.0e30
    estimator.step(features)
    newest = estimator.history.window().view(8, 2, 3)[:, -1]
    assert torch.isfinite(newest).all()
    assert newest[0, 0] == 0.0 and newest[1, 1] == 0.0
    assert newest[2, 2] == -FEATURE_LIMIT
    torch.testing.assert_close(newest[3:], torch.ones(5, 3))
    assert int(estimator.replaced_values) == 3
    assert torch.isfinite(estimator.mu_hat).all()
    assert torch.isfinite(estimator.v_hat).all()


def test_estimator_update_drops_samples_without_a_finite_label():
    estimator = _estimator()
    estimator.learning = True
    targets = torch.zeros(8, 4)
    targets[:3] = float("nan")
    for _ in range(4):
        estimator.step(torch.randn(8, 3), targets)
    assert estimator.updates == 2
    assert int(estimator.dropped_samples) > 0
    assert torch.isfinite(_weights(estimator)).all()


def test_single_usable_sample_keeps_normalization_and_learning_finite():
    estimator = _estimator(num_envs=2)
    estimator.learning = True
    estimator.unnormalized_inputs = torch.ones(6, dtype=torch.bool)
    valid = torch.tensor([0.1, 0.2, -0.1, 0.1, -0.2, 0.2])
    estimator._windows = [torch.stack((valid, torch.ones(6)))]
    estimator._targets = [torch.tensor([[0.5, 0.0, 0.0, 0.0], [float("nan")] * 4])]

    estimator._update()

    assert int(estimator.dropped_samples) == 1
    torch.testing.assert_close(estimator.model.input_mean, valid)
    torch.testing.assert_close(estimator.model.input_std, torch.full((6,), 1e-3))
    assert torch.isfinite(_weights(estimator)).all()
    estimator.reset(torch.ones(2, dtype=torch.bool), torch.zeros(2, 3))
    assert torch.isfinite(estimator.mu_hat).all()
    assert torch.isfinite(estimator.v_hat).all()

    before = _weights(estimator).detach().clone()
    for _ in range(2):
        estimator.step(0.1 * torch.randn(2, 3), torch.zeros(2, 4))
    estimator.step(torch.zeros(2, 3))  # predict with the updated weights
    assert estimator.updates == 2 and estimator.skipped_steps == 0
    assert not torch.equal(_weights(estimator), before)
    assert torch.isfinite(_weights(estimator)).all()
    assert torch.isfinite(estimator.mu_hat).all()
    assert torch.isfinite(estimator.v_hat).all()


def test_estimator_skips_an_update_with_no_usable_samples():
    estimator = _estimator()
    estimator.learning = True
    before = _weights(estimator).clone()
    for _ in range(2):
        estimator.step(torch.randn(8, 3), torch.full((8, 4), float("nan")))
    assert estimator.updates == 0 and estimator.skipped_steps == 2
    torch.testing.assert_close(_weights(estimator), before)


def test_estimator_skips_a_nonfinite_gradient_step():
    estimator = _estimator()
    estimator.learning = True
    before = _weights(estimator).clone()
    for _ in range(2):
        # Finite labels whose squared error overflows float32.
        estimator.step(torch.randn(8, 3), torch.full((8, 4), 1.0e30))
    assert estimator.skipped_steps == 2
    torch.testing.assert_close(_weights(estimator), before)
    # Normal samples train again afterwards.
    for _ in range(2):
        estimator.step(torch.randn(8, 3), torch.zeros(8, 4))
    assert not torch.equal(_weights(estimator), before)
    assert torch.isfinite(_weights(estimator)).all()


_LEG_FEATURES = Go2OscPushGlideEst._leg_features


def _nan_force_for_one_robot(self, torques, dof_vel):
    """The estimator task's leg features with NaN foot forces for ROBOT."""
    force, ratio, velocity = _LEG_FEATURES(self, torques, dof_vel)
    force = force.clone()
    force[ROBOT] = float("nan")
    return force, ratio, velocity


def _train(env, runner_cfg, log_dir, monkeypatch, iterations=2):
    # The nested config classes are shared: put their values back afterwards.
    monkeypatch.setattr(runner_cfg.runner, "device", "cpu", raising=False)
    monkeypatch.setattr(runner_cfg.runner, "max_iterations", iterations)
    monkeypatch.setattr(runner_cfg.algorithm, "rollout_size", 24 * NUM_ENVS)
    monkeypatch.setattr(runner_cfg.algorithm, "batch_size", 24)
    monkeypatch.setattr(runner_cfg.logging, "enable_local_saving", False)
    runner_cfg.log_dir = str(log_dir)
    runner = task_registry.make_alg_runner(env, runner_cfg)
    runner.learn()
    return runner


def test_unguarded_one_robots_nan_gives_every_robot_nan_actions(monkeypatch, tmp_path):
    """Unprotected features poison shared normalization and every action row."""
    monkeypatch.setattr(
        Go2OscPushGlideDeploy, "_leg_features", _nan_force_for_one_robot
    )
    env, runner_cfg, backend = _make()
    try:
        with pytest.raises(ValueError, match="Expected parameter loc") as error:
            _train(env, runner_cfg, tmp_path, monkeypatch)
        assert f"({NUM_ENVS}, 12)" in str(error.value)
    finally:
        backend.close()


def test_training_goes_on_when_one_robots_foot_forces_are_nan(monkeypatch, tmp_path):
    monkeypatch.setattr(Go2OscPushGlideEst, "_leg_features", _nan_force_for_one_robot)
    env, runner_cfg, backend = _make()
    try:
        runner = _train(env, runner_cfg, tmp_path, monkeypatch)
        assert runner.it == 2
        actor = torch.cat([value.flatten() for value in runner.alg.actor.parameters()])
        assert torch.isfinite(actor).all()
        assert torch.isfinite(env.oscillators).all()
        assert torch.isfinite(env._push_signal).all()
        counts = env.guard_counts()
        assert counts["leg_values_replaced"] > 0
        assert counts["estimator_steps_skipped"] == 0
        model = torch.cat([v.flatten() for v in env.estimator.model.parameters()])
        assert torch.isfinite(model).all()
    finally:
        backend.close()


def test_a_broken_simulation_stops_with_the_robot_and_signal_named(monkeypatch):
    env, _, backend = _make()
    try:
        env.step()
        signals = env._simulator_signals

        def broken():
            values = {name: value.clone() for name, value in signals().items()}
            values["dof_vel"][2, 4] = float("nan")
            return values

        monkeypatch.setattr(env, "_simulator_signals", broken)
        with pytest.raises(FloatingPointError, match=r"robot 2 .*\(dof_vel\)"):
            env.step()
    finally:
        backend.close()


def test_guard_report_is_printed_only_when_something_was_handled(capsys):
    env, _, backend = _make()
    try:
        env.estimator.updates = 50
        env.estimator.last_loss = None
        env._report_estimator_loss()
        out = capsys.readouterr().out
        assert "loss none yet" in out and "go2osc guard" not in out
        env.estimator.updates = 100
        env.estimator.last_loss = 0.5
        env.estimator.dropped_samples += 3
        env._report_estimator_loss()
        out = capsys.readouterr().out
        assert "loss 0.5000" in out
        assert "'estimator_samples_dropped': 3" in out
    finally:
        backend.close()
