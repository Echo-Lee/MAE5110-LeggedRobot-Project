from types import SimpleNamespace

import pytest
import torch

from learning.modules.history_estimator import (
    FRICTION_CLIP,
    TARGET_SCALES,
    EstimatorInTheLoop,
    HistoryStateEstimator,
    ProprioHistory,
)


def test_history_window_takes_every_stride_th_frame_newest_last():
    history = ProprioHistory(
        num_envs=1, num_features=1, frames=2, stride=2, device="cpu"
    )
    for value in (1.0, 2.0, 3.0, 4.0):
        history.append(torch.tensor([[value]]))
    # Buffer [1, 2, 3, 4]: every 2nd frame counting back from the newest.
    assert history.window().tolist() == [[2.0, 4.0]]


def test_reset_fills_only_selected_envs_with_current_features():
    history = ProprioHistory(
        num_envs=2, num_features=1, frames=3, stride=1, device="cpu"
    )
    history.append(torch.tensor([[5.0], [5.0]]))
    history.reset(torch.tensor([True, False]), torch.tensor([[9.0], [9.0]]))
    assert history.window()[0].tolist() == [9.0, 9.0, 9.0]
    assert history.window()[1].tolist() == [0.0, 0.0, 5.0]


def _estimator():
    torch.manual_seed(0)
    return HistoryStateEstimator(
        features=("a", "b"), feature_dims=(2, 1), frames=3, stride=2, hidden_dims=(16,)
    )


def test_predict_returns_mu_and_velocity_in_physical_units():
    model = _estimator()
    window = torch.randn(5, 9)
    mu, v = model.predict(window)
    assert mu.shape == (5,) and v.shape == (5, 3)
    raw = model(window)
    torch.testing.assert_close(mu, raw[:, 0] * TARGET_SCALES[0])
    torch.testing.assert_close(v, raw[:, 1:] * torch.tensor(TARGET_SCALES[1:]))


def test_loss_is_zero_for_a_perfect_prediction():
    model = _estimator()
    window = torch.randn(4, 9)
    mu, v = model.predict(window)
    perfect = torch.cat((mu.unsqueeze(1), v), dim=1).detach()
    assert model.loss(window, perfect).item() == pytest.approx(0.0, abs=1e-10)


def test_input_normalization_is_frozen_and_saved(tmp_path):
    model = _estimator()
    data = 3.0 + 2.0 * torch.randn(1000, 9)
    model.set_input_normalization(data)
    torch.testing.assert_close(model.input_mean, data.mean(dim=0))
    path = tmp_path / "estimator.pt"
    model.save(path, extra={"note": "x"})
    loaded, extra = HistoryStateEstimator.load(path)
    window = torch.randn(3, 9)
    torch.testing.assert_close(loaded.predict(window)[0], model.predict(window)[0])
    assert extra == {"note": "x"}
    assert loaded.config["frames"] == 3 and loaded.config["stride"] == 2


def test_in_the_loop_overwrites_the_policy_friction_input(tmp_path):
    model = _estimator()
    path = tmp_path / "estimator.pt"
    model.save(path)
    true_view = torch.full((2,), 0.1)
    env = SimpleNamespace(
        num_envs=2,
        get_states=lambda names: torch.randn(2, 3),
        contact_friction=true_view.unsqueeze(1),
        _backend=SimpleNamespace(contact_friction=true_view),
    )
    loop = EstimatorInTheLoop(env, path, "cpu")
    loop.step()
    assert env.contact_friction.shape == (2, 1)
    torch.testing.assert_close(
        env.contact_friction[:, 0], loop.mu_hat.clamp(*FRICTION_CLIP)
    )
    assert loop.v_hat.shape == (2, 3)
    # Physics keeps the true friction.
    assert torch.equal(env._backend.contact_friction, torch.full((2,), 0.1))


def test_observe_only_leaves_the_policy_friction_input(tmp_path):
    model = _estimator()
    path = tmp_path / "estimator.pt"
    model.save(path)
    true_input = torch.full((2, 1), 0.1)
    env = SimpleNamespace(
        num_envs=2,
        get_states=lambda names: torch.randn(2, 3),
        contact_friction=true_input,
    )
    loop = EstimatorInTheLoop(env, path, "cpu", feed_policy=False)
    loop.step()
    assert env.contact_friction is true_input
    assert loop.mu_hat.shape == (2,)


def test_external_estimator_refuses_a_task_with_its_own(tmp_path):
    model = _estimator()
    path = tmp_path / "estimator.pt"
    model.save(path)
    env = SimpleNamespace(num_envs=2, contact_friction_hat=torch.zeros(2, 1))
    with pytest.raises(ValueError, match="own estimator"):
        EstimatorInTheLoop(env, path, "cpu")


def test_concurrent_estimator_collects_and_updates_on_schedule():
    from learning.modules.history_estimator import ConcurrentEstimator

    model = _estimator()  # 3 features (a: 2, b: 1), 3 frames, stride 2
    loop = ConcurrentEstimator(
        model, num_envs=5, device="cpu", samples_per_step=3, update_interval=2
    )
    loop.reset(torch.ones(5, dtype=torch.bool), torch.zeros(5, 3))
    targets = torch.zeros(5, 4)
    loop.step(torch.randn(5, 3), targets)
    assert loop.updates == 0 and loop._windows == []  # not learning
    loop.learning = True
    loop.step(torch.randn(5, 3), targets)
    assert len(loop._windows) == 1 and loop._windows[0].shape == (3, 9)
    loop.step(torch.randn(5, 3), targets)
    assert loop.updates == 1 and loop._windows == []
    assert loop.mu_hat.shape == (5,) and loop.v_hat.shape == (5, 3)


def test_widened_estimator_predicts_exactly_as_before():
    from learning.modules.history_estimator import expand_estimator_inputs

    model = _estimator()  # features a (2), b (1); 3 frames
    model.set_input_normalization(torch.randn(200, 9) * 2 + 1)
    wide, new = expand_estimator_inputs(model, ["a", "acc", "b"], [2, 3, 1])
    assert wide.config["features"] == ["a", "acc", "b"]
    assert new.sum() == 3 * 3  # 3 acc inputs per frame
    old_window = torch.randn(6, 9)
    # Same frames with arbitrary accelerometer values inserted per frame.
    frames = old_window.view(6, 3, 3)
    acc = torch.randn(6, 3, 3) * 5
    wide_window = torch.cat((frames[..., :2], acc, frames[..., 2:]), dim=2)
    torch.testing.assert_close(
        wide.predict(wide_window.reshape(6, -1))[0], model.predict(old_window)[0]
    )
    with pytest.raises(ValueError, match="drop"):
        expand_estimator_inputs(model, ["a"], [2])


def test_new_inputs_get_normalized_on_the_first_update():
    from learning.modules.history_estimator import (
        ConcurrentEstimator,
        expand_estimator_inputs,
    )

    wide, new = expand_estimator_inputs(_estimator(), ["a", "b", "c"], [2, 1, 1])
    loop = ConcurrentEstimator(
        wide, num_envs=8, device="cpu", samples_per_step=8, update_interval=2
    )
    loop.unnormalized_inputs = new
    loop.reset(torch.ones(8, dtype=torch.bool), torch.zeros(8, 4))
    loop.learning = True
    for _ in range(2):
        features = torch.zeros(8, 4)
        features[:, 3] = 5.0 + torch.randn(8)
        loop.step(features, torch.zeros(8, 4))
    assert loop.unnormalized_inputs is None
    # Only the newest frame has seen the new input yet (older: reset zeros).
    newest_c = 2 * 4 + 3  # frame 2 of 3, feature c (4 features per frame)
    assert loop.model.input_mean[newest_c] > 2.0
    old_inputs = ~new
    torch.testing.assert_close(  # untouched
        loop.model.input_mean[old_inputs], torch.zeros(int(old_inputs.sum()))
    )
