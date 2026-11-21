"""Widening a trained actor's inputs keeps its actions unchanged."""

import pytest
import torch

from learning.modules import Actor
from learning.modules.input_widening import widen_actor_state

OLD = [("a", 3), ("b", 2)]


def _trained_actor(num_obs):
    torch.manual_seed(0)
    actor = Actor(num_obs, 4, hidden_dims=[16, 8], activation="elu")
    actor.train()
    for _ in range(5):  # give the normalizer non-trivial statistics
        actor.act_inference(3.0 * torch.randn(64, num_obs) + 1.0)
    return actor.eval()


def test_widened_actor_acts_exactly_like_the_original_whatever_the_new_inputs():
    old = _trained_actor(5)
    new_inputs = [
        ("a", 3),
        ("mu_hat", 1),
        ("b", 2),
        ("v_hat", 3),
    ]  # inserted + appended
    wide = Actor(9, 4, hidden_dims=[16, 8], activation="elu").eval()
    wide.load_state_dict(widen_actor_state(old.state_dict(), OLD, new_inputs))
    obs_a, obs_b = torch.randn(32, 3), torch.randn(32, 2)
    for _ in range(3):
        extra_mu, extra_v = 10 * torch.randn(32, 1), 10 * torch.randn(32, 3)
        wide_obs = torch.cat((obs_a, extra_mu, obs_b, extra_v), dim=1)
        torch.testing.assert_close(
            wide.act_inference(wide_obs),
            old.act_inference(torch.cat((obs_a, obs_b), dim=1)),
        )


def test_new_inputs_take_the_given_statistics_and_old_ones_keep_theirs():
    old = _trained_actor(5)
    state = old.state_dict()
    new_inputs = OLD + [("mu_hat", 1)]
    stats = {"mu_hat": (torch.tensor([0.5]), torch.tensor([0.04]))}
    wide = widen_actor_state(state, OLD, new_inputs, new_stats=stats)
    torch.testing.assert_close(
        wide["obs_rms.running_mean"][:5], state["obs_rms.running_mean"]
    )
    assert wide["obs_rms.running_mean"][5].item() == pytest.approx(0.5)
    assert wide["obs_rms.running_var"][5].item() == pytest.approx(0.04)
    assert torch.all(wide["NN.0.weight"][:, 5] == 0)


def test_dropping_or_resizing_an_input_is_refused():
    state = _trained_actor(5).state_dict()
    with pytest.raises(ValueError, match="drop"):
        widen_actor_state(state, OLD, [("a", 3), ("c", 2)])
    with pytest.raises(ValueError, match="size"):
        widen_actor_state(state, OLD, [("a", 3), ("b", 3)])


def test_dropping_an_input_equals_feeding_the_original_its_running_mean():
    old = _trained_actor(5)
    state = old.state_dict()
    new_inputs = [("a", 3), ("mu_hat", 1)]  # drop "b", add "mu_hat"
    narrow = Actor(4, 4, hidden_dims=[16, 8], activation="elu").eval()
    narrow.load_state_dict(widen_actor_state(state, OLD, new_inputs, drop=["b"]))
    obs_a = torch.randn(32, 3)
    b_mean = state["obs_rms.running_mean"][3:5].float().expand(32, 2)
    torch.testing.assert_close(
        narrow.act_inference(torch.cat((obs_a, 5 * torch.randn(32, 1)), dim=1)),
        old.act_inference(torch.cat((obs_a, b_mean), dim=1)),
    )


def test_only_explicitly_listed_inputs_can_be_dropped():
    state = _trained_actor(5).state_dict()
    with pytest.raises(ValueError, match="not in drop"):
        widen_actor_state(state, OLD, [("a", 3)])
    with pytest.raises(ValueError, match="lacks"):
        widen_actor_state(state, OLD, [("a", 3)], drop=["b", "zzz"])
    with pytest.raises(ValueError, match="still listed"):
        widen_actor_state(state, OLD, OLD, drop=["b"])
