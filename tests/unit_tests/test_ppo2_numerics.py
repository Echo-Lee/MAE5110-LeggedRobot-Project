"""PPO updates must not corrupt parameters when likelihoods or gradients spike."""

import math

import pytest
import torch
from tensordict import TensorDict

from learning.algorithms import ppo2
from learning.modules import Actor, Critic


def _algorithm():
    torch.manual_seed(0)
    actor = Actor(1, 1, hidden_dims=[], normalize_obs=False)
    critic = Critic(1, hidden_dims=[], normalize_obs=False)
    return ppo2.PPO2(
        actor,
        critic,
        batch_size=2,
        max_gradient_steps=1,
        entropy_coef=0.0,
        use_clipped_value_loss=False,
    )


def _data():
    return TensorDict(
        {
            "actor_obs": torch.ones(2, 1, 1),
            "critic_obs": torch.ones(2, 1, 1),
            "actions": torch.ones(2, 1, 1),
            "advantages": torch.ones(2, 1),
            "values": torch.zeros(2, 1),
            "returns": torch.full((2, 1), 3.0),
        },
        batch_size=[2, 1],
    )


def _parameters(network):
    return torch.cat(
        [parameter.detach().flatten() for parameter in network.parameters()]
    )


def test_extreme_likelihood_change_keeps_the_ppo_objective_finite(monkeypatch):
    algorithm = _algorithm()
    linear = next(
        layer
        for layer in algorithm.actor.NN.modules()
        if isinstance(layer, torch.nn.Linear)
    )
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(1000.0)
    data = _data()
    data["actions"].zero_()
    data["advantages"].fill_(-1.0)

    def changed_policy_batches(data, *args, **kwargs):
        # PPO compares cached old log probabilities with an updated policy.
        # This controlled change makes that log ratio exceed float32's exp range.
        with torch.no_grad():
            linear.bias.zero_()
        yield data.flatten(0, 1)

    monkeypatch.setattr(ppo2, "create_uniform_generator", changed_policy_batches)
    algorithm.update_actor(data)
    assert torch.isfinite(_parameters(algorithm.actor)).all()
    assert math.isfinite(algorithm.mean_surrogate_loss)
    assert algorithm.mean_surrogate_loss == pytest.approx(math.exp(20.0), rel=1e-6)
    assert algorithm.skipped_updates == 0


@pytest.mark.parametrize("network_name", ["actor", "critic"])
def test_nonfinite_gradients_leave_parameters_intact_and_allow_recovery(network_name):
    algorithm = _algorithm()
    network = getattr(algorithm, network_name)
    before = _parameters(network).clone()
    parameter = next(network.parameters())
    handle = parameter.register_hook(
        lambda gradient: torch.full_like(gradient, math.nan)
    )
    update = getattr(algorithm, f"update_{network_name}")
    try:
        with pytest.raises(FloatingPointError, match=f"All {network_name} minibatches"):
            update(_data())
    finally:
        handle.remove()
    torch.testing.assert_close(_parameters(network), before, atol=0.0, rtol=0.0)
    assert algorithm.skipped_updates == 1
    update(_data())
    after = _parameters(network)
    assert torch.isfinite(after).all()
    assert not torch.equal(after, before)
