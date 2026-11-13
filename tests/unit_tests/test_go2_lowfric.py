"""Go2LowFric: privileged friction observation and episodic bimodal friction."""

import pytest
import torch

from gym.envs.go2.go2_lowfric_config import (
    GROUND_FRICTION,
    ICE_FRICTION,
    Go2LowFricCfg,
    Go2LowFricRunnerCfg,
)


@pytest.fixture
def env():
    from gym.envs.go2.go2_lowfric import Go2LowFric
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2LowFricCfg()
    runner_cfg = Go2LowFricRunnerCfg()
    cfg.env.num_envs = 8
    cfg.seed = 3
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, "cpu", "mujoco")
    task = Go2LowFric(cfg, "cpu", True, backend)
    yield task
    task._backend.close()


def _in_bins(values):
    ice = (ICE_FRICTION[0] <= values) & (values <= ICE_FRICTION[1])
    ground = (GROUND_FRICTION[0] <= values) & (values <= GROUND_FRICTION[1])
    return ice | ground


def test_friction_is_an_actor_and_critic_observation():
    runner_cfg = Go2LowFricRunnerCfg()
    assert "contact_friction" in runner_cfg.actor.obs
    assert "contact_friction" in runner_cfg.critic.obs


def test_friction_observation_is_live_view_of_backend(env):
    obs = env.get_state("contact_friction")
    assert obs.shape == (8, 1)
    assert _in_bins(obs[:, 0]).all()
    torch.testing.assert_close(obs[:, 0], env._backend.contact_friction)

    mask = torch.zeros(8, dtype=torch.bool)
    mask[[2, 5]] = True
    before = env.contact_friction.clone()
    env._reset_idx(mask)

    after = env.get_state("contact_friction")
    torch.testing.assert_close(after[:, 0], env._backend.contact_friction)
    assert torch.equal(after[~mask], before[~mask])
    assert not torch.equal(after[mask], before[mask])
    assert _in_bins(after[:, 0]).all()


def test_actor_observation_includes_friction_and_steps(env):
    runner_cfg = Go2LowFricRunnerCfg()
    actor_obs = env.get_states(runner_cfg.actor.obs)
    assert actor_obs.shape[0] == 8
    torch.testing.assert_close(actor_obs[:, -1], env._backend.contact_friction)
    env.step()
    assert torch.isfinite(env.get_states(runner_cfg.critic.obs)).all()
