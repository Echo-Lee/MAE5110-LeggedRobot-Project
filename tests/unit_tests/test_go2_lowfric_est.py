"""go2_lowfric_est: actor reads mu-hat from an estimator the env runs."""

import warnings

import pytest
import torch

from gym.envs.go2.go2_lowfric_config import Go2LowFricRunnerCfg
from gym.envs.go2.go2_lowfric_est_config import (
    Go2LowFricEstCfg,
    Go2LowFricEstRunnerCfg,
)
from learning.modules.history_estimator import FRICTION_CLIP


def _make(cfg_edit=None):
    from gym.envs.go2.go2_lowfric_est import Go2LowFricEst
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2LowFricEstCfg()
    runner_cfg = Go2LowFricEstRunnerCfg()
    cfg.env.num_envs = 4
    cfg.seed = 5
    # Untrained shell (no file needed); small updates so tests are fast.
    cfg.state_estimator.init_path = None
    cfg.state_estimator.hidden_dims = [16]
    cfg.state_estimator.samples_per_step = 4
    cfg.state_estimator.update_interval = 2
    cfg.state_estimator.gradient_steps = 1
    if cfg_edit:
        cfg_edit(cfg)
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, "cpu", "mujoco")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = Go2LowFricEst(cfg, "cpu", True, backend)
    return env, runner_cfg


@pytest.fixture
def env():
    task, _ = _make()
    yield task
    task._backend.close()


def test_actor_reads_mu_hat_in_the_teacher_slot_and_critic_keeps_mu():
    teacher = Go2LowFricRunnerCfg.actor.obs
    student = Go2LowFricEstRunnerCfg.actor.obs
    i = teacher.index("contact_friction")
    assert student[i] == "contact_friction_hat"
    assert student[:i] + student[i + 1 :] == teacher[:i] + teacher[i + 1 :]
    assert "contact_friction" in Go2LowFricEstRunnerCfg.critic.obs
    assert Go2LowFricEstCfg.friction_switch.probability > 0


def test_mu_hat_input_is_the_clipped_estimate_not_the_true_mu(env):
    env.step()
    env._reset_idx(torch.zeros(env.num_envs, dtype=torch.bool))
    hat = env.get_state("contact_friction_hat")
    assert hat.shape == (env.num_envs, 1)
    torch.testing.assert_close(hat[:, 0], env.estimator.mu_hat.clamp(*FRICTION_CLIP))
    assert not torch.allclose(hat[:, 0], env._backend.contact_friction)


def _weights(env):
    return [p.detach().clone() for p in env.estimator.model.parameters()]


def test_estimator_trains_only_while_learning_and_under_inference_mode(env):
    before = _weights(env)
    for _ in range(4):
        env.step()
    assert env.estimator.updates == 0  # learning is off by default
    env.set_learning(True)
    # As in the PPO rollout. (Env buffers rebuilt under inference mode stay
    # inference tensors, so the rest of this test stays inside it too.)
    with torch.inference_mode():
        for _ in range(4):
            env.step()
        assert env.estimator.updates == 2
        env.set_learning(False)
        env.step()
        env.step()
    assert env.estimator.updates == 2
    after = _weights(env)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_learn_flag_in_config_can_freeze_the_estimator():
    def freeze(cfg):
        cfg.state_estimator.learn = False

    env, _ = _make(freeze)
    try:
        env.set_learning(True)
        for _ in range(4):
            env.step()
        assert env.estimator.updates == 0
    finally:
        env._backend.close()


def test_policy_checkpoint_carries_the_estimator(env, tmp_path):
    from gym.utils.task_registry import task_registry

    runner_cfg = Go2LowFricEstRunnerCfg()
    runner_cfg.runner.resume = False
    runner_cfg.runner.device = "cpu"
    runner_cfg.log_dir = str(tmp_path)
    runner = task_registry.make_alg_runner(env, runner_cfg)
    runner.it = 7
    runner.save()
    saved = _weights(env)
    path = tmp_path / "model_7.pt"
    assert "env_state_dict" in torch.load(path, weights_only=True)
    with torch.no_grad():
        for p in env.estimator.model.parameters():
            p.add_(1.0)
    runner.load(path)
    for a, b in zip(saved, _weights(env)):
        torch.testing.assert_close(a, b)


def test_student_friction_covers_the_whole_range_without_gaps():
    startup = Go2LowFricEstCfg.domain_randomization.startup
    bins = startup.contact_friction_bins
    assert bins[0][0] == startup.contact_friction_range[0]
    assert bins[-1][1] == startup.contact_friction_range[1]
    for (_, high), (low, _) in zip(bins, bins[1:]):
        assert high == low
    assert len(startup.contact_friction_bin_weights) == len(bins)


def test_latch_holds_slide_start_mu_hat_until_the_phase_ends(env):
    env.latch_in_slide = True
    env.slide_phase[:, 0] = 0.0
    env.estimator.mu_hat = torch.full((env.num_envs,), 0.1)
    env._publish_estimate()
    env.slide_phase[:, 0] = 1.0  # slide starts: latch 0.1
    env._publish_estimate()
    env.estimator.mu_hat = torch.full((env.num_envs,), 0.6)  # drifts mid-slide
    env._publish_estimate()
    env._publish_estimate()  # a second call in the same step keeps the latch
    torch.testing.assert_close(env.contact_friction_hat[:, 0], torch.full((4,), 0.1))
    env.slide_phase[:, 0] = 0.0  # slide over: live estimate again
    env._publish_estimate()
    torch.testing.assert_close(env.contact_friction_hat[:, 0], torch.full((4,), 0.6))


def test_external_slide_phase_survives_the_step(env):
    env.external_slide_phase = True
    env.slide_phase[:, 0] = 1.0
    env.step()
    assert (env.slide_phase == 1.0).all()
    env.external_slide_phase = False
    env.step()  # no slide episodes running: the env zeroes it
    assert (env.slide_phase == 0.0).all()


def test_legacy_checkpoint_estimator_is_widened_with_the_accelerometer(env):
    from learning.modules.history_estimator import (
        DEFAULT_FEATURES,
        LEGACY_FEATURES,
        HistoryStateEstimator,
    )

    assert list(env.estimator_features) == list(DEFAULT_FEATURES)
    config = env._legacy_estimator_config()
    legacy = HistoryStateEstimator(
        features=config["features"],
        feature_dims=config["feature_dims"],
        frames=25,
        stride=2,
        hidden_dims=[16],
    )
    state = {"state_estimator": {"model": legacy.state_dict(), "optimizer": {}}}
    env.load_checkpoint_state(state)  # no estimator_config: a run 15-17 file
    assert env.estimator.unnormalized_inputs is not None
    assert list(env.estimator_features) == list(DEFAULT_FEATURES)
    assert "base_lin_acc" not in LEGACY_FEATURES
    # Saved again, it carries its (widened) config.
    saved = env.checkpoint_state()
    assert saved["estimator_config"]["features"] == list(DEFAULT_FEATURES)
