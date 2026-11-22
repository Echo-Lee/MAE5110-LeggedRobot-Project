"""scripts/pretrain_go2osc_estimator.py resets finished envs like PPO does."""

import warnings

import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2 import go2osc_est_config as configs
from gym.envs.go2.go2osc_est import Go2OscPushGlideEst
from gym.utils.task_registry import task_registry
from scripts import pretrain_go2osc_estimator as pretrain

NUM_ENVS = 3
EPISODE_S = 0.8  # longer than SETTLE_S, so metrics keep samples
FORCED_STEP = 5  # env 0 is made to fall on this step


def _make():
    cfg = configs.Go2OscPushGlideEstCfg()
    runner_cfg = configs.Go2OscPushGlideEstRunnerCfg()
    cfg.seed = runner_cfg.seed = 3
    cfg.env.num_envs = NUM_ENVS
    cfg.env.episode_length_s = EPISODE_S
    cfg.push_robots.toggle = False
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.friction_switch.probability = 0.0
    cfg.state_estimator.hidden_dims = [16]
    runner_cfg.runner.device = "cpu"
    runner_cfg.logging.enable_local_saving = False
    runner_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = MuJocoCPUBackend()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # untrained-estimator warning
        env = Go2OscPushGlideEst(cfg, "cpu", True, backend)
    runner = task_registry.make_alg_runner(env, runner_cfg)
    runner.switch_to_eval()
    # Only the forced fall below should end an episode early.
    original = env._check_terminations_and_timeouts
    count = {"steps": 0}

    def check():
        original()
        env.terminated[:] = False
        count["steps"] += 1
        if count["steps"] == FORCED_STEP:
            env.terminated[0] = True

    env._check_terminations_and_timeouts = check
    return env, runner, backend


def test_act_resets_timed_out_and_fallen_envs():
    env, runner, backend = _make()
    try:
        limit = int(env.max_episode_length)
        friction_before = env._backend.contact_friction.clone()
        lengths, done = [], []
        with torch.no_grad():
            for _ in range(limit + 10):  # crosses the timeout
                timed_out, terminated = pretrain.act(env, runner)
                lengths.append(env.episode_length_buf.clone())
                done.append(timed_out | terminated)
        lengths, done = torch.stack(lengths), torch.stack(done)
        assert done[FORCED_STEP - 1, 0]  # the forced fall was reported
        assert lengths[FORCED_STEP - 1, 0] == 0  # ... and env 0 restarted
        assert done.any(dim=0).all()  # every env finished an episode
        assert lengths.max() <= limit  # no env ran past its episode
        # Resets resample friction (the est task redraws it per episode).
        assert not torch.allclose(env._backend.contact_friction, friction_before)
    finally:
        backend.close()


def test_evaluation_records_terminal_steps_and_counts_episodes():
    env, runner, backend = _make()
    try:
        limit = int(env.max_episode_length)
        with torch.no_grad():
            trace = pretrain.evaluate(env, runner, limit + 10)
        t_reset = trace["t_reset"]
        assert t_reset.max() <= limit * env.dt + 1e-6
        # Each finished episode is recorded at its last step, then restarts.
        finished = trace["timed_out"] | trace["terminated"]
        restarts = t_reset[1:] < t_reset[:-1]
        assert (finished[:-1] == restarts).all()
        assert trace["terminated"][:, 0].sum() == 1
        result = pretrain.metrics(trace)
        assert result["episodes_terminated"] == 1
        assert result["episodes_timed_out"] == int(trace["timed_out"].sum()) >= 2
        assert result["samples"] == int((t_reset >= pretrain.SETTLE_S).sum()) > 0
    finally:
        backend.close()
