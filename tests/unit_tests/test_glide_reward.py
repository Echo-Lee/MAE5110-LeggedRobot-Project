"""Glide reward and ice touchdown penalty for Go2LowFric."""

from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2_lowfric import Go2LowFric
from gym.envs.go2.go2_lowfric_config import Go2LowFricCfg, Go2LowFricRunnerCfg
from gym.utils.locomotion_metrics import GRAVITY_M_S2

FEET = torch.tensor([1, 2, 3, 4])
RADIUS = 0.022
MASS = 15.0
QUARTER_WEIGHT = 0.25 * MASS * GRAVITY_M_S2  # full-credit load per foot


def _state(
    mu=0.1,
    body_v=(1.0, 0.0),
    foot_v=(1.0, 0.0),
    foot_w=(0.0, 0.0, 0.0),
    load=QUARTER_WEIGHT,
    contact_time=0.2,
    touchdown=(False, False, False, False),
    command=(1.0, 0.0, 0.0),
):
    """One environment, five bodies (base + four feet), every foot alike."""
    lin = torch.zeros(1, 5, 3)
    lin[0, 1:, :2] = torch.tensor(foot_v)
    ang = torch.zeros(1, 5, 3)
    ang[0, 1:] = torch.tensor(foot_w)
    root = torch.zeros(1, 13)
    root[0, 7:9] = torch.tensor(body_v)
    forces = torch.zeros(1, 5, 3)
    forces[0, 1:, 2] = load
    settings = Go2LowFricCfg.reward_settings
    state = SimpleNamespace(
        cfg=SimpleNamespace(
            reward_settings=SimpleNamespace(
                glide_friction_threshold=settings.glide_friction_threshold,
                glide_friction_width=settings.glide_friction_width,
                glide_min_command=settings.glide_min_command,
                glide_full_load_fraction=settings.glide_full_load_fraction,
                glide_min_contact_time=settings.glide_min_contact_time,
                foot_radius=RADIUS,
            )
        ),
        _backend=SimpleNamespace(
            contact_friction=torch.tensor([mu]),
            # Mass split over bodies; only the total matters.
            link_mass=torch.full((1, 5), MASS / 5),
        ),
        feet_indices=FEET,
        _rigid_body_lin_vel=lin,
        _rigid_body_ang_vel=ang,
        root_states=root,
        contact_forces=forces,
        foot_contact_time=torch.full((1, 4), contact_time),
        foot_touchdown=torch.tensor([touchdown]),
        commands=torch.tensor([command]),
    )
    state._ice_gate = lambda: Go2LowFric._ice_gate(state)
    return state


def _glide(**kwargs):
    return Go2LowFric._reward_glide(_state(**kwargs)).item()


def _gate(mu):
    settings = Go2LowFricCfg.reward_settings
    return torch.sigmoid(
        torch.tensor(
            (settings.glide_friction_threshold - mu) / settings.glide_friction_width
        )
    ).item()


# --- glide: what earns full credit -------------------------------------------


def test_sustained_weight_bearing_glide_on_ice_earns_the_gate():
    assert _glide(mu=0.1) == pytest.approx(_gate(0.1))
    assert _gate(0.1) > 0.98


def test_feet_sliding_faster_than_the_body_are_capped_at_one():
    assert _glide(foot_v=(3.0, 0.0)) == pytest.approx(_gate(0.1))


def test_carrying_more_than_a_quarter_of_body_weight_is_capped_at_one():
    assert _glide(load=2 * QUARTER_WEIGHT) == pytest.approx(_gate(0.1))


# --- glide: the loopholes from run Nov15_04-33-19_ ---------------------------


def test_lightly_loaded_scuffing_foot_earns_only_its_load_share():
    # 10 N is enough to count as contact but is a small share of the weight.
    assert _glide(load=10.0) == pytest.approx(10.0 / QUARTER_WEIGHT * _gate(0.1))
    assert _glide(load=10.0) < 0.3


def test_brief_contact_earns_only_its_time_share():
    assert _glide(contact_time=0.02) == pytest.approx(0.2 * _gate(0.1))
    assert _glide(contact_time=0.0) == 0.0  # just landed / in the air


def test_light_brief_scuff_earns_almost_nothing():
    assert _glide(load=10.0, contact_time=0.02) < 0.06


# --- glide: other zero points ---------------------------------------------------


def test_planted_feet_earn_nothing():
    assert _glide(foot_v=(0.0, 0.0)) == 0.0


def test_ground_friction_switches_the_reward_off():
    assert _glide(mu=1.0) < 1e-5
    assert _glide(mu=0.6) < 0.003


def test_half_speed_slide_earns_half():
    assert _glide(foot_v=(0.5, 0.0)) == pytest.approx(0.5 * _gate(0.1))


def test_backward_and_sideways_slip_earn_nothing():
    assert _glide(foot_v=(-1.0, 0.0)) == 0.0
    assert _glide(foot_v=(0.0, 1.0)) == 0.0


def test_unloaded_feet_earn_nothing():
    assert _glide(load=0.0) == 0.0


def test_rolling_foot_is_not_gliding():
    spin = 1.0 / RADIUS  # center moves forward, contact point stays put
    assert _glide(foot_w=(0.0, spin, 0.0)) == pytest.approx(0.0, abs=1e-6)


def test_stand_still_and_turn_in_place_commands_earn_nothing():
    assert _glide(command=(0.0, 0.0, 0.0)) == 0.0
    assert _glide(command=(0.0, 0.0, 1.0)) == 0.0


def test_shuffling_under_a_slow_body_is_scaled_down():
    slow = _glide(body_v=(0.06, 0.0), foot_v=(0.06, 0.0))
    assert slow == pytest.approx(0.2 * _gate(0.1))


def test_misreported_friction_observation_does_not_change_the_gate():
    state = _state(mu=0.1)
    state.contact_friction = torch.tensor([[1.0]])  # what the policy is told
    assert Go2LowFric._reward_glide(state).item() == pytest.approx(_gate(0.1))


# --- ice touchdown penalty -------------------------------------------------------


def test_each_touchdown_on_ice_costs_one_gated_unit():
    state = _state(mu=0.1, touchdown=(True, False, True, False))
    assert Go2LowFric._reward_ice_touchdowns(state).item() == pytest.approx(
        -2 * _gate(0.1)
    )


def test_touchdowns_on_ground_are_free_and_none_cost_nothing():
    on_ground = _state(mu=1.0, touchdown=(True, True, True, True))
    assert Go2LowFric._reward_ice_touchdowns(on_ground).item() > -1e-4
    assert Go2LowFric._reward_ice_touchdowns(_state(mu=0.1)).item() == 0.0


# --- configuration and the real environment ---------------------------------


def test_new_weights_are_only_in_go2_lowfric():
    from gym.envs.go2.go2_config import Go2RunnerCfg

    weights = Go2LowFricRunnerCfg().critic.reward.weights
    assert weights.glide == pytest.approx(1.0)
    assert weights.ice_touchdowns == pytest.approx(0.5)
    go2 = Go2RunnerCfg().critic.reward.weights
    assert not hasattr(go2, "glide") and not hasattr(go2, "ice_touchdowns")


@pytest.fixture
def env():
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2LowFricCfg()
    cfg.env.num_envs = 4
    cfg.seed = 2
    task_registry.convert_frequencies_to_params(cfg, Go2LowFricRunnerCfg())
    task = Go2LowFric(cfg, "cpu", True, select_backend(cfg, "cpu", "mujoco"))
    yield task
    task._backend.close()


def test_contact_timer_counts_continuous_load_and_flags_touchdowns(env):
    previous = env.foot_contact_time.clone()
    for _ in range(40):
        previous = env.foot_contact_time.clone()
        env.step()
        loaded = env.contact_forces[:, env.feet_indices, 2] > 5.0
        expected = (previous + env.dt) * loaded
        torch.testing.assert_close(env.foot_contact_time, expected)
        assert torch.equal(env.foot_touchdown, loaded & (previous == 0))
    # A standing robot has feet that have stayed down for many steps.
    assert (env.foot_contact_time > 10 * env.dt).any()


def test_reset_clears_the_contact_timer_for_reset_envs_only(env):
    for _ in range(20):
        env.step()
    before = env.foot_contact_time.clone()
    mask = torch.tensor([True, False, True, False])
    env._reset_idx(mask)
    assert (env.foot_contact_time[mask] == 0).all()
    assert not env.foot_touchdown[mask].any()
    torch.testing.assert_close(env.foot_contact_time[~mask], before[~mask])


def test_rewards_run_on_a_real_go2_lowfric_step(env):
    env.commands[:] = torch.tensor([1.0, 0.0, 0.0])
    for _ in range(5):
        env.step()
    glide = env._reward_glide()
    touchdowns = env._reward_ice_touchdowns()
    assert glide.shape == touchdowns.shape == (4,)
    assert ((glide >= 0) & (glide <= 1)).all()
    assert ((touchdowns <= 0) & (touchdowns >= -4)).all()
