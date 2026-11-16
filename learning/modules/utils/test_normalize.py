import pytest
import torch

from learning.modules.utils.normalize import RunningMeanStd


def test_update_moments():
    rms = RunningMeanStd(num_items=3)
    running_mean = 0.95 * torch.ones(3)
    running_var = 0.2 * torch.ones(3)
    batch_mean = 0.35 * torch.ones(3)
    batch_var = 0.1 * torch.ones(3)

    new_mean, new_var, total_count = rms._update_mean_var_from_moments(
        running_mean,
        running_var,
        torch.tensor(10.0),
        batch_mean,
        batch_var,
        batch_count=5,
    )

    torch.testing.assert_close(new_mean, 0.75 * torch.ones(3))
    torch.testing.assert_close(new_var, (3.7 / 15.0) * torch.ones(3))
    assert total_count == 15


def test_forward_updates_statistics_and_normalizes_input():
    rms = RunningMeanStd(num_items=2)
    normalized = rms(torch.ones(2, 2))

    expected = (
        (1.0 / 3.0) / torch.sqrt(torch.tensor(5.0 / 9.0 + rms.epsilon))
    ) * torch.ones_like(normalized)
    torch.testing.assert_close(normalized, expected)


@pytest.mark.parametrize("sizes", [(1024, 1024), (1024, 1025), (1000, 7)])
def test_parallel_moments_match_concatenated_samples(sizes):
    generator = torch.Generator().manual_seed(13)
    first = torch.randn(sizes[0], 3, generator=generator, dtype=torch.float64)
    second = 2 * torch.randn(sizes[1], 3, generator=generator, dtype=torch.float64) + 4
    combined = torch.cat((first, second))
    rms = RunningMeanStd(3)
    mean, variance, count = rms._update_mean_var_from_moments(
        first.mean(0),
        first.var(0, correction=0),
        torch.tensor(float(len(first))),
        second.mean(0),
        second.var(0, correction=0),
        len(second),
    )
    torch.testing.assert_close(mean, combined.mean(0), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        variance, combined.var(0, correction=0), atol=1e-12, rtol=1e-12
    )
    assert count == len(combined)


def test_equal_zero_mean_batches_do_not_create_a_mean_offset():
    rms = RunningMeanStd(3)
    mean, variance, count = rms._update_mean_var_from_moments(
        torch.zeros(3),
        torch.ones(3),
        torch.tensor(1024.0),
        torch.zeros(3),
        torch.ones(3),
        1024,
    )
    torch.testing.assert_close(mean, torch.zeros(3), atol=0, rtol=0)
    torch.testing.assert_close(variance, torch.ones(3), atol=0, rtol=0)
    assert count == 2048


@pytest.mark.parametrize("sizes", [(1024, 1024), (1, 1, 3, 7), (3, 5, 2)])
def test_repeated_forward_updates_match_all_observed_samples(sizes):
    rms = RunningMeanStd(3)
    # Remove the module's unit-variance prior to compare actual samples directly.
    rms.count.zero_()
    generator = torch.Generator().manual_seed(29)
    batches = []
    for index, size in enumerate(sizes):
        batch = torch.randn(size, 3, generator=generator, dtype=torch.float64) + index
        batches.append(batch)
        normalized = rms(batch)
        combined = torch.cat(batches)
        torch.testing.assert_close(
            rms.running_mean, combined.mean(0), atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            rms.running_var, combined.var(0, correction=0), atol=1e-12, rtol=1e-12
        )
        assert rms.count == len(combined)
        assert torch.isfinite(normalized).all()


def test_multidimensional_batches_count_every_sample_axis():
    rms = RunningMeanStd(3)
    rms.count.zero_()
    first = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)
    second = torch.arange(18, 30, dtype=torch.float64).reshape(2, 2, 3)
    rms(first)
    rms(second)
    combined = torch.cat((first.reshape(-1, 3), second.reshape(-1, 3)))
    torch.testing.assert_close(rms.running_mean, combined.mean(0))
    torch.testing.assert_close(rms.running_var, combined.var(0, correction=0))
    assert rms.count == 10


def test_eval_preserves_loaded_moments_and_count():
    trained = RunningMeanStd(2)
    trained(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    restored = RunningMeanStd(2).eval()
    restored.load_state_dict(trained.state_dict())
    before = {name: value.clone() for name, value in restored.state_dict().items()}
    sample = torch.tensor([[5.0, 6.0]])
    actual = restored(sample)
    expected = (sample - before["running_mean"].float()) / torch.sqrt(
        before["running_var"].float() + restored.epsilon
    )
    torch.testing.assert_close(actual, expected)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, before[name])
