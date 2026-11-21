"""Give a trained actor new inputs, or remove some, without retraining.

The first layer gets zero weights for the new input columns, so the widened
actor's actions are identical to the original's whatever the new inputs
read. Inputs listed in ``drop`` lose their columns; since the actor
normalizes each input first, that is exactly the original actor fed the
dropped input's running mean. The running input normalization keeps the
old statistics; the new columns take statistics measured from data
(``new_stats``), because a
normalizer that has already counted millions of samples would barely move
for them during fine-tuning.
"""

import torch

FIRST_LAYER = "NN.0.weight"


def input_columns(names, dims):
    """{name: (start, end)} column range of each named input."""
    columns, start = {}, 0
    for name, dim in zip(names, dims):
        columns[name] = (start, start + dim)
        start += dim
    return columns


def widen_actor_state(state, old_inputs, new_inputs, new_stats=None, drop=()):
    """Copy of an actor ``state_dict`` taking ``new_inputs``.

    ``old_inputs`` and ``new_inputs`` are lists of (name, dim). Every old
    input not named in ``drop`` must appear in the new list with the same
    size, in any position. ``new_stats`` maps an added input's name to
    (mean, var) tensors.
    """
    old_names = [name for name, _ in old_inputs]
    new_names = [name for name, _ in new_inputs]
    old_cols = input_columns(old_names, [dim for _, dim in old_inputs])
    new_cols = input_columns(new_names, [dim for _, dim in new_inputs])
    drop = set(drop)
    unknown = drop - set(old_names)
    if unknown:
        raise ValueError(f"cannot drop inputs the actor lacks: {sorted(unknown)}")
    if drop & set(new_names):
        raise ValueError(
            f"dropped inputs are still listed: {sorted(drop & set(new_names))}"
        )
    missing = set(old_names) - set(new_names) - drop
    if missing:
        raise ValueError(f"cannot drop actor inputs {sorted(missing)} (not in drop)")
    kept = [name for name in old_names if name not in drop]
    for name in kept:
        if old_cols[name][1] - old_cols[name][0] != (
            new_cols[name][1] - new_cols[name][0]
        ):
            raise ValueError(f"input {name} changed size")
    width = sum(dim for _, dim in new_inputs)
    weight = state[FIRST_LAYER]
    if weight.shape[1] != sum(dim for _, dim in old_inputs):
        raise ValueError("old inputs do not match the actor's first layer")

    widened = {key: value.clone() for key, value in state.items()}
    new_weight = torch.zeros(weight.shape[0], width, dtype=weight.dtype)
    has_rms = "obs_rms.running_mean" in state
    if has_rms:
        mean = torch.zeros(width, dtype=state["obs_rms.running_mean"].dtype)
        var = torch.ones(width, dtype=state["obs_rms.running_var"].dtype)
    for name in kept:
        (a, b), (c, d) = old_cols[name], new_cols[name]
        new_weight[:, c:d] = weight[:, a:b]
        if has_rms:
            mean[c:d] = state["obs_rms.running_mean"][a:b]
            var[c:d] = state["obs_rms.running_var"][a:b]
    for name, (low, high) in new_cols.items():
        if name in old_cols or not has_rms or not new_stats or name not in new_stats:
            continue
        stat_mean, stat_var = new_stats[name]
        mean[low:high] = torch.as_tensor(stat_mean, dtype=mean.dtype)
        var[low:high] = torch.as_tensor(stat_var, dtype=var.dtype)
    widened[FIRST_LAYER] = new_weight
    if has_rms:
        widened["obs_rms.running_mean"] = mean
        widened["obs_rms.running_var"] = var
    return widened
