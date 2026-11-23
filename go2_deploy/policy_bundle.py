"""A policy bundle keeps a network together with everything that must match it.

Layout of a bundle directory:

    policy.pt           TorchScript actor (obs normalization traced in)
    deploy_meta.json    obs layout, scaling, posture, gains, provenance, hash

Deploying from a bundle means the obs/action scaling can never drift away
from the weights it was trained with, even if the training config changes.
"""

import copy
import hashlib
import json
import math
import os
from datetime import datetime

import torch

from go2_deploy.deploy_config import DeployConfig
from go2_deploy.utility.import_custom_impl import load_custom_impl

META_FILE = "deploy_meta.json"
POLICY_FILE = "policy.pt"
NUM_ACTIONS = 12
ACTION_SCALE_KEY = "dof_pos_target"


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _as_number_or_list(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return float(value)


def _scaling_snapshot(cfg):
    """Every scale the obs builder and action path read, as plain numbers."""
    keys = list(cfg.obs_vector) + [ACTION_SCALE_KEY]
    return {
        key: _as_number_or_list(getattr(cfg.DeployScaling, key, 1.0)) for key in keys
    }


def _save_torchscript(actor, path):
    # Trace a CPU eval copy: eval() freezes the obs_rms statistics in the graph.
    # (Actor.export also writes ONNX, which needs onnxscript; the robot doesn't.)
    model = copy.deepcopy(actor).to("cpu").eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, torch.zeros(actor.num_obs))
    torch.jit.save(traced, path)


def export_bundle(actor, cfg, out_dir, provenance=None):
    """Write actor + cfg as a bundle. Returns the path to deploy_meta.json."""
    if actor.num_obs != cfg.obs_dim():
        raise ValueError(
            f"actor takes {actor.num_obs} obs, but cfg.obs_vector gives {cfg.obs_dim()}"
        )
    os.makedirs(out_dir, exist_ok=True)
    policy_path = os.path.join(out_dir, POLICY_FILE)
    _save_torchscript(actor, policy_path)
    meta = {
        "task_name": cfg.task_name,
        "obs_vector": list(cfg.obs_vector),
        "obs_sizes": {obs: cfg.obs_sizes[obs] for obs in cfg.obs_vector},
        "scaling": _scaling_snapshot(cfg),
        "default_dof_pos": _as_number_or_list(cfg.default_dof_pos),
        "ctrl_freq": cfg.ctrl_freq,
        "exp_moving_avg": cfg.exp_moving_avg,
        "ema_time_const": cfg.ema_time_const,
        "kp": cfg.kp,
        "kd": cfg.kd,
        "command_limits": dict(cfg.command_limits),
        "num_obs": actor.num_obs,
        "num_actions": actor.num_actions,
        "policy_sha256": _sha256(policy_path),
        "provenance": {
            "exported_at": datetime.now().isoformat(),
            **(provenance or {}),
        },
    }
    meta_path = os.path.join(out_dir, META_FILE)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def config_from_meta(meta, custom_impl=None):
    """DeployConfig with every policy-coupled field taken from the bundle.

    Timing and filtering use the bundle's recorded values, or the task's
    explicit defaults for older bundles. custom_impl supplies the custom
    buffers/obs pieces (default: the one for the bundle's task_name).
    """
    if custom_impl is None:
        custom_impl = load_custom_impl(meta["task_name"])
    elif custom_impl.task_name != meta["task_name"]:
        raise ValueError(
            f"bundle is for task '{meta['task_name']}', "
            f"but custom_impl is for '{custom_impl.task_name}'"
        )
    cfg = DeployConfig(custom_impl)
    # The custom configuration defines observations; a bundle records and
    # verifies that contract rather than silently replacing it.
    if list(meta["obs_vector"]) != cfg.obs_vector:
        raise ValueError("bundle observation order differs from CustomConfig")
    for name in cfg.obs_vector:
        if meta["obs_sizes"][name] != cfg.obs_sizes[name]:
            raise ValueError(
                f"bundle observation size for {name} differs from CustomConfig"
            )
    for name in cfg.obs_vector + [ACTION_SCALE_KEY]:
        expected = torch.as_tensor(
            getattr(cfg.DeployScaling, name, 1.0), dtype=torch.float
        )
        recorded = torch.as_tensor(meta["scaling"][name], dtype=torch.float)
        if not torch.equal(expected, recorded):
            raise ValueError(
                f"bundle observation scale for {name} differs from CustomConfig"
            )
    if not torch.equal(
        cfg.default_dof_pos, torch.tensor(meta["default_dof_pos"], dtype=torch.float)
    ):
        raise ValueError("bundle default posture differs from CustomConfig")
    cfg.task_name = meta["task_name"]
    cfg.kp = meta["kp"]
    cfg.kd = meta["kd"]
    cfg.command_limits = dict(meta["command_limits"])
    # Older bundles use the task's explicit runner settings. New exports record
    # timing and filtering alongside gains so they cannot drift independently.
    for name in ("ctrl_freq", "exp_moving_avg", "ema_time_const"):
        if name in meta:
            setattr(cfg, name, meta[name])
    cfg.ema_smoothing_factor = -math.expm1(-(1.0 / cfg.ctrl_freq) / cfg.ema_time_const)

    scaling = meta["scaling"]
    for key in cfg.obs_vector + [ACTION_SCALE_KEY]:
        if key not in scaling:
            raise KeyError(f"bundle has no scaling for '{key}'")
    cfg.bundle_meta = meta
    cfg.validate()
    return cfg


def load_bundle(bundle_dir, custom_impl=None):
    """Returns (policy, cfg). Raises if the bundle is inconsistent."""
    with open(os.path.join(bundle_dir, META_FILE)) as f:
        meta = json.load(f)

    policy_path = os.path.join(bundle_dir, POLICY_FILE)
    if _sha256(policy_path) != meta["policy_sha256"]:
        raise ValueError(f"{policy_path} does not match the hash in {META_FILE}")

    cfg = config_from_meta(meta, custom_impl)
    if cfg.obs_dim() != meta["num_obs"]:
        raise ValueError(
            f"obs_vector adds up to {cfg.obs_dim()} values, "
            f"but the policy expects {meta['num_obs']}"
        )

    policy = torch.jit.load(policy_path, map_location="cpu").eval()
    with torch.no_grad():
        action = policy(torch.zeros(cfg.obs_dim()))
    if action.shape != (NUM_ACTIONS,):
        raise ValueError(f"policy output shape {tuple(action.shape)} != (12,)")

    return policy, cfg
