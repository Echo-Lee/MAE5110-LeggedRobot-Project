"""Load the group-specific deployment adapter selected by GO2_CUSTOM_IMPL."""

import importlib.util
import os
from pathlib import Path


def get_custom_impl_module(name=None):
    name = name or os.environ.get("GO2_CUSTOM_IMPL", "xuyi_chenyu_thomas.py")
    path = Path(__file__).resolve().parents[1] / "custom_implementation" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom implementation {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


custom_impl = get_custom_impl_module()
