import importlib.util
import os
from pathlib import Path

# The team's implementation in go2_deploy/custom_implementation/. Set
# GO2_CUSTOM_IMPL to deploy another file from that folder (e.g. go2trot_base.py).
DEFAULT_CUSTOM_IMPL = "go2osc_push_glide.py"


def get_custom_impl_module(name=None):
    if name is None:
        name = os.environ.get("GO2_CUSTOM_IMPL", DEFAULT_CUSTOM_IMPL)
    repo_path = Path(__file__).resolve().parent.parent
    path = Path(repo_path / "custom_implementation" / name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load file at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

custom_impl = get_custom_impl_module()
