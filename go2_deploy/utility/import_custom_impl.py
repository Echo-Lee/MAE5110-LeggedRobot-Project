import importlib.util
from pathlib import Path

def get_custom_impl_module(name="go2trot_base.py"):
    repo_path = Path(__file__).resolve().parent.parent  
    path = Path(repo_path / "custom_implementation" / name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load file at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

custom_impl = get_custom_impl_module()