import importlib.util
from pathlib import Path

from go2_deploy.custom_implementation.custom_config import CustomConfig

CUSTOM_IMPL_DIR = Path(__file__).resolve().parent.parent / "custom_implementation"
INTERFACE_FILE = "custom_config.py"


def _load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load file at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _implementations(module):
    # Only classes defined in this file, not CustomConfig or imported ones
    return [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, CustomConfig)
        and obj is not CustomConfig
        and obj.__module__ == module.__name__
    ]


def load_custom_impl(task_name):
    """Instance of the CustomConfig subclass in custom_implementation/ whose
    task_name matches. Raises if there is not exactly one."""
    if not task_name:
        raise ValueError(f"Must specify a task name, got {task_name!r}")

    matches, errors = [], []
    for path in sorted(CUSTOM_IMPL_DIR.glob("*.py")):
        if path.name == INTERFACE_FILE:
            continue
        try:
            module = _load_module(path)
        except Exception as error:  # report, so one broken file can't hide others
            errors.append(f"  {path.name}: {type(error).__name__}: {error}")
            continue
        matches += [
            (path.name, cls)
            for cls in _implementations(module)
            if cls.task_name == task_name
        ]

    if len(matches) > 1:
        found = ", ".join(f"{cls.__name__} ({name})" for name, cls in matches)
        raise ValueError(f"Multiple custom implementations for '{task_name}': {found}")
    if not matches:
        message = f"No CustomConfig subclass with task_name '{task_name}' in {CUSTOM_IMPL_DIR}"
        if errors:
            message += "\nThese files failed to load:\n" + "\n".join(errors)
        raise LookupError(message)
    return matches[0][1]()
