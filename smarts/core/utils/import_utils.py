import importlib.util
import sys
from pathlib import Path


def import_module_from_file(module_name, path: Path):
    path = str(path)  # Get one directory up
    if path not in sys.path:
        sys.path.append(path)

    spec = importlib.util.spec_from_file_location(module_name, f"{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
