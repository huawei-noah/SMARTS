# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import importlib.resources as pkg_resources
import os
import re
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

import smarts.assets.vehicles
from smarts.core import config


def load_vehicle_list(vehicle_list_filepath: str):
    """Load a vehicle definition list file."""
    if (vehicle_list_filepath is None) or not os.path.exists(vehicle_list_filepath):
        with pkg_resources.path(smarts.assets.vehicles, "vehicle_list.yaml") as vd_path:
            vehicle_list_filepath = str(vd_path.absolute())
    return load_yaml_config_with_substitution(Path(vehicle_list_filepath))


def load_vehicle_definition(vehicle_definition_filepath: str):
    """Load a vehicle definition file."""
    return load_yaml_config_with_substitution(Path(vehicle_definition_filepath))


def load_yaml_config(path: Path) -> Optional[Dict[str, Any]]:
    """Read in a yaml configuration to dictionary format."""
    config = None
    if path.exists():
        assert path.suffix in (".yaml", ".yml"), f"`{str(path)}` is not a YAML file."
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    return config


def _replace_with_module_path(base: str, module_str: str):
    spec = find_spec(module_str)
    if not spec:
        raise RuntimeError(
            f"Spec cannot be found for `${{{module_str}}}` in config:\n{base}"
        )
    origin = spec.origin
    if origin.endswith("__init__.py"):
        origin = origin[: -len("__init__.py")]
    return base.replace(f"${{{{{module_str}}}}}", origin)


def load_yaml_config_with_substitution(path: Path) -> Optional[Dict[str, Any]]:
    """Read in a yaml configuration to dictionary format replacing instances of ${{module}} with
    module's file path."""
    smarts_config = config()
    out_config = None
    if path.exists():
        assert path.suffix in (".yaml", ".yml"), f"`{str(path)}` is not a YAML file."
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=path.parent) as c:
            with open(str(path), "r", encoding="utf-8") as o:
                pattern = re.compile(r"\$\{\{(.+)\}\}")
                conf = o.read()
                match = pattern.findall(conf)
                if match:
                    for val in match:
                        conf = _replace_with_module_path(conf, val)
                conf = smarts_config.substitute_settings(conf, path.__str__())
                c.write(conf)

            c.flush()
            with open(c.name, "r", encoding="utf-8") as file:
                out_config = yaml.safe_load(file)
    return out_config
