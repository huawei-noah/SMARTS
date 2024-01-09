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
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

import smarts.assets.vehicles
from smarts.core import config


def load_vehicle_definitions_list(vehicle_list_filepath: Optional[str]):
    """Load a vehicle definition list file."""
    if (vehicle_list_filepath is None) or not os.path.exists(vehicle_list_filepath):
        vehicle_list_filepath = config()("assets", "default_vehicle_definitions_list")
    vehicle_list_filepath = Path(vehicle_list_filepath).absolute()

    return VehicleDefinitions(
        data=load_yaml_config_with_substitution(vehicle_list_filepath),
        filepath=vehicle_list_filepath,
    )


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


def load_yaml_config_with_substitution(
    path: Union[str, Path]
) -> Optional[Dict[str, Any]]:
    """Read in a yaml configuration to dictionary format replacing instances of ${{module}} with
    module's file path and ${} with the SMARTS environment variable."""
    smarts_config = config()
    out_config = None
    if isinstance(path, str):
        path = Path(path)
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


@dataclass(frozen=True)
class VehicleDefinitions:
    """This defines a set of vehicle definitions and loading utilities."""

    data: Dict[str, Any]
    """The data associated with the vehicle definitions. This is generally vehicle type keys."""
    filepath: Union[str, Path]
    """The path to the vehicle definitions file."""

    def __post_init__(self):
        if isinstance(self.filepath, Path):
            object.__setattr__(self, "filepath", self.filepath.__str__())

    @lru_cache(maxsize=20)
    def load_vehicle_definition(self, vehicle_class: str):
        """Loads in a particular vehicle definition."""
        if vehicle_definition_filepath := self.data.get(vehicle_class):
            return load_yaml_config_with_substitution(Path(vehicle_definition_filepath))
        raise OSError(
            f"Vehicle '{vehicle_class}' is not defined in {list(self.data.keys())}"
        )

    @lru_cache(maxsize=20)
    def controller_params_for_vehicle_class(self, vehicle_class: str):
        """Get the controller parameters for the given vehicle type"""
        vehicle_definition = self.load_vehicle_definition(vehicle_class)
        controller_params = Path(vehicle_definition["controller_params"])
        return load_yaml_config_with_substitution(controller_params)

    @lru_cache(maxsize=20)
    def chassis_params_for_vehicle_class(self, vehicle_class: str):
        """Get the controller parameters for the given vehicle type"""
        vehicle_definition = self.load_vehicle_definition(vehicle_class)
        chassis_parms = Path(vehicle_definition["chassis_params"])
        return load_yaml_config_with_substitution(chassis_parms)

    def __hash__(self) -> int:
        return hash(self.filepath)
