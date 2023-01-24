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

from .. import models


def load_controller_params(controller_filepath: str):
    """Load a controller parameters file."""
    if (controller_filepath is None) or not os.path.exists(controller_filepath):
        with pkg_resources.path(
            models, "controller_parameters.yaml"
        ) as controller_path:
            controller_filepath = str(controller_path.absolute())
    with open(controller_filepath, "r", encoding="utf-8") as controller_file:
        return yaml.safe_load(controller_file)


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
    config = None
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
                c.write(conf)

            c.flush()
            with open(c.name, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
    return config
