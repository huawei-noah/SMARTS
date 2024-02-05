# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
"""smarts.core
===========

Core functionality of the SMARTS simulator
"""

import random
import uuid
from functools import lru_cache, partial
from pathlib import Path

import numpy as np

from smarts.core.configuration import Config

_current_seed = None


def current_seed():
    """Get the last used seed."""
    return _current_seed


def seed(a):
    """Seed common pseudo-random generators."""
    global _current_seed
    _current_seed = a
    random.seed(a)
    np.random.seed(a)


def gen_id():
    """Generates a unique but deterministic id if `smarts.core.seed` has set the core seed."""
    id_ = uuid.UUID(int=random.getrandbits(128))
    return str(id_)[:8]


@lru_cache(maxsize=2)
def config(
    config_path: str = "./smarts_engine.ini", environment_prefix="SMARTS"
) -> Config:
    """Get the SMARTS environment configuration for the smarts engine.

    .. note::

        This searches the following locations and loads the first one it finds:
        Supplied ``config_path``. Default: `./smarts_engine.ini`
        `~/.smarts/engine.ini`
        `/etc/smarts/engine.ini`
        `$PYTHON_PATH/smarts/engine.ini`

    Args:
        config_path (str, optional): The configurable location. Defaults to `./smarts_engine.ini`.

    Returns:
        Config: A configuration utility that allows resolving environment and `engine.ini` configuration.
    """
    from smarts.core.utils.file import smarts_global_user_dir, smarts_local_user_dir

    def get_file(config_file: Path):
        try:
            if not config_file.is_file():
                return ""
        except PermissionError:
            return ""

        return str(config_file)

    conf = partial(Config, environment_prefix=environment_prefix)

    file = get_file(Path(config_path).absolute())
    if file:
        return conf(file)

    try:
        local_dir = smarts_local_user_dir()
    except PermissionError:
        file = ""
    else:
        file = get_file(Path(local_dir) / "engine.ini")
    if file:
        return conf(file)

    global_dir = smarts_global_user_dir()
    file = get_file(Path(global_dir) / "engine.ini")
    if file:
        return conf(file)

    default_path = Path(__file__).parents[1].resolve() / "engine.ini"
    return conf(get_file(default_path))
