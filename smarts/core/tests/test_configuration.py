# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import configparser
import functools
import os
import tempfile
from pathlib import Path

import pytest

from smarts.core.configuration import Config


@pytest.fixture
def config_path():
    config = configparser.ConfigParser()
    config["section_1"] = {"string_option": "value", "option_2": "value_2"}
    config["section_2"] = {"bool_option": "True", "float_option": "3.14"}
    with tempfile.TemporaryDirectory() as tmpdir:
        file = Path(tmpdir) / "config.ini"
        with file.open("w") as file_pointer:
            config.write(fp=file_pointer)
        yield str(file)


def test_get_setting_with_file(config_path):
    config = Config(config_path)
    assert config.get_setting("section_1", "string_option") == "value"
    partition_string = functools.partial(
        lambda source, sep: str.partition(source, sep), sep="_"
    )
    assert config.get_setting("section_1", "option_2", cast=partition_string) == (
        "value",
        "_",
        "2",
    )
    assert config.get_setting("section_2", "bool_option", cast=bool) is True
    assert config.get_setting("section_2", "float_option", cast=float) == 3.14
    with pytest.raises(KeyError):
        config.get_setting("nonexistent", "option")


def test_get_setting_with_environment_variables(config_path):
    config = Config(config_path, "smarts")
    assert config.get_setting("nonexistent", "option", default=None) is None

    os.environ["SMARTS_NONEXISTENT_OPTION"] = "now_exists"
    config = Config(config_path, "smarts")
    assert config.get_setting("nonexistent", "option", default=None) == "now_exists"
    del os.environ["SMARTS_NONEXISTENT_OPTION"]


def test_get_missing_section_and_missing_option():
    from smarts.core import config as core_conf

    core_conf.cache_clear()

    config: Config = core_conf()

    with pytest.raises(KeyError):
        config.get_setting("core", "not_a_setting")

    with pytest.raises(KeyError):
        config.get_setting("not_a_section", "bar")
