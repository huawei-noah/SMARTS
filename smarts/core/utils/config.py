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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from typing import Dict, Set

_valid_configs: Dict[str, Set[str]] = {
    "renderer-debug-mode": {"spam", "debug", "info", "warning", "error"}
}


def _validate_config(config_dict):
    for key, value in config_dict.items():
        assert (
            key in _valid_configs
        ), f"Invalid SMARTS simulation configuration: `{key}`"
        assert (
            value in _valid_configs[key]
        ), f"Invalid SMARTS configuration: `{key}`:=`{value}`\nValid configurations are: `{_valid_configs[key]}`"


class SmartsConfig:
    """"""

    def __init__(self) -> None:
        self._configurations = dict()

    @staticmethod
    def from_dictionary(config_dict):
        config = SmartsConfig()
        config.update_config(config_dict)
        return config

    def update_config(self, config_dict):
        _validate_config(config_dict=config_dict)
        self._configurations.update(config_dict)

    def get(self, key, default=None):
        return self._configurations.get(key, default)
