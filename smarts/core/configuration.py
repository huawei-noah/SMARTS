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
from pathlib import Path
from typing import Any, Callable, Optional, Union

_UNSET = object()


class Config:
    """A configuration utility that handles configuration from file and environment variable.

    Args:
        config_file (Union[str, Path]): The path to the configuration file.
        environment_prefix (str, optional): The prefix given to the environment variables. Defaults to "".

    Raises:

    """

    def __init__(
        self, config_file: Union[str, Path], environment_prefix: str = ""
    ) -> None:
        if isinstance(config_file, str):
            config_file = Path(config_file)
        if not config_file.is_file():
            raise FileNotFoundError(f"Configuration file not found at {config_file}")

        self._config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        self._config.read(str(config_file.absolute()))
        self._environment_prefix = environment_prefix.upper()
        self._format_string = self._environment_prefix + "_{}_{}"

    @property
    def environment_prefix(self):
        """The prefix that environment variables configuration is provided with."""
        return self._environment_prefix

    @functools.lru_cache(maxsize=100)
    def get_setting(
        self,
        section: str,
        option: str,
        default: Any = _UNSET,
        cast: Callable[[str], Any] = str,
    ) -> Optional[Any]:
        """Finds the given configuration checking the following in order: environment variable,
        configuration file, and default.

        Args:
            section (str): The grouping that the configuration option is under.
            option (str): The specific configuration option.
            default (Any, optional): The default if the requested configuration option is not found. Defaults to _UNSET.
            cast (Callable, optional): A function that takes a string and returns the desired type. Defaults to str.


        Returns:
            Optional[str]: The value of the configuration.

        Raises:
            KeyError: If the configuration option is not found in the configuration file and no default is provided.
            configparser.NoSectionError: If the section in the configuration file is not found and no default is provided.
        """
        env_variable = self._format_string.format(section.upper(), option.upper())
        setting = os.getenv(env_variable)
        if setting is not None:
            return cast(setting)
        try:
            value = self._config[section][option]
        except (configparser.NoSectionError, KeyError):
            if default is _UNSET:
                raise
            return default
        return cast(value)
