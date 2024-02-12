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
import ast
import configparser
import functools
import logging
import os
import pathlib
import re
import warnings
from typing import Any, Callable, Final, List, Optional, Union

import smarts

_UNSET = object()

logger = logging.getLogger(__name__)


def _passthrough_cast(val):
    return val


def _convert_truthy(t: str) -> bool:
    """Convert value to a boolean. This should only allow ([Tt]rue)|([Ff]alse)|[\\d].

    This is necessary because bool("false") == True.
    Args:
        t (str): The value to convert.

    Returns:
        bool: The truth value.
    """
    # ast literal_eval will parse python literals int, str, e.t.c.
    out = ast.literal_eval(t.strip().title())
    assert isinstance(out, (bool, int))
    return bool(out)


_assets_path = os.path.join(list(smarts.__path__)[0], "assets")
_config_defaults: Final = {
    ("assets", "path"): _assets_path,
    ("assets", "default_agent_vehicle"): "sedan",
    ("assets", "default_vehicle_definitions_list"): os.path.join(
        _assets_path, "vehicles/vehicle_definitions_list.yaml"
    ),
    ("core", "observation_workers"): 0,
    ("core", "max_custom_image_sensors"): 32,
    ("core", "sensor_parallelization"): "mp",
    ("core", "debug"): False,
    ("core", "reset_retries"): 0,
    ("physics", "max_pybullet_freq"): 240,
    ("ray", "num_gpus"): 0,
    ("ray", "num_cpus"): None,
    ("ray", "log_to_driver"): False,
    ("sumo", "central_port"): 8619,
    ("sumo", "central_host"): "localhost",
    ("sumo", "traci_serve_mode"): "local",  # local|central
    ("traffic", "traci_retries"): 5,
    ("visdom", "enabled"): False,
    ("visdom", "hostname"): "http://localhost",
    ("visdom", "port"): 8097,
}


class Config:
    """A configuration utility that handles configuration from file and environment variable.

    Args:
        config_file (Union[str, pathlib.Path]): The path to the configuration file.
        environment_prefix (str, optional): The prefix given to the environment variables. Defaults to "".

    Raises:
        FileNotFoundError: If the configuration file cannot be found at the given file location.
    """

    def __init__(
        self, config_file: Union[str, pathlib.Path], environment_prefix: str = ""
    ) -> None:
        self._config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        self._environment_prefix = environment_prefix.upper()
        self._environment_variable_format_string = (
            self._environment_prefix + "_{}_{}" if self._environment_prefix else "{}_{}"
        )
        self.env_variable_substitution_pattern = re.compile(r"\$\{(.+)\}")

        if isinstance(config_file, str):
            config_file = pathlib.Path(config_file)
        config_file = config_file.resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Configuration file not found at {config_file}")

        self._config.read(str(config_file.absolute()))
        logger.info(msg=f"Using engine configuration from: {config_file.absolute()}")

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
        cast: Callable[[Any], Any] = _passthrough_cast,
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
        env_variable = self._environment_variable_format_string.format(
            section.upper(), option.upper()
        )
        setting = os.getenv(env_variable)
        if cast is bool:
            # This is necessary because bool("false") == True.
            cast = _convert_truthy
        if setting is not None:
            return cast(setting)
        try:
            value = self._config[section][option]
        except (configparser.NoSectionError, KeyError) as exc:
            if default is _UNSET:
                if (value := _config_defaults.get((section, option), _UNSET)) != _UNSET:
                    return value
                raise EnvironmentError(
                    f"Setting `${env_variable}` cannot be found in environment or configuration."
                ) from exc
            return default
        return cast(value)

    def substitute_settings(self, input: str, source: Optional[str] = "") -> str:
        """Given a string, substitutes in configuration settings if they exist."""

        m: List[str] = self.env_variable_substitution_pattern.findall(input)

        if not m:
            return input
        output = input
        for val in set(m):
            if self.environment_prefix:
                environment_prefix, _, setting = val.partition("_")
                if environment_prefix != self.environment_prefix:
                    warnings.warn(
                        f"Unable to substitute environment variable `{val}` from `{source}`"
                    )
                    continue
            else:
                setting = val

            section, _, option_name = setting.lower().partition("_")
            env_value = self(section, option_name)

            output = output.replace(f"${{{val}}}", env_value)
        return output

    def __call__(
        self,
        section: str,
        option: str,
        /,
        default: Any = _UNSET,
        cast: Callable[[str], Any] = str,
    ) -> Optional[Any]:
        return self.get_setting(section, option, default, cast)

    def __repr__(self) -> str:
        return f"Config(config_file={ {k: dict(v.items()) for k, v in self._config.items(raw=True)} }, environment_prefix={self._environment_prefix})"
