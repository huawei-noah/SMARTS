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
        config_file = str(config_file)
        if not Path(config_file).is_file():
            raise FileNotFoundError(f"Configuration file not found at {config_file}")

        self._config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation
        )
        self._config.read(config_file)
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
