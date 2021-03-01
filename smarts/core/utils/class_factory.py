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
# ====================
# Heavily derived from https://github.com/openai/gym/blob/v0.10.5/gym/envs/registration.py
# See gym license in THIRD_PARTY_OPEN_SOURCE_SOFTWARE_NOTICE
import importlib
import re
from urllib.parse import urlparse

# Taken from OpenAI gym's name constraints
NAME_CONSTRAINT_REGEX = re.compile(r"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$")


def is_valid_locator(locator: str):
    # Handle non-URL-based agents (e.g. open_agent-v0)
    return NAME_CONSTRAINT_REGEX.search(locator)


def find_attribute_spec(name):
    """Finds the attribute specification from a reachable module.
    Args:
        name:
            The module and attribute name (i.e. smarts.core.lidar:Lidar, ...)
    """
    module_name, attribute_name = name.split(":")
    module = importlib.import_module(module_name)
    attribute_spec = getattr(module, attribute_name)
    return attribute_spec


class ClassFactory:
    def __init__(self, name, entrypoint=None, **kwargs):
        self.name = name
        self.entrypoint = entrypoint
        self._kwargs = kwargs

        if self.entrypoint is None:
            raise EnvironmentError(
                f"Entry-point is empty for: '{self.name}'. Provide an entry-point"
            )

    def make(self, **kwargs):
        if self.entrypoint is None:
            raise AttributeError(f"Entry-point does not exist for name `{self.name}`")
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entrypoint):
            instance = self.entrypoint(**_kwargs)
        else:
            type_spec = find_attribute_spec(self.entrypoint)
            instance = type_spec(**_kwargs)

        return instance

    def __repr__(self):
        return f"""ClassFactory(
  name={self.name},
  entrypoint={self.entrypoint},
  kwargs={self._kwargs},
)"""


class ClassRegister:
    def __init__(self):
        self.index = {}

    def register(self, locator, entry_point=None, **kwargs):
        # TODO: locator is being used for both module:name and just name. The former
        #       is the locator, and the latter is simply name. Update the signature of
        #       this method to be register(name, entrypoint, ...)
        name = locator
        if name not in self.index:
            self.index[name] = ClassFactory(locator, entry_point, **kwargs)

    def find_factory(self, locator):
        self._raise_on_invalid_locator(locator)

        mod_name, name = locator.split(":", 1)
        # `name` could be simple name string (e.g. <open_agent-v0> or a URL
        try:
            # Import the module so that the agent may register it self in our self.index
            module = importlib.import_module(mod_name)
        except ImportError:
            raise ImportError(
                f"Ensure that `{mod_name}` module can be found from your "
                f"PYTHONPATH and name=`{locator}` exists (e.g. was registered "
                "manually or downloaded."
            )

        try:
            return self.index[name]
        except KeyError:
            raise NameError(f"Locator not registered in lookup: {locator}")

    def make(self, locator, **kwargs):
        factory = self.find_factory(locator)
        instance = factory.make(**kwargs)
        return instance

    def all(self):
        return self.index.values()

    def _raise_on_invalid_locator(self, locator: str):
        if not is_valid_locator(locator):
            # TODO: Give clearer instructions/examples of the locator syntax
            raise ValueError(
                f"Cannot register invalid locator={locator}. E.g. syntax: "
                '"module:name-v0".'
            )
