# MIT License
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
from dataclasses import dataclass

# TODO MTA: Start to use this
@dataclass(frozen=True)
class SimulationGlobalConstants:
    """This is state that should not ever change."""

    OBSERVATION_WORKERS: int

    _SMARTS_ENVIRONMENT_PREFIX: str = "SEV_"

    @classmethod
    def from_environment(cls, environ):
        """This is intended to be used in the following way:
        >>> sgc = SimulationGlobalConstants.from_environment(os.environ)
        """
        SEV = cls._SMARTS_ENVIRONMENT_PREFIX

        def environ_get(NAME, data_type, default):
            nonlocal SEV
            assert isinstance(default, data_type)
            return data_type(environ.get(f"{SEV}{NAME}", default))

        # TODO MTA: consider a different option where defaults are in the object:
        # and the typehints are used to determine the type
        # cls(
        #   **environ_get_all(cls._SMARTS_ENVIRONMENT_PREFIX)
        # )
        return cls(
            OBSERVATION_WORKERS=environ_get("OBSERVATION_WORKERS", int, 0),
        )
