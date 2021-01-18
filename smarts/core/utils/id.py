# MIT License
#
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""A helper to ensure consistent naming of IDs within SMARTS Platform."""
import uuid


class Id(str):
    def __init__(self, dtype: str, identifier: str):
        self._dtype = dtype
        self._identifier = identifier

    def __new__(cls, dtype: str, identifier: str):
        return super(Id, cls).__new__(cls, f"{dtype}-{identifier}")

    def __getnewargs__(self):
        return (self._dtype, self._identifier)

    @classmethod
    def new(cls, dtype: str):
        """E.g. boid-93572825"""
        return cls(dtype=dtype, identifier=str(uuid.uuid4())[:8])

    @classmethod
    def parse(cls, id_: str):
        split = -8 - 1  # should be "-"
        if id_[split] != "-":
            raise ValueError(
                f"id={id_} is invalid, format should be <type>-<8_char_uuid>"
            )

        return cls(dtype=id_[:split], identifier=id_[split + 1 :])

    @property
    def dtype(self):
        return self._dtype


class SocialAgentId(Id):
    """
    >>> SocialAgentId.new("keep-lane", group="all")
    'social-agent-all-keep-lane'
    >>> isinstance(SocialAgentId.new("keep-lane"), str)
    True
    """

    DTYPE = "social-agent"

    @classmethod
    def new(cls, name: str, group: str = None):
        identifier = "-".join([group, name]) if group is not None else name
        return cls(dtype=SocialAgentId.DTYPE, identifier=identifier)
