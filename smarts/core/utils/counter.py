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
from typing import Any, Dict


class ReferenceCounter:
    def __init__(self, on_remove_last=lambda v: None) -> None:
        self._entries: Dict[Any, int] = {}
        self._on_remove_last = on_remove_last

    def increment(self, value: Any) -> int:
        if not hasattr(value, "__hash__"):
            assert ValueError(f"Value {value} is not hashable")
        count = self._entries.get(value, 0) + 1
        self._entries[value] = count

        return count

    def decrement(self, value: Any) -> int:
        if not hasattr(value, "__hash__"):
            assert ValueError(f"Value {value} is not hashable")
        count = self._entries.get(value, 0) - 1
        if count < 1:
            self._on_remove_last(value)
            del self._entries[value]
        else:
            self._entries[value] = count

        return count

    def count(self, value: Any):
        return self._entries.get(value, 0)

    def clear(self):
        self._entries.clear()
