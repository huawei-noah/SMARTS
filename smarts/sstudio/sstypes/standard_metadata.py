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


from collections.abc import Mapping
from functools import cached_property
from typing import Any, Dict, Iterator, Optional


class StandardMetadata(Mapping):
    """Metadata that does not have direct influence on the simulation."""

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if metadata is None:
            metadata = {}
        self._standard_metadata = tuple(
            (
                setting_key,
                setting_value,
            )
            for setting_key, setting_value in metadata.items()
            if setting_value is not None
        )

    def __iter__(self) -> Iterator:
        return iter(self._standard_metadata)

    def __len__(self) -> int:
        return len(self._standard_metadata)

    def __getitem__(self, __key: Any) -> Any:
        return self._dict_metadata[__key]

    def get(self, __key, __default=None):
        """Retrieve the value or a default.

        Args:
            __key (Any): The key to find.
            __default (Any, optional): The default if the key does not exist. Defaults to None.

        Returns:
            Optional[Any]: The value or default.
        """
        return self._dict_metadata.get(__key, __default)

    def __hash__(self) -> int:
        return self._hash_id

    @cached_property
    def _hash_id(self) -> int:
        return hash(frozenset(self._standard_metadata))

    @cached_property
    def _dict_metadata(self) -> Dict[str, Any]:
        return dict(self._standard_metadata)
