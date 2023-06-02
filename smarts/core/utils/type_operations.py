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
import warnings
from collections import defaultdict
from inspect import getclasstree
from typing import Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar("T")


def get_type_chain(target_subclass: type, target_super_class: type):
    """Finds an inheritance chain from the current subtype to the target super type.

    Args:
        target_subclass (type): The subclass type.
        target_super_class (type): The superclass type.

    Returns:
        List[type]: The inheritance chain from the current class to the target superclass.
    """
    if not issubclass(target_subclass, target_super_class):
        raise TypeError(f"The first type must be a subclass of the second type.")

    if target_subclass is target_super_class:
        return [target_subclass]

    result = []
    q = []
    seen = {target_subclass}

    def unpack_super_types(subtype):
        class_tree = getclasstree([subtype])
        types = []
        for i in range(0, len(class_tree), 2):
            parent = class_tree[i]
            t = parent[0]
            types.insert(0, t)
        return types

    q.extend(
        (
            ([target_subclass, next_super_type], next_super_type)
            for next_super_type in unpack_super_types(target_subclass)
        )
    )
    while len(q) > 0:
        subtypes, next_super_type = q.pop()
        if next_super_type in seen:
            continue
        seen.add(next_super_type)
        # found the correct inheritance chain
        if next_super_type == target_super_class:
            result = subtypes
            break

        q.extend(
            (
                (subtypes + [next_super_type], next_super_type)
                for next_super_type in unpack_super_types(subtypes[-1])
            )
        )

    return result


class _TypeGroup(Generic[T]):
    def __init__(self) -> None:
        self._instances = []
        self._instances_by_type = {}
        self._instances_by_id = {}

    def insert(self, instance: T):
        if instance.__class__ in self._instances_by_type:
            # XXX: consider allowing multiple instances of the same type.
            warnings.warn("Duplicate item added", category=UserWarning, stacklevel=1)
            return
        self._instances.append(instance)
        self._instances_by_type[instance.__class__] = instance
        self._instances_by_id[instance.__class__.__name__] = instance

    def remove(self, instance: T) -> Optional[T]:
        try:
            self._instances.remove(instance)
            self._instances_by_type.pop(instance.__class__)
            self._instances_by_id.pop(instance.__class__.__name__)
        except (KeyError, ValueError):
            return None
        return instance

    def remove_instance_by_id(self, instance_id: str) -> Optional[T]:
        provider = self._instances_by_id.get(instance_id)
        if provider is not None:
            self.remove(provider)
        return provider

    def remove_instance_by_type(self, instance_type: Type[T]) -> Optional[T]:
        provider = self._instances_by_type.get(instance_type)
        if provider is not None:
            self.remove(provider)
        return provider

    def get_instance_by_id(self, provider_id: str) -> Optional[T]:
        return self._instances_by_id.get(provider_id)

    def get_instance_by_type(self, requested_type: Type[T]) -> Optional[T]:
        return self._instances_by_type.get(requested_type)

    @property
    def instances(self) -> List[T]:
        return self._instances


class TypeSuite(Generic[T]):
    """A utility that manages subtypes of the given base type.

    Args:
        base_type (Type[T]): The base type this will manage.
    """

    def __init__(self, base_type: Type[T]) -> None:
        self._type_groups: Dict[Type[T], _TypeGroup] = defaultdict(_TypeGroup)
        self._associated_groups: Dict[T, List[Type[T]]] = defaultdict(list)
        self._base_type = base_type

    def clear_type(self, base_type: Type[T]):
        t = self._associated_groups.get(base_type)
        if t is None:
            return
        for inst in self._type_groups[base_type].instances[:]:
            self.remove(inst)

    def insert(self, instance: T):
        t = instance.__class__
        if t not in self._associated_groups:
            types = get_type_chain(t, self._base_type)
            self._associated_groups[t] = types
        for group_type in self._associated_groups[t]:
            self._type_groups[group_type].insert(instance)

    def remove(self, instance: T) -> T:
        t = instance.__class__
        for group_type in self._associated_groups[t]:
            self._type_groups[group_type].remove(instance)
        return instance

    def remove_by_name(self, instance_id: str) -> T:
        assert isinstance(instance_id, str)
        instance = self._type_groups[self._base_type].get_instance_by_id(instance_id)
        return self.remove(instance=instance)

    def remove_by_type(self, requested_type: Type[T]) -> T:
        if not issubclass(requested_type, self._base_type):
            raise TypeError(f"{requested_type} must be a subclass of {self._base_type}")
        instance = self._type_groups[self._base_type].get_instance_by_type(
            requested_type
        )
        return self.remove(instance=instance)

    def get_by_id(self, instance_id: str) -> Optional[T]:
        return self._type_groups[self._base_type].get_instance_by_id(instance_id)

    def get_by_type(self, requested_type: Type[T]) -> Optional[T]:
        if not issubclass(requested_type, self._base_type):
            raise TypeError(f"{requested_type} must be a subclass of {self._base_type}")
        return self._type_groups[self._base_type].get_instance_by_type(
            requested_type=requested_type
        )

    def get_all_by_type(self, requested_type: Type[T]) -> Optional[T]:
        if not issubclass(requested_type, self._base_type):
            raise TypeError(f"{requested_type} must be a subclass of {self._base_type}")
        return self._type_groups[requested_type].instances

    @property
    def instances(self) -> List[T]:
        if len(self._type_groups) == 0:
            return []
        return self._type_groups[self._base_type].instances
