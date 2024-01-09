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
S = TypeVar("S")


def get_type_chain(target_subclass: type, target_super_class: type) -> List[type]:
    """Finds an inheritance chain from the current sub-type to the target super-type.

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
        """Inserts the given instance into the set of managed instances. This must be a sub-class
        of the type that this type group manages.

        Args:
            instance (T): The instance to add.
        """
        if instance.__class__ in self._instances_by_type:
            # XXX: consider allowing multiple instances of the same type.
            warnings.warn("Duplicate item added", category=UserWarning, stacklevel=1)
            return
        self._instances.append(instance)
        self._instances_by_type[instance.__class__] = instance
        # pytype: disable=attribute-error
        self._instances_by_id[instance.__class__.__name__] = instance
        # pytype: enable=attribute-error

    def remove(self, instance: T) -> Optional[T]:
        """Removes the given instance from this type group.

        Args:
            instance (T): The instance to remove.
        """
        try:
            self._instances.remove(instance)
            self._instances_by_type.pop(instance.__class__)
            # pytype: disable=attribute-error
            self._instances_by_id.pop(instance.__class__.__name__)
            # pytype: enable=attribute-error
        except (KeyError, ValueError):
            return None
        return instance

    def remove_instance_by_id(self, instance_id: str) -> Optional[T]:
        """Attempt to remove the instance by id.

        Args:
            instance_id (str): The name of the instance to remove.

        Returns:
            Optional[T]: The instance that was removed.
        """
        instance = self._instances_by_id.get(instance_id)
        if instance is not None:
            self.remove(instance)
        return instance

    def remove_instance_by_type(self, instance_type: Type[T]) -> Optional[T]:
        """Attempt to remove an instance by its type.

        Args:
            instance_type (Type[T]): The type of the instance to remove.

        Returns:
            Optional[T]: The instance that was removed.
        """
        instance = self._instances_by_type.get(instance_type)
        if instance is not None:
            self.remove(instance)
        return instance

    def get_instance_by_id(self, instance_id: str) -> Optional[T]:
        """Find an instance with the given id.

        Args:
            instance_id (str): The name of the instance.

        Returns:
            Optional[T]: An instance with the given name if found or else `None`.
        """
        return self._instances_by_id.get(instance_id)

    def get_instance_by_type(self, instance_type: Type[T]) -> Optional[T]:
        """Find an instance of the given type.

        Args:
            instance_type (Type[T]): The type of the instance to find.

        Returns:
            Optional[T]: An instance of that type if found or else `None`.
        """
        return self._instances_by_type.get(instance_type)

    @property
    def instances(self) -> List[T]:
        """All instances currently managed in this type group."""
        return self._instances


class TypeSuite(Generic[T]):
    """A utility that manages sub-classes of the given base type.

    Args:
        base_type (Type[T]): The base type this suite will manage.
    """

    def __init__(self, base_type: Type[T]) -> None:
        self._type_groups: Dict[Type[T], _TypeGroup] = defaultdict(_TypeGroup)
        self._associated_groups: Dict[T, List[Type[T]]] = defaultdict(list)
        self._base_type = base_type

    def clear_type(self, type_to_clear: Type):
        """Clear all instances of the given type from this suite. This includes
        all sub-classes. This should be an sub-class of the type that this suite
        manages.

        Args:
            type_to_clear (Type[S]): The type to clear.
        """
        self._assert_is_managed(type_to_clear)
        t = self._associated_groups.get(type_to_clear)
        if t is None:
            return
        for inst in self._type_groups[type_to_clear].instances[:]:
            self.remove(inst)

    def insert(self, instance: T):
        """Adds the instance to the suite of managed instances.

        Args:
            instance (T): The instance to add.
        """
        t = instance.__class__
        if t not in self._associated_groups:
            types = get_type_chain(t, self._base_type)
            self._associated_groups[t] = types
        for group_type in self._associated_groups[t]:
            self._type_groups[group_type].insert(instance)

    def remove(self, instance: T) -> T:
        """Removes the given instance from the suite.

        Args:
            instance (T): The instance to remove.

        Returns:
            T: The removed instance.
        """
        t = instance.__class__
        for group_type in self._associated_groups[t]:
            self._type_groups[group_type].remove(instance)
        return instance

    def remove_by_name(self, instance_id: str) -> T:
        """Attempts to remove an instance from the suite by its name.

        Args:
            instance_id (str): The instance to remove from the suite.

        Returns:
            T: The instance that was removed.
        """
        assert isinstance(instance_id, str)
        instance = self._type_groups[self._base_type].get_instance_by_id(instance_id)
        return self.remove(instance=instance)

    def remove_by_type(self, requested_type: Type[T]) -> T:
        """Attempts to remove an instance from the suite by its type.

        Args:
            requested_type (Type[T]): The type of instance to remove.

        Raises:
            TypeError: The type is not a sub-class of the type this suite manages.

        Returns:
            T: The instance that was removed.
        """
        self._assert_is_managed(requested_type=requested_type)
        instance = self._type_groups[self._base_type].get_instance_by_type(
            requested_type
        )
        return self.remove(instance=instance)

    def get_by_id(self, instance_id: str) -> Optional[T]:
        """Get an instance by its name.

        Args:
            instance_id (str): The name of the instance to retrieve.

        Returns:
            Optional[T]: The instance if it exists.
        """
        assert isinstance(instance_id, str), "Id must be a string."
        return self._type_groups[self._base_type].get_instance_by_id(instance_id)

    def get_by_type(self, requested_type: Type[T]) -> Optional[T]:
        """Get an instance of the exact given type.

        Args:
            requested_type (Type[T]): The type of instance to find.

        Raises:
            TypeError: The type is not a sub-class of the type this suite manages.

        Returns:
            Optional[T]: The instance if it exists.
        """
        self._assert_is_managed(requested_type=requested_type)
        return self._type_groups[self._base_type].get_instance_by_type(
            instance_type=requested_type
        )

    def get_all_by_type(self, requested_type: Type[S]) -> List[S]:
        """Gets all instances that are a sub-type of the given type.

        Args:
            requested_type (Type[T]): The type to query for.

        Raises:
            TypeError: The type is not a sub-class of the type this suite manages.

        Returns:
            Optional[T]:
        """
        self._assert_is_managed(requested_type=requested_type)
        return self._type_groups[requested_type].instances

    def _assert_is_managed(self, requested_type):
        if not issubclass(requested_type, self._base_type):
            raise TypeError(f"{requested_type} must be a subclass of {self._base_type}")

    @property
    def instances(self) -> List[T]:
        """Gets all instances that this suite manages. This will contain all instances that
        that are instances of the base class T.

        Returns:
            List[T]: A list of instances this suite manages.
        """
        if len(self._type_groups) == 0:
            return []
        return self._type_groups[self._base_type].instances
