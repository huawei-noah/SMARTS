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
import functools
from typing import Any
from types import FunctionType
from threading import RLock

_CACHE_KEY_PREFIX = "_cache_decorator"


# Taken from https://git.io/JI4PW
class _HashedSeq(list):
    """This class guarantees that hash() will be called no more than once per
    element.  This is important because the lru_cache() will hash the key multiple
    times on a cache miss.
    """

    __slots__ = "hashvalue"

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


# Taken from https://git.io/JI4PW
def _make_key(
    args,
    kwds,
    typed=False,
    kwd_mark=(object(),),
    fasttypes={int, str},
    tuple=tuple,
    type=type,
    len=len,
):
    """Make a cache key from optionally typed positional and keyword arguments. The key
    is constructed in a way that is flat as possible rather than as a nested structure
    that would take more memory. If there is only a single argument and its data type
    is known to cache its hash value, then that argument is returned without a wrapper.
    This saves space and improves lookup speed.
    """
    # All of code below relies on kwds preserving the order input by the user.
    # Formerly, we sorted() the kwds before looping.  The new way is *much*
    # faster; however, it means that f(x=1, y=2) will now be treated as a
    # distinct call from f(y=2, x=1) which will be cached separately.
    key = args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for v in kwds.values())
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


# Inspired by https://strandmark.wordpress.com/2018/10/01/clearable-per-instance-method-cache-in-python/
class _CacheCallable:
    def __init__(self, cache_key: str, method: FunctionType, instance: Any):
        self._method = method
        self._instance = instance
        self._cache = instance.__dict__
        self._cache_key = cache_key
        self._lock = RLock()

    def __call__(self, *args, **kwargs) -> Any:
        cached = self._cache.get(self._cache_key, {})

        key = _make_key(args, kwargs)
        if key not in cached:
            with self._lock:
                # Check if another thread filled cache while we awaited lock
                cached = self._cache.get(self._cache_key, {})
                if key not in cached:
                    cached[key] = self._method(self._instance, *args, **kwargs)
                    self._cache[self._cache_key] = cached

        return cached[key]

    def clear_cache(self):
        _CacheCallable.external_clear_cache(self._instance, self._cache_key)

    @staticmethod
    def external_clear_cache(instance, cache_key):
        setattr(instance, cache_key, {})


class cache:
    def __init__(self, method: FunctionType):
        self._method = method
        self._cache_key = f"{_CACHE_KEY_PREFIX}_{method.__name__}"

    def __get__(self, instance: Any, _):
        assert instance, "Method must be called on an object"
        return _CacheCallable(self._cache_key, self._method, instance)


def clear_cache(func):
    def _clear_caches(self):
        for key in self.__dict__:
            if key.startswith(_CACHE_KEY_PREFIX):
                _CacheCallable.external_clear_cache(self, key)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        _clear_caches(self)
        return func(self, *args, **kwargs)

    return wrapper
