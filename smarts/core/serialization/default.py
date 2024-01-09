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
from typing import Any

_proxies = {}


def dumps(__o):
    """Serializes the given object."""
    import cloudpickle

    _lazy_init()
    r = __o
    type_ = type(__o)
    # TODO: Add a formatter parameter instead of handling proxies internal to serialization
    proxy_func = _proxies.get(type_)
    if proxy_func:
        r = proxy_func(__o)
    return cloudpickle.dumps(r)


def loads(__o):
    """Deserializes the given object."""
    import cloudpickle

    r = cloudpickle.loads(__o)
    if hasattr(r, "deproxy"):
        r = r.deproxy()
    return r


class Proxy:
    """Defines a proxy object used to facilitate serialization of a non-serializable object."""

    def deproxy(self):
        """Convert the proxy back into the original object."""
        raise NotImplementedError()


@dataclass(frozen=True)
class _SimulationLocalConstantsProxy(Proxy):
    road_map_spec: Any
    road_map_hash: int

    def __eq__(self, __o: object) -> bool:
        if __o is None:
            return False
        return self.road_map_hash == getattr(__o, "road_map_hash")

    def deproxy(self):
        import smarts.sstudio.sstypes
        from smarts.core.simulation_local_constants import SimulationLocalConstants

        assert isinstance(self.road_map_spec, smarts.sstudio.sstypes.MapSpec)
        road_map, _ = self.road_map_spec.builder_fn(self.road_map_spec)
        return SimulationLocalConstants(road_map, self.road_map_hash)


def _proxy_slc(v):
    return _SimulationLocalConstantsProxy(v.road_map.map_spec, v.road_map_hash)


def _lazy_init():
    if len(_proxies) > 0:
        return
    from smarts.core.simulation_local_constants import SimulationLocalConstants

    _proxies[SimulationLocalConstants] = _proxy_slc
